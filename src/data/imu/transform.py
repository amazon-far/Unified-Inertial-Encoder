"""IMU sequence transforms: gravity, sensor corruption, device dropping and motion augmentation."""

import math
import torch
import random
import pypose as pp
import jaxtyping as jt

from typing import Sequence

from ...interface.sequence import ISequence
from ...interface.data_joint import JointTrait
from ...interface.data_imu import IMUData_w_GT
from ...interface.human    import T_CANONICAL_NAMES, CANONICAL_NAMES, CANONICAL_IDX


def _exclude_ids(exclude_devices: Sequence[str]) -> torch.Tensor:
    """Canonical ids (uint8) for ``exclude_devices``; empty means "exclude nothing" (no-op)."""
    unknown = [n for n in exclude_devices if n not in CANONICAL_IDX]
    assert not unknown, f"unknown canonical device name(s): {unknown}"
    # Enumerate: a plain str cannot index the literal-keyed CANONICAL_IDX.
    exclude = set(exclude_devices)
    return torch.tensor([i for i, n in enumerate(CANONICAL_NAMES) if n in exclude], dtype=torch.uint8)


def _include_multiplier(raw_data: IMUData_w_GT, exclude_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(1, N, 1, 1) multiplier: 0 for devices whose canonical id is in ``exclude_ids``, else 1."""
    dev_ids = raw_data.sensor_id[0, :, 0, 0]                 # (N,) canonical id per device
    return (~torch.isin(dev_ids, exclude_ids)).to(dtype).view(1, -1, 1, 1)


def _rescale_kinematic(
    acc           : jt.Float[torch.Tensor, "1 N T 3"],
    gravity_world : jt.Float[torch.Tensor, "1 N 3"],
    R_wb          : pp.LieTensor,                        # SO3 (1 N T), body->world
    c             : float,
) -> jt.Float[torch.Tensor, "1 N T 3"]:
    """Scale ONLY the kinematic part of a body-frame acceleration by ``c``, gravity held fixed.

    With ``a_kin_world = R_wb @ acc + g_world``, this is ``c * acc + (c - 1) * (R_bw @ g_world)``.
    Only the fp32 gravity vector is rotated: ``acc`` may be fp16, which the SO3 Act rejects.
    """
    g_body = R_wb.Inv() @ gravity_world.unsqueeze(2)           # world gravity in body frame (1,N,T,3), fp32
    return c * acc + (c - 1.0) * g_body                        # promotes to fp32, as AddGravity also does


def _lowpass_taps(cutoff_norm: float, num_taps: int) -> jt.Float[torch.Tensor, "K"]:
    """Hamming-windowed-sinc lowpass, unit DC gain.

    ``cutoff_norm`` is the -6 dB cutoff as a fraction of the SAMPLE RATE (Nyquist = 0.5).
    ``num_taps`` must be odd, so that centred convolution cancels the group delay exactly.
    """
    assert num_taps % 2 == 1, f"num_taps must be odd to stay linear-phase, got {num_taps}"
    n = torch.arange(num_taps, dtype=torch.float32) - (num_taps - 1) / 2
    h = 2. * cutoff_norm * torch.special.sinc(2. * cutoff_norm * n)
    h = h * torch.hamming_window(num_taps, periodic=False, dtype=torch.float32)
    return h / h.sum()


def _lowpass(x: jt.Float[torch.Tensor, "B N T C"], taps: jt.Float[torch.Tensor, "K"]) -> jt.Float[torch.Tensor, "B N T C"]:
    """Apply ``taps`` along time, reflect-padded: zero padding would ring a gravity-sized step."""
    B, N, T, C = x.shape
    pad = (taps.numel() - 1) // 2
    y   = x.permute(0, 1, 3, 2).reshape(B * N * C, 1, T)
    y   = torch.nn.functional.pad(y, (pad, pad), mode="reflect")
    y   = torch.nn.functional.conv1d(y, taps.view(1, 1, -1).to(y.dtype))
    return y.reshape(B, N, C, T).permute(0, 1, 3, 2)


class AddGravity(ISequence[IMUData_w_GT]):
    """Add gravity to a kinematic acceleration, giving the specific force an accelerometer measures.

    Args:
        source: Upstream sequence with ``kind == "filtered-no-g"``.
        gravity_range: ``(low, high)`` bounds, in m/s^2; |g| is drawn once per instance.
    """
    def __init__(self, source: ISequence[IMUData_w_GT], gravity_range: tuple[float, float]=(9.76, 9.83)):
        self.source  = source

        self.g       = random.uniform(*gravity_range)
        self.gravity = torch.tensor([0., 0., self.g], dtype=torch.float32).view(1, 1, 1, 3)

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        raw_data = self.source[index]
        assert raw_data.kind == "filtered-no-g", f"Doesn't make sense to add gravity on {raw_data.kind=} since g should already exist."

        acc_w_g = raw_data.acc + raw_data.gt_pose.rotation().Inv() @ self.gravity
        raw_data.acc            = acc_w_g
        # `correct_acc` stays kinematic, so `correct_gravity` keeps the source's (zero) value.
        raw_data.gravity        = torch.tensor(
            [0., 0., -self.g], dtype=raw_data.gravity.dtype, device=raw_data.gravity.device,
        ).expand_as(raw_data.gravity).contiguous()
        raw_data.kind           = "sim"
        raw_data.correct_mask   = torch.ones_like(raw_data.correct_mask)
        return raw_data

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps


class AddBias(ISequence[IMUData_w_GT]):
    """Add a slowly drifting random-walk bias to acc/gyro and to the GT bias channels.

    The walk is pre-generated as fp16 knots, one per ``knot_period_s``, and linearly interpolated;
    its std grows like ``randwalk_coeff * sqrt(elapsed_time)``. Output ``kind`` becomes ``"sim"``.

    Args:
        source: Upstream sequence to corrupt.
        acc_bias_randwalk: Accelerometer bias random-walk coefficient, in (m/s^2)/sqrt(s).
        gyro_bias_randwalk: Gyroscope bias random-walk coefficient, in (rad/s)/sqrt(s).
        acc_bias_init: Initial accelerometer bias, in m/s^2, per axis.
        gyro_bias_init: Initial gyroscope bias, in rad/s, per axis.
        knot_period_s: Spacing of the stored knots, in seconds.
        exclude_devices: Canonical device names to leave bias-free, e.g. a real measurement the
            other body IMUs are synthesized from.
    """
    def __init__(
        self,
        source: ISequence[IMUData_w_GT],
        acc_bias_randwalk       : float,
        gyro_bias_randwalk      : float,
        acc_bias_init           : tuple[float, float, float]=(0., 0., 0.),
        gyro_bias_init          : tuple[float, float, float]=(0., 0., 0.),
        knot_period_s           : float = 1.0,
        exclude_devices         : Sequence[str] = (),
    ) -> None:
        super().__init__()

        self.exclude_ids = _exclude_ids(exclude_devices)

        self.approx_dt = source.sample_dt_s
        n_device       = source.num_devices

        self.knot_stride = max(1, round(knot_period_s / self.approx_dt))   # samples per knot
        knot_dt          = self.knot_stride * self.approx_dt
        # Cover sample T-1 plus one extra knot so interpolation's i0+1 lookup never overflows.
        n_knots          = (len(source) - 1) // self.knot_stride + 2

        self.acc_bias_step_std = acc_bias_randwalk  * math.sqrt(knot_dt)
        self.gyr_bias_step_std = gyro_bias_randwalk * math.sqrt(knot_dt)

        self.acc_bias_knots = self._generate_3d_brownian_trajectory(
            start        =torch.tensor(acc_bias_init, dtype=torch.float32).view(1, 1, 3).repeat(1, n_device, 1),
            step_std     =self.acc_bias_step_std,
            t_steps      =n_knots,
            n_device     =n_device
        ).to(torch.float16)

        self.gyro_bias_knots = self._generate_3d_brownian_trajectory(
            start        =torch.tensor(gyro_bias_init, dtype=torch.float32).view(1, 1, 3).repeat(1, n_device, 1),
            step_std     =self.gyr_bias_step_std,
            t_steps      =n_knots,
            n_device     =n_device
        ).to(torch.float16)

        self.source = source

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    @staticmethod
    def _generate_3d_brownian_trajectory(
        start           : jt.Float[torch.Tensor, "1 N 3"],
        step_std        : float,
        t_steps : int,
        n_device: int
    ) -> jt.Float[torch.Tensor, "1 N T 3"]:
        sigma = torch.randn((1, n_device, t_steps - 1, 3), dtype=torch.float32) * step_std
        path  = torch.empty(1, n_device, t_steps, 3, dtype=torch.float32)
        path[:, :, 0 ]  = start
        torch.cumsum(sigma, dim=2, out=path[:, :, 1:])
        path[:, :, 1:] += start.unsqueeze(2)
        return path

    def _bias_at(self, knots: jt.Float[torch.Tensor, "1 N K 3"], start: int, stop: int) -> jt.Float[torch.Tensor, "1 N T 3"]:
        """Linearly interpolate the fp16 bias knots to per-sample fp32 values for samples [start, stop)."""
        pos  = torch.arange(start, stop, dtype=torch.float32) / self.knot_stride
        i0   = pos.long()
        frac = (pos - i0.float()).view(1, 1, -1, 1)
        k0   = knots[:, :, i0    ].float()
        k1   = knots[:, :, i0 + 1].float()
        return k0 + (k1 - k0) * frac

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        if isinstance(index, int):
            index = slice(index, index + 1)
        start, stop, step = index.indices(len(self))
        assert step == 1, "AddBias requires contiguous slices"

        raw_data    = self.source[index]
        acc_bias    = self._bias_at(self.acc_bias_knots,  start, stop)
        gyro_bias   = self._bias_at(self.gyro_bias_knots, start, stop)

        if self.exclude_ids.numel():
            include   = _include_multiplier(raw_data, self.exclude_ids, acc_bias.dtype)
            acc_bias  = acc_bias  * include
            gyro_bias = gyro_bias * include

        # Out-of-place: the source may return views of its storage.
        raw_data.acc        = raw_data.acc       + acc_bias
        raw_data.gyro       = raw_data.gyro      + gyro_bias
        raw_data.acc_bias   = raw_data.acc_bias  + acc_bias
        raw_data.gyro_bias  = raw_data.gyro_bias + gyro_bias

        raw_data.correct_mask   = torch.ones_like(raw_data.correct_mask)
        raw_data.kind           = "sim"
        return raw_data


class AddNoise(ISequence[IMUData_w_GT]):
    """Add zero-mean white Gaussian noise to acc/gyro, independently per sample, axis and device.

    Densities discretize to a per-sample std of ``density / sqrt(dt)``, so noise integrated over
    a fixed window is rate-invariant. Output ``kind`` becomes ``"sim"``.

    Args:
        source: Upstream sequence to corrupt.
        acc_noise_density: Accelerometer white-noise density, in (m/s^2)/sqrt(Hz).
        gyro_noise_density: Gyroscope white-noise density, in (rad/s)/sqrt(Hz).
        exclude_devices: Canonical device names to leave noise-free (see ``AddBias``).
    """
    def __init__(
        self,
        source: ISequence[IMUData_w_GT],
        acc_noise_density   : float,
        gyro_noise_density  : float,
        exclude_devices     : Sequence[str] = (),
    ) -> None:
        self.source  = source

        self.exclude_ids = _exclude_ids(exclude_devices)

        dt = source.sample_dt_s

        self.acc_std = acc_noise_density / math.sqrt(dt)
        self.gyr_std = gyro_noise_density / math.sqrt(dt)

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        raw_data = self.source[index]
        acc_noise  = torch.randn_like(raw_data.acc)  * self.acc_std
        gyro_noise = torch.randn_like(raw_data.gyro) * self.gyr_std

        if self.exclude_ids.numel():
            include    = _include_multiplier(raw_data, self.exclude_ids, raw_data.acc.dtype)
            acc_noise  = acc_noise  * include
            gyro_noise = gyro_noise * include

        # Out-of-place: the source may return views of its storage.
        raw_data.acc  = raw_data.acc  + acc_noise
        raw_data.gyro = raw_data.gyro + gyro_noise

        raw_data.correct_mask   = torch.ones_like(raw_data.correct_mask)
        raw_data.kind           = "sim"
        return raw_data


class RandomDropDevice(ISequence[IMUData_w_GT]):
    """Randomly drop whole devices from each sample, as a missing-sensor augmentation.

    Device count and order vary per sample, so consume the result through the jagged path.
    ``sensor_id`` preserves device identity, and ``model`` stays the full device superset.
    Must run after ``AddBias``, whose knots are sized to the full device count.
    """
    def __init__(self, source: ISequence[IMUData_w_GT]):
        self.source = source

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        raw_data    = self.source[index]
        n_device    = raw_data.acc.size(1)
        n_keep      = random.randint(1, n_device)
        keep_idx    = torch.as_tensor(random.sample(range(n_device), n_keep), dtype=torch.long, device=raw_data.acc.device,)

        # Advanced indexing copies, leaving the source intact. `model` carries over unchanged.
        raw_data.time_ns      = raw_data.time_ns[:, keep_idx]
        raw_data.acc          = raw_data.acc[:, keep_idx]
        raw_data.gyro         = raw_data.gyro[:, keep_idx]
        raw_data.correct_mask = raw_data.correct_mask[:, keep_idx]
        raw_data.correct_acc  = raw_data.correct_acc[:, keep_idx]
        raw_data.correct_gyro = raw_data.correct_gyro[:, keep_idx]
        raw_data.gravity         = raw_data.gravity[:, keep_idx]
        raw_data.correct_gravity = raw_data.correct_gravity[:, keep_idx]
        raw_data.acc_bias     = raw_data.acc_bias[:, keep_idx]
        raw_data.gyro_bias    = raw_data.gyro_bias[:, keep_idx]
        raw_data.sensor_id    = raw_data.sensor_id[:, keep_idx]
        raw_data.labels       = raw_data.labels[:, keep_idx]

        # Re-wrap the SE3 so the LieTensor `ltype` survives advanced indexing.
        raw_data.gt_pose      = pp.SE3(raw_data.gt_pose.tensor()[:, keep_idx])
        if raw_data.gt_vel is not None:
            raw_data.gt_vel   = raw_data.gt_vel[:, keep_idx]

        return raw_data


class DeterministicDropDevice(ISequence[IMUData_w_GT]):
    """Prune each sample's devices to a fixed canonical subset, so all corpora share one layout.

    ``sensor_id`` preserves device identity, and ``model`` stays the full device superset.
    Must run after ``AddBias``, whose knots are sized to the full device count.

    Args:
        source: Upstream sequence to prune.
        keep_devices: Canonical site names to retain. ``None`` keeps all but ``torso_t8``, the
            shoulders and the hands.
    """
    def __init__(self, source: ISequence[IMUData_w_GT], keep_devices: Sequence[T_CANONICAL_NAMES] | None = None):
        self.source = source

        if keep_devices is None:
            keep_devices = tuple(
                n for n in CANONICAL_NAMES 
                if n not in (
                    "torso_t8", "left_shoulder", "right_shoulder", "left_hand", "right_hand"
                )
            )
        assert all(n in CANONICAL_IDX for n in keep_devices), \
            f"keep_devices has non-canonical name(s): {[n for n in keep_devices if n not in CANONICAL_IDX]}"

        # uint8 to match `sensor_id`'s dtype for torch.isin below.
        self.keep_ids = torch.tensor([CANONICAL_IDX[n] for n in keep_devices], dtype=torch.uint8)

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        raw_data    = self.source[index]

        dev_ids     = raw_data.sensor_id[0, :, 0, 0]
        keep_mask   = torch.isin(dev_ids, self.keep_ids.to(dev_ids.device))
        keep_idx    = keep_mask.nonzero(as_tuple=True)[0]
        assert keep_idx.numel() > 0, "DeterministicDropDevice pruned every device; keep_devices matched nothing in this sample."

        # Advanced indexing copies, leaving the source intact. `model` carries over unchanged.
        raw_data.time_ns      = raw_data.time_ns[:, keep_idx]
        raw_data.acc          = raw_data.acc[:, keep_idx]
        raw_data.gyro         = raw_data.gyro[:, keep_idx]
        raw_data.correct_mask = raw_data.correct_mask[:, keep_idx]
        raw_data.correct_acc  = raw_data.correct_acc[:, keep_idx]
        raw_data.correct_gyro = raw_data.correct_gyro[:, keep_idx]
        raw_data.gravity         = raw_data.gravity[:, keep_idx]
        raw_data.correct_gravity = raw_data.correct_gravity[:, keep_idx]
        raw_data.acc_bias     = raw_data.acc_bias[:, keep_idx]
        raw_data.gyro_bias    = raw_data.gyro_bias[:, keep_idx]
        raw_data.sensor_id    = raw_data.sensor_id[:, keep_idx]
        raw_data.labels       = raw_data.labels[:, keep_idx]

        # Re-wrap the SE3 so the LieTensor `ltype` survives advanced indexing.
        raw_data.gt_pose      = pp.SE3(raw_data.gt_pose.tensor()[:, keep_idx])
        if raw_data.gt_vel is not None:
            raw_data.gt_vel   = raw_data.gt_vel[:, keep_idx]

        return raw_data


class RandomFrequencyReduction(ISequence[IMUData_w_GT]):
    """Randomly lower each window's IMU sample rate: anti-alias lowpass, then decimate.

    Per window, a target rate ~U(freq_range) gives the integer factor ``m = round(native_hz /
    target_hz)``; every ``m``-th sample is kept from a random phase, and retained stamps pass
    through. A target at or above the native rate is a no-op. ``T`` becomes window-dependent, so
    consume the result through the jagged path. Window planning stays on the source's grid.

    The lowpass at the post-decimation Nyquist mirrors a real sensor's anti-alias filter:
    subsampling alone folds high frequencies in-band and leaves ``sqrt(m)`` too much ``AddNoise``
    noise. It covers ``acc``/``gyro``, ``correct_*`` and the bias channels, but not
    ``gt_pose``/``gt_vel``, which are labels rather than sensor readings.

    Args:
        source: Upstream sequence to decimate.
        freq_range: ``(low, high)`` bounds, in Hz, for the per-window target rate.
        anti_alias: Lowpass before thinning. ``False`` subsamples only, which aliases.
    """
    def __init__(
        self,
        source: ISequence[IMUData_w_GT],
        freq_range: tuple[float, float]=(60., 240.),
        anti_alias: bool = True,
    ):
        self.source     = source
        self.freq_range = freq_range
        self.anti_alias = anti_alias

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        raw_data = self.source[index]
        n_time   = raw_data.acc.size(2)

        dt_s      = float((raw_data.time_ns[:, :, 1:] - raw_data.time_ns[:, :, :-1]).float().mean() * 1e-9)
        native_hz = 1. / dt_s

        # Clamp m so at least two samples survive.
        target_hz = random.uniform(*self.freq_range)
        m         = max(1, round(native_hz / target_hz))
        m         = min(m, max(1, n_time - 1))

        # Post-decimation Nyquist is 0.5/m of the native rate; m == 1 stays an exact no-op.
        if self.anti_alias and m > 1:
            # Taps scale with m to hold the relative transition width. The cap keeps the reflect
            # pad inside the window; 2*n_time-1 stays odd, as _lowpass_taps needs.
            taps = _lowpass_taps(0.5 / m, min(24 * m + 1, 2 * n_time - 1))
            raw_data.acc          = _lowpass(raw_data.acc,          taps)
            raw_data.gyro         = _lowpass(raw_data.gyro,         taps)
            raw_data.correct_acc  = _lowpass(raw_data.correct_acc,  taps)
            raw_data.correct_gyro = _lowpass(raw_data.correct_gyro, taps)
            raw_data.acc_bias     = _lowpass(raw_data.acc_bias,     taps)
            raw_data.gyro_bias    = _lowpass(raw_data.gyro_bias,    taps)

        # Random phase so the retained grid is not always anchored at t0.
        offset    = random.randrange(m)
        sl        = (slice(None), slice(None), slice(offset, None, m))

        # Per-time fields only: `gravity`, `correct_gravity`, `model`, `kind` and `labels` have no
        # time axis. Fields are rebound, never written, so the source stays intact.
        raw_data.time_ns      = raw_data.time_ns[sl]
        raw_data.acc          = raw_data.acc[sl]
        raw_data.gyro         = raw_data.gyro[sl]
        raw_data.correct_mask = raw_data.correct_mask[sl]
        raw_data.correct_acc  = raw_data.correct_acc[sl]
        raw_data.correct_gyro = raw_data.correct_gyro[sl]
        raw_data.acc_bias     = raw_data.acc_bias[sl]
        raw_data.gyro_bias    = raw_data.gyro_bias[sl]
        raw_data.sensor_id    = raw_data.sensor_id[sl]

        # Re-wrap the SE3 so the LieTensor `ltype` survives slicing.
        raw_data.gt_pose      = pp.SE3(raw_data.gt_pose.tensor()[sl])
        if raw_data.gt_vel is not None:
            raw_data.gt_vel   = raw_data.gt_vel[sl]

        # A joint stream shares the IMU sample grid: thin it identically, with no lowpass, since
        # encoder rates do not pass through an ADC anti-alias filter.
        if isinstance(raw_data, JointTrait):
            raw_data.joint_time_subset(slice(offset, None, m))

        return raw_data


class AddTimestampJitter(ISequence[IMUData_w_GT]):
    """Wobble each sample's timestamp about its nominal instant, leaving every value intact.

    Stamps are perturbed, not intervals, so drift stays bounded: ``t'_i = t_i + e_i`` with iid
    ``e_i`` of std ``s`` gives spacing std ``sqrt(2)*s``, hence ``s = cv * dt / sqrt(2)``. ``e_i``
    is clamped to ``+-0.49 * min(dt)`` per device, which keeps stamps strictly increasing; a device
    whose window already has a non-positive ``dt`` is left untouched. Devices jitter independently,
    and each device's first stamp stays fixed.

    Place after every pass that reads a rate off ``time_ns``.

    Args:
        source: Upstream sequence to jitter.
        cv_range: ``(low, high)`` bounds for the per-window spacing CV, ``std(dt)/mean(dt)``.
    """
    def __init__(self, source: ISequence[IMUData_w_GT], cv_range: tuple[float, float]=(0.0, 0.15)):
        self.source   = source
        self.cv_range = cv_range

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        raw_data = self.source[index]
        t        = raw_data.time_ns                        # (1, N, T, 1) int64 ns
        cv       = random.uniform(*self.cv_range)
        if t.size(2) < 3 or cv <= 0.:
            return raw_data

        dt = (t[:, :, 1:] - t[:, :, :-1]).float()
        # Median spacing is the nominal grid; min spacing bounds the perturbation so order holds.
        sigma = (cv / math.sqrt(2.)) * dt.median(dim=2, keepdim=True).values
        bound = 0.49 * dt.min(dim=2, keepdim=True).values
        bound = bound.clamp_min(0.)      # non-positive dt somewhere -> this device stays put

        eps = torch.randn(t.shape, device=t.device, dtype=torch.float32) * sigma
        eps = torch.maximum(torch.minimum(eps, bound), -bound)
        eps[:, :, 0] = 0.                # hold each device's window origin fixed

        raw_data.time_ns = t + eps.round().to(t.dtype)
        return raw_data


class RandomTimeWarp(ISequence[IMUData_w_GT]):
    """Speed augmentation: relabel a window's clock as if the motion ran ``s`` times faster.

    No resampling. Stamps compress about each device's first stamp (``t' = t0 + (t - t0)/s``),
    gyro and ``gt_vel`` scale by ``s``, and the kinematic part of acc scales by ``s²``; ``gt_pose``
    is unchanged.

    Place before ``AddGravity``/``AddNoise``/``AddBias`` so no noise or bias gets scaled. This is
    a motion augmentation, not a corruption: ``correct_mask`` and ``kind`` are left untouched.

    Args:
        source: Upstream pre-corruption sequence to warp.
        speed_range: ``(low, high)`` bounds for the per-window speed factor ``s``.
    """
    def __init__(self, source: ISequence[IMUData_w_GT], speed_range: tuple[float, float]=(0.6, 1.5)):
        self.source      = source
        self.speed_range = speed_range

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        raw_data = self.source[index]
        s        = random.uniform(*self.speed_range)
        c        = s * s
        R_wb     = raw_data.gt_pose.rotation()   # SO3 (1,N,T), body->world

        # Out-of-place: the source may return views of its storage.
        # Divide in float64 for ns precision.
        t0                = raw_data.time_ns[:, :, :1, :]                     # (1,N,1,1)
        delta             = (raw_data.time_ns - t0).to(torch.float64)
        raw_data.time_ns  = t0 + torch.round(delta / s).to(torch.int64)

        raw_data.gyro         = s * raw_data.gyro
        raw_data.correct_gyro = s * raw_data.correct_gyro

        raw_data.acc          = _rescale_kinematic(raw_data.acc,         raw_data.gravity,         R_wb, c)
        raw_data.correct_acc  = _rescale_kinematic(raw_data.correct_acc, raw_data.correct_gravity, R_wb, c)

        if raw_data.gt_vel is not None:
            raw_data.gt_vel = s * raw_data.gt_vel

        if isinstance(raw_data, JointTrait):
            raw_data.joint_time_scale(s)

        return raw_data


def _window_dt_s(time_ns: jt.Int64[torch.Tensor, "1 N T 1"]) -> float:
    """Mean per-sample spacing of a window, in seconds.

    Diff first, THEN cast: epoch-ns stamps are far outside float32's exact range.
    """
    return float((time_ns[:, :, 1:] - time_ns[:, :, :-1]).float().mean() * 1e-9)


def _intensity_modulation(
    gyro_clean : jt.Float[torch.Tensor, "1 N T 3"],
    dt         : float,
    smooth_s   : float,
    floor      : float,
    cap        : float,
) -> jt.Float[torch.Tensor, "1 N T 1"]:
    """Motion-intensity envelope from the clean angular rate, normalized to ~unit mean per device.

    Lowpassed over ~``smooth_s``, clamped to ``cap``, and floored so quiet stretches keep a
    residual disturbance.
    """
    env = gyro_clean.float().norm(dim=-1, keepdim=True)                     # (1,N,T,1)
    T   = env.size(2)
    num_taps = min(max(int(round(2. * smooth_s / dt)) | 1, 3), 2 * T - 1)
    env = _lowpass(env, _lowpass_taps(dt / smooth_s, num_taps))
    # Side-lobes undershoot zero next to bursts; the clamp keeps a valid amplitude and a valid
    # multinomial weight for AddImpact.
    m   = (env / (env.mean(dim=2, keepdim=True) + 1e-6)).clamp(min=0., max=cap)
    return floor + (1. - floor) * m


class AddVibration(ISequence[IMUData_w_GT]):
    """Add colored, motion-modulated vibration noise, mimicking a real robot's structural hum.

    Per window, Lorentzian resonance modes shape unit-RMS noise, which is modulated by the
    clean-gyro envelope and added to ``acc``/``gyro``. The response is zero below ``highpass_hz``,
    where the velocity signal lives. ``correct_*`` stay clean; output ``kind`` becomes ``"sim"``.

    Place after the motion augmentations and before any decimation.

    Args:
        source: Upstream sequence to corrupt.
        amp_range: ``(low, high)`` m/s² RMS of the per-window acc vibration, log-uniform.
        gyro_per_acc: ``(low, high)`` gyro-to-acc ratio, in (rad/s)/(m/s²), ~U per window.
        freq_range: ``(low, high)`` Hz of each mode's center (clamped to 90% of Nyquist).
        bandwidth_range: ``(low, high)`` Hz of each mode's Lorentzian half-width.
        n_modes_range: ``(low, high)`` inclusive number of modes per window.
        highpass_hz: Frequencies below this get zero response.
        smooth_s / mod_floor / mod_cap: Envelope parameters (see ``_intensity_modulation``).
        exclude_devices: Canonical device names to leave clean (see ``AddBias``).
    """
    def __init__(
        self,
        source: ISequence[IMUData_w_GT],
        amp_range       : tuple[float, float] = (0.3, 3.0),
        gyro_per_acc    : tuple[float, float] = (0.015, 0.05),
        freq_range      : tuple[float, float] = (5., 60.),
        bandwidth_range : tuple[float, float] = (2., 15.),
        n_modes_range   : tuple[int, int]     = (1, 3),
        highpass_hz     : float = 3.0,
        smooth_s        : float = 0.25,
        mod_floor       : float = 0.15,
        mod_cap         : float = 4.0,
        exclude_devices : Sequence[str] = (),
    ) -> None:
        self.source          = source
        self.amp_range       = amp_range
        self.gyro_per_acc    = gyro_per_acc
        self.freq_range      = freq_range
        self.bandwidth_range = bandwidth_range
        self.n_modes_range   = n_modes_range
        self.highpass_hz     = highpass_hz
        self.smooth_s        = smooth_s
        self.mod_floor       = mod_floor
        self.mod_cap         = mod_cap

        self.exclude_ids     = _exclude_ids(exclude_devices)

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    def _colored_unit_noise(
        self, shape: torch.Size, response: jt.Float[torch.Tensor, "F"],
    ) -> jt.Float[torch.Tensor, "1 N T 3"]:
        """Unit-RMS (per device-axis) Gaussian noise shaped by ``response`` along time."""
        spec = torch.fft.rfft(torch.randn(shape, dtype=torch.float32), dim=2)
        v    = torch.fft.irfft(spec * response.view(1, 1, -1, 1), n=shape[2], dim=2)
        return v / (v.std(dim=2, keepdim=True) + 1e-6)

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        raw_data = self.source[index]
        T        = raw_data.acc.size(2)

        # Bookkeeping happens even for windows too short to corrupt, so the `correct_mask`/`kind`
        # contract stays uniform.
        raw_data.correct_mask = torch.ones_like(raw_data.correct_mask)
        raw_data.kind         = "sim"
        if T < 8:
            return raw_data

        dt  = _window_dt_s(raw_data.time_ns)
        nyq = 0.5 / dt

        # Sum of Lorentzian magnitude responses; DC-free by the highpass gate.
        freqs    = torch.fft.rfftfreq(T, d=dt)
        response = torch.zeros_like(freqs)
        for _ in range(random.randint(*self.n_modes_range)):
            f0 = min(random.uniform(*self.freq_range), 0.9 * nyq)
            bw = random.uniform(*self.bandwidth_range)
            response = response + (1. + ((freqs - f0) / bw).square()).rsqrt()
        response = response * (freqs >= self.highpass_hz)
        if not bool(response.any()):
            return raw_data

        lo, hi   = self.amp_range
        acc_amp  = math.exp(random.uniform(math.log(lo), math.log(hi)))
        gyro_amp = acc_amp * random.uniform(*self.gyro_per_acc)
        gain     = torch.empty(1, raw_data.acc.size(1), 1, 1).uniform_(0.5, 1.5)
        mod      = _intensity_modulation(raw_data.correct_gyro, dt, self.smooth_s, self.mod_floor, self.mod_cap)

        shape    = torch.Size((1, raw_data.acc.size(1), T, 3))
        acc_vib  = acc_amp  * gain * mod * self._colored_unit_noise(shape, response)
        gyro_vib = gyro_amp * gain * mod * self._colored_unit_noise(shape, response)

        if self.exclude_ids.numel():
            include  = _include_multiplier(raw_data, self.exclude_ids, acc_vib.dtype)
            acc_vib  = acc_vib  * include
            gyro_vib = gyro_vib * include

        # Out-of-place: the source may return views of its storage. fp16 caches promote to fp32.
        raw_data.acc  = raw_data.acc  + acc_vib
        raw_data.gyro = raw_data.gyro + gyro_vib
        return raw_data


class AddImpact(ISequence[IMUData_w_GT]):
    """Add sparse ground-impact transients: damped-sinusoid shocks at motion-weighted times.

    Per window, a Poisson count of impulses with log-normal peaks lands preferentially where the
    clean-gyro envelope is high, pointing along body-frame up plus jitter. The train is convolved
    with one causal ``exp(-t/tau)*sin(2*pi*f*t)`` kernel; a gyro transient is added the same way.

    Ordering and contract as ``AddVibration``.

    Args:
        source: Upstream sequence to corrupt.
        rate_range: ``(low, high)`` events/s of the per-window mean impact rate, ~U.
        amp_range: ``(low, high)`` m/s² PEAK base amplitude, log-uniform; each event multiplies it
            by exp(0.6*N(0,1)).
        decay_range: ``(low, high)`` seconds of the ring-down time constant tau.
        freq_range: ``(low, high)`` Hz of the ring frequency (clamped to 90% of Nyquist).
        gyro_per_acc: ``(low, high)`` peak ratio of the gyro transient, in (rad/s)/(m/s²).
        vertical_jitter: Std of the Gaussian jitter mixed into the up direction; 0 is purely vertical.
        smooth_s / weight_floor: Event-placement envelope smoothing and quiet-time floor.
        exclude_devices: Canonical device names to leave clean (see ``AddBias``).
    """
    def __init__(
        self,
        source: ISequence[IMUData_w_GT],
        rate_range      : tuple[float, float] = (0.5, 4.0),
        amp_range       : tuple[float, float] = (3., 30.),
        decay_range     : tuple[float, float] = (0.015, 0.05),
        freq_range      : tuple[float, float] = (8., 30.),
        gyro_per_acc    : tuple[float, float] = (0.02, 0.08),
        vertical_jitter : float = 0.4,
        smooth_s        : float = 0.25,
        weight_floor    : float = 0.15,
        exclude_devices : Sequence[str] = (),
    ) -> None:
        self.source          = source
        self.rate_range      = rate_range
        self.amp_range       = amp_range
        self.decay_range     = decay_range
        self.freq_range      = freq_range
        self.gyro_per_acc    = gyro_per_acc
        self.vertical_jitter = vertical_jitter
        self.smooth_s        = smooth_s
        self.weight_floor    = weight_floor

        self.exclude_ids     = _exclude_ids(exclude_devices)

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    @staticmethod
    def _causal_ring(
        impulses: jt.Float[torch.Tensor, "1 N T 3"], kernel: jt.Float[torch.Tensor, "K"],
    ) -> jt.Float[torch.Tensor, "1 N T 3"]:
        """Convolve the impulse train with a causal ring-down kernel along time.

        conv1d cross-correlates, so the kernel is flipped and the input left-padded by K-1.
        """
        B, N, T, C = impulses.shape
        x = impulses.permute(0, 1, 3, 2).reshape(B * N * C, 1, T)
        x = torch.nn.functional.pad(x, (kernel.numel() - 1, 0))
        y = torch.nn.functional.conv1d(x, kernel.flip(0).view(1, 1, -1))
        return y.reshape(B, N, C, T).permute(0, 1, 3, 2)

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        raw_data = self.source[index]
        N, T     = raw_data.acc.size(1), raw_data.acc.size(2)

        # Bookkeeping happens even for windows too short to corrupt, so the `correct_mask`/`kind`
        # contract stays uniform.
        raw_data.correct_mask = torch.ones_like(raw_data.correct_mask)
        raw_data.kind         = "sim"
        if T < 8:
            return raw_data

        dt = _window_dt_s(raw_data.time_ns)

        # Poisson event count at a sampled rate; cap only as a runaway guard.
        n_events = int(torch.poisson(torch.tensor(random.uniform(*self.rate_range) * T * dt)))
        n_events = min(n_events, 256)
        if n_events == 0:
            return raw_data

        # Weights are shared across devices (one body, one gait); only relative mass matters.
        mod = _intensity_modulation(raw_data.correct_gyro, dt, self.smooth_s, self.weight_floor, cap=8.0)
        idx = torch.multinomial(mod.mean(dim=(0, 1, 3)), n_events, replacement=True)

        lo, hi = self.amp_range
        amp    = math.exp(random.uniform(math.log(lo), math.log(hi)))
        amp    = amp * torch.exp(0.6 * torch.randn(1, 1, n_events, 1))
        gain   = torch.empty(1, N, 1, 1).uniform_(0.4, 1.0)

        up     = torch.tensor([0., 0., 1.], dtype=torch.float32).view(1, 1, 1, 3)
        R_evt  = pp.SO3(raw_data.gt_pose.rotation().tensor()[:, :, idx])         # (1,N,E)
        d_acc  = R_evt.Inv() @ up + self.vertical_jitter * torch.randn(1, N, n_events, 3)
        d_acc  = d_acc / (d_acc.norm(dim=-1, keepdim=True) + 1e-6)
        d_gyro = torch.randn(1, N, n_events, 3)
        d_gyro = d_gyro / (d_gyro.norm(dim=-1, keepdim=True) + 1e-6)

        acc_imp  = torch.zeros(1, N, T, 3)
        gyro_imp = torch.zeros(1, N, T, 3)
        acc_imp .index_add_(2, idx, amp * gain * d_acc)
        gyro_imp.index_add_(2, idx, amp * gain * random.uniform(*self.gyro_per_acc) * d_gyro)

        # One ring-down kernel per window, unit peak so `amp_range` is in output units.
        tau    = random.uniform(*self.decay_range)
        f_ring = min(random.uniform(*self.freq_range), 0.9 * 0.5 / dt)
        t      = torch.arange(min(int(5. * tau / dt) + 1, T), dtype=torch.float32) * dt
        kernel = torch.exp(-t / tau) * torch.sin(2. * math.pi * f_ring * t)
        kernel = kernel / (kernel.abs().max() + 1e-6)

        acc_imp  = self._causal_ring(acc_imp,  kernel)
        gyro_imp = self._causal_ring(gyro_imp, kernel)

        if self.exclude_ids.numel():
            include  = _include_multiplier(raw_data, self.exclude_ids, acc_imp.dtype)
            acc_imp  = acc_imp  * include
            gyro_imp = gyro_imp * include

        # Out-of-place: the source may return views of its storage. fp16 caches promote to fp32.
        raw_data.acc  = raw_data.acc  + acc_imp
        raw_data.gyro = raw_data.gyro + gyro_imp
        return raw_data


class RandomEmbodimentScale(ISequence[IMUData_w_GT]):
    """Randomly rescale the whole embodiment by a uniform size factor ``k``, physically consistently.

    Makes human and humanoid size regimes overlap in training. Per window, the kinematic part of
    acc (``_rescale_kinematic``), ``gt_vel`` and the pose translation scale by ``k``; gyro, rotation
    and ``time_ns`` are unchanged.

    Place before ``AddGravity``/``AddNoise``/``AddBias`` so no noise or bias gets scaled. This is
    a motion augmentation, not a corruption: ``correct_mask`` and ``kind`` are left untouched.

    Args:
        source: Upstream sequence to rescale.
        scale_range: ``(low, high)`` bounds for the per-window size factor ``k``.
    """
    def __init__(self, source: ISequence[IMUData_w_GT], scale_range: tuple[float, float]=(0.75, 1.25)):
        self.source      = source
        self.scale_range = scale_range

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        raw_data = self.source[index]
        k        = random.uniform(*self.scale_range)

        # Rotation is size-invariant: it is both the rescale frame and the new pose's rotation.
        R_wb = raw_data.gt_pose.rotation()

        # Out-of-place: the source may return views of its storage.
        raw_data.acc         = _rescale_kinematic(raw_data.acc,         raw_data.gravity,         R_wb, k)
        raw_data.correct_acc = _rescale_kinematic(raw_data.correct_acc, raw_data.correct_gravity, R_wb, k)

        # Rebuild the SE3 to preserve the LieTensor ltype.
        Tr = raw_data.gt_pose.translation() * k        # (...,3)
        raw_data.gt_pose = pp.SE3(torch.cat([Tr, R_wb.tensor()], dim=-1))

        if raw_data.gt_vel is not None:
            raw_data.gt_vel = raw_data.gt_vel * k

        return raw_data
