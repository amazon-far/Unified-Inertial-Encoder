"""Corruption transforms for the joint stream of root-IMU + joint samples (``JointTrait``)."""
from __future__ import annotations

import math
import torch
import random
import pypose as pp
import jaxtyping as jt

from ...interface.sequence   import ISequence
from ...interface.data_joint import JointTrait
from ...interface.data_imu   import IMUData_w_GT
from .transform import _window_dt_s, _intensity_modulation


def _apply(sample: JointTrait, d_angle: torch.Tensor | None, d_rate: torch.Tensor | None) -> None:
    """Add an angle and/or rate perturbation, each (1, T, J, 3), honouring the joint kind.

    ``so3``: the angle applies in the CHILD frame, ``q' = q * Exp(e)``, matching ``dq``; the rate
    is additive. ``revolute``: both are additive on the first channel. Out-of-place on the fields,
    since the source may hand out views of its storage.
    """
    match sample.joint_kind:
        case "so3":
            if d_angle is not None:
                q = pp.SO3(sample.joint_q.float()) @ pp.so3(d_angle.to(torch.float32)).Exp()
                sample.joint_q = q.tensor().to(sample.joint_q.dtype)
            if d_rate is not None:
                sample.joint_dq = sample.joint_dq + d_rate.to(sample.joint_dq.dtype)
        case "revolute":
            if d_angle is not None:
                sample.joint_q = sample.joint_q + d_angle[..., :1].to(sample.joint_q.dtype)
            if d_rate is not None:
                sample.joint_dq = sample.joint_dq + d_rate[..., :1].to(sample.joint_dq.dtype)
        case other:
            raise ValueError(f"unknown joint kind {other!r}")


def _dt_s(sample: JointTrait) -> float:
    """Per-window sample spacing off the (pelvis) clock the joints share."""
    time_ns = getattr(sample, "time_ns")                # IMUJointData carries the IMU fields too
    return _window_dt_s(time_ns[:, :1])


def _rate_clean(sample: JointTrait) -> jt.Float[torch.Tensor, "1 J T 3"]:
    """Joint rates as (1, J, T, 3), J as devices, for the motion envelope; revolute is zero-padded."""
    dq = sample.joint_dq.float()
    if dq.shape[-1] == 1:
        dq = torch.cat([dq, dq.new_zeros(*dq.shape[:-1], 2)], dim=-1)
    return dq.permute(0, 2, 1, 3)


class _JointTransform(ISequence[IMUData_w_GT]):
    """Base joint corruption: touches only joint fields; samples without joints pass through.

    Place after the motion augmentations and, since rates are read off ``time_ns``, before
    ``AddTimestampJitter``.
    """
    def __init__(self, source: ISequence[IMUData_w_GT]):
        self.source = source

    def __len__(self) -> int: return len(self.source)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]: return self.source.timestamps

    def corrupt(self, sample: JointTrait, index: slice) -> None:
        raise NotImplementedError

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        if isinstance(index, int):
            index = slice(index, index + 1)
        raw_data = self.source[index]
        if isinstance(raw_data, JointTrait) and raw_data.joint_q.shape[1] > 0:
            self.corrupt(raw_data, index)
        return raw_data


class AddJointNoise(_JointTransform):
    """White noise on joint angle and rate, discretized as ``density / sqrt(dt)`` like ``AddNoise``.

    Args:
        angle_noise_density: rad/sqrt(Hz).
        rate_noise_density : (rad/s)/sqrt(Hz).
    """
    def __init__(self, source: ISequence[IMUData_w_GT], angle_noise_density: float, rate_noise_density: float):
        super().__init__(source)
        dt = source.sample_dt_s
        self.angle_std = angle_noise_density / math.sqrt(dt)
        self.rate_std  = rate_noise_density  / math.sqrt(dt)

    def corrupt(self, sample: JointTrait, index: slice) -> None:
        shape = (*sample.joint_q.shape[:3], 3)
        _apply(sample,
               torch.randn(shape) * self.angle_std,
               torch.randn(shape) * self.rate_std)


class AddJointBias(_JointTransform):
    """Drifting encoder offset and rate bias: interpolated random-walk knots, like ``AddBias``.

    Args:
        angle_bias_randwalk: rad/sqrt(s).
        rate_bias_randwalk : (rad/s)/sqrt(s).
        angle_bias_init_std: rad, std of the per-joint constant offset; 0 disables it.
        knot_period_s: knot spacing in seconds.
    """
    def __init__(
        self,
        source: ISequence[IMUData_w_GT],
        angle_bias_randwalk: float,
        rate_bias_randwalk : float,
        angle_bias_init_std: float = 0.0,
        knot_period_s      : float = 1.0,
    ):
        super().__init__(source)
        self.approx_dt   = source.sample_dt_s
        self.knot_stride = max(1, round(knot_period_s / self.approx_dt))
        knot_dt          = self.knot_stride * self.approx_dt
        n_knots          = (len(source) - 1) // self.knot_stride + 2
        # Joint count is a property of the sample, so knots are drawn on first fetch, per process.
        self._n_knots  = n_knots
        self._angle_step = angle_bias_randwalk * math.sqrt(knot_dt)
        self._rate_step  = rate_bias_randwalk  * math.sqrt(knot_dt)
        self._init_std   = angle_bias_init_std
        self._angle_knots: torch.Tensor | None = None   # (J, K, 3)
        self._rate_knots : torch.Tensor | None = None

    def _knots(self, J: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._angle_knots is None or self._angle_knots.shape[0] != J:
            def walk(step: float, start: torch.Tensor) -> torch.Tensor:
                path = torch.empty(J, self._n_knots, 3)
                path[:, 0] = start
                path[:, 1:] = start.unsqueeze(1) + (torch.randn(J, self._n_knots - 1, 3) * step).cumsum(1)
                return path.to(torch.float16)
            self._angle_knots = walk(self._angle_step, torch.randn(J, 3) * self._init_std)
            self._rate_knots  = walk(self._rate_step,  torch.zeros(J, 3))
        assert self._rate_knots is not None
        return self._angle_knots, self._rate_knots

    def _at(self, knots: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """Interpolate the fp16 knots (J, K, 3) to samples [start, stop) -> (1, T, J, 3)."""
        pos  = torch.arange(start, stop, dtype=torch.float32) / self.knot_stride
        i0   = pos.long().clamp_max(knots.shape[1] - 2)
        frac = (pos - i0.float()).view(1, -1, 1)                                  # (1, T, 1)
        k0, k1 = knots[:, i0].float(), knots[:, i0 + 1].float()                   # (J, T, 3)
        return (k0 + (k1 - k0) * frac).permute(1, 0, 2).unsqueeze(0)              # (1, T, J, 3)

    def corrupt(self, sample: JointTrait, index: slice) -> None:
        start, stop, step = index.indices(len(self))
        assert step == 1, "AddJointBias requires contiguous slices"
        T = sample.joint_q.shape[1]
        # The source may have thinned the window: interpolate on the window's own sample count.
        a_knots, r_knots = self._knots(sample.joint_q.shape[2])
        if stop - start != T:
            stop = start + T
        _apply(sample, self._at(a_knots, start, stop), self._at(r_knots, start, stop))


class AddJointVibration(_JointTransform):
    """Colored, motion-modulated vibration on the joint rate, integrated onto the joint angle.

    Same construction as ``AddVibration``.

    Args:
        amp_range      : (low, high) rad/s RMS of the rate vibration, log-uniform per window.
        angle_per_rate : s; angle vibration = rate vibration * this (~1/(2 pi f)).
        freq_range / bandwidth_range / n_modes_range / highpass_hz / smooth_s / mod_floor /
        mod_cap: as ``AddVibration``.
    """
    def __init__(
        self,
        source: ISequence[IMUData_w_GT],
        amp_range       : tuple[float, float] = (0.02, 0.2),
        angle_per_rate  : float = 0.008,
        freq_range      : tuple[float, float] = (5., 60.),
        bandwidth_range : tuple[float, float] = (2., 15.),
        n_modes_range   : tuple[int, int]     = (1, 3),
        highpass_hz     : float = 3.0,
        smooth_s        : float = 0.25,
        mod_floor       : float = 0.15,
        mod_cap         : float = 4.0,
    ):
        super().__init__(source)
        self.amp_range, self.angle_per_rate = amp_range, angle_per_rate
        self.freq_range, self.bandwidth_range, self.n_modes_range = freq_range, bandwidth_range, n_modes_range
        self.highpass_hz, self.smooth_s, self.mod_floor, self.mod_cap = highpass_hz, smooth_s, mod_floor, mod_cap

    def corrupt(self, sample: JointTrait, index: slice) -> None:
        T = sample.joint_q.shape[1]
        if T < 8:
            return
        dt  = _dt_s(sample)
        nyq = 0.5 / dt
        freqs    = torch.fft.rfftfreq(T, d=dt)
        response = torch.zeros_like(freqs)
        for _ in range(random.randint(*self.n_modes_range)):
            f0 = min(random.uniform(*self.freq_range), 0.9 * nyq)
            bw = random.uniform(*self.bandwidth_range)
            response = response + (1. + ((freqs - f0) / bw).square()).rsqrt()
        response = response * (freqs >= self.highpass_hz)
        if not bool(response.any()):
            return

        J   = sample.joint_q.shape[2]
        lo, hi = self.amp_range
        amp = math.exp(random.uniform(math.log(lo), math.log(hi)))
        gain = torch.empty(1, J, 1, 1).uniform_(0.5, 1.5)
        mod  = _intensity_modulation(_rate_clean(sample), dt, self.smooth_s, self.mod_floor, self.mod_cap)   # (1, J, T, 1)

        spec = torch.fft.rfft(torch.randn(1, J, T, 3), dim=2)
        v    = torch.fft.irfft(spec * response.view(1, 1, -1, 1), n=T, dim=2)
        v    = v / (v.std(dim=2, keepdim=True) + 1e-6)
        rate_vib = (amp * gain * mod * v).permute(0, 2, 1, 3)                 # (1, T, J, 3)
        _apply(sample, self.angle_per_rate * rate_vib, rate_vib)


class AddJointImpact(_JointTransform):
    """Sparse ground-impact transients on the joint rate, integrated onto the joint angle.

    Same construction as ``AddImpact``, with one causal ring-down kernel per window.

    Args:
        rate_range  : events/s ~U per window (Poisson counts).
        amp_range   : rad/s PEAK base amplitude, log-uniform; each event x exp(0.6 N(0,1)).
        decay_range : s, ring-down time constant.
        freq_range  : Hz, ring frequency (clamped to 90% Nyquist).
        angle_per_rate: s; angle transient = rate transient * this.
        smooth_s / weight_floor: event-placement envelope, as ``AddImpact``.
    """
    def __init__(
        self,
        source: ISequence[IMUData_w_GT],
        rate_range     : tuple[float, float] = (0.5, 4.0),
        amp_range      : tuple[float, float] = (0.3, 3.0),
        decay_range    : tuple[float, float] = (0.015, 0.05),
        freq_range     : tuple[float, float] = (8., 30.),
        angle_per_rate : float = 0.008,
        smooth_s       : float = 0.25,
        weight_floor   : float = 0.1,
    ):
        super().__init__(source)
        self.rate_range, self.amp_range, self.decay_range, self.freq_range = rate_range, amp_range, decay_range, freq_range
        self.angle_per_rate, self.smooth_s, self.weight_floor = angle_per_rate, smooth_s, weight_floor

    def corrupt(self, sample: JointTrait, index: slice) -> None:
        T = sample.joint_q.shape[1]
        if T < 8:
            return
        dt  = _dt_s(sample)
        nyq = 0.5 / dt
        n_events = int(torch.poisson(torch.tensor(random.uniform(*self.rate_range) * T * dt)).item())
        if n_events == 0:
            return
        J = sample.joint_q.shape[2]

        # Motion-weighted event times (clean joint-rate envelope, pooled over joints).
        mod = _intensity_modulation(_rate_clean(sample), dt, self.smooth_s, self.weight_floor, 1e6)   # (1, J, T, 1)
        w   = mod.mean(dim=1).reshape(T).clamp_min(1e-6)
        t_ev = torch.multinomial(w, n_events, replacement=True)                # (E,)

        lo, hi = self.amp_range
        base = math.exp(random.uniform(math.log(lo), math.log(hi)))
        amp  = base * torch.exp(0.6 * torch.randn(n_events))                   # heavy tail
        dirs = torch.nn.functional.normalize(torch.randn(n_events, J, 3), dim=-1)
        gain = torch.empty(J, 1).uniform_(0.5, 1.5)

        impulse = torch.zeros(T, J, 3)
        impulse.index_add_(0, t_ev, amp.view(-1, 1, 1) * dirs * gain)         # (T, J, 3)

        tau = random.uniform(*self.decay_range)
        f0  = min(random.uniform(*self.freq_range), 0.9 * nyq)
        K   = min(T, max(8, int(round(5. * tau / dt))))
        tk  = torch.arange(K, dtype=torch.float32) * dt
        kernel = torch.exp(-tk / tau) * torch.sin(2. * math.pi * f0 * tk)      # causal ring-down
        x   = impulse.permute(1, 2, 0).reshape(1, J * 3, T)
        y   = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (K - 1, 0)), kernel.view(1, 1, -1).expand(J * 3, 1, K), groups=J * 3)
        rate_imp = y.reshape(J, 3, T).permute(2, 0, 1).unsqueeze(0)            # (1, T, J, 3)
        _apply(sample, self.angle_per_rate * rate_imp, rate_imp)
