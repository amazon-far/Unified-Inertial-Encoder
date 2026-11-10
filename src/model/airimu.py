"""AirIMU re-implemented against this repo's data pipeline: per-interval IMU corrections and
measurement covariances, supervised through IMU preintegration."""

import math

from dataclasses import dataclass

import torch
import torch.nn as nn
import pypose as pp
import jaxtyping as jt
import typing as tp

from typing import TYPE_CHECKING

from ..interface.data_imu import IMUData_w_GT_Jagged, GRAVITY_STANDARD


class _CNNEncoder(nn.Module):
    """Port of ``thirdparty/airimu/model/cnn.py:CNNEncoder`` with the official CodeNet arguments."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        for c_in, c_out in [(6, 32), (32, 64)]:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=7, stride=3, padding=3),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
                nn.Dropout(0.1),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class AirIMU_Output:
    # Chunk-end integrated states (C chunks) and the propagated 9x9 rot/vel/pos covariance.
    pos     : jt.Float[torch.Tensor, "n_keep C 3"]
    vel     : jt.Float[torch.Tensor, "n_keep C 3"]
    rot     : pp.LieTensor                          # SO3 [n_keep C 4]
    cov     : jt.Float[torch.Tensor, "n_keep C 9 9"]
    # Row index (within each run) of each chunk-end state, for gathering GT.
    state_idx: jt.Int64[torch.Tensor, "C"]
    run_keep : jt.Bool[torch.Tensor, "pack_BN"]


class AirIMU(nn.Module):
    INTERVAL = 9          # one shared correction per 9-sample block (official CodeNet)
    SAMPLING = 50         # integration chunk length of the official training recipe

    def __init__(self, window_size: int, acc_std: float = 0.1, gyro_std: float = math.pi / 180):
        super().__init__()
        self.window_size = int(window_size)
        self.register_buffer("acc_std" , torch.tensor(acc_std))
        self.register_buffer("gyro_std", torch.tensor(gyro_std))

        self.cnn  = _CNNEncoder()
        self.gru1 = nn.GRU(input_size=64 , hidden_size=128, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)

        def head() -> nn.Sequential:
            return nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.accdecoder      = head()
        self.acccov_decoder  = head()
        self.gyrodecoder     = head()
        self.gyrocov_decoder = head()

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.accdecoder[0].weight.dtype  # type: ignore[union-attr]

    def configure_trainable_params(self) -> None:
        """Everything always trains; exists only for parity with ``FoundationIMU``."""

    def _encoder(self, x: jt.Float[torch.Tensor, "B T 6"]) -> jt.Float[torch.Tensor, "B F 256"]:
        x = self.cnn(x.transpose(-1, -2)).transpose(-1, -2)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x

    def _spread(self, feat: jt.Float[torch.Tensor, "B F 3"], frame_len: int) -> jt.Float[torch.Tensor, "B frame_len 3"]:
        """Official ``CodeNet._update``: feature ``i`` covers samples ``[9i - 4, 9i + 5)`` (clipped)."""
        head = self.INTERVAL // 2
        tail = self.INTERVAL - head
        out  = feat.new_zeros(feat.shape[0], frame_len, 3)
        n_blk = math.ceil((frame_len - head) / self.INTERVAL) + 1
        for i in range(n_blk):
            s = min(max(i * self.INTERVAL - head, 0), frame_len)
            e = min(max(i * self.INTERVAL + tail, 0), frame_len)
            idx = min(max(i, 0), feat.shape[1] - 1)
            out[:, s:e, :] += feat[:, idx : idx + 1, :]
        return out

    def correct(
        self, acc: jt.Float[torch.Tensor, "B T 3"], gyro: jt.Float[torch.Tensor, "B T 3"]
    ) -> dict[str, torch.Tensor]:
        """Official ``CodeNet.inference``: corrections + covariances for samples
        ``[INTERVAL:]`` of the input window (the first block has no output)."""
        frame_len = acc.shape[1] - self.INTERVAL
        feat = self._encoder(torch.cat([acc, gyro], dim=-1))[:, 1:, :]   # skip first padded feature
        corr_acc  = self._spread(self.accdecoder (feat) * self.acc_std , frame_len)
        corr_gyro = self._spread(self.gyrodecoder(feat) * self.gyro_std, frame_len)
        acc_cov   = self._spread(torch.exp(self.acccov_decoder (feat) - 5.0), frame_len)
        gyro_cov  = self._spread(torch.exp(self.gyrocov_decoder(feat) - 5.0), frame_len)
        return {
            "corrected_acc" : acc [:, self.INTERVAL:, :] + corr_acc,
            "corrected_gyro": gyro[:, self.INTERVAL:, :] + corr_gyro,
            "acc_cov"       : acc_cov,
            "gyro_cov"      : gyro_cov,
        }

    @torch.inference_mode()
    def correct_stream(
        self, acc: jt.Float[torch.Tensor, "L 3"], gyro: jt.Float[torch.Tensor, "L 3"],
        chunk: int = 200_000, pad: int = 2_400,
    ) -> dict[str, torch.Tensor]:
        """``correct`` over one stream in overlapping chunks (cuDNN rejects GRU inputs beyond ~60k
        timesteps). Returns rows aligned to samples ``[INTERVAL:L]``."""
        L   = acc.shape[0]
        out: dict[str, list[torch.Tensor]] = {}
        for c0 in range(0, L, chunk):
            c1 = min(c0 + chunk, L)
            p0 = max(c0 - pad, 0)
            p1 = min(c1 + pad, L)
            st = self.correct(acc[None, p0:p1], gyro[None, p0:p1])
            # st rows cover global samples [p0 + INTERVAL, p1); keep [c0, c1).
            lo = max(c0, p0 + self.INTERVAL)
            s, e = lo - (p0 + self.INTERVAL), c1 - (p0 + self.INTERVAL)
            for k, v in st.items():
                out.setdefault(k, []).append(v[0, s:e])
        return {k: torch.cat(v, dim=0) for k, v in out.items()}

    def body_features(
        self, batch: IMUData_w_GT_Jagged, row_keep: jt.Bool[torch.Tensor, "pack"], run_keep: jt.Bool[torch.Tensor, "N"], T: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row body-frame ``(acc, gyro)`` for the kept runs, fp32, with the acc
        completed to the body-frame specific force (same convention as ``AirIO``)."""
        R_wb = pp.SO3(batch.gt_pose.rotation().tensor()[row_keep])
        g_up = torch.tensor([0.0, 0.0, GRAVITY_STANDARD], device=batch.acc.device)
        g_w  = (batch.gravity.float()[run_keep] + g_up).repeat_interleave(T, dim=0)
        acc  = batch.acc.float()[row_keep] + tp.cast(torch.Tensor, R_wb.Inv() @ g_w).float()
        return acc, batch.gyro.float()[row_keep]

    def forward(self, batch: IMUData_w_GT_Jagged) -> AirIMU_Output:
        run_len  = batch.offsets_N.diff()
        run_keep = run_len == self.window_size
        assert bool(run_keep.any()), (
            f"no device run has the {self.window_size} samples AirIMU was built for "
            f"(set data_conf.window_unit=sample, window_size={self.window_size}); "
            f"got run lengths {run_len.unique().tolist()}"
        )
        row_keep = torch.repeat_interleave(run_keep, run_len)
        n, T, I  = int(run_keep.sum()), self.window_size, self.INTERVAL

        acc, gyro = self.body_features(batch, row_keep, run_keep, T)
        state = self.correct(
            acc.reshape(n, T, 3).to(self.compute_dtype),
            gyro.reshape(n, T, 3).to(self.compute_dtype),
        )

        # Frames [I : T-1] integrate with dt_t = t_{t+1} - t_t; the state after frame I+k lands on
        # sample I+1+k. Integration runs in fp64, the network stays fp32. dt is floored at 1 us: some
        # caches carry duplicate or backward timestamps, and pypose's covariance propagation returns
        # NaN for dt <= 0.
        t_ns   = batch.time_ns[:, 0][row_keep].reshape(n, T)
        dt     = (t_ns[:, I:] .diff(dim=-1).double() * 1e-9).clamp_min(1e-6).unsqueeze(-1)
        gt_R   = pp.SO3(batch.gt_pose.rotation().tensor()[row_keep].reshape(n, T, 4).double())
        gt_p   = batch.gt_pose.translation()[row_keep].reshape(n, T, 3).double()
        gt_v_w = (gt_R @ batch.gt_vel[row_keep].reshape(n, T, 3).double())          # body -> world

        integrator = pp.module.IMUPreintegrator(
            gravity=GRAVITY_STANDARD, prop_cov=True, reset=True
        ).to(dt.device).double()

        init_state: dict = {
            "pos": gt_p  [:, I : I + 1],
            "rot": gt_R  [:, I : I + 1],
            "vel": gt_v_w[:, I : I + 1],
        }
        F = T - I - 1
        pos, vel, rot, cov, idx = [], [], [], [], []
        inte_state = None
        for s in range(0, F, self.SAMPLING):
            if F - s < self.SAMPLING:   # official recipe drops the partial tail chunk
                continue
            e   = s + self.SAMPLING
            sel = slice(s, e)
            if inte_state is not None:
                init_state = {
                    "pos": inte_state["pos"][:, -1:],
                    "vel": inte_state["vel"][:, -1:],
                    "rot": inte_state["rot"][:, -1:],
                    "Rij": inte_state["Rij"],
                    "cov": inte_state["cov"],
                }
            inte_state = integrator(
                init_state = init_state,
                dt         = dt[:, sel],
                gyro       = state["corrected_gyro"][:, sel].double(),
                acc        = state["corrected_acc" ][:, sel].double(),
                rot        = gt_R[:, I + s : I + e],                     # gtrot: True
                gyro_cov   = state["gyro_cov"][:, sel].double(),
                acc_cov    = state["acc_cov" ][:, sel].double(),
            )
            pos.append(inte_state["pos"][:, -1])
            vel.append(inte_state["vel"][:, -1])
            rot.append(inte_state["rot"].tensor()[:, -1])
            cov.append(inte_state["cov"])
            idx.append(I + e)     # integrated state after frame I+e-1 == sample I+e

        return AirIMU_Output(
            pos       = torch.stack(pos, dim=1),
            vel       = torch.stack(vel, dim=1),
            rot       = pp.SO3(torch.stack(rot, dim=1)),
            cov       = torch.stack([c.squeeze(1) if c.dim() == 4 else c for c in cov], dim=1),
            state_idx = torch.tensor(idx, dtype=torch.int64, device=dt.device),
            run_keep  = run_keep,
        )


class AirIMU_Loss(nn.Module):
    """Official AirIMU loss: Huber on chunk-end rotation-Log / velocity / position errors plus the
    Gaussian NLL against the propagated covariance diagonal (errors not detached).

    Returns a graph-connected zero when no run has a finite target (DDP rank consistency).
    """

    def __init__(
        self,
        rot_weight : float = 1.0e3,
        vel_weight : float = 1.0e1,
        pos_weight : float = 1.0e2,
        cov_weight : float = 1.0e-4,
        huber_delta: float = 0.005,
    ):
        super().__init__()
        self.rot_weight  = rot_weight
        self.vel_weight  = vel_weight
        self.pos_weight  = pos_weight
        self.cov_weight  = cov_weight
        self.huber_delta = huber_delta

    def forward(
        self,
        model: AirIMU,
        data : IMUData_w_GT_Jagged,
        pred : AirIMU_Output,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        row_keep = torch.repeat_interleave(pred.run_keep, data.offsets_N.diff())
        n, T     = pred.pos.shape[0], model.window_size

        gt_R   = pp.SO3(data.gt_pose.rotation().tensor()[row_keep].reshape(n, T, 4).double())
        gt_p   = data.gt_pose.translation()[row_keep].reshape(n, T, 3).double()
        gt_v_w = gt_R @ data.gt_vel[row_keep].reshape(n, T, 3).double()

        sel        = pred.state_idx
        targ_p     = gt_p  [:, sel]
        targ_v     = gt_v_w[:, sel]
        targ_R     = pp.SO3(gt_R.tensor()[:, sel])

        rot_dist = (pred.rot @ targ_R.Inv()).Log().tensor()
        vel_dist = pred.vel - targ_v
        pos_dist = pred.pos - targ_p

        valid = torch.cat([rot_dist, vel_dist, pos_dist, targ_p], dim=-1).isfinite().all(-1).all(-1)
        if not bool(valid.any()):
            return (pred.pos.sum() + pred.vel.sum() + pred.cov.sum()).float() * 0.0, {}
        rot_dist, vel_dist, pos_dist = rot_dist[valid], vel_dist[valid], pos_dist[valid]
        cov_diag = torch.diagonal(pred.cov[valid], dim1=-2, dim2=-1)

        def huber(dist: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.huber_loss(dist, torch.zeros_like(dist), delta=self.huber_delta)

        def nll(dist: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
            return ((dist.pow(2) / cov) + cov.log()).mean()

        rot_loss = huber(rot_dist) + self.cov_weight * nll(rot_dist, cov_diag[..., 0:3])
        vel_loss = huber(vel_dist) + self.cov_weight * nll(vel_dist, cov_diag[..., 3:6])
        pos_loss = huber(pos_dist) + self.cov_weight * nll(pos_dist, cov_diag[..., 6:9])
        loss = (self.rot_weight * rot_loss + self.vel_weight * vel_loss + self.pos_weight * pos_loss).float()

        metric = {
            "loss"           : loss.item(),
            "airimu/rot_deg" : rot_dist[:, -1].norm(dim=-1).mean().rad2deg().item(),
            "airimu/vel_mps" : vel_dist[:, -1].norm(dim=-1).mean().item(),
            "airimu/pos_m"   : pos_dist[:, -1].norm(dim=-1).mean().item(),
        }
        return loss, metric

    if TYPE_CHECKING: __call__ = forward  # so pyright sees forward's signature on calls
