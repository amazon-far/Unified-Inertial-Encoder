"""Adapter around the TLIO displacement network vendored at ``thirdparty/tlio``, with its official
augmentation and Gaussian NLL loss re-implemented for this repo's jagged batches."""

import sys
import math
import importlib.util

from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import pypose as pp
import jaxtyping as jt

from typing import TYPE_CHECKING

from ..interface.data_imu import IMUData_w_GT_Jagged, GRAVITY_STANDARD


def _load_tlio_resnet() -> tuple[type[nn.Module], type[nn.Module]]:
    """Load ``(BasicBlock1D, ResNet1D)`` by file location: ``thirdparty/tlio/src`` on ``sys.path``
    would collide on its generic ``network``/``utils`` package names."""
    mod_name = "_tlio_model_resnet"
    if mod_name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "thirdparty/tlio/src/network/model_resnet.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec is not None and spec.loader is not None, f"cannot load TLIO network from {path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[mod_name] = module
    module = sys.modules[mod_name]
    return module.BasicBlock1D, module.ResNet1D


def _fc_in_dim(window_size: int) -> int:
    """Temporal length after ResNet1D's five stride-2 stages, the FC head's ``in_dim`` (TLIO's
    ``window // 32 + 1`` shortcut is only exact for its default 200-sample window)."""
    L = window_size
    for _ in range(5):
        L = (L - 1) // 2 + 1
    return L


@dataclass
class TLIO_Output:
    disp    : jt.Float[torch.Tensor, "n_keep 3"]    # predicted window displacement, augmented world frame [m]
    logstd  : jt.Float[torch.Tensor, "n_keep 3"]    # predicted per-axis log(sigma) of the displacement
    R_aug   : jt.Float[torch.Tensor, "n_keep 3 3"]  # world -> augmented-frame yaw (identity at eval); the loss rotates GT by it
    run_keep: jt.Bool [torch.Tensor, "pack_BN"]     # runs of exactly window_size samples; True count == n_keep


class TLIO(nn.Module):
    def __init__(
        self,
        window_size     : int,
        yaw_aug         : bool  = True,
        grav_perturb_deg: float = 5.0,   # max input tilt [deg]; 0 disables
        accel_bias_range: float = 0.2,   # bias shift U(-r, r) [m/s^2]; 0 disables
        gyro_bias_range : float = 0.05,  # bias shift U(-r, r) [rad/s]; 0 disables
        accel_noise_std : float = 0.0,   # white noise std [m/s^2]
        gyro_noise_std  : float = 0.0,   # white noise std [rad/s]
    ):
        super().__init__()
        self.window_size      = int(window_size)
        self.yaw_aug          = yaw_aug
        self.grav_perturb_deg = grav_perturb_deg
        self.accel_bias_range = accel_bias_range
        self.gyro_bias_range  = gyro_bias_range
        self.accel_noise_std  = accel_noise_std
        self.gyro_noise_std   = gyro_noise_std

        BasicBlock1D, ResNet1D = _load_tlio_resnet()
        # 6-channel [gyro; acc] input, 3D output, [2,2,2,2] groups.
        self.net = ResNet1D(BasicBlock1D, 6, 3, [2, 2, 2, 2], _fc_in_dim(self.window_size))

    @property
    def compute_dtype(self) -> torch.dtype:
        return next(self.net.parameters()).dtype

    def configure_trainable_params(self) -> None:
        """Everything always trains; exists only for parity with ``FoundationIMU``."""

    def _augment_rotations(
        self, n: int, device: torch.device
    ) -> tuple[jt.Float[torch.Tensor, "n 3 3"], jt.Float[torch.Tensor, "n 3 3"]]:
        """Per-run ``(R_yaw, R_tilt)``: yaw U(-pi, pi), applied to input and target, and a tilt of
        angle U(0, grav_perturb_deg) about a random horizontal axis, input only."""
        eye = torch.eye(3, device=device).expand(n, 3, 3)

        if self.yaw_aug:
            yaw  = (torch.rand(n, device=device) * 2 - 1) * math.pi
            c, s = yaw.cos(), yaw.sin()
            z, o = torch.zeros_like(c), torch.ones_like(c)
            R_yaw = torch.stack([c, -s, z, s, c, z, z, z, o], dim=-1).reshape(n, 3, 3)
        else:
            R_yaw = eye

        if self.grav_perturb_deg > 0:
            phi   = torch.rand(n, device=device) * 2 * math.pi
            theta = torch.rand(n, device=device) * math.radians(self.grav_perturb_deg)
            rvec  = theta[:, None] * torch.stack([phi.cos(), phi.sin(), torch.zeros_like(phi)], dim=-1)
            R_tilt = pp.so3(rvec).Exp().matrix()
        else:
            R_tilt = eye

        return R_yaw, R_tilt

    def forward(self, batch: IMUData_w_GT_Jagged) -> TLIO_Output:
        # The conv/FC stack needs a fixed length: keep only runs of exactly window_size samples.
        run_len  = batch.offsets_N.diff()
        run_keep = run_len == self.window_size
        assert bool(run_keep.any()), (
            f"no device run has the {self.window_size} samples TLIO was built for "
            f"(set data_conf.window_unit=sample, window_size={self.window_size}); "
            f"got run lengths {run_len.unique().tolist()}"
        )
        row_keep = torch.repeat_interleave(run_keep, run_len)
        n, T     = int(run_keep.sum()), self.window_size

        # GT orientation rotates the body-frame measurements into the world frame. acc_w is the world-frame
        # specific force: `gravity + (0,0,+g)` vanishes for kind="real" and restores standard gravity
        # for gravity-free channels.
        R_wb   = batch.gt_pose.rotation().matrix().float()[row_keep]
        g_up   = torch.tensor([0.0, 0.0, GRAVITY_STANDARD], device=R_wb.device)
        gyro_w = torch.einsum("pij,pj->pi", R_wb, batch.gyro.float()[row_keep])
        acc_w  = torch.einsum("pij,pj->pi", R_wb, batch.acc.float()[row_keep]) \
               + (batch.gravity.float()[run_keep] + g_up).repeat_interleave(T, dim=0)

        if self.training:
            # Official transform order: bias/noise on the world-frame feats, then the tilt/yaw rotations.
            if self.gyro_bias_range > 0:
                gyro_w = gyro_w + ((torch.rand(n, 3, device=R_wb.device) * 2 - 1)
                                   * self.gyro_bias_range).repeat_interleave(T, dim=0)
            if self.accel_bias_range > 0:
                acc_w  = acc_w  + ((torch.rand(n, 3, device=R_wb.device) * 2 - 1)
                                   * self.accel_bias_range).repeat_interleave(T, dim=0)
            if self.gyro_noise_std > 0:
                gyro_w = gyro_w + torch.randn_like(gyro_w) * self.gyro_noise_std
            if self.accel_noise_std > 0:
                acc_w  = acc_w  + torch.randn_like(acc_w) * self.accel_noise_std

            R_yaw, R_tilt = self._augment_rotations(n, R_wb.device)
            R_in   = (R_yaw @ R_tilt).repeat_interleave(T, dim=0)
            gyro_w = torch.einsum("pij,pj->pi", R_in, gyro_w)
            acc_w  = torch.einsum("pij,pj->pi", R_in, acc_w)
        else:
            R_yaw = torch.eye(3, device=R_wb.device).expand(n, 3, 3)

        x = torch.cat([gyro_w, acc_w], dim=-1).reshape(n, T, 6).transpose(1, 2)  # [n 6 T]
        disp, logstd = self.net(x.to(self.compute_dtype))

        return TLIO_Output(disp=disp, logstd=logstd, R_aug=R_yaw, run_keep=run_keep)


class TLIO_Loss(nn.Module):
    """TLIO's Gaussian NLL on the window displacement, with ``log(sigma)`` clamped from below and
    detached for the first ``logstd_warmup_epochs`` epochs.

    The target is the run's end-minus-start world translation, rotated by the yaw augmentation.
    Returns a graph-connected zero when no run has a finite target (DDP rank consistency).
    """

    def __init__(self, logstd_warmup_epochs: int = 10, min_sigma: float = 1.0e-3):
        super().__init__()
        self.logstd_warmup_epochs = logstd_warmup_epochs
        self.min_log_std          = math.log(min_sigma)
        self.current_epoch        = 0   # refreshed by LitTLIO.on_train_epoch_start

    def forward(
        self,
        model : nn.Module,
        data  : IMUData_w_GT_Jagged,
        pred  : TLIO_Output,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        row_keep = torch.repeat_interleave(pred.run_keep, data.offsets_N.diff())
        pos  = data.gt_pose.translation().float()[row_keep].reshape(pred.disp.shape[0], -1, 3)
        targ = torch.einsum("nij,nj->ni", pred.R_aug, pos[:, -1] - pos[:, 0])

        valid = targ.isfinite().all(dim=-1)
        if not bool(valid.any()):
            return (pred.disp.float().sum() + pred.logstd.float().sum()) * 0.0, {}

        disp   = pred.disp.float()[valid]
        logstd = pred.logstd.float()[valid]
        targ   = targ[valid]

        if self.current_epoch < self.logstd_warmup_epochs:
            logstd = logstd.detach()
        logstd = logstd.clamp(min=self.min_log_std)

        sq_err = (disp - targ).pow(2)
        loss   = (sq_err / (2 * torch.exp(2 * logstd)) + logstd).mean()

        metric = {
            "loss"           : loss.item(),
            "tlio/mse"       : sq_err.mean().item(),
            "tlio/disp_err_m": sq_err.sum(dim=-1).sqrt().mean().item(),
            "tlio/sigma_m"   : logstd.exp().mean().item(),
        }
        return loss, metric

    if TYPE_CHECKING: __call__ = forward  # so pyright sees forward's signature on calls
