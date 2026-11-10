"""Adapter around the AirIO motion network vendored at ``thirdparty/airio``: body-frame velocity
and diagonal covariance from body-frame IMU plus an attitude encoding, with the official loss."""

import sys
import importlib.util

from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import pypose as pp
import jaxtyping as jt
import typing as tp

from typing import TYPE_CHECKING

from ..interface.data_imu import IMUData_w_GT_Jagged, GRAVITY_STANDARD


def _load_airio_codenet() -> type[nn.Module]:
    """Load ``CodeNetMotionwithRot`` by file location: ``thirdparty/airio`` on ``sys.path`` would
    collide on its generic ``model``/``utils`` package names."""
    mod_name = "_airio_model_code"
    if mod_name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "thirdparty/airio/model/code.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec is not None and spec.loader is not None, f"cannot load AirIO network from {path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[mod_name] = module
    return sys.modules[mod_name].CodeNetMotionwithRot


@dataclass
class AirIO_Output:
    net_vel : jt.Float[torch.Tensor, "n_keep K 3"]  # body-frame velocity per CNN output frame
    cov     : jt.Float[torch.Tensor, "n_keep K 3"]  # per-axis velocity variance
    run_keep: jt.Bool [torch.Tensor, "pack_BN"]     # runs of exactly window_size samples; True count == n_keep


class AirIO(nn.Module):
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = int(window_size)
        # propcov=True: velocity + covariance heads.
        self.net = _load_airio_codenet()(SimpleNamespace(propcov=True))

    @property
    def compute_dtype(self) -> torch.dtype:
        return next(self.net.parameters()).dtype

    def configure_trainable_params(self) -> None:
        """Everything always trains; exists only for parity with ``FoundationIMU``."""

    def _body_features(
        self, batch: IMUData_w_GT_Jagged, row_keep: jt.Bool[torch.Tensor, "pack"], run_keep: jt.Bool[torch.Tensor, "N"], T: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-row body-frame ``(acc, gyro, xi)`` for the kept runs, fp32; ``xi`` is the so(3) Log attitude.

        ``acc`` is completed to specific force: ``gravity + (0,0,+g)`` is zero for measured channels
        and restores standard gravity for gravity-free ones.
        """
        R_wb = pp.SO3(batch.gt_pose.rotation().tensor()[row_keep])
        g_up = torch.tensor([0.0, 0.0, GRAVITY_STANDARD], device=batch.acc.device)
        g_w  = (batch.gravity.float()[run_keep] + g_up).repeat_interleave(T, dim=0)
        acc  = batch.acc.float()[row_keep] + tp.cast(torch.Tensor, R_wb.Inv() @ g_w).float()
        gyro = batch.gyro.float()[row_keep]
        xi   = R_wb.Log().tensor().float()
        return acc, gyro, xi

    def forward(self, batch: IMUData_w_GT_Jagged) -> AirIO_Output:
        # Batching needs a fixed length: keep only runs of exactly window_size samples.
        run_len  = batch.offsets_N.diff()
        run_keep = run_len == self.window_size
        assert bool(run_keep.any()), (
            f"no device run has the {self.window_size} samples AirIO was built for "
            f"(set data_conf.window_unit=sample, window_size={self.window_size}); "
            f"got run lengths {run_len.unique().tolist()}"
        )
        row_keep = torch.repeat_interleave(run_keep, run_len)
        n, T     = int(run_keep.sum()), self.window_size

        acc, gyro, xi = self._body_features(batch, row_keep, run_keep, T)
        data = {
            "acc" : acc.reshape(n, T, 3).to(self.compute_dtype),
            "gyro": gyro.reshape(n, T, 3).to(self.compute_dtype),
        }
        out = self.net(data, xi.reshape(n, T, 3).to(self.compute_dtype))
        return AirIO_Output(net_vel=out["net_vel"], cov=out["cov"], run_keep=run_keep)

    def frame_centers(self, length: int, n_frames: int, device: torch.device) -> jt.Int64[torch.Tensor, "K"]:
        """Input-sample index each CNN output frame supervises: the official ``get_label`` stride
        mapping, clamped at the window edge."""
        k_list = tp.cast(list[int], self.net.k_list)
        p_list = tp.cast(list[int], self.net.p_list)
        s_list = tp.cast(list[int], self.net.s_list)
        s_idx  = (k_list[0] - p_list[0]) + s_list[0] * (k_list[1] - 1 - p_list[1]) + 1
        stride = s_list[0] * s_list[1]
        return (s_idx + stride * torch.arange(n_frames, device=device)).clamp_max(length - 1)

    def select_label(self, batch: IMUData_w_GT_Jagged, pred: AirIO_Output) -> jt.Float[torch.Tensor, "n_keep K 3"]:
        """GT body-frame velocity at the CNN output frames."""
        row_keep = torch.repeat_interleave(pred.run_keep, batch.offsets_N.diff())
        gt_vel   = batch.gt_vel.float()[row_keep].reshape(pred.net_vel.shape[0], self.window_size, 3)
        centers  = self.frame_centers(self.window_size, pred.net_vel.shape[1], gt_vel.device)
        return gt_vel[:, centers]


class AirIO_Loss(nn.Module):
    """Official AirIO motion loss: Huber on the body-velocity error plus the covariance NLL (error
    not detached). ``rmse`` is the official validation criterion, RMSE of the window-mean error.

    Returns a graph-connected zero when no run has a finite target (DDP rank consistency).
    """

    def __init__(
        self,
        weight     : float = 1.0e2,
        cov_weight : float = 1.0e-4,
        huber_delta: float = 0.005,
    ):
        super().__init__()
        self.weight      = weight
        self.cov_weight  = cov_weight
        self.huber_delta = huber_delta

    def forward(
        self,
        model: AirIO,
        data : IMUData_w_GT_Jagged,
        pred : AirIO_Output,
        rmse : bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        targ  = model.select_label(data, pred)
        valid = targ.isfinite().all(dim=-1).all(dim=-1)
        if not bool(valid.any()):
            return (pred.net_vel.float().sum() + pred.cov.float().sum()) * 0.0, {}

        net_vel = pred.net_vel.float()[valid]
        cov     = pred.cov.float()[valid]
        targ    = targ[valid]
        err     = net_vel - targ

        if rmse:
            dist = err.mean(dim=-2)
            loss = dist.norm(dim=-1).pow(2).mean().sqrt()
        else:
            loss = torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=self.huber_delta)
            loss = loss + self.cov_weight * ((err.pow(2) / cov) + cov.log()).mean()
            loss = self.weight * loss

        metric = {
            "loss"             : loss.item(),
            "airio/vel_rmse"   : err.pow(2).sum(-1).mean().sqrt().item(),
            "airio/vel_l1"     : err.norm(dim=-1).mean().item(),
            "airio/sigma_mps"  : cov.mean().sqrt().item(),
        }
        return loss, metric

    if TYPE_CHECKING: __call__ = forward  # so pyright sees forward's signature on calls
