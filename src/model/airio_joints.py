"""Adapter around IMUnet's ``JointsAirIONetworkIMUOnly`` vendored at ``thirdparty/IMUnet``: body-frame
velocity and full covariance from the pelvis IMU and joint streams, with no attitude input."""
from __future__ import annotations

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pypose as pp
import jaxtyping as jt
import typing as tp

from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

from ..interface.data_imu   import GRAVITY_STANDARD
from ..interface.data_joint import IMUJointData_w_GT_Jagged


def _load_imunet() -> tuple[type[nn.Module], tp.Any]:
    """Import ``JointsAirIONetworkIMUOnly`` and ``INetwork``; IMUnet's root is appended to
    ``sys.path``, never prepended, because its top-level package names are generic."""
    root = str(Path(__file__).resolve().parents[2] / "thirdparty" / "IMUnet")
    if root not in sys.path:
        sys.path.append(root)
    from Model.airio   import JointsAirIONetworkIMUOnly   # type: ignore[import-not-found]
    from Model.Network import INetwork                    # type: ignore[import-not-found]
    return JointsAirIONetworkIMUOnly, INetwork


@dataclass
class AirIOJoints_Output:
    net_vel : jt.Float[torch.Tensor, "n_keep M 3"]     # body-frame velocity per anchor
    cov     : jt.Float[torch.Tensor, "n_keep M 3 3"]   # full SPD velocity covariance per anchor
    run_keep: jt.Bool [torch.Tensor, "pack_BN"]        # device runs (== samples) fed to the net


class AirIOJoints(nn.Module):
    def __init__(
        self,
        window_size           : int,
        num_joints            : int,
        joint_channels        : tp.Sequence[str] = ("position", "velocity"),
        k_list                : tp.Sequence[int] = (7, 7),
        s_list                : tp.Sequence[int] = (3, 3),
        p_list                : tp.Sequence[int] = (3, 3),
        imu_encoder_channels  : tp.Sequence[int] = (32, 64),
        joint_encoder_channels: tp.Sequence[int] = (32, 64),
        fusion_dim            : int  = 64,
        fusion_hidden         : int  = 128,
        gru1_hidden           : int  = 64,
        gru2_hidden           : int  = 128,
        decoder_hidden        : int  = 128,
        bidirectional         : bool = True,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.num_joints  = int(num_joints)
        net_cls, inetwork = _load_imunet()
        self._Input = inetwork.Input
        self.net: tp.Any = net_cls(           # untyped for pyright: anchor geometry attrs

            k_list=list(k_list), s_list=list(s_list), p_list=list(p_list),
            imu_encoder_channels=list(imu_encoder_channels), joint_encoder_channels=list(joint_encoder_channels),
            joint_channels=list(joint_channels), fusion_dim=fusion_dim, fusion_hidden=fusion_hidden,
            gru1_hidden=gru1_hidden, gru2_hidden=gru2_hidden, decoder_hidden=decoder_hidden,
            bidirectional=bidirectional, num_imus=1, num_joints=self.num_joints, imu_input="raw",
        )

    @property
    def compute_dtype(self) -> torch.dtype:
        return next(self.net.parameters()).dtype

    def configure_trainable_params(self) -> None:
        """Everything always trains; parity with ``FoundationIMU`` / ``AirIO``."""

    @staticmethod
    def body_features(
        batch: IMUJointData_w_GT_Jagged, row_keep: torch.Tensor, run_keep: torch.Tensor, T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Kept rows' body-frame ``(acc, gyro)``, fp32, acc completed to specific force as in
        ``AirIO._body_features``."""
        R_wb = pp.SO3(batch.gt_pose.rotation().tensor()[row_keep])
        g_up = torch.tensor([0.0, 0.0, GRAVITY_STANDARD], device=batch.acc.device)
        g_w  = (batch.gravity.float()[run_keep] + g_up).repeat_interleave(T, dim=0)
        acc  = batch.acc.float()[row_keep] + tp.cast(torch.Tensor, R_wb.Inv() @ g_w).float()
        gyro = batch.gyro.float()[row_keep]
        return acc, gyro

    def joint_stack(
        self, joint_q: torch.Tensor, joint_dq: torch.Tensor, kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IMUnet scalar-joint stacks ``(position, velocity)``, each ``(n, T, J', 1)``, from ``(n, T, J, C)``
        fields: ``so3`` joints give ``Log(q)`` and child-frame rate (``J' = 3J``), ``revolute`` ``J' = J``."""
        n, T, J, _ = joint_q.shape
        match kind:
            case "so3":
                pos = pp.SO3(joint_q.float()).Log().tensor()            # (n, T, J, 3) rotation vector
                vel = joint_dq.float()
            case "revolute":
                pos, vel = joint_q.float(), joint_dq.float()            # (n, T, J, 1)
            case _:
                raise ValueError(kind)
        pos = pos.reshape(n, T, -1, 1)
        vel = vel.reshape(n, T, -1, 1)
        assert pos.shape[2] == self.num_joints, (
            f"network built for num_joints={self.num_joints} scalar joints, batch carries "
            f"{pos.shape[2]} ({J} '{kind}' joints)")
        return pos, vel

    def net_input(
        self, acc: torch.Tensor, gyro: torch.Tensor, time_ns: torch.Tensor,
        joint_q: torch.Tensor, joint_dq: torch.Tensor, kind: str,
    ):
        """``INetwork.Input`` for ``(n, T, ...)`` tensors; ``rot`` is a placeholder the network ignores."""
        n, T = acc.shape[:2]
        dtype = self.compute_dtype
        pos, vel = self.joint_stack(joint_q, joint_dq, kind)
        return self._Input(
            imu    = SimpleNamespace(acc=acc.to(dtype).unsqueeze(2), gyro=gyro.to(dtype).unsqueeze(2)),   # (n, T, 1, 3)
            rot    = pp.so3(torch.zeros(n, T, 3, device=acc.device, dtype=dtype)),
            joints = SimpleNamespace(joint_time_ns=time_ns.reshape(n, T, 1), position=pos.to(dtype),
                                     velocity=vel.to(dtype), effort=None),
        )

    def forward(self, batch: IMUJointData_w_GT_Jagged) -> AirIOJoints_Output:
        assert torch.equal(batch.offsets_N, batch.offsets_B), \
            "AirIOJoints needs exactly one IMU device (the pelvis) per sample so joint rows align with IMU rows"
        run_len  = batch.offsets_N.diff()
        run_keep = run_len == self.window_size
        assert bool(run_keep.any()), (
            f"no run has the {self.window_size} samples AirIOJoints was built for "
            f"(set data_conf.window_unit=sample); got run lengths {run_len.unique().tolist()}")
        row_keep = torch.repeat_interleave(run_keep, run_len)
        n, T     = int(run_keep.sum()), self.window_size

        acc, gyro = self.body_features(batch, row_keep, run_keep, T)
        x = self.net_input(
            acc.reshape(n, T, 3), gyro.reshape(n, T, 3), batch.time_ns[row_keep].reshape(n, T, 1),
            batch.joint_q[row_keep].reshape(n, T, *batch.joint_q.shape[1:]),
            batch.joint_dq[row_keep].reshape(n, T, *batch.joint_dq.shape[1:]),
            batch.joint_kind,
        )
        out = self.net(x)
        return AirIOJoints_Output(net_vel=out.vel_b, cov=out.vel_cov_b, run_keep=run_keep)

    def anchors(self, length: int, device: torch.device) -> jt.Int64[torch.Tensor, "M"]:
        """Input-sample index of each output anchor for a ``length``-sample stream."""
        M = self.net.anchor_count(length)
        return self.net.start_index + self.net.stride * torch.arange(M, device=device)

    def select_label(self, batch: IMUJointData_w_GT_Jagged, pred: AirIOJoints_Output) -> jt.Float[torch.Tensor, "n_keep M 3"]:
        """GT body-frame velocity at the anchors (IMUnet's ``downsample``)."""
        row_keep = torch.repeat_interleave(pred.run_keep, batch.offsets_N.diff())
        gt_vel   = batch.gt_vel.float()[row_keep].reshape(pred.net_vel.shape[0], self.window_size, 3)
        return self.net.downsample(gt_vel)


class AirIOJoints_Loss(nn.Module):
    """Official AirIO motion loss with IMUnet's full-covariance NLL; ``rmse`` is the window-mean
    velocity RMSE. IMUnet detaches the trunk before the covariance head, so the NLL trains only it."""

    def __init__(self, weight: float = 1.0e2, cov_weight: float = 1.0e-4, huber_delta: float = 0.005, jitter: float = 1e-6):
        super().__init__()
        self.weight     : float = weight
        self.cov_weight : float = cov_weight
        self.huber_delta: float = huber_delta
        self.jitter     : float = jitter

    def forward(
        self, model: AirIOJoints, data: IMUJointData_w_GT_Jagged, pred: AirIOJoints_Output, rmse: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        targ  = model.select_label(data, pred)
        valid = targ.isfinite().all(dim=-1).all(dim=-1)
        if not bool(valid.any()):
            return (pred.net_vel.float().sum() + pred.cov.float().sum()) * 0.0, {}

        net_vel = pred.net_vel.float()[valid]
        cov     = pred.cov.float()[valid]
        err     = net_vel - targ[valid]

        if rmse:
            dist = err.mean(dim=-2)
            loss = dist.norm(dim=-1).pow(2).mean().sqrt()
        else:
            huber = F.huber_loss(err, torch.zeros_like(err), delta=float(self.huber_delta))
            eye   = torch.eye(3, device=cov.device, dtype=cov.dtype)
            L     = torch.linalg.cholesky(cov + float(self.jitter) * eye)
            y     = torch.linalg.solve_triangular(L, err.unsqueeze(-1), upper=False).squeeze(-1)
            nll   = (y.pow(2).sum(-1) + 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)).mean()
            loss  = float(self.weight) * (huber + float(self.cov_weight) * nll)

        metric = {
            "loss"                   : loss.item(),
            "airio_joints/vel_rmse"  : err.pow(2).sum(-1).mean().sqrt().item(),
            "airio_joints/vel_l1"    : err.norm(dim=-1).mean().item(),
            "airio_joints/sigma_mps" : torch.diagonal(cov, dim1=-2, dim2=-1).mean().sqrt().item(),
        }
        return loss, metric

    if TYPE_CHECKING: __call__ = forward
