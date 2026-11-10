"""Whole-body virtual IMU synthesis from G1 joint and pelvis-IMU trajectories."""

import torch
import pypose as pp
import jaxtyping as jt

from functools import cached_property
from rich.progress import track
from typing import Literal

from ...interface.sequence      import ISequence
from ...interface.data          import ArticulatedModel
from ...interface.data_humanoid import ArticulatedJointMeasures_wGT
from ...interface.data_imu      import IMUData_w_GT


class Unitree_G1_SuperOdom_IMU(ISequence[IMUData_w_GT]):
    """Whole-body *virtual* IMU synthesis for a G1 joint and pelvis-IMU trajectory.

    Subclasses set ``joint_store``, ``base_acc``, ``base_gyro`` and ``base_frame``, then call
    ``synthesize_at``.
    """
    joint_store: ArticulatedJointMeasures_wGT
    base_acc   : jt.Float[torch.Tensor, "1 T 3"]
    base_gyro  : jt.Float[torch.Tensor, "1 T 3"]
    base_frame : str
    imu_store  : IMUData_w_GT

    # Timesteps per as_imu call; chunking bounds FK and autograd peak memory.
    SYNTH_CHUNK = 4096
    # Chunk overlap: must cover the widest stencil in as_imu so chunked output equals unchunked.
    CHUNK_MARGIN = 64

    # Half-windows (s) of the two numerical derivatives; <= 0 means a 2-sample central difference.
    # q_ddot is always derived from dq because lowstate ddq is zero-filled. Windows must stay
    # <= CHUNK_MARGIN samples at the raw rate.
    GYRO_DERIV_WINDOW_S = 0.031
    JOINT_DD_WINDOW_S   = 0.031
    
    def synthesize_at(
        self,
        target_frames: list[str],
        sensor_ids   : list[int],
        imu_model    : "ArticulatedModel",
        gyro_deriv_window_s: float | None = None,
        joint_dd_window_s  : float | None = None,
        kind: Literal["real", "filtered-no-g", "sim"] = "sim",
    ) -> IMUData_w_GT:
        """Synthesize virtual IMUs at URDF links from the stored pelvis signals (chunked).

        The smoothing windows default to the class constants.
        """
        return self._synthesize_chunked(
            self.base_acc, self.base_gyro, target_frames, sensor_ids, imu_model,
            gyro_deriv_window_s = self.GYRO_DERIV_WINDOW_S if gyro_deriv_window_s is None else gyro_deriv_window_s,
            joint_dd_window_s   = self.JOINT_DD_WINDOW_S if joint_dd_window_s is None else joint_dd_window_s,
            kind = kind,
        )

    def _synthesize_chunked(
        self,
        base_acc     : jt.Float[torch.Tensor, "1 T 3"],
        base_gyro    : jt.Float[torch.Tensor, "1 T 3"],
        target_frames: list[str],
        sensor_ids   : list[int],
        imu_model    : "ArticulatedModel",
        gyro_deriv_window_s: float = 0.0,
        joint_dd_window_s  : float = 0.0,
        kind: Literal["real", "filtered-no-g", "sim"] = "sim",
    ) -> IMUData_w_GT:
        """Run ``as_imu`` in overlapping time chunks and concatenate along time.

        Chunks overlap by ``CHUNK_MARGIN`` and the overlap is trimmed, so the result is exactly
        that of an unchunked call.
        """
        s = self.joint_store
        T = s.time_ns.size(1)

        base_fields = ["time_ns", "acc", "gyro", "correct_mask", "correct_acc",
                       "correct_gyro", "acc_bias", "gyro_bias", "sensor_id"]
        parts: dict[str, list[torch.Tensor]] = {k: [] for k in [*base_fields, "gt_pose", "gt_vel"]}
        out = None   # last chunk; carries the per-device fields (identical across chunks)

        starts = range(0, T, self.SYNTH_CHUNK)
        for start in track(starts, description="Synthesizing whole-body IMU...", transient=True):
            end = min(start + self.SYNTH_CHUNK, T)
            lo  = max(start - self.CHUNK_MARGIN, 0)
            hi  = min(end + self.CHUNK_MARGIN, T)
            sl  = slice(lo, hi)

            measure = ArticulatedJointMeasures_wGT(
                root_frame = s.root_frame,
                root_pose  = pp.SE3(s.root_pose[:, sl]),
                root_vel   = s.root_vel  [:, sl],
                root_omega = s.root_omega[:, sl],
                joints     = s.joints [:, :, sl],
                joint_d    = s.joint_d[:, :, sl],
                time_ns    = s.time_ns[:, sl],
                model      = s.model,
                kind       = s.kind,
            )
            
            out = measure.as_imu(
                label         = self.get_loss_label(torch.Size((base_acc.shape[0], len(target_frames)))),
                base_acc      = base_acc[:, sl],
                base_gyro     = base_gyro[:, sl],
                base_frame    = self.base_frame,
                target_frames = target_frames,
                sensor_ids    = sensor_ids,
                imu_model     = imu_model,
                kind          = kind,
                gyro_deriv_window_s = gyro_deriv_window_s,
                joint_dd_window_s   = joint_dd_window_s,
            )

            trim = slice(start - lo, out.acc.size(2) - (hi - end))
            for k in base_fields:
                parts[k].append(getattr(out, k)[:, :, trim])
            parts["gt_pose"].append(out.gt_pose.tensor()[:, :, trim])
            assert out.gt_vel is not None
            parts["gt_vel"].append(out.gt_vel[:, :, trim])

        base = {k: torch.cat(parts[k], dim=2) for k in base_fields}
        assert out is not None
        return IMUData_w_GT(
            **base,
            gravity         = out.gravity,          # per-device, no time axis to chunk
            correct_gravity = out.correct_gravity,
            labels  = self.get_loss_label(base["time_ns"].shape[:2]),
            gt_pose = pp.SE3(torch.cat(parts["gt_pose"], dim=2)),
            gt_vel  = torch.cat(parts["gt_vel"], dim=2),
            model   = imu_model,
            kind    = kind,
        )

    def __len__(self) -> int:
        return self.imu_store.acc.size(2)   # fields are device-major (1, N, T, C); T is axis 2

    @cached_property
    def num_devices(self) -> int:
        return self.imu_store.model.num_joint

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        return self.imu_store.time_ns[0, 0, :, 0]   # all devices share the lowstate clock

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        if isinstance(index, int): index = slice(index, index + 1)

        s = self.imu_store   # synthesized once in __init__; windows are time slices of it
        time_ns = s.time_ns     [:, :, index]
        
        return IMUData_w_GT(
            labels       = self.get_loss_label(time_ns.shape[:2]),
            time_ns      = time_ns,
            acc          = s.acc         [:, :, index],
            gyro         = s.gyro        [:, :, index],
            correct_mask = s.correct_mask[:, :, index],
            correct_acc  = s.correct_acc [:, :, index],
            correct_gyro = s.correct_gyro[:, :, index],
            gravity         = s.gravity,           # per-device, no time axis
            correct_gravity = s.correct_gravity,
            acc_bias     = s.acc_bias    [:, :, index],
            gyro_bias    = s.gyro_bias   [:, :, index],
            sensor_id    = s.sensor_id   [:, :, index],
            gt_pose      = pp.SE3(s.gt_pose[:, :, index]),
            gt_vel       = s.gt_vel      [:, :, index],
            model        = s.model,
            kind         = s.kind,
        )
