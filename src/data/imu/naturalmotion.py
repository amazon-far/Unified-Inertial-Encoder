"""Reader for raw VT Natural Motion Dataset XSens recordings (HDF5)."""

import h5py
import torch
import numpy as np
import typing as tp
import pypose as pp
import jaxtyping as jt

from pathlib import Path

from ...interface.sequence   import ISequence
from ...interface.data       import ArticulatedModel
from ...interface.data_imu   import IMUData_w_GT, Loss_Flags, gravity_of_kind
from ...interface.human      import T_CANONICAL_NAMES, CANONICAL_IDX, CANONICAL_NAMES


# XSens MVN Link capture: same 23-segment layout and conventions as Nymeria.
SEGMENT_NAMES: tuple[str, ...] = (
    "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
    "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
    "LeftShoulder",  "LeftUpperArm",  "LeftForeArm",  "LeftHand",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe",
    "LeftUpperLeg",  "LeftLowerLeg",  "LeftFoot",  "LeftToe",
)

CANONICALIZED_NATURALMOTION: dict[T_CANONICAL_NAMES, str] = {
     "pelvis"          : "Pelvis"       ,
     "head"            : "Head"         ,
     "torso_t8"        : "T8"           ,
     "right_shoulder"  : "RightShoulder",
     "right_upper_arm" : "RightUpperArm",
     "right_forearm"   : "RightForeArm" ,
     "right_hand"      : "RightHand"    ,
     "left_shoulder"   : "LeftShoulder" ,
     "left_upper_arm"  : "LeftUpperArm" ,
     "left_forearm"    : "LeftForeArm"  ,
     "left_hand"       : "LeftHand"     ,
     "right_upper_leg" : "RightUpperLeg",
     "right_lower_leg" : "RightLowerLeg",
     "right_foot"      : "RightFoot"    ,
     "left_upper_leg"  : "LeftUpperLeg" ,
     "left_lower_leg"  : "LeftLowerLeg" ,
     "left_foot"       : "LeftFoot"     ,
}

# XSens MVN runs at 240 Hz; the .h5 export carries no timestamps.
DEFAULT_FPS: float = 240.0


class NaturalMotion_XSens_IMU(ISequence[IMUData_w_GT]):
    
    APPLICABLE_LOSS = Loss_Flags.JEPA \
                | Loss_Flags.Vel_L2 \
                | Loss_Flags.Denoise_L2
    
    def __init__(
        self,
        path: str,
        target_sensors: tp.Sequence[T_CANONICAL_NAMES],
        fps: float = DEFAULT_FPS,
    ):
        assert all(n in CANONICAL_NAMES for n in target_sensors)

        self.path   = Path(path)
        self.model  = ArticulatedModel(names=tuple(target_sensors), urdf=None)

        # MATLAB writes column-major, so the (23 * dims, T) that h5py surfaces is really the
        # logical (T, 23 * dims) array: recover it with a reshape. A transpose scrambles segments.
        with h5py.File(self.path, "r") as h5:
            def seg(name: str, dims: int) -> torch.Tensor:
                arr = np.asarray(h5[name])                                # (23 * dims, T)
                T   = arr.shape[1]
                return torch.from_numpy(arr).reshape(T, 23, dims)

            free_accel = seg("acceleration",    3)                        # global, gravity-free
            seg_q_wxyz = seg("orientation",     4)                        # wxyz, global
            seg_t      = seg("position",        3)                        # global, meters
            seg_linvel = seg("velocity",        3)                        # global
            seg_angvel = seg("angularVelocity", 3)                        # global

        T = free_accel.shape[0]

        seg_idx    = tuple(SEGMENT_NAMES.index(CANONICALIZED_NATURALMOTION[s]) for s in target_sensors)
        free_accel = free_accel[:, seg_idx, :]
        seg_q_wxyz = seg_q_wxyz[:, seg_idx, :]
        seg_t      = seg_t[:, seg_idx, :]
        seg_linvel = seg_linvel[:, seg_idx, :]
        seg_angvel = seg_angvel[:, seg_idx, :]

        seg_q_xyzw = seg_q_wxyz[..., [1, 2, 3, 0]]
        seg_q_xyzw = seg_q_xyzw / seg_q_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        R = pp.SO3(seg_q_xyzw)

        # Body-frame acc/gyro/lin_vel and global pose, as for Nymeria.
        self.acc        : torch.Tensor = (R.Inv() @ free_accel).float()
        self.gyro       : torch.Tensor = (R.Inv() @ seg_angvel).float()
        self.lin_vel    : torch.Tensor = (R.Inv() @ seg_linvel).float()
        self.gt_pose    : pp.LieTensor = pp.SE3(torch.cat([seg_t, seg_q_xyzw], dim=-1))
        self.time_ns    : torch.Tensor = (torch.arange(T, dtype=torch.int64) * round(1e9 / fps))
        self.sensor_id  : torch.Tensor = torch.tensor([CANONICAL_IDX[s] for s in target_sensors], dtype=torch.uint8).view(1, -1, 1, 1)

        self._T = T

    def __len__(self) -> int:
        return self._T

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        return self.time_ns

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        if isinstance(index, int):
            i = index + self._T if index < 0 else index
            if i < 0 or i >= self._T:
                raise IndexError(f"index {index} out of range for length {self._T}")
            sl = slice(i, i + 1)
        else:
            sl = index

        acc         = self.acc[sl].transpose(0, 1).unsqueeze(0)                             # (1, N, T, 3)
        gyro        = self.gyro[sl].transpose(0, 1).unsqueeze(0)
        acc_bias    = torch.zeros_like(acc)
        gyro_bias   = torch.zeros_like(gyro)
        gt_vel      = self.lin_vel[sl].transpose(0, 1).unsqueeze(0)
        gt_pose     = tp.cast(pp.LieTensor, self.gt_pose[sl].transpose(0, 1).unsqueeze(0))  # (1, N, T, 7)

        n_sensor = self.acc.shape[1]
        t_slice  = self.time_ns[sl]                                         # (T,)
        time_ns  = t_slice.view(1, 1, -1, 1).expand(1, n_sensor, t_slice.shape[0], 1).contiguous()

        gravity, correct_gravity = gravity_of_kind("filtered-no-g", 1, n_sensor)

        return IMUData_w_GT(
            acc         = acc,
            gyro        = gyro,
            time_ns     = time_ns,
            acc_bias    = acc_bias,
            gyro_bias   = gyro_bias,

            correct_mask= torch.zeros(acc.shape[:-1], dtype=torch.bool, device=acc.device),
            correct_acc = acc.clone(),
            correct_gyro= gyro.clone(),
            gravity         = gravity,
            correct_gravity = correct_gravity,

            gt_pose     = gt_pose,
            gt_vel      = gt_vel,
            labels      = self.get_loss_label(time_ns.shape[:2]),

            sensor_id   = self.sensor_id.expand(1, -1, t_slice.shape[0], -1).contiguous(),
            model       = self.model,
            kind        = "filtered-no-g"
        )
