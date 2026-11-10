"""Reader for the VT Natural Motion format-v3 cache."""

import json
import torch
import numpy as np
import typing as tp
import pypose as pp
import jaxtyping as jt

from pathlib import Path

from ...interface.sequence  import ISequence
from ...interface.data      import ArticulatedModel
from ...interface.data_imu  import IMUData_w_GT, Loss_Flags, gravity_of_kind
from ...interface.human     import T_CANONICAL_NAMES, CANONICAL_IDX


class NaturalMotion_XSens_Processed_IMU(ISequence[IMUData_w_GT]):
    """Memory-mapped reader for caches written by ``src.scripts.naturalmotion_imu_process``.

    Format v3: device-major fp16 ``measure.npy`` (N, T, F) and fp32 ``pose.npy`` (N, T, 7). The
    all-zero bias fields are not stored. Arrays map lazily, so DDP ranks and workers share pages.
    """
    
    APPLICABLE_LOSS = Loss_Flags.JEPA \
                    | Loss_Flags.Vel_L2 \
                    | Loss_Flags.Denoise_L2

    def __init__(self, path: str, target_sensors: tp.Sequence[T_CANONICAL_NAMES]):
        self.path = Path(path)
        with open(self.path / "spec.json") as f:
            spec = json.load(f)

        names = spec["names"]
        assert len(set(target_sensors) - set(names)) == 0

        self.model  = ArticulatedModel(names=tuple(target_sensors), urdf=None)
        self.kind: tp.Literal["real", "filtered-no-g", "sim"] = spec["kind"]

        self._T          = int(spec["length"])
        self._n_all      = len(names)
        self._sensor_idx = [names.index(s) for s in target_sensors]
        self._fields     = spec["measure"]["fields"]            # name -> {col: [s, e), shape: [N, C]}

        self._measure_file = self.path / spec["measure"]["file"]
        self._pose_file    = self.path / spec["pose"]["file"]
        self._time_file    = self.path / spec["time_ns"]["file"]
        # Opened lazily per process (see __getstate__).
        self._measure: np.ndarray | None = None
        self._pose:    np.ndarray | None = None
        self._time_ns: np.ndarray | None = None

        self.sensor_id     = torch.tensor([CANONICAL_IDX[s] for s in target_sensors], dtype=torch.uint8).view(1, -1, 1, 1)

    def __getstate__(self):
        # A live np.memmap pickles as a full ndarray, so workers re-open via _open().
        return {**self.__dict__, "_measure": None, "_pose": None, "_time_ns": None}

    def _open(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        measure, pose, time_ns = self._measure, self._pose, self._time_ns
        if measure is None or pose is None or time_ns is None:
            measure = self._measure = np.load(self._measure_file, mmap_mode="r")
            pose    = self._pose    = np.load(self._pose_file,    mmap_mode="r")
            time_ns = self._time_ns = np.load(self._time_file,    mmap_mode="r")
        return measure, pose, time_ns

    def __len__(self) -> int:
        return self._T

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        *_, time_ns = self._open()
        return torch.from_numpy(np.array(time_ns))

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        if isinstance(index, int):
            i = index + self._T if index < 0 else index
            if i < 0 or i >= self._T:
                raise IndexError(f"index {index} out of range for length {self._T}")
            sl = slice(i, i + 1)
        else:
            sl = index

        measure, pose, time_ns = self._open()

        # Device-major (N_all, T, ...): slice time, no transpose. The copy detaches the sample
        # from the read-only mapping.
        m = torch.from_numpy(np.array(measure[:, sl]))                     # (N_all, t, F)
        p = torch.from_numpy(np.array(pose[:, sl]))                        # (N_all, t, 7)
        t = m.shape[1]

        def field(name: str) -> torch.Tensor:                              # (1, n, t, C)
            s, e = self._fields[name]["col"]
            return m[self._sensor_idx, :, s:e].unsqueeze(0)

        acc     = field("acc")
        gyro    = field("gyro")
        gt_pose = p[self._sensor_idx].unsqueeze(0)                         # (1, n, t, 7)

        n        = len(self._sensor_idx)
        t_slice  = torch.from_numpy(np.array(time_ns[sl]))                  # (T,)
        time_exp = t_slice.view(1, 1, -1, 1).expand(1, n, t, 1).contiguous()

        gravity, correct_gravity = gravity_of_kind(self.kind, 1, n)

        return IMUData_w_GT(
            acc         = acc,
            gyro        = gyro,
            time_ns     = time_exp,
            acc_bias    = torch.zeros_like(acc),
            gyro_bias   = torch.zeros_like(gyro),

            correct_mask=torch.zeros(acc.shape[:-1], dtype=torch.bool, device=acc.device),
            correct_acc =acc.clone(),
            correct_gyro=gyro.clone(),
            gravity         = gravity,
            correct_gravity = correct_gravity,

            gt_pose     = pp.SE3(gt_pose),
            gt_vel      = field("gt_vel"),
            labels      = self.get_loss_label(time_exp.shape[:2]),

            sensor_id   = self.sensor_id.expand(1, -1, acc.shape[2], -1).clone(),
            model       = self.model,
            kind        = self.kind,
        )
