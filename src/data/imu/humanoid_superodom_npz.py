"""Language-typed wrapper of the real-G1 clip cache, which has no narration."""

import torch
import dataclasses
import numpy as np
import pypose as pp
import typing as tp
import jaxtyping as jt

from pathlib import Path

from ...interface.sequence      import ISequence
from ...interface.data_humanoid import ArticulatedJointMeasures_wGT
from ...interface.data_imu      import IMUData_w_GT_Lang, Loss_Flags
from ...interface.human         import T_CANONICAL_NAMES, CANONICAL_IDX

from ..humanoid.g1_superodom import (
    G1_29DOF_Model, G1_WholeBody_Link_Mapping, G1_WholeBody_IMU_Model,
)
from .g1_superodom_processed import Unitree_G1_SuperOdom_IMU_Processed


class Humanoid_SuperOdom_NPZ_Processed_Language(ISequence[IMUData_w_GT_Lang]):
    """``Unitree_G1_SuperOdom_IMU_Processed`` re-typed to share a batch with language readers.

    Every window carries an empty feature (``lang_feat_idx == -1``); ``Lang_CLIP`` never applies.
    """

    # CLIP ViT-L/14 embedding width; shapes the empty feature.
    _LANG_DIM = 768

    # Same as the wrapped reader: label-free losses only.
    APPLICABLE_LOSS = Loss_Flags.JEPA \
                    | Loss_Flags.Vel_SSIO

    def __init__(self, path: str, target_sensors: tp.Sequence[T_CANONICAL_NAMES] | None = None):
        self.path = Path(path)
        self._imu = Unitree_G1_SuperOdom_IMU_Processed(path, target_sensors)

    def __len__(self) -> int:
        return len(self._imu)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        return self._imu.timestamps

    def __getitem__(self, index: slice | int) -> IMUData_w_GT_Lang:
        base = self._imu[index]

        fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        return IMUData_w_GT_Lang(
            **fields,
            lang_feat_idx = torch.full((1,), -1, dtype=torch.int64),
            lang_feature  = torch.empty(0, self._LANG_DIM),
        )
