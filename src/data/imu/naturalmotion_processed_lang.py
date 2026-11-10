"""Language-typed wrapper of the VT Natural Motion cache, which has no narration."""

import torch
import dataclasses
import typing as tp
import jaxtyping as jt

from pathlib import Path

from .naturalmotion_processed import NaturalMotion_XSens_Processed_IMU
from ...interface.sequence    import ISequence
from ...interface.data_imu    import IMUData_w_GT_Lang, Loss_Flags
from ...interface.human       import T_CANONICAL_NAMES


class NaturalMotion_XSens_Processed_IMU_Language(ISequence[IMUData_w_GT_Lang]):
    """``NaturalMotion_XSens_Processed_IMU`` re-typed to share a batch with language readers.

    Every window carries an empty feature (``lang_feat_idx == -1``); ``Lang_CLIP`` never applies.
    """

    # CLIP ViT-L/14 embedding width; shapes the empty feature.
    _LANG_DIM = 768

    APPLICABLE_LOSS = Loss_Flags.JEPA \
                    | Loss_Flags.Vel_L2 \
                    | Loss_Flags.Denoise_L2 \
                    | Loss_Flags.Traj_Denoise \
                    | Loss_Flags.Traj_Vel

    def __init__(self, path: str, target_sensors: tp.Sequence[T_CANONICAL_NAMES]):
        self.path = Path(path)
        self._imu = NaturalMotion_XSens_Processed_IMU(path, target_sensors)

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
