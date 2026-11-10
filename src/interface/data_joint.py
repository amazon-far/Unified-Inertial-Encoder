"""IMU samples that also carry a proprioceptive joint stream."""
from __future__ import annotations

import torch
import typing as tp
import jaxtyping as jt

from dataclasses import dataclass
from typing import Sequence, Self

from .data     import _field_to
from .data_imu import IMUData_w_GT, IMUData_w_GT_Jagged


T_JOINT_KIND = tp.Literal["so3", "revolute"]


@dataclass
class JointTrait:
    """Per-frame joints on the IMU sample grid, padded ``(B, T, J, C)``.

    ``so3`` is the relative joints of ``interface.joint``; ``revolute`` is native robot encoders.
    """
    joint_q    : jt.Float[torch.Tensor, "B T J Cq"]   # so3: quaternion xyzw (4); revolute: angle rad (1)
    joint_dq   : jt.Float[torch.Tensor, "B T J Cd"]   # so3: child-frame rad/s (3); revolute: rad/s (1)
    joint_names: tuple[str, ...]
    joint_kind : T_JOINT_KIND

    def _validate_joint_fields(self, T: int) -> None:
        assert self.joint_q.shape[1] == self.joint_dq.shape[1] == T, \
            f"joint time axis {self.joint_q.shape[1]} != IMU time axis {T}"
        assert self.joint_q.shape[2] == self.joint_dq.shape[2] == len(self.joint_names)

    # --- hooks used by the time-axis transforms --------------------------------------------
    def joint_time_subset(self, sl: slice) -> None:
        """Keep the time samples ``sl``, mirroring ``RandomFrequencyReduction``'s decimation."""
        self.joint_q  = self.joint_q [:, sl]
        self.joint_dq = self.joint_dq[:, sl]

    def joint_time_scale(self, s: float) -> None:
        """Play the motion ``s``x faster: angles are unchanged, rates scale by ``s``."""
        self.joint_dq = s * self.joint_dq

    @staticmethod
    def _collate_joint_fields(batch: Sequence[JointTrait]) -> dict:
        return dict(
            joint_q     = torch.cat([s.joint_q  for s in batch], dim=0),
            joint_dq    = torch.cat([s.joint_dq for s in batch], dim=0),
            joint_names = batch[0].joint_names,
            joint_kind  = batch[0].joint_kind,
        )

    def _clone_joint_fields(self) -> dict:
        return dict(joint_q=self.joint_q.clone(), joint_dq=self.joint_dq.clone(),
                    joint_names=self.joint_names, joint_kind=self.joint_kind)

    def _to_joint_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        return dict(joint_q=_field_to(self.joint_q, device, dtype), joint_dq=_field_to(self.joint_dq, device, dtype),
                    joint_names=self.joint_names, joint_kind=self.joint_kind)

    def _as_jagged_joint_fields(self) -> dict:
        B, T = self.joint_q.shape[:2]
        return dict(
            joint_q     = self.joint_q.flatten(0, 1),
            joint_dq    = self.joint_dq.flatten(0, 1),
            offsets_J   = torch.arange(B + 1, device=self.joint_q.device, dtype=torch.int64) * T,
            joint_names = self.joint_names,
            joint_kind  = self.joint_kind,
        )


def _merge_offsets(offsets: list[torch.Tensor], sizes: list[int]) -> torch.Tensor:
    """Concatenate per-sample CSR offsets into one global offsets vector."""
    parts, shift = [offsets[0]], int(sizes[0])
    for o, sz in zip(offsets[1:], sizes[1:]):
        parts.append(o[1:] + shift)
        shift += int(sz)
    return torch.cat(parts)


@dataclass
class JointTrait_Jagged:
    joint_q    : jt.Float[torch.Tensor, "pack_BT J Cq"]
    joint_dq   : jt.Float[torch.Tensor, "pack_BT J Cd"]
    offsets_J  : jt.Int64[torch.Tensor, "B+1"]          # sample boundaries of the joint rows
    joint_names: tuple[str, ...]
    joint_kind : T_JOINT_KIND

    @staticmethod
    def _collate_joint_fields(batch: Sequence[JointTrait_Jagged]) -> dict:
        sizes = [s.joint_q.shape[0] for s in batch]
        return dict(
            joint_q     = torch.cat([s.joint_q  for s in batch], dim=0),
            joint_dq    = torch.cat([s.joint_dq for s in batch], dim=0),
            offsets_J   = _merge_offsets([s.offsets_J for s in batch], sizes),
            joint_names = batch[0].joint_names,
            joint_kind  = batch[0].joint_kind,
        )

    def _clone_joint_fields(self) -> dict:
        return dict(joint_q=self.joint_q.clone(), joint_dq=self.joint_dq.clone(), offsets_J=self.offsets_J.clone(),
                    joint_names=self.joint_names, joint_kind=self.joint_kind)

    def _to_joint_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        return dict(joint_q=_field_to(self.joint_q, device, dtype), joint_dq=_field_to(self.joint_dq, device, dtype),
                    offsets_J=_field_to(self.offsets_J, device, dtype),
                    joint_names=self.joint_names, joint_kind=self.joint_kind)


@dataclass
class IMUJointData_w_GT(IMUData_w_GT, JointTrait):
    """``IMUData_w_GT`` window plus a joint stream on the same time grid."""

    def __post_init__(self) -> None:
        self._validate_joint_fields(self.acc.shape[2])

    @classmethod
    def collate(cls, batch: Sequence[tp.Any]) -> Self:
        return cls(**cls._collate_base_fields(batch), **cls._collate_label_fields(batch),
                   **cls._collate_joint_fields(batch))

    def clone(self) -> IMUJointData_w_GT:
        return IMUJointData_w_GT(**self._clone_base_fields(), **self._clone_label_fields(),
                                 **self._clone_joint_fields())

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUJointData_w_GT:
        return IMUJointData_w_GT(**self._to_base_fields(device, dtype), **self._to_label_fields(device, dtype),
                                 **self._to_joint_fields(device, dtype))

    def as_jagged(self) -> IMUJointData_w_GT_Jagged:
        return IMUJointData_w_GT_Jagged(**self._as_jagged_base_fields(), **self._as_jagged_label_fields(),
                                        **self._as_jagged_joint_fields())


@dataclass
class IMUJointData_w_GT_Jagged(IMUData_w_GT_Jagged, JointTrait_Jagged):
    """Jagged ``IMUJointData_w_GT``: IMU rows are device-major ``pack_BNT``, joint rows ``pack_BT``.

    They coincide row-for-row only when every sample has one device (``offsets_N == offsets_B``).
    """

    @classmethod
    def collate(cls, batch: Sequence[tp.Any]) -> Self:
        return cls(**cls._collate_base_fields(batch), **cls._collate_label_fields(batch),
                   **cls._collate_joint_fields(batch))

    def clone(self) -> IMUJointData_w_GT_Jagged:
        return IMUJointData_w_GT_Jagged(**self._clone_base_fields(), **self._clone_label_fields(),
                                        **self._clone_joint_fields())

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUJointData_w_GT_Jagged:
        return IMUJointData_w_GT_Jagged(**self._to_base_fields(device, dtype), **self._to_label_fields(device, dtype),
                                        **self._to_joint_fields(device, dtype))


# Language-carrying variants.
from .data_imu import IMUData_w_GT_Lang, IMUData_w_GT_Lang_Jagged   # noqa: E402


@dataclass
class IMUJointData_w_GT_Lang(IMUData_w_GT_Lang, JointTrait):
    """``IMUData_w_GT_Lang`` window plus a joint stream on the same time grid."""

    def __post_init__(self) -> None:
        self._validate_joint_fields(self.acc.shape[2])

    @classmethod
    def collate(cls, batch: Sequence[tp.Any]) -> Self:
        return cls(**cls._collate_base_fields(batch), **cls._collate_label_fields(batch),
                   **cls._collate_lang_fields(batch), **cls._collate_joint_fields(batch))

    def clone(self) -> IMUJointData_w_GT_Lang:
        return IMUJointData_w_GT_Lang(**self._clone_base_fields(), **self._clone_label_fields(),
                                      **self._clone_lang_fields(), **self._clone_joint_fields())

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUJointData_w_GT_Lang:
        return IMUJointData_w_GT_Lang(**self._to_base_fields(device, dtype), **self._to_label_fields(device, dtype),
                                      **self._to_lang_fields(device, dtype), **self._to_joint_fields(device, dtype))

    def as_jagged(self) -> IMUJointData_w_GT_Lang_Jagged:
        return IMUJointData_w_GT_Lang_Jagged(**self._as_jagged_base_fields(), **self._as_jagged_label_fields(),
                                             **self._as_jagged_lang_fields(), **self._as_jagged_joint_fields())


@dataclass
class IMUJointData_w_GT_Lang_Jagged(IMUData_w_GT_Lang_Jagged, JointTrait_Jagged):
    """Jagged form of ``IMUJointData_w_GT_Lang``."""

    @classmethod
    def collate(cls, batch: Sequence[tp.Any]) -> Self:
        return cls(**cls._collate_base_fields(batch), **cls._collate_label_fields(batch),
                   **cls._collate_lang_fields(batch), **cls._collate_joint_fields(batch))

    def clone(self) -> IMUJointData_w_GT_Lang_Jagged:
        return IMUJointData_w_GT_Lang_Jagged(**self._clone_base_fields(), **self._clone_label_fields(),
                                             **self._clone_lang_fields(), **self._clone_joint_fields())

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUJointData_w_GT_Lang_Jagged:
        return IMUJointData_w_GT_Lang_Jagged(**self._to_base_fields(device, dtype), **self._to_label_fields(device, dtype),
                                             **self._to_lang_fields(device, dtype), **self._to_joint_fields(device, dtype))
