"""IMU sample containers in padded and jagged layouts, with labels and gravity conventions."""
from __future__ import annotations

import torch
import rerun as rr
import pypose as pp
import jaxtyping as jt

from enum import IntFlag
from dataclasses import dataclass
from typing import Any, Literal, Self, Sequence

from .data      import ArticulatedModel, _field_to
from .data_lang import LanguageFeature_Trait

# "sample" counts raw samples; "ms" masks whole windows aligned to the tokenizer's
# ``window_span_ms`` (a half-masked window would leak visible samples via per-window pooling).
TokenUnit = Literal["sample", "ms"]


class Loss_Flags(IntFlag):
    EMPTY       = 0b0000_0000
    JEPA        = 0b0000_0001
    Vel_L2      = 0b0000_0010
    Vel_SSIO    = 0b0000_0100
    Denoise_L2  = 0b0000_1000
    Lang_CLIP   = 0b0001_0000
    # Integrated-trajectory supervision: needs finite gt_pose on every opted-in device
    # (Traj_Denoise also finite gt_vel at run starts).
    Traj_Denoise= 0b0010_0000
    Traj_Vel    = 0b0100_0000


# Nominal gravity magnitude [m/s²], for datasets that carry gravity but not its local value.
GRAVITY_STANDARD = 9.80665


def _label_dtype(dtype: torch.dtype | None) -> torch.dtype | None:
    """Dtype for ground-truth label fields: never below fp32.

    ``gt_pose`` holds absolute world coordinates, which a sub-fp32 cast quantizes badly enough to
    poison any loss built on GT displacements.
    """
    if dtype is not None and dtype.is_floating_point and torch.finfo(dtype).bits < 32:
        return torch.float32
    return dtype


def gravity_of_kind(
    kind: Literal["real", "filtered-no-g", "sim"],
    batch: int,
    n_device: int,
) -> tuple[jt.Float[torch.Tensor, "B N 3"], jt.Float[torch.Tensor, "B N 3"]]:
    """Default world-frame ``(gravity, correct_gravity)`` implied by a dataset ``kind``.

    Each is what an integrator adds back (``v_w += (R_wb @ a_b + g_w) dt``): ``(0, 0, -|g|)`` for a
    specific force (``"real"``/``"sim"``), zero for kinematic acceleration (``"filtered-no-g"``).
    Pipelines whose channels diverge (e.g. ``AddGravity``) must overwrite the fields themselves.
    """
    match kind:
        case "filtered-no-g": g = 0.0
        case "real" | "sim" : g = -GRAVITY_STANDARD
        case _              : raise ValueError(f"Unexpected data kind {kind}")
    gravity = torch.tensor([0., 0., g], dtype=torch.float32).expand(batch, n_device, 3).contiguous()
    return gravity, gravity.clone()


def _merge_offsets(
    offsets: Sequence[torch.Tensor],
    pack_sizes: Sequence[int],
) -> torch.Tensor:
    """Concatenate per-sample CSR offsets into one monotone offset array over the joined pack."""
    shifted: list[torch.Tensor] = []
    running = 0
    for off, p in zip(offsets, pack_sizes):
        shifted.append(off[:-1] + running)
        running += p
    shifted.append(offsets[0].new_tensor([running]))
    return torch.cat(shifted)


@dataclass
class IMUData_BaseTrait:
    time_ns     : jt.Int64[torch.Tensor, "B N T 1"]
    acc         : jt.Float[torch.Tensor, "B N T 3"]
    gyro        : jt.Float[torch.Tensor, "B N T 3"]

    correct_mask: jt.Bool[torch.Tensor , "B N T"]       # True where the entry is corrected.
    correct_acc : jt.Float[torch.Tensor, "B N T 3"]     # Corrected value; defaults to acc
    correct_gyro: jt.Float[torch.Tensor, "B N T 3"]     # Corrected value; defaults to gyro

    # World-frame gravity to add back when integrating each channel; see ``gravity_of_kind``.
    gravity         : jt.Float[torch.Tensor, "B N 3"]   # gravity carried by `acc`
    correct_gravity : jt.Float[torch.Tensor, "B N 3"]   # gravity carried by `correct_acc`

    acc_bias    : jt.Float[torch.Tensor, "B N T 3"]
    gyro_bias   : jt.Float[torch.Tensor, "B N T 3"]

    sensor_id   : jt.UInt8[torch.Tensor, "B N T 1"]     # Index into CANONICAL_NAMES (interface/human.py)
    model       : ArticulatedModel
    kind        : Literal["real", "filtered-no-g", "sim"]

    def clone(self) -> IMUData_BaseTrait:
        """Return a deep copy of the concrete type."""
        raise NotImplementedError("clone() must be implemented by the concrete IMUData class")

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUData_BaseTrait:
        """Return a copy on `device`, floating-point fields cast to `dtype`; integer/bool fields keep
        their dtype. Follows ``Tensor.to``, so this is not a deep copy.
        """
        raise NotImplementedError("to() must be implemented by the concrete IMUData class")

    @classmethod
    def collate(cls, batch: Sequence[Any]) -> Self:
        """Merge single-window samples (B=1) along the batch axis; ``batch`` is ``Any`` so overrides can narrow it to their own class."""
        raise NotImplementedError("collate() must be implemented by the concrete IMUData class")

    def as_jagged(self) -> IMUData_BaseTrait_Jagged:
        """Pack into the uniform jagged (``pack_BNT``) form; any pad slots count as real samples."""
        raise NotImplementedError("as_jagged() must be implemented by the concrete IMUData class")

    @staticmethod
    def _collate_base_fields(batch: Sequence[IMUData_BaseTrait]) -> dict:
        """Concatenate base fields into a kwargs dict; `model` and `kind` come from the first sample."""
        return dict(
            time_ns      = torch.cat([s.time_ns      for s in batch], dim=0),
            acc          = torch.cat([s.acc          for s in batch], dim=0),
            gyro         = torch.cat([s.gyro         for s in batch], dim=0),
            correct_mask = torch.cat([s.correct_mask for s in batch], dim=0),
            correct_acc  = torch.cat([s.correct_acc  for s in batch], dim=0),
            correct_gyro = torch.cat([s.correct_gyro for s in batch], dim=0),
            gravity         = torch.cat([s.gravity         for s in batch], dim=0),
            correct_gravity = torch.cat([s.correct_gravity for s in batch], dim=0),
            acc_bias     = torch.cat([s.acc_bias     for s in batch], dim=0),
            gyro_bias    = torch.cat([s.gyro_bias    for s in batch], dim=0),
            sensor_id    = torch.cat([s.sensor_id    for s in batch], dim=0),
            model        = batch[0].model,
            kind         = batch[0].kind,
        )

    def _clone_base_fields(self) -> dict:
        """Deep-clone base tensor fields into a kwargs dict; `model` is shared by reference."""
        return dict(
            time_ns      = self.time_ns.clone(),
            acc          = self.acc.clone(),
            gyro         = self.gyro.clone(),
            correct_mask = self.correct_mask.clone(),
            correct_acc  = self.correct_acc.clone(),
            correct_gyro = self.correct_gyro.clone(),
            gravity         = self.gravity.clone(),
            correct_gravity = self.correct_gravity.clone(),
            acc_bias     = self.acc_bias.clone(),
            gyro_bias    = self.gyro_bias.clone(),
            sensor_id    = self.sensor_id.clone(),
            model        = self.model,
            kind         = self.kind,
        )

    def _to_base_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        return dict(
            time_ns      = _field_to(self.time_ns,      device, dtype),
            acc          = _field_to(self.acc,          device, dtype),
            gyro         = _field_to(self.gyro,         device, dtype),
            correct_mask = _field_to(self.correct_mask, device, dtype),
            correct_acc  = _field_to(self.correct_acc,  device, dtype),
            correct_gyro = _field_to(self.correct_gyro, device, dtype),
            gravity         = _field_to(self.gravity,         device, dtype),
            correct_gravity = _field_to(self.correct_gravity, device, dtype),
            acc_bias     = _field_to(self.acc_bias,     device, dtype),
            gyro_bias    = _field_to(self.gyro_bias,    device, dtype),
            sensor_id    = _field_to(self.sensor_id,    device, dtype),
            model        = self.model,
            kind         = self.kind,
        )

    def _as_jagged_base_fields(self) -> dict:
        """Flatten ``(B, N, T, F)`` fields sample-, device-, then time-major, with uniform CSR offsets.

        Per-device fields (``gravity``/``correct_gravity``) flatten to ``pack_BN`` instead.
        """
        B, N, T = self.acc.shape[:3]
        device  = self.acc.device
        flat    = lambda f: f.flatten(0, 2)
        return dict(
            time_ns      = flat(self.time_ns),
            acc          = flat(self.acc),
            gyro         = flat(self.gyro),
            correct_mask = flat(self.correct_mask),
            correct_acc  = flat(self.correct_acc),
            correct_gyro = flat(self.correct_gyro),
            gravity         = self.gravity.flatten(0, 1),
            correct_gravity = self.correct_gravity.flatten(0, 1),
            acc_bias     = flat(self.acc_bias),
            gyro_bias    = flat(self.gyro_bias),
            offsets_B    = torch.arange(B + 1,     device=device, dtype=torch.int64) * (N * T),
            offsets_N    = torch.arange(B * N + 1, device=device, dtype=torch.int64) * T,
            sensor_id    = flat(self.sensor_id),
            model        = self.model,
            kind         = self.kind,
        )

    @staticmethod
    def visualize_initial(paths: list[str]):
        for path in paths:
            rr.log(
                path,
                rr.Boxes3D(
                    sizes=[0.015, 0.015, 0.015], centers=[[0., 0., 0.]], quaternions=rr.Quaternion(xyzw=[0., 0., 0., 1.]),
                    fill_mode=rr.components.FillMode.Solid
                ),
                rr.TransformAxes3D(axis_length=0.05),
                static=True
            )

    def visualize(self, paths: list[str], index: int=0):
        assert len(paths) == self.model.num_joint

        for i, path in enumerate(paths):
            time_ns    = self.time_ns[index, i, :, 0].numpy().astype("int64").view("datetime64[ns]")
            time_index = [rr.TimeColumn("ros_time_ns", timestamp=time_ns)]

            rr.send_columns(
                f"{path}/acc", indexes=time_index,
                columns=rr.Scalars.columns(scalars=self.acc[index, i]),
            )
            rr.send_columns(
                f"{path}/gyro", indexes=time_index,
                columns=rr.Scalars.columns(scalars=self.gyro[index, i]),
            )
            rr.send_columns(
                f"{path}/acc_bias", indexes=time_index,
                columns=rr.Scalars.columns(scalars=self.acc_bias[index, i]),
            )
            rr.send_columns(
                f"{path}/gyro_bias", indexes=time_index,
                columns=rr.Scalars.columns(scalars=self.gyro_bias[index, i]),
            )


def collate_imu_data(batch: Sequence[IMUData_BaseTrait_Jagged]) -> IMUData_BaseTrait_Jagged:
    """``DataLoader`` collate for jagged samples."""
    return type(batch[0]).collate(batch)


@dataclass
class IMUData_LabelTrait:
    labels      : jt.UInt8[torch.Tensor, "B N"] # Loss_Flags bitmask
    gt_pose     : jt.Float[pp.LieTensor, "B N T 7"]
    gt_vel      : jt.Float[torch.Tensor, "B N T 3"]

    @staticmethod
    def _collate_label_fields(batch: Sequence[IMUData_LabelTrait]) -> dict:
        return dict(
            labels  = torch.cat([s.labels for s in batch], dim=0),
            gt_pose = pp.SE3(torch.cat([s.gt_pose.tensor() for s in batch], dim=0)),
            gt_vel  = torch.cat([s.gt_vel for s in batch], dim=0),
        )

    def _clone_label_fields(self) -> dict:
        return dict(
            labels  = self.labels.clone(),
            gt_pose = self.gt_pose.clone(),
            gt_vel  = self.gt_vel.clone(),
        )

    def _to_label_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        """Apply ``.to`` into a kwargs dict, clamping sub-fp32 casts to fp32 (see `_label_dtype`)."""
        dtype = _label_dtype(dtype)
        return dict(
            labels  = _field_to(self.labels, device, None), # labels field is always uint8.
            gt_pose = _field_to(self.gt_pose, device, dtype),
            gt_vel  = _field_to(self.gt_vel, device, dtype),
        )

    def _as_jagged_label_fields(self) -> dict:
        """Flatten label fields to match the layout of ``_as_jagged_base_fields``."""
        return dict(
            labels  = self.labels.flatten(),
            gt_pose = pp.SE3(self.gt_pose.tensor().flatten(0, 2)),
            gt_vel  = self.gt_vel.flatten(0, 2),
        )


@dataclass
class IMUData(IMUData_BaseTrait):
    @classmethod
    def collate(cls, batch: Sequence[Self]) -> Self:
        """Merge single-window samples into one batched ``IMUData``."""
        return cls(**cls._collate_base_fields(batch))

    def clone(self) -> IMUData:
        """Return a deep copy with all tensor fields cloned."""
        return IMUData(**self._clone_base_fields())

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUData:
        """Return a copy on `device`, floating-point fields cast to `dtype`."""
        return IMUData(**self._to_base_fields(device, dtype))

    def as_jagged(self) -> IMUData_Jagged:
        """Pack into a uniform ``IMUData_Jagged``."""
        return IMUData_Jagged(**self._as_jagged_base_fields())


@dataclass
class IMUData_w_GT(IMUData_BaseTrait, IMUData_LabelTrait):
    @classmethod
    def collate(cls, batch: Sequence[Self]) -> Self:
        """Merge single-window samples into one batched ``IMUData_w_GT``."""
        return cls(
            **cls._collate_base_fields(batch),
            **cls._collate_label_fields(batch),
        )

    def clone(self) -> IMUData_w_GT:
        """Return a deep copy with all tensor fields cloned."""
        return IMUData_w_GT(
            **self._clone_base_fields(),
            **self._clone_label_fields(),
        )

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUData_w_GT:
        """Return a copy on `device`, floating-point fields cast to `dtype`."""
        return IMUData_w_GT(
            **self._to_base_fields(device, dtype),
            **self._to_label_fields(device, dtype),
        )

    def as_jagged(self) -> IMUData_w_GT_Jagged:
        """Pack into a uniform ``IMUData_w_GT_Jagged``."""
        return IMUData_w_GT_Jagged(
            **self._as_jagged_base_fields(),
            **self._as_jagged_label_fields(),
        )

    @staticmethod
    def visualize_initial(paths: list[str]):
        IMUData_BaseTrait.visualize_initial(paths)

        for path in paths:
            rr.log(
                path,
                rr.Arrows3D.from_fields(origins=(0., 0., 0.)),
                static=True
            )

    def visualize(self, paths: list[str], index: int = 0):
        IMUData_BaseTrait.visualize(self, paths, index)

        VEL_VIS_SCALE_FACTOR = 5.
        
        for i, path in enumerate(paths):
            time_ns    = self.time_ns[index, i, :, 0].numpy().astype("int64").view("datetime64[ns]")
            time_index = [rr.TimeColumn("ros_time_ns", timestamp=time_ns)]
            pose = pp.SE3(self.gt_pose[index, i])

            rr.send_columns(
                f"{path}", indexes=time_index,
                columns=rr.Transform3D.columns(
                    translation=pose.translation().numpy(),
                    quaternion =pose.rotation().tensor().numpy(),
                )
            )

            if self.gt_vel is not None:
                rr.send_columns(
                    f"{path}", indexes=time_index,
                    columns=rr.Arrows3D.columns(vectors=self.gt_vel[index, i] / VEL_VIS_SCALE_FACTOR)
                )
  

@dataclass
class IMUData_w_GT_Lang(IMUData_BaseTrait, IMUData_LabelTrait, LanguageFeature_Trait):
    @classmethod
    def collate(cls, batch: Sequence[Self]) -> Self:
        """Merge single-window samples into one batched ``IMUData_w_GT_Lang``."""
        return cls(
            **cls._collate_base_fields(batch),
            **cls._collate_label_fields(batch),
            **cls._collate_lang_fields(batch),
        )

    def clone(self) -> IMUData_w_GT_Lang:
        """Return a deep copy with all tensor fields cloned."""
        return IMUData_w_GT_Lang(
            **self._clone_base_fields(),
            **self._clone_label_fields(),
            **self._clone_lang_fields(),
        )

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUData_w_GT_Lang:
        """Return a copy on `device`, floating-point fields cast to `dtype`."""
        return IMUData_w_GT_Lang(
            **self._to_base_fields(device, dtype),
            **self._to_label_fields(device, dtype),
            **self._to_lang_fields(device, dtype),
        )

    def as_jagged(self) -> IMUData_w_GT_Lang_Jagged:
        """Pack into a uniform ``IMUData_w_GT_Lang_Jagged``; language fields carry over unchanged."""
        return IMUData_w_GT_Lang_Jagged(
            **self._as_jagged_base_fields(),
            **self._as_jagged_label_fields(),
            **self._as_jagged_lang_fields(),
        )

    # Composes the traits directly, so without these delegations lookup lands on the scalars-only
    # ``IMUData_BaseTrait.visualize`` and gt_pose/gt_vel are never logged.
    @staticmethod
    def visualize_initial(paths: list[str]):
        IMUData_w_GT.visualize_initial(paths)

    def visualize(self, paths: list[str], index: int = 0):
        # Unbound call: touches only the fields shared with ``IMUData_w_GT``.
        IMUData_w_GT.visualize(self, paths, index)  # type: ignore[arg-type]


# Jagged Structure to support non-uniform batching / compute

@dataclass
class IMUData_BaseTrait_Jagged:
    time_ns     : jt.Int64[torch.Tensor, "pack_BNT 1"]
    acc         : jt.Float[torch.Tensor, "pack_BNT 3"]
    gyro        : jt.Float[torch.Tensor, "pack_BNT 3"]

    correct_mask: jt.Bool [torch.Tensor, "pack_BNT"]       # True where the entry is corrected.
    correct_acc : jt.Float[torch.Tensor, "pack_BNT 3"]     # Corrected value; defaults to acc
    correct_gyro: jt.Float[torch.Tensor, "pack_BNT 3"]     # Corrected value; defaults to gyro

    # World-frame gravity per device run (``offsets_N`` order); see ``gravity_of_kind``.
    gravity         : jt.Float[torch.Tensor, "pack_BN 3"]  # gravity carried by `acc`
    correct_gravity : jt.Float[torch.Tensor, "pack_BN 3"]  # gravity carried by `correct_acc`

    acc_bias    : jt.Float[torch.Tensor, "pack_BNT 3"]
    gyro_bias   : jt.Float[torch.Tensor, "pack_BNT 3"]

    offsets_B   : jt.Int64[torch.Tensor, "B+1"]            # Start row of each sample.
    offsets_N   : jt.Int64[torch.Tensor, "N+1"]            # Start row of each device.

    sensor_id   : jt.UInt8[torch.Tensor, "pack_BNT 1"]     # Index into CANONICAL_NAMES (interface/human.py)
    model       : ArticulatedModel
    kind        : Literal["real", "filtered-no-g", "sim"]

    def clone(self) -> IMUData_BaseTrait_Jagged:
        """Return a deep copy of the concrete type."""
        raise NotImplementedError("clone() must be implemented by the concrete IMUData class")

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUData_BaseTrait_Jagged:
        """Return a copy on `device`, floating-point fields cast to `dtype`; integer/bool fields keep
        their dtype. Follows ``Tensor.to``, so this is not a deep copy.
        """
        raise NotImplementedError("to() must be implemented by the concrete IMUData class")

    @classmethod
    def collate(cls, batch: Sequence[Any]) -> Self:
        """Merge single-sample jagged windows, packing rows and merging ``offsets_*``; ``batch`` is ``Any`` so overrides can narrow it to their own class."""
        raise NotImplementedError("collate() must be implemented by the concrete IMUData class")

    def _clone_base_fields(self) -> dict:
        return dict(
            time_ns      = self.time_ns.clone(),
            acc          = self.acc.clone(),
            gyro         = self.gyro.clone(),
            correct_mask = self.correct_mask.clone(),
            correct_acc  = self.correct_acc.clone(),
            correct_gyro = self.correct_gyro.clone(),
            gravity         = self.gravity.clone(),
            correct_gravity = self.correct_gravity.clone(),
            acc_bias     = self.acc_bias.clone(),
            gyro_bias    = self.gyro_bias.clone(),
            offsets_B    = self.offsets_B.clone(),
            offsets_N    = self.offsets_N.clone(),
            sensor_id    = self.sensor_id.clone(),
            model        = self.model,
            kind         = self.kind,
        )

    def _to_base_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        return dict(
            time_ns      = _field_to(self.time_ns,      device, dtype),
            acc          = _field_to(self.acc,          device, dtype),
            gyro         = _field_to(self.gyro,         device, dtype),
            correct_mask = _field_to(self.correct_mask, device, dtype),
            correct_acc  = _field_to(self.correct_acc,  device, dtype),
            correct_gyro = _field_to(self.correct_gyro, device, dtype),
            gravity         = _field_to(self.gravity,         device, dtype),
            correct_gravity = _field_to(self.correct_gravity, device, dtype),
            acc_bias     = _field_to(self.acc_bias,     device, dtype),
            gyro_bias    = _field_to(self.gyro_bias,    device, dtype),
            offsets_B    = _field_to(self.offsets_B,    device, dtype),
            offsets_N    = _field_to(self.offsets_N,    device, dtype),
            sensor_id    = _field_to(self.sensor_id,    device, dtype),
            model        = self.model,
            kind         = self.kind,
        )

    @staticmethod
    def _collate_base_fields(batch: Sequence[IMUData_BaseTrait_Jagged]) -> dict:
        """Concatenate packed fields and merge ``offsets_*``; `model` and `kind` come from the first sample."""
        pack_sizes = [s.acc.shape[0] for s in batch]
        return dict(
            time_ns      = torch.cat([s.time_ns      for s in batch], dim=0),
            acc          = torch.cat([s.acc          for s in batch], dim=0),
            gyro         = torch.cat([s.gyro         for s in batch], dim=0),
            correct_mask = torch.cat([s.correct_mask for s in batch], dim=0),
            correct_acc  = torch.cat([s.correct_acc  for s in batch], dim=0),
            correct_gyro = torch.cat([s.correct_gyro for s in batch], dim=0),
            gravity         = torch.cat([s.gravity         for s in batch], dim=0),
            correct_gravity = torch.cat([s.correct_gravity for s in batch], dim=0),
            acc_bias     = torch.cat([s.acc_bias     for s in batch], dim=0),
            gyro_bias    = torch.cat([s.gyro_bias    for s in batch], dim=0),
            offsets_B    = _merge_offsets([s.offsets_B for s in batch], pack_sizes),
            offsets_N    = _merge_offsets([s.offsets_N for s in batch], pack_sizes),
            sensor_id    = torch.cat([s.sensor_id    for s in batch], dim=0),
            model        = batch[0].model,
            kind         = batch[0].kind,
        )


@dataclass
class IMUData_LabelTrait_Jagged:
    labels      : jt.UInt8[torch.Tensor, "pack_BN"]
    gt_pose     : jt.Float[pp.LieTensor, "pack_BNT 7"]
    gt_vel      : jt.Float[torch.Tensor, "pack_BNT 3"]

    @staticmethod
    def _collate_label_fields(batch: Sequence[IMUData_LabelTrait_Jagged]) -> dict:
        return dict(
            labels  = torch.cat([s.labels for s in batch], dim=0),
            gt_pose = pp.SE3(torch.cat([s.gt_pose.tensor() for s in batch], dim=0)),
            gt_vel  = torch.cat([s.gt_vel for s in batch], dim=0),
        )

    def _clone_label_fields(self) -> dict:
        return dict(
            labels  = self.labels.clone(),
            gt_pose = self.gt_pose.clone(),
            gt_vel  = self.gt_vel.clone(),
        )

    def _to_label_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        """Apply ``.to`` into a kwargs dict, clamping sub-fp32 casts to fp32 (see `_label_dtype`)."""
        dtype = _label_dtype(dtype)
        return dict(
            labels  = _field_to(self.labels, device, None),
            gt_pose = _field_to(self.gt_pose, device, dtype),
            gt_vel  = _field_to(self.gt_vel, device, dtype),
        )


@dataclass
class IMUData_Jagged(IMUData_BaseTrait_Jagged):
    @classmethod
    def collate(cls, batch: Sequence[Self]) -> Self:
        """Merge single-sample jagged windows into one batched ``IMUData_Jagged``."""
        return cls(**cls._collate_base_fields(batch))

    def clone(self) -> IMUData_Jagged:
        """Return a deep copy with all tensor fields cloned."""
        return IMUData_Jagged(**self._clone_base_fields())

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUData_Jagged:
        """Return a copy on `device`, floating-point fields cast to `dtype`."""
        return IMUData_Jagged(**self._to_base_fields(device, dtype))


@dataclass
class IMUData_w_GT_Jagged(IMUData_BaseTrait_Jagged, IMUData_LabelTrait_Jagged):
    @classmethod
    def collate(cls, batch: Sequence[Self]) -> Self:
        """Merge single-sample jagged windows into one batched ``IMUData_w_GT_Jagged``."""
        return cls(
            **cls._collate_base_fields(batch),
            **cls._collate_label_fields(batch),
        )

    def clone(self) -> IMUData_w_GT_Jagged:
        """Return a deep copy with all tensor fields cloned."""
        return IMUData_w_GT_Jagged(
            **self._clone_base_fields(),
            **self._clone_label_fields(),
        )

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUData_w_GT_Jagged:
        """Return a copy on `device`, floating-point fields cast to `dtype`."""
        return IMUData_w_GT_Jagged(
            **self._to_base_fields(device, dtype),
            **self._to_label_fields(device, dtype),
        )


@dataclass
class IMUData_w_GT_Lang_Jagged(IMUData_BaseTrait_Jagged, IMUData_LabelTrait_Jagged, LanguageFeature_Trait):
    @classmethod
    def collate(cls, batch: Sequence[Self]) -> Self:
        """Merge single-sample jagged windows into one batched ``IMUData_w_GT_Lang_Jagged``."""
        return cls(
            **cls._collate_base_fields(batch),
            **cls._collate_label_fields(batch),
            **cls._collate_lang_fields(batch),
        )

    def clone(self) -> IMUData_w_GT_Lang_Jagged:
        """Return a deep copy with all tensor fields cloned."""
        return IMUData_w_GT_Lang_Jagged(
            **self._clone_base_fields(),
            **self._clone_label_fields(),
            **self._clone_lang_fields(),
        )

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> IMUData_w_GT_Lang_Jagged:
        """Return a copy on `device`, floating-point fields cast to `dtype`."""
        return IMUData_w_GT_Lang_Jagged(
            **self._to_base_fields(device, dtype),
            **self._to_label_fields(device, dtype),
            **self._to_lang_fields(device, dtype),
        )
