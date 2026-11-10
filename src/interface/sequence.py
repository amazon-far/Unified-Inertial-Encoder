"""Sequence dataset interface shared by leaf datasets and wrappers."""
import torch
import typing as tp
import jaxtyping as jt

from functools import cached_property
from torch.utils.data import Dataset

from .base import Base
from .data_imu import Loss_Flags


T_Data = tp.TypeVar("T_Data")


class ISequence(Dataset[T_Data], Base):
    APPLICABLE_LOSS = Loss_Flags.EMPTY
    
    def __getitem__(self, index: slice | int) -> T_Data:
        ...

    def __len__(self) -> int:
        ...

    def get_loss_label(self, shape: torch.Size) -> jt.UInt8[torch.Tensor, "..."]:
        return torch.empty(shape, dtype=torch.uint8).fill_(self.APPLICABLE_LOSS)

    def _forward_target(self) -> "ISequence | None":
        """The wrapped sequence, if any; wrappers hold it as ``source`` or ``_imu``."""
        target = getattr(self, "source", None)
        if target is None:
            target = getattr(self, "_imu", None)
        return target

    @cached_property
    def sample_dt_s(self) -> float:
        """Representative native sample interval in seconds; wrappers forward to their source."""
        target = self._forward_target()
        if target is not None:
            return float(target.sample_dt_s)
        time_ns = self.timestamps
        if time_ns.numel() < 2:
            raise ValueError(f"{type(self).__name__} needs at least two timestamps to estimate sample_dt_s")
        return float(time_ns.diff().to(torch.float64).median() * 1e-9)

    @cached_property
    def num_devices(self) -> int:
        """Number of devices emitted before runtime augmentation; wrappers forward to their source."""
        target = self._forward_target()
        if target is not None:
            return int(target.num_devices)
        model = getattr(self, "model", None)
        if model is None:
            raise AttributeError(f"{type(self).__name__} exposes neither source nor model for num_devices")
        return int(model.num_joint)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        """Per-index timestamps in ns, available without materializing the measurements.

        Leaf datasets must override this. Not forwarded automatically: for index-remapping wrappers
        (e.g. windowing) a pass-through would hand out a misaligned time axis.
        """
        raise NotImplementedError(f"{type(self).__name__} does not expose per-index timestamps")
