"""Language-embedding traits for IMU samples."""
from __future__ import annotations

import torch
import jaxtyping as jt

from dataclasses import dataclass
from typing import Sequence, Any, Self

from .data import _field_to


@dataclass
class LanguageFeature_Trait:
    """Compact per-sample language embeddings: ``lang_feature`` has one row per labelled sample.

    Fields have no time/device axis, so padded and jagged layouts coincide.
    Gather with ``lang_feature[lang_feat_idx[lang_feat_idx >= 0]]``.
    """
    lang_feat_idx   : jt.Int64[torch.Tensor, "B"]         # Row into lang_feature; -1 -> no embedding.
    lang_feature    : jt.Float[torch.Tensor, "V C_lang"]  # Compact storage, one row per valid sample.

    def clone(self) -> LanguageFeature_Trait:
        """Return a deep copy of the concrete type."""
        raise NotImplementedError("clone() must be implemented by the concrete LanguageFeature class")

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> LanguageFeature_Trait:
        """Return a copy on `device`; `lang_feature` casts to `dtype`, `lang_feat_idx` never does."""
        raise NotImplementedError("to() must be implemented by the concrete LanguageFeature class")

    @classmethod
    def collate(cls, batch: Sequence[Any]) -> Self:
        """Merge samples, concatenating storages and re-basing indices; ``batch`` is ``Any`` so overrides can narrow it to their own class."""
        raise NotImplementedError("collate() must be implemented by the concrete LanguageFeature class")

    def as_jagged(self) -> LanguageFeature_Trait:
        """Return the jagged form; concrete classes rebuild the matching jagged type."""
        raise NotImplementedError("as_jagged() must be implemented by the concrete LanguageFeature class")

    @staticmethod
    def _collate_lang_fields(batch: Sequence[LanguageFeature_Trait]) -> dict:
        """Concatenate compact storages and shift each sample's valid indices; ``-1`` entries stay."""
        idx_shifted: list[torch.Tensor] = []
        running = 0
        for s in batch:
            idx = s.lang_feat_idx
            idx_shifted.append(torch.where(idx >= 0, idx + running, idx))
            running += s.lang_feature.shape[0]
        return dict(
            lang_feat_idx = torch.cat(idx_shifted, dim=0),
            lang_feature  = torch.cat([s.lang_feature for s in batch], dim=0),
        )

    def _clone_lang_fields(self) -> dict:
        return dict(
            lang_feat_idx = self.lang_feat_idx.clone(),
            lang_feature  = self.lang_feature.clone(),
        )

    def _to_lang_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        return dict(
            lang_feat_idx = _field_to(self.lang_feat_idx, device, None),
            lang_feature  = _field_to(self.lang_feature,  device, dtype),
        )

    def _as_jagged_lang_fields(self) -> dict:
        return dict(
            lang_feat_idx = self.lang_feat_idx,
            lang_feature  = self.lang_feature,
        )


@dataclass
class LanguageFeature_TimeTrait:
    """Per-sample language annotation span in ns, on the same clock as the IMU ``time_ns``.

    Dense over B, not per ``lang_feature`` row; undefined where ``lang_feat_idx == -1``.
    """
    lang_time_ns     : jt.Int64[torch.Tensor, "B"]  # Annotation span start.
    lang_time_end_ns : jt.Int64[torch.Tensor, "B"]  # Annotation span end.

    @staticmethod
    def _collate_time_fields(batch: Sequence[LanguageFeature_TimeTrait]) -> dict:
        return dict(
            lang_time_ns     = torch.cat([s.lang_time_ns     for s in batch], dim=0),
            lang_time_end_ns = torch.cat([s.lang_time_end_ns for s in batch], dim=0),
        )

    def _clone_time_fields(self) -> dict:
        return dict(
            lang_time_ns     = self.lang_time_ns.clone(),
            lang_time_end_ns = self.lang_time_end_ns.clone(),
        )

    def _to_time_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        return dict(
            lang_time_ns     = _field_to(self.lang_time_ns,     device, None),
            lang_time_end_ns = _field_to(self.lang_time_end_ns, device, None),
        )

    def _as_jagged_time_fields(self) -> dict:
        return dict(
            lang_time_ns     = self.lang_time_ns,
            lang_time_end_ns = self.lang_time_end_ns,
        )


@dataclass
class LanguageFeature_Data(LanguageFeature_Trait, LanguageFeature_TimeTrait):
    @classmethod
    def collate(cls, batch: Sequence[Self]) -> Self:
        """Merge single-window samples into one batched ``LanguageFeature_Data``."""
        return cls(
            **cls._collate_lang_fields(batch),
            **cls._collate_time_fields(batch),
        )

    def clone(self) -> LanguageFeature_Data:
        """Return a deep copy with all tensor fields cloned."""
        return LanguageFeature_Data(
            **self._clone_lang_fields(),
            **self._clone_time_fields(),
        )

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> LanguageFeature_Data:
        """Return a copy on `device`, floating-point fields cast to `dtype`."""
        return LanguageFeature_Data(
            **self._to_lang_fields(device, dtype),
            **self._to_time_fields(device, dtype),
        )

    def as_jagged(self) -> LanguageFeature_Data:
        """Rebuild as-is: per-sample fields are layout-identical when jagged."""
        return LanguageFeature_Data(
            **self._as_jagged_lang_fields(),
            **self._as_jagged_time_fields(),
        )
