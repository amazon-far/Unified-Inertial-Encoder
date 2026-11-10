"""Nymeria cache reader with precomputed CLIP language features."""

import json
import torch
import dataclasses
import numpy as np
import typing as tp
import jaxtyping as jt

from pathlib import Path

from .nymeria_processed    import Nymeria_XSens_Processed_IMU
from ...interface.sequence import ISequence
from ...interface.data_imu import IMUData_w_GT_Lang, Loss_Flags
from ...interface.human    import T_CANONICAL_NAMES


class Nymeria_XSens_Processed_IMU_Language(ISequence[IMUData_w_GT_Lang]):
    """``Nymeria_XSens_Processed_IMU`` plus the features of ``src.scripts.nymeria_lang_process``.

    Annotations are sparse ``[start, end)`` ns spans on the IMU clock. A window gets a feature
    only if spans fully cover it and all map to one valid row (see ``_lang_row``); otherwise
    ``lang_feat_idx == -1``, the feature is empty and ``Lang_CLIP`` is cleared. A sequence
    without a lang spec is usable IMU-only and behaves like an unannotated one.
    """

    # CLIP ViT-L/14 embedding width; shapes the empty feature.
    _LANG_DIM = 768

    APPLICABLE_LOSS = Loss_Flags.Denoise_L2 \
                    | Loss_Flags.Lang_CLIP \
                    | Loss_Flags.Vel_L2 \
                    | Loss_Flags.JEPA \
                    | Loss_Flags.Traj_Denoise \
                    | Loss_Flags.Traj_Vel

    def __init__(self, path: str, target_sensors: tp.Sequence[T_CANONICAL_NAMES],
                 lang_variant: str | None = None):
        self.path = Path(path)
        self._imu = Nymeria_XSens_Processed_IMU(path, target_sensors)

        # `lang_variant` selects a sidecar set written by src.scripts.nymeria_lang_process --variant.
        spec_file = self.path / (f"lang_spec.{lang_variant}.json" if lang_variant else "lang_spec.json")
        self._has_lang = spec_file.exists()
        if not self._has_lang:
            self._lang_dim  = self._LANG_DIM
            self._lang_file = None
            self._lang_idx   = np.empty(0, dtype=np.int64)
            self._lang_start = np.empty(0, dtype=np.int64)
            self._lang_end   = np.empty(0, dtype=np.int64)
            self._lang_feature: np.ndarray | None = None
            return

        with open(spec_file) as f:
            lang_spec = json.load(f)
        self._lang_dim  = int(lang_spec["feature"]["shape"][1])
        self._lang_file = self.path / lang_spec["feature"]["file"]

        # (N_ann,) row into lang_feature.npy (-1 = no embedding) and (N_ann,) span bounds.
        self._lang_idx   = np.load(self.path / lang_spec["index"]["file"])
        spans            = np.load(self.path / lang_spec["time_ns"]["file"])
        self._lang_start = spans[:, 0]
        self._lang_end   = spans[:, 1]
        assert self._lang_idx.shape[0] == spans.shape[0], \
            f"{self.path}: lang_index ({self._lang_idx.shape[0]}) and lang_time_ns ({spans.shape[0]}) disagree"
        # `_lang_row` assumes spans sorted by start.
        assert bool((np.diff(self._lang_start) >= 0).all()), \
            f"{self.path}: lang_time_ns spans must be sorted by start"

        self._lang_feature = None    # Lazy mmap, like the wrapped reader.

    def __getstate__(self):
        # A live memmap pickles as a full ndarray; the wrapped `_imu` drops its own handles.
        return {**self.__dict__, "_lang_feature": None}

    def _open_lang(self) -> np.ndarray:
        assert self._lang_file is not None
        feature = self._lang_feature
        if feature is None:
            feature = self._lang_feature = np.load(self._lang_file, mmap_mode="r")
        return feature

    def __len__(self) -> int:
        return len(self._imu)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        return self._imu.timestamps

    def _lang_row(self, w0: int, w1: int) -> int:
        """The feature row covering window ``[w0, w1]`` (inclusive ns), or ``-1``.

        Spans are half-open. Valid iff every intersecting span maps to one non-negative row and
        they cover the window with no gap (spans may overlap, hence the running max).
        """
        hit = (self._lang_start <= w1) & (self._lang_end > w0)
        rows = np.unique(self._lang_idx[hit])
        if rows.size != 1 or rows[0] < 0:
            return -1

        s, e = self._lang_start[hit], self._lang_end[hit]
        if s[0] > w0 or int(e.max()) <= w1:
            return -1
        if bool((np.maximum.accumulate(e)[:-1] < s[1:]).any()):
            return -1
        return int(rows[0])

    def __getitem__(self, index: slice | int) -> IMUData_w_GT_Lang:
        base = self._imu[index]

        t = base.acc.shape[2]
        w0 = int(base.time_ns[0, 0,  0, 0]) if t > 0 else 0
        w1 = int(base.time_ns[0, 0, -1, 0]) if t > 0 else -1
        row = self._lang_row(w0, w1) if t > 0 else -1

        if row >= 0:
            # bfloat16 is opaque |V2 to numpy: reinterpret through uint16, then widen to fp32.
            raw  = np.array(self._open_lang()[row])
            feat = torch.from_numpy(raw.view(np.uint16)).view(torch.bfloat16).float().unsqueeze(0)
            feat_idx = torch.zeros(1, dtype=torch.int64)
            flags    = self.APPLICABLE_LOSS
        else:
            feat     = torch.empty(0, self._lang_dim)
            feat_idx = torch.full((1,), -1, dtype=torch.int64)
            flags    = self.APPLICABLE_LOSS & ~Loss_Flags.Lang_CLIP

        # Labels are re-stamped so Lang_CLIP is set only when the window has a feature.
        fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        fields["labels"] = torch.full_like(base.labels, int(flags))
        return IMUData_w_GT_Lang(
            **fields,
            lang_feat_idx = feat_idx,
            lang_feature  = feat,
        )
