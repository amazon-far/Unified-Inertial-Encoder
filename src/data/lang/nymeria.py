"""On-the-fly CLIP language features for Nymeria narration annotations."""

import csv
import torch
import numpy as np
import jaxtyping as jt

from functools import cached_property
from pathlib import Path

from ...interface.sequence  import ISequence
from ...interface.data_imu  import Loss_Flags
from ...interface.data_lang import LanguageFeature_Data
from ...utility.clip import CLIP_Model, _FEAT_DIM_

_CSV_NAME_  = "transcribed_activity_summarization.csv"
_TEXT_COL_  = "Describe my activity"
_XDATA_NAME_ = "xdata.npz"


class Nymeria_Language(ISequence[LanguageFeature_Data]):
    """Per-annotation CLIP text features for a Nymeria sequence, encoded on the fly.

    One index is one row of the ``src.scripts.nymeria_lang_transcribe`` CSV, sorted by span start.
    CSV times are on the Aria clock; ``__init__`` re-bases them onto the XSens IMU clock, so
    ``timestamps`` and the emitted spans need no further correction.
    """

    APPLICABLE_LOSS = Loss_Flags.Lang_CLIP

    # One frozen CLIP text tower per process, built lazily on first encode. As a class attribute
    # it is shared by all instances and never pickled into DataLoader workers.
    _clip: CLIP_Model | None = None

    def __init__(self, path: str, csv_name: str = _CSV_NAME_):
        # csv_name selects the transcription style (see _STYLES_ in src.scripts.nymeria_lang_transcribe).
        self.path = Path(path, "narration", csv_name)
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None and _TEXT_COL_ in reader.fieldnames, \
                f"{self.path} lacks the '{_TEXT_COL_}' column"
            rows = sorted(reader, key=lambda row: float(row["start_time"]))

        # Span bounds in seconds, on the Aria clock.
        self.texts = [row[_TEXT_COL_].strip() for row in rows]
        start_s = [float(row["start_time"]) for row in rows]
        end_s   = [float(row["end_time"])   for row in rows]

        # Re-base onto the IMU clock by anchoring the earliest annotation to the XSens start.
        with np.load(Path(path, "body", _XDATA_NAME_), mmap_mode="r") as xdata:
            imu_t0_ns = int(xdata["timestamps_us"][0]) * 1000
        offset_ns = imu_t0_ns - round(start_s[0] * 1e9)
        self.start_ns = torch.tensor([round(s * 1e9) + offset_ns for s in start_s], dtype=torch.int64)
        self.end_ns   = torch.tensor([round(s * 1e9) + offset_ns for s in end_s],   dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.texts)

    @cached_property
    def num_devices(self) -> int:
        return 0

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        return self.start_ns

    @classmethod
    def _encoder(cls) -> CLIP_Model:
        if cls._clip is None:
            cls._clip = CLIP_Model()
        return cls._clip

    def __getitem__(self, index: slice | int) -> LanguageFeature_Data:
        """Computes the language CLIP feature on-the-fly for the queried batch"""
        if isinstance(index, int):
            i = index + len(self) if index < 0 else index
            if i < 0 or i >= len(self):
                raise IndexError(f"index {index} out of range for length {len(self)}")
            sl = slice(i, i + 1)
        else:
            sl = index

        texts = self.texts[sl]

        # Duplicate phrases share one feature row; empty phrases have no embedding and map to -1.
        feat_idx = torch.full((len(texts),), -1, dtype=torch.int64)
        row_of: dict[str, int] = {}
        for i, text in enumerate(texts):
            if text:
                feat_idx[i] = row_of.setdefault(text, len(row_of))

        feature = self._encoder()(list(row_of)).cpu() \
            if row_of else torch.empty(0, _FEAT_DIM_)

        return LanguageFeature_Data(
            lang_feat_idx    = feat_idx,
            lang_feature     = feature,
            lang_time_ns     = self.start_ns[sl].clone(),
            lang_time_end_ns = self.end_ns[sl].clone(),
        )
