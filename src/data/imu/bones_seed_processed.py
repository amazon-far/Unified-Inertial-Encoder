"""Readers for the BONES-SEED format-v3 caches (G1 and SOMA) with optional language sidecars."""

import json
import re
import torch
import typing as tp

from pathlib import Path
from functools import cached_property

import jaxtyping as jt
import numpy as np
import pypose as pp

from ...interface.data      import ArticulatedModel
from ...interface.data_imu  import IMUData_w_GT_Lang, Loss_Flags, gravity_of_kind
from ...interface.human     import CANONICAL_IDX, T_CANONICAL_NAMES
from ...interface.sequence  import ISequence
from ...utility.clip        import _FEAT_DIM_

from .bones_seed_soma         import CANONICALIZED_BONES_SOMA
from ..humanoid.g1_superodom  import G1_WholeBody_Link_Mapping


# Within-motion variant tokens: mirror/version, turn angle, travel direction.
# Matched against the lowercased stem, so alternatives are all lowercase.
VARIANT_SUFFIX = re.compile(
    r"_(r|l|v\d+"
    r"|45|90|135|180|225|270|315|360"
    r"|cw|ccw|left|right|front|back|forward|backward|up|down)$"
)


def canonical_motion(path: str) -> str:
    """Collapse a listing path to its motion identity, the key of bones_seed_split_groups.json.

    Lowercases the stem, then strips the extension, actor id, take token and every trailing
    variant suffix, so mirrored/rotated/directional takes collapse together.
    """
    stem = re.sub(r"\.(csv|bvh)$", "", path.rsplit("/", 1)[-1].lower())
    stem = re.sub(r"__a\d+$", "", stem)
    stem = re.sub(r"_?0\d\d[a-z]?(0\d\d)?$", "", stem)
    assert stem, f"canonicalization emptied {path!r}"
    while (shorter := VARIANT_SUFFIX.sub("", stem)) != stem:
        stem = shorter
    return stem


def _decode_bf16_row(raw: np.ndarray) -> torch.Tensor:
    """A ``(C,)`` ``ml_dtypes.bfloat16`` row (opaque ``|V2`` to numpy) as fp32."""
    return torch.from_numpy(raw.view(np.uint16)).view(torch.bfloat16).float()


def _first_containing_row(
    starts: np.ndarray, ends: np.ndarray, idx: np.ndarray, w0: int, w1: int,
) -> int:
    """The CLIP row of the first event fully containing window ``[w0, w1]`` (inclusive ns), or ``-1``."""
    hits = np.nonzero((starts <= w0) & (w1 < ends))[0]
    return int(idx[hits[0]]) if hits.size else -1


class _Bones_Seed_Processed_IMU_Lang(ISequence[IMUData_w_GT_Lang]):
    """Shared implementation of the concrete readers, which differ only in ``_DEFAULT_SITES``."""

    APPLICABLE_LOSS = Loss_Flags.JEPA | Loss_Flags.Vel_SSIO | Loss_Flags.Lang_CLIP | Loss_Flags.Vel_L2 | Loss_Flags.Denoise_L2 \
                    | Loss_Flags.Traj_Denoise | Loss_Flags.Traj_Vel

    _DEFAULT_SITES: tp.ClassVar[tuple[T_CANONICAL_NAMES, ...]]

    def __init__(self, path: str, target_sensors: tp.Sequence[T_CANONICAL_NAMES] | None = None):
        self.path = Path(path)
        target_sensors = self._DEFAULT_SITES if target_sensors is None else tuple(target_sensors)

        with open(self.path / "spec.json") as f:
            spec = json.load(f)
        names = spec["names"]
        missing = set(target_sensors) - set(names)
        assert not missing, f"{self.path}: cache lacks site(s) {sorted(missing)}"

        self.kind: tp.Literal["real", "filtered-no-g", "sim"] = spec["kind"]
        self.model = ArticulatedModel(names=tuple(target_sensors), urdf=None)

        self._T          = int(spec["length"])
        self._sensor_idx = [names.index(s) for s in target_sensors]
        self._fields     = spec["measure"]["fields"]          # name -> {col: [s, e), shape}
        self._measure_file = self.path / spec["measure"]["file"]
        self._pose_file    = self.path / spec["pose"]["file"]
        self._time_file    = self.path / spec["time_ns"]["file"]
        # Opened lazily per process (see __getstate__).
        self._measure: np.ndarray | None = None
        self._pose:    np.ndarray | None = None
        self._time_ns: np.ndarray | None = None

        self.sensor_id = torch.tensor(
            [CANONICAL_IDX[s] for s in target_sensors], dtype=torch.uint8
        ).view(1, -1, 1, 1)

        lang_spec_file = self.path / "lang_spec.json"
        self._has_lang = lang_spec_file.exists()
        if not self._has_lang:
            self._lang_dim   = _FEAT_DIM_
            self._lang_file  = None
            self._lang_idx   = np.empty(0, dtype=np.int64)
            self._lang_start = np.empty(0, dtype=np.int64)
            self._lang_end   = np.empty(0, dtype=np.int64)
            self._lang_feature: np.ndarray | None = None
            return

        with open(lang_spec_file) as f:
            lang_spec = json.load(f)
        self._lang_dim  = int(lang_spec["feature"]["shape"][1])
        self._lang_file = self.path / lang_spec["feature"]["file"]
        # (N_events,) rows into lang_feature.npy (-1 = blank) and (N_events, 2) ns spans, in event order.
        self._lang_idx   = np.load(self.path / lang_spec["index"]["file"])
        spans            = np.load(self.path / lang_spec["time_ns"]["file"]).reshape(-1, 2)
        self._lang_start = spans[:, 0]
        self._lang_end   = spans[:, 1]
        assert self._lang_idx.shape[0] == spans.shape[0], \
            f"{self.path}: lang_index ({self._lang_idx.shape[0]}) and spans ({spans.shape[0]}) disagree"
        self._lang_feature = None    # lazy mmap, like the IMU arrays

    @cached_property
    def sample_dt_s(self) -> float:
        """Fixed 120 Hz source clock, so construction needs no array I/O."""
        return 1.0 / 120.0

    @cached_property
    def num_devices(self) -> int:
        """Number of selected cached sites, available without reading a sample."""
        return len(self._sensor_idx)

    def __getstate__(self):
        # A live np.memmap pickles as the whole mapped file, so workers re-open lazily.
        return {**self.__dict__, "_measure": None, "_pose": None, "_time_ns": None, "_lang_feature": None}

    def _open_time(self) -> np.ndarray:
        time_ns = self._time_ns
        if time_ns is None:
            time_ns = self._time_ns = np.load(self._time_file, mmap_mode="r")
        return time_ns

    def _open(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        measure, pose = self._measure, self._pose
        if measure is None or pose is None:
            measure = self._measure = np.load(self._measure_file, mmap_mode="r")
            pose    = self._pose    = np.load(self._pose_file,    mmap_mode="r")
        return measure, pose, self._open_time()

    def _open_lang(self) -> np.ndarray:
        assert self._lang_file is not None      # only reached for a valid row
        feature = self._lang_feature
        if feature is None:
            feature = self._lang_feature = np.load(self._lang_file, mmap_mode="r")
        return feature

    def __len__(self) -> int:
        return self._T

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        # Windowing reads this at construction: open only the time axis.
        return torch.from_numpy(np.array(self._open_time(), dtype=np.int64))

    def __getitem__(self, index: slice | int) -> IMUData_w_GT_Lang:
        if isinstance(index, int):
            i = index + self._T if index < 0 else index
            if i < 0 or i >= self._T:
                raise IndexError(f"index {index} out of range for length {self._T}")
            sl = slice(i, i + 1)
        else:
            sl = index

        measure, pose, time_ns = self._open()

        # Device-major (N_all, T, ...): slice time, no transpose. np.array copies the window off
        # the read-only mmap.
        m = torch.from_numpy(np.array(measure[:, sl]))                      # (N_all, t, F)
        p = torch.from_numpy(np.array(pose[:, sl]))                        # (N_all, t, 7)
        n = len(self._sensor_idx)
        t = m.shape[1]

        def field(name: str) -> torch.Tensor:                              # (1, n, t, C)
            s, e = self._fields[name]["col"]
            return m[self._sensor_idx, :, s:e].unsqueeze(0)

        acc     = field("acc")
        gyro    = field("gyro")
        gt_vel  = field("gt_vel")
        gt_pose = pp.SE3(p[self._sensor_idx].unsqueeze(0))                 # (1, n, t, 7)

        t_axis   = torch.from_numpy(np.array(time_ns[sl], dtype=np.int64))  # (t,)
        time_exp = t_axis.view(1, 1, -1, 1).expand(1, n, t, 1).contiguous()
        gravity, correct_gravity = gravity_of_kind(self.kind, 1, n)

        # Language: the first event containing this window, if any.
        w0 = int(t_axis[0])  if t > 0 else 0
        w1 = int(t_axis[-1]) if t > 0 else -1
        row = _first_containing_row(self._lang_start, self._lang_end, self._lang_idx, w0, w1) \
            if (self._has_lang and t > 0) else -1
        if row >= 0:
            feature  = _decode_bf16_row(np.array(self._open_lang()[row])).unsqueeze(0)
            feat_idx = torch.zeros(1, dtype=torch.int64)
            flags    = self.APPLICABLE_LOSS
        else:
            feature  = torch.empty(0, self._lang_dim)
            feat_idx = torch.full((1,), -1, dtype=torch.int64)
            flags    = self.APPLICABLE_LOSS & ~Loss_Flags.Lang_CLIP

        return IMUData_w_GT_Lang(
            acc          = acc,
            gyro         = gyro,
            time_ns      = time_exp,
            labels       = torch.full(time_exp.shape[:2], int(flags), dtype=torch.uint8),
            acc_bias     = torch.zeros_like(acc),
            gyro_bias    = torch.zeros_like(gyro),

            correct_mask = torch.zeros(acc.shape[:-1], dtype=torch.bool),
            correct_acc  = acc.clone(),
            correct_gyro = gyro.clone(),
            gravity         = gravity,
            correct_gravity = correct_gravity,

            gt_pose      = gt_pose,
            gt_vel       = gt_vel,

            sensor_id    = self.sensor_id.expand(1, -1, t, -1).contiguous(),
            model        = self.model,
            kind         = self.kind,

            lang_feat_idx = feat_idx,
            lang_feature  = feature,
        )


class Bones_Seed_G1_Processed_IMU_w_Lang(_Bones_Seed_Processed_IMU_Lang):
    """G1 humanoid cache from ``src.scripts.bonesseed_imu_process`` plus its temporal language.

    ``kind`` is "filtered-no-g" (gravity-free acc).
    """

    _DEFAULT_SITES = tuple(G1_WholeBody_Link_Mapping.keys())


class Bones_Seed_SOMA_Processed_IMU_w_Lang(_Bones_Seed_Processed_IMU_Lang):
    """SOMA human cache from ``src.scripts.bonesseed_soma_process`` plus its temporal language."""

    _DEFAULT_SITES = tuple(CANONICALIZED_BONES_SOMA)
