"""Virtual IMU synthesis and temporal language for BONES-SEED G1 trajectories."""

import dataclasses
import json

from pathlib import Path

import jaxtyping as jt
import numpy as np
import pypose as pp
import torch

from ...interface.sequence      import ISequence
from ...interface.data_humanoid import ArticulatedJointMeasures_wGT, _windowed_time_derivative
from ...interface.data_imu      import IMUData_w_GT, IMUData_w_GT_Lang, Loss_Flags
from ...interface.human         import CANONICAL_IDX
from ...utility.clip            import CLIP_Model, _FEAT_DIM_

from ..humanoid.g1_superodom import (
    G1_29DOF_Model, G1_WholeBody_IMU_Model, G1_WholeBody_Link_Mapping,
)
from .g1_superodom           import Unitree_G1_SuperOdom_IMU


_FPS_         = 120.0
_CM_TO_M_     = 0.01
_LABELS_NAME_ = "seed_metadata_v002_temporal_labels.jsonl"

_ROOT_COLUMNS_ = (
    "Frame",
    "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX",    "root_rotateY",    "root_rotateZ",
)


def _expected_header() -> tuple[str, ...]:
    return _ROOT_COLUMNS_ + tuple(f"{name}_dof" for name in G1_29DOF_Model.names)


class Bones_Seed_G1_IMU(Unitree_G1_SuperOdom_IMU):
    """Synthesize whole-body virtual IMUs from one BONES-SEED G1 trajectory.

    Root and joint derivatives come from smoothed finite differences of the 120 Hz CSV.
    Acceleration stays gravity-free; ``AddGravity`` adds gravity downstream.
    """

    Link_Mapping        = G1_WholeBody_Link_Mapping
    WholeBody_IMU_Model = G1_WholeBody_IMU_Model

    APPLICABLE_LOSS = Loss_Flags.JEPA | Loss_Flags.Vel_SSIO

    ROOT_DERIV_WINDOW_S = 0.031

    @classmethod
    def _imu_sites(cls) -> tuple[list[str], list[int]]:
        """Synthesis targets: URDF link frames and per-device sensor ids, index-aligned with
        ``WholeBody_IMU_Model.names``."""
        return list(cls.Link_Mapping.values()), [CANONICAL_IDX[site] for site in cls.Link_Mapping]

    def __init__(self, path: str):
        self.path       = Path(path)
        self.root_frame = "pelvis"
        self.base_frame = "pelvis"

        q_deg, root_t_cm, root_e_deg = self._read_csv(self.path)
        T = q_deg.shape[0]
        if T < 2:
            raise ValueError(f"{self.path}: expected at least 2 frames, got {T}")

        time_ns = torch.from_numpy(
            np.round(np.arange(T) * (1e9 / _FPS_)).astype(np.int64)
        ).view(1, T, 1)

        p_w = torch.from_numpy(root_t_cm * _CM_TO_M_).float().view(1, T, 3)
        # BONES-SEED uses intrinsic ZYX Euler angles; pypose accepts roll, pitch, yaw.
        R_w  = pp.euler2SO3(torch.from_numpy(np.deg2rad(root_e_deg)).float().view(1, T, 3))
        R_bw = R_w.matrix().transpose(-1, -2)

        root_pose = pp.SE3(torch.cat([p_w, R_w.tensor()], dim=-1))
        v_w = _windowed_time_derivative(p_w, time_ns, self.ROOT_DERIV_WINDOW_S)
        a_w = _windowed_time_derivative(v_w, time_ns, self.ROOT_DERIV_WINDOW_S)

        root_omega = self._body_angular_velocity(R_w, time_ns, self.ROOT_DERIV_WINDOW_S)
        root_vel   = (R_bw @ v_w.unsqueeze(-1)).squeeze(-1)

        self.base_acc  = (R_bw @ a_w.unsqueeze(-1)).squeeze(-1)
        self.base_gyro = root_omega

        q_tm = torch.from_numpy(np.deg2rad(q_deg)).float().view(
            1, T, G1_29DOF_Model.num_joint
        )
        qd_tm = _windowed_time_derivative(q_tm, time_ns, self.JOINT_DD_WINDOW_S)

        self.joint_store = ArticulatedJointMeasures_wGT(
            root_frame = self.root_frame,
            root_pose  = root_pose,
            root_vel   = root_vel,
            root_omega = root_omega,
            joints     = q_tm.unsqueeze(-1).transpose(1, 2),
            joint_d    = qd_tm.unsqueeze(-1).transpose(1, 2),
            time_ns    = time_ns,
            model      = G1_29DOF_Model,
            kind       = "sim",
        )

        target_frames, sensor_ids = self._imu_sites()
        self.imu_store = self.synthesize_at(
            target_frames = target_frames,
            sensor_ids    = sensor_ids,
            imu_model     = self.WholeBody_IMU_Model,
            kind          = "filtered-no-g",
        )

    @staticmethod
    def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read joint angles, root translation, and root Euler angles in native units."""
        with open(path, newline="") as f:
            header = tuple(f.readline().strip().split(","))

        expected = _expected_header()
        if header != expected:
            raise ValueError(
                f"{path}: unexpected CSV header; expected root fields followed by "
                "G1_29DOF_Model.names"
            )

        raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[None]
        if raw.shape[1] != len(expected):
            raise ValueError(f"{path}: expected {len(expected)} columns, got {raw.shape[1]}")
        if not np.isfinite(raw).all():
            raise ValueError(f"{path}: CSV contains non-finite values")
        if not np.array_equal(raw[:, 0], np.arange(raw.shape[0])):
            raise ValueError(f"{path}: Frame must be a contiguous zero-based index")

        return raw[:, 7:], raw[:, 1:4], raw[:, 4:7]

    @staticmethod
    def _body_angular_velocity(
        R            : jt.Float[pp.LieTensor, "B T 4"],
        time_ns      : jt.Int64[torch.Tensor, "B T 1"],
        half_window_s: float,
    ) -> jt.Float[torch.Tensor, "B T 3"]:
        """Compute body angular velocity with a symmetric SO3 finite difference."""
        t      = time_ns[..., 0]
        dt_med = float(t.diff(dim=1).median()) / 1e9
        k      = max(1, round(half_window_s / dt_med)) if half_window_s > 0 else 1

        T   = R.shape[1]
        idx = torch.arange(T, device=R.device)
        hi  = (idx + k).clamp(max=T - 1)
        lo  = (idx - k).clamp(min=0)
        den = ((t[:, hi] - t[:, lo]).float() / 1e9).unsqueeze(-1)

        rel = pp.SO3(R[:, lo]).Inv() @ pp.SO3(R[:, hi])
        return rel.Log().tensor() / den

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        if isinstance(index, int):
            original = index
            index = index + len(self) if index < 0 else index
            if index < 0 or index >= len(self):
                raise IndexError(f"index {original} out of range for length {len(self)}")
        return super().__getitem__(index)


class Bones_Seed_IMU_w_GT_Lang(ISequence[IMUData_w_GT_Lang]):
    """Add BONES-SEED temporal descriptions to the synthesized IMU sequence.

    A window gets a CLIP feature only if it lies fully inside one half-open event; otherwise
    ``lang_feat_idx == -1`` and ``Lang_CLIP`` is cleared.
    """

    APPLICABLE_LOSS = Loss_Flags.JEPA | Loss_Flags.Vel_SSIO | Loss_Flags.Lang_CLIP

    _clip: CLIP_Model | None = None
    _labels: dict[Path, dict[str, list[tuple[int, int, str]]]] = {}

    def __init__(self, path: str, labels_path: str | None = None):
        """Load ``<root>/g1/csv/<date>/<motion>.csv`` and its temporal labels."""
        self.path = Path(path)
        self._imu = Bones_Seed_G1_IMU(path)

        self._labels_path = (
            Path(labels_path) if labels_path is not None
            else self.path.parents[3] / "metadata" / _LABELS_NAME_
        ).expanduser().resolve()
        self._events = self._label_table(self._labels_path).get(self.path.stem, [])
        self._feature_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self._imu)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        return self._imu.timestamps

    @classmethod
    def _encoder(cls) -> CLIP_Model:
        if cls._clip is None:
            cls._clip = CLIP_Model()
        return cls._clip

    @classmethod
    def _label_table(cls, labels_path: Path) -> dict[str, list[tuple[int, int, str]]]:
        """Parse and cache temporal events by motion stem."""
        table = cls._labels.get(labels_path)
        if table is not None:
            return table

        if not labels_path.is_file():
            raise FileNotFoundError(f"temporal labels not found at {labels_path}")

        table = {}
        with open(labels_path) as f:
            for line in f:
                rec = json.loads(line)
                table[rec["filename"]] = [
                    (
                        round(event["start_time"] * 1e9),
                        round(event["end_time"] * 1e9),
                        event["description"].strip(),
                    )
                    for event in rec["events"]
                ]
        cls._labels[labels_path] = table
        return table

    def _text_for(self, w0: int, w1: int) -> str:
        """Return the event text containing the inclusive sample range ``[w0, w1]``."""
        for start, end, text in self._events:
            if start <= w0 and w1 < end:
                return text
        return ""

    def _feature_for(self, text: str) -> torch.Tensor:
        feature = self._feature_cache.get(text)
        if feature is None:
            feature = self._encoder()([text]).cpu()
            self._feature_cache[text] = feature
        return feature.clone()

    def __getitem__(self, index: slice | int) -> IMUData_w_GT_Lang:
        base = self._imu[index]

        T = base.acc.shape[2]
        w0 = int(base.time_ns[0, 0,  0, 0]) if T else 0
        w1 = int(base.time_ns[0, 0, -1, 0]) if T else -1
        text = self._text_for(w0, w1) if T else ""

        if text:
            feature  = self._feature_for(text)
            feat_idx = torch.zeros(1, dtype=torch.int64)
            flags    = self.APPLICABLE_LOSS
        else:
            feature  = torch.empty(0, _FEAT_DIM_)
            feat_idx = torch.full((1,), -1, dtype=torch.int64)
            flags    = self.APPLICABLE_LOSS & ~Loss_Flags.Lang_CLIP

        fields = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
        fields["labels"] = torch.full_like(base.labels, int(flags))
        return IMUData_w_GT_Lang(
            **fields,
            lang_feat_idx = feat_idx,
            lang_feature  = feature,
        )
