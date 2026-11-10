"""Training readers for the root-IMU + joint caches written by ``src.scripts.rootjoint_process``."""
from __future__ import annotations

import json
import torch
import numpy as np
import typing as tp
import jaxtyping as jt

from pathlib import Path
from dataclasses import fields

from ...interface.data_imu   import IMUData_w_GT, Loss_Flags
from ...interface.data_joint import IMUJointData_w_GT, IMUJointData_w_GT_Lang
from ...interface.sequence   import ISequence
from ..imu.nymeria_processed import Nymeria_XSens_Processed_IMU


class RootJoint_IMU_Sequence(Nymeria_XSens_Processed_IMU):
    """Pelvis IMU plus joint stream, as ``IMUJointData_w_GT``.

    ``joints``: ``"relative"`` is the 11 embodiment-agnostic joints (quaternion + child-frame
    rate, in every corpus); ``"encoder"`` is the native 29 G1 encoders (angle + rate, G1 only).
    """
    APPLICABLE_LOSS = Loss_Flags.JEPA | Loss_Flags.Vel_L2 | Loss_Flags.Denoise_L2

    def __init__(self, path: str, joints: tp.Literal["relative", "encoder"] = "relative"):
        super().__init__(path, target_sensors=("pelvis",))
        with open(Path(path) / "spec.json") as f:
            spec = json.load(f)
        assert spec["format"] == "rootjoint-1", f"{path}: not a rootjoint cache ({spec['format']!r})"

        self.joints_source = joints
        match joints:
            case "relative":
                sec = spec["joint"]
                self._joint_file  = Path(path) / sec["file"]
                self._joint_cols  = (sec["fields"]["q"], sec["fields"]["dq"])     # [s, e) each
                self.joint_names  = tuple(sec["names"])
                self.joint_kind: tp.Literal["so3", "revolute"] = "so3"
            case "encoder":
                assert "encoder" in spec, f"{path}: no native encoder stream (human corpus?)"
                sec = spec["encoder"]
                self._joint_file  = Path(path) / sec["file"]
                self._joint_cols  = (sec["fields"]["q"], sec["fields"]["dq"])
                self.joint_names  = tuple(sec["names"])
                self.joint_kind   = "revolute"
            case _:
                raise ValueError(f"joints must be 'relative' or 'encoder', got {joints!r}")
        self._joint: np.ndarray | None = None

    def __getstate__(self):
        # Like the parent: never pickle a live memmap.
        return {**super().__getstate__(), "_joint": None}

    def _open_joint(self) -> np.ndarray:
        joint = self._joint
        if joint is None:
            joint = self._joint = np.load(self._joint_file, mmap_mode="r")    # (J, T, C)
        return joint

    def __getitem__(self, index: slice | int) -> IMUJointData_w_GT:
        base: IMUData_w_GT = super().__getitem__(index)
        if isinstance(index, int):
            i  = index + self._T if index < 0 else index
            sl = slice(i, i + 1)
        else:
            sl = index

        j = torch.from_numpy(np.array(self._open_joint()[:, sl]))              # (J, t, C)
        (qs, qe), (ds, de) = self._joint_cols
        joint_q  = j[..., qs:qe].permute(1, 0, 2).unsqueeze(0)                 # (1, t, J, Cq)
        joint_dq = j[..., ds:de].permute(1, 0, 2).unsqueeze(0)                 # (1, t, J, Cd)

        return IMUJointData_w_GT(
            **{f.name: getattr(base, f.name) for f in fields(base)},
            joint_q     = joint_q,
            joint_dq    = joint_dq,
            joint_names = self.joint_names,
            joint_kind  = self.joint_kind,
        )


class RootJoint_IMU_Sequence_Lang(ISequence[IMUJointData_w_GT_Lang]):
    """Root-joint cache plus the language of the corpus's whole-body cache under the same stem.

    ``DELEGATE``, opened on ``<lang_root>/<stem>``, is queried on the same slice and its language
    and per-window labels are copied over; the two caches share one time axis. Per-corpus
    subclasses exist so eval tooling can group pipelines by reader class name.
    """
    DELEGATE: tp.ClassVar[type]   # the corpus's language reader class (path, target_sensors)

    def __init__(self, path: str, lang_root: str, joints: tp.Literal["relative", "encoder"] = "relative"):
        self.path  = Path(path)
        self._imu  = RootJoint_IMU_Sequence(path, joints=joints)
        self._lang = self.DELEGATE(str(Path(lang_root) / self.path.name), target_sensors=["pelvis"])
        assert len(self._lang) == len(self._imu), \
            f"{self.path}: rootjoint cache ({len(self._imu)}) and language cache ({len(self._lang)}) disagree in length"
        self.APPLICABLE_LOSS = self._lang.APPLICABLE_LOSS

    def __len__(self) -> int:
        return len(self._imu)

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        return self._imu.timestamps

    def __getitem__(self, index: slice | int) -> IMUJointData_w_GT_Lang:
        base = self._imu[index]
        lang = self._lang[index]
        return IMUJointData_w_GT_Lang(
            **{f.name: getattr(base, f.name) for f in fields(base) if f.name != "labels"},
            labels        = lang.labels,          # the corpus's own per-window loss flags
            lang_feat_idx = lang.lang_feat_idx,
            lang_feature  = lang.lang_feature,
        )


def _delegate(cls_path: str) -> type:
    module, name = cls_path.rsplit(".", 1)
    import importlib
    return getattr(importlib.import_module(module, package=__package__), name)


class RootJoint_Nymeria_Lang(RootJoint_IMU_Sequence_Lang):
    DELEGATE = _delegate("..imu.nymeria_processed_lang.Nymeria_XSens_Processed_IMU_Language")

class RootJoint_NaturalMotion_Lang(RootJoint_IMU_Sequence_Lang):
    DELEGATE = _delegate("..imu.naturalmotion_processed_lang.NaturalMotion_XSens_Processed_IMU_Language")

class RootJoint_BonesSOMA_Lang(RootJoint_IMU_Sequence_Lang):
    DELEGATE = _delegate("..imu.bones_seed_processed.Bones_Seed_SOMA_Processed_IMU_w_Lang")

class RootJoint_BonesG1_Lang(RootJoint_IMU_Sequence_Lang):
    DELEGATE = _delegate("..imu.bones_seed_processed.Bones_Seed_G1_Processed_IMU_w_Lang")

class RootJoint_SuperOdom_Lang(RootJoint_IMU_Sequence_Lang):
    DELEGATE = _delegate("..imu.humanoid_superodom_npz.Humanoid_SuperOdom_NPZ_Processed_Language")
