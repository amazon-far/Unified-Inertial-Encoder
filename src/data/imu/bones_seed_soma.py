"""Virtual IMU synthesis and temporal language for BONES-SEED SOMA BVH motions."""

import math
import typing as tp

from pathlib import Path

import jaxtyping as jt
import numpy as np
import pypose as pp
import torch

from ...interface.data          import ArticulatedModel
from ...interface.data_humanoid import _windowed_time_derivative
from ...interface.data_imu      import IMUData_w_GT, Loss_Flags, gravity_of_kind
from ...interface.human         import CANONICAL_IDX, CANONICAL_NAMES, T_CANONICAL_NAMES
from ...interface.sequence      import ISequence

from .bones_seed_g1 import Bones_Seed_IMU_w_GT_Lang, _CM_TO_M_, _FPS_, _LABELS_NAME_

# +90 degrees around X maps the SOMA BVH Y-up world to right-handed Z-up.
_R_YUP_TO_ZUP = pp.SO3(torch.tensor([math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]))


def _yup_to_zup(v: torch.Tensor) -> torch.Tensor:
    """Map Y-up world vectors (..., 3) to Z-up: (x, y, z) -> (x, -z, y)."""
    return torch.stack([v[..., 0], -v[..., 2], v[..., 1]], dim=-1)


# Canonical human IMU site -> SOMA BVH joint. SOMA calls the *thigh* `LeftLeg` and the shank
# `LeftShin`. `torso_t8` is absent: T8 falls mid-bone with no joint there, and mapping it to
# Chest would offset the device from the other corpora.
CANONICALIZED_BONES_SOMA: dict[T_CANONICAL_NAMES, str] = {
    "pelvis"          : "Hips"         ,
    "head"            : "Head"         ,
    "left_shoulder"   : "LeftShoulder" ,
    "left_upper_arm"  : "LeftArm"      ,
    "left_forearm"    : "LeftForeArm"  ,
    "left_hand"       : "LeftHand"     ,
    "right_shoulder"  : "RightShoulder",
    "right_upper_arm" : "RightArm"     ,
    "right_forearm"   : "RightForeArm" ,
    "right_hand"      : "RightHand"    ,
    "left_upper_leg"  : "LeftLeg"      ,
    "left_lower_leg"  : "LeftShin"     ,
    "left_foot"       : "LeftFoot"     ,
    "right_upper_leg" : "RightLeg"     ,
    "right_lower_leg" : "RightShin"    ,
    "right_foot"      : "RightFoot"    ,
}

_AXIS_OF = {"X": 0, "Y": 1, "Z": 2}


class _BVHSkeleton(tp.NamedTuple):
    names   : tuple[str, ...]                # joints in channel-declaration order
    parents : tuple[int, ...]                # index into `names`; -1 for the hierarchy root
    offsets : torch.Tensor                   # (J, 3) metres, in the parent's frame
    channels: tuple[tuple[str, ...], ...]    # per joint, in declaration order
    motion  : torch.Tensor                   # (T, sum(len(channels)))


def _axis_so3(angle_rad: torch.Tensor, axis: int) -> pp.LieTensor:
    """Rotation of `angle_rad` about a single basis `axis`, as SO3."""
    q = torch.zeros(*angle_rad.shape, 4)
    q[..., axis] = torch.sin(angle_rad * 0.5)
    q[...,    3] = torch.cos(angle_rad * 0.5)
    return pp.SO3(q)


def _parse_bvh(path: Path) -> _BVHSkeleton:
    """Parse a BVH file into its hierarchy and motion block."""
    text = path.read_text()
    if "MOTION" not in text:
        raise ValueError(f"{path}: no MOTION block")
    head, motion = text.split("MOTION", 1)

    names   : list[str] = []
    parents : list[int] = []
    offsets : list[list[float] | None] = []
    channels: list[tuple[str, ...]] = []

    stack  : list[int] = []          # open joints; -1 marks an End Site
    pending: str | None = None
    in_end = False
    for raw in head.splitlines():
        line = raw.strip()
        if line.startswith(("ROOT ", "JOINT ")):
            pending, in_end = line.split()[1], False
        elif line.startswith("End Site"):
            pending, in_end = None, True
        elif line == "{":
            if in_end:
                stack.append(-1)     # End Site carries an OFFSET that names no joint
            else:
                if pending is None:
                    raise ValueError(f"{path}: '{{' with no preceding ROOT/JOINT")
                parents.append(stack[-1] if stack else -1)
                names.append(pending)
                offsets.append(None)
                channels.append(())
                stack.append(len(names) - 1)
            pending, in_end = None, False
        elif line == "}":
            if not stack:
                raise ValueError(f"{path}: unbalanced '}}' in HIERARCHY")
            stack.pop()
        elif stack and stack[-1] >= 0:
            if line.startswith("OFFSET"):
                offsets[stack[-1]] = [float(v) for v in line.split()[1:4]]
            elif line.startswith("CHANNELS"):
                channels[stack[-1]] = tuple(line.split()[2:])

    if stack:
        raise ValueError(f"{path}: unbalanced '{{' in HIERARCHY")
    if any(off is None for off in offsets):
        raise ValueError(f"{path}: joint without an OFFSET")

    rows = [
        line for line in motion.splitlines()
        if line.strip() and not line.lstrip().startswith(("Frames", "Frame Time"))
    ]
    frames = torch.from_numpy(np.array([[float(v) for v in r.split()] for r in rows], dtype=np.float64))

    n_channel = sum(len(c) for c in channels)
    if frames.ndim != 2 or frames.shape[1] != n_channel:
        raise ValueError(f"{path}: motion has {tuple(frames.shape)} values, expected (T, {n_channel})")
    if not torch.isfinite(frames).all():
        raise ValueError(f"{path}: motion contains non-finite values")

    return _BVHSkeleton(
        names    = tuple(names),
        parents  = tuple(parents),
        offsets  = torch.tensor(offsets, dtype=torch.float32) * _CM_TO_M_,
        channels = tuple(channels),
        motion   = frames,
    )


def _forward_kinematics(
    skel: _BVHSkeleton,
    keep: tp.Sequence[int],
) -> tuple[jt.Float[torch.Tensor, "T N 3"], jt.Float[pp.LieTensor, "T N 4"]]:
    """World pose of the joints `keep`, in the BVH's own (Y-up, metre) frame.

    A joint's local translation is its ``*position`` channels when present, else its ``OFFSET``;
    the two are never summed (`Hips` has both, equal in the bind pose). Rotation channels compose
    intrinsically in declaration order: `Zrotation Yrotation Xrotation` is ``R = Rz @ Ry @ Rx``.
    """
    starts, col = [], 0
    for chans in skel.channels:
        starts.append(col)
        col += len(chans)

    T       = skel.motion.shape[0]
    needed  = _ancestor_closure(skel.parents, keep)
    world_p: list[torch.Tensor | None]  = [None] * len(skel.names)
    world_R: list[pp.LieTensor | None]  = [None] * len(skel.names)

    for j in range(len(skel.names)):
        if j not in needed:
            continue                                    # fingers/eyes: never a sensor site
        chans = skel.channels[j]
        value = {c: skel.motion[:, starts[j] + k].float() for k, c in enumerate(chans)}

        rot = None
        for c in chans:                                 # declaration order == compose order
            if not c.endswith("rotation"):
                continue
            step = _axis_so3(torch.deg2rad(value[c]), _AXIS_OF[c[0]])
            rot  = step if rot is None else rot @ step
        if rot is None:
            rot = pp.identity_SO3(T)

        pos_c = [c for c in chans if c.endswith("position")]
        if pos_c:
            local_t = torch.stack([value[f"{a}position"] for a in "XYZ"], dim=-1) * _CM_TO_M_
        else:
            local_t = skel.offsets[j].expand(T, 3)

        parent = skel.parents[j]
        if parent < 0:
            world_p[j], world_R[j] = local_t, rot
        else:
            p_p, p_R = world_p[parent], world_R[parent]
            assert p_p is not None and p_R is not None   # pre-order: parents resolve first
            world_p[j] = p_p + (p_R @ local_t)
            world_R[j] = p_R @ rot

    pos = torch.stack([tp.cast(torch.Tensor, world_p[j]) for j in keep], dim=1)          # (T, N, 3)
    rot = pp.SO3(torch.stack([tp.cast(pp.LieTensor, world_R[j]).tensor() for j in keep], dim=1))
    return pos, rot


def _ancestor_closure(parents: tuple[int, ...], keep: tp.Sequence[int]) -> set[int]:
    """`keep` plus every ancestor, i.e. the joints FK must actually evaluate."""
    needed: set[int] = set()
    for j in keep:
        while j >= 0 and j not in needed:
            needed.add(j)
            j = parents[j]
    return needed


class Bones_Seed_SOMA_IMU(ISequence[IMUData_w_GT]):
    """Synthesize whole-body virtual IMUs from one BONES-SEED SOMA BVH.

    Each site is differentiated directly from FK, with no base-sensor transport or URDF. Output
    is Z-up right-handed metres. Acceleration stays gravity-free (``kind="filtered-no-g"``), and
    the second derivative amplifies mocap ripple, so ``acc`` is kept out of label-bearing losses.
    """

    APPLICABLE_LOSS = Loss_Flags.JEPA | Loss_Flags.Vel_SSIO

    # Half-window (s) of every derivative stencil here.
    DERIV_WINDOW_S = 0.031

    def __init__(self, path: str, target_sensors: tp.Sequence[T_CANONICAL_NAMES] | None = None):
        if target_sensors is None:
            target_sensors = tuple(CANONICALIZED_BONES_SOMA)
        unknown = [n for n in target_sensors if n not in CANONICAL_NAMES]
        if unknown:
            raise ValueError(f"non-canonical sensor name(s): {unknown}")
        missing = [n for n in target_sensors if n not in CANONICALIZED_BONES_SOMA]
        if missing:
            raise ValueError(f"SOMA has no joint for site(s): {missing}")

        self.path  = Path(path)
        self.model = ArticulatedModel(names=tuple(target_sensors), urdf=None)

        skel = _parse_bvh(self.path)
        T    = skel.motion.shape[0]
        if T < 2:
            raise ValueError(f"{self.path}: expected at least 2 frames, got {T}")

        index = {name: i for i, name in enumerate(skel.names)}
        absent = [CANONICALIZED_BONES_SOMA[n] for n in target_sensors if CANONICALIZED_BONES_SOMA[n] not in index]
        if absent:
            raise ValueError(f"{self.path}: BVH lacks joint(s) {absent}")
        keep = [index[CANONICALIZED_BONES_SOMA[n]] for n in target_sensors]

        # BVH is Y-up, the package Z-up. A world-frame rotation reframes as R_zup = C @ R_yup, which
        # leaves body-frame channels unchanged; gt_pose must be reframed for AddGravity.
        pos_yup, rot_yup = _forward_kinematics(skel, keep)
        pos = _yup_to_zup(pos_yup)                                          # (T, N, 3)
        rot = _R_YUP_TO_ZUP @ rot_yup                                       # (T, N, 4)

        time_ns = torch.from_numpy(np.round(np.arange(T) * (1e9 / _FPS_)).astype(np.int64))

        # The stencils are time-major over a flat batch of sites.
        p_w   = pos.transpose(0, 1)                                         # (N, T, 3)
        t_ns  = time_ns.view(1, T, 1).expand(len(keep), T, 1)
        v_w   = _windowed_time_derivative(p_w, t_ns, self.DERIV_WINDOW_S)
        a_w   = _windowed_time_derivative(v_w, t_ns, self.DERIV_WINDOW_S)

        R_wb  = pp.SO3(rot.tensor().transpose(0, 1))                        # (N, T, 4)
        R_bw  = R_wb.matrix().transpose(-1, -2)                             # (N, T, 3, 3)

        self.acc       : torch.Tensor = (R_bw @ a_w.unsqueeze(-1)).squeeze(-1).float()
        self.gyro      : torch.Tensor = _body_angular_velocity(R_wb, t_ns, self.DERIV_WINDOW_S).float()
        self.lin_vel   : torch.Tensor = (R_bw @ v_w.unsqueeze(-1)).squeeze(-1).float()
        self.gt_pose   : pp.LieTensor = pp.SE3(torch.cat([p_w, R_wb.tensor()], dim=-1))
        self.time_ns   : torch.Tensor = time_ns
        self.sensor_id : torch.Tensor = torch.tensor(
            [CANONICAL_IDX[n] for n in target_sensors], dtype=torch.uint8
        ).view(1, -1, 1, 1)

        self._T = T

    def __len__(self) -> int:
        return self._T

    @property
    def timestamps(self) -> jt.Int64[torch.Tensor, "T"]:
        return self.time_ns

    def __getitem__(self, index: slice | int) -> IMUData_w_GT:
        if isinstance(index, int):
            original = index
            index = index + self._T if index < 0 else index
            if index < 0 or index >= self._T:
                raise IndexError(f"index {original} out of range for length {self._T}")
            index = slice(index, index + 1)

        acc     = self.acc    [:, index].unsqueeze(0)                       # (1, N, T, 3)
        gyro    = self.gyro   [:, index].unsqueeze(0)
        gt_vel  = self.lin_vel[:, index].unsqueeze(0)
        gt_pose = tp.cast(pp.LieTensor, self.gt_pose[:, index].unsqueeze(0))

        n_sensor = self.acc.shape[0]
        t_slice  = self.time_ns[index]                                      # (T,)
        time_ns  = t_slice.view(1, 1, -1, 1).expand(1, n_sensor, t_slice.shape[0], 1).contiguous()

        # Both channels are kinematic accelerations -> zero gravity.
        gravity, correct_gravity = gravity_of_kind("filtered-no-g", 1, n_sensor)

        return IMUData_w_GT(
            acc          = acc,
            gyro         = gyro,
            time_ns      = time_ns,
            labels       = self.get_loss_label(time_ns.shape[:2]),
            acc_bias     = torch.zeros_like(acc),
            gyro_bias    = torch.zeros_like(gyro),

            correct_mask = torch.zeros(acc.shape[:-1], dtype=torch.bool, device=acc.device),
            correct_acc  = acc.clone(),
            correct_gyro = gyro.clone(),

            gravity         = gravity,
            correct_gravity = correct_gravity,

            gt_pose      = gt_pose,
            gt_vel       = gt_vel,

            sensor_id    = self.sensor_id.expand(1, -1, t_slice.shape[0], -1).contiguous(),
            model        = self.model,
            kind         = "filtered-no-g",
        )


def _body_angular_velocity(
    R            : jt.Float[pp.LieTensor, "B T 4"],
    time_ns      : jt.Int64[torch.Tensor, "B T 1"],
    half_window_s: float,
) -> jt.Float[torch.Tensor, "B T 3"]:
    """Body angular velocity from a symmetric SO3 finite difference."""
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


class Bones_Seed_SOMA_IMU_w_GT_Lang(Bones_Seed_IMU_w_GT_Lang):
    """``Bones_Seed_SOMA_IMU`` with the same temporal descriptions as the G1 variant.

    The SOMA BVH and G1 CSV of a motion share a stem and clock, so the language logic is inherited.
    """

    def __init__(
        self,
        path          : str,
        labels_path   : str | None = None,
        target_sensors: tp.Sequence[T_CANONICAL_NAMES] | None = None,
    ):
        """Load ``<root>/soma_proportional/bvh/<date>/<motion>.bvh`` and its temporal labels."""
        self.path = Path(path)
        self._imu = Bones_Seed_SOMA_IMU(path, target_sensors)

        # Same directory depth as the G1 layout.
        self._labels_path = (
            Path(labels_path) if labels_path is not None
            else self.path.parents[3] / "metadata" / _LABELS_NAME_
        ).expanduser().resolve()
        self._events = self._label_table(self._labels_path).get(self.path.stem, [])
        self._feature_cache: dict[str, torch.Tensor] = {}
