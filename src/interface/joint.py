"""Embodiment-agnostic joint angles between adjacent canonical IMU sites."""
from __future__ import annotations

import torch
import pypose as pp
import typing as tp
import jaxtyping as jt

from .human import T_CANONICAL_NAMES


# The 12-site layout: all 17 canonical sites except torso_t8, both shoulders and both hands.
JOINT_SITES: tuple[T_CANONICAL_NAMES, ...] = (
    "pelvis", "head",
    "left_upper_arm", "left_forearm",
    "right_upper_arm", "right_forearm",
    "left_upper_leg", "left_lower_leg", "left_foot",
    "right_upper_leg", "right_lower_leg", "right_foot",
)

# (joint name, parent site, child site). No torso site survives the prune, so torso-level joints
# hang off the pelvis: "left_shoulder" is the whole pelvis -> upper-arm chain.
JOINT_TREE: tuple[tuple[str, T_CANONICAL_NAMES, T_CANONICAL_NAMES], ...] = (
    ("spine",          "pelvis",          "head"),
    ("left_shoulder",  "pelvis",          "left_upper_arm"),
    ("left_elbow",     "left_upper_arm",  "left_forearm"),
    ("right_shoulder", "pelvis",          "right_upper_arm"),
    ("right_elbow",    "right_upper_arm", "right_forearm"),
    ("left_hip",       "pelvis",          "left_upper_leg"),
    ("left_knee",      "left_upper_leg",  "left_lower_leg"),
    ("left_ankle",     "left_lower_leg",  "left_foot"),
    ("right_hip",      "pelvis",          "right_upper_leg"),
    ("right_knee",     "right_upper_leg", "right_lower_leg"),
    ("right_ankle",    "right_lower_leg", "right_foot"),
)
JOINT_NAMES: tuple[str, ...] = tuple(j[0] for j in JOINT_TREE)

assert all(p in JOINT_SITES and c in JOINT_SITES for _, p, c in JOINT_TREE)
assert len({c for _, _, c in JOINT_TREE}) == len(JOINT_TREE) == len(JOINT_SITES) - 1


def continuous_sign(q: jt.Float[torch.Tensor, "T 4"]) -> jt.Float[torch.Tensor, "T 4"]:
    """Flip quaternion signs so consecutive samples share a hemisphere, with ``w >= 0`` at t = 0."""
    sign0 = torch.where(q[0, 3] < 0, -1.0, 1.0)
    dots  = (q[1:] * q[:-1]).sum(-1)
    flips = torch.where(dots < 0, -1.0, 1.0)
    signs = torch.cat([sign0.view(1), sign0 * torch.cumprod(flips, dim=0)])
    return q * signs.unsqueeze(-1)


def relative_joints(
    rot_ws: jt.Float[pp.LieTensor, "N T 4"],
    gyro  : jt.Float[torch.Tensor, "N T 3"],
    names : tp.Sequence[str],
) -> tuple[jt.Float[torch.Tensor, "J T 4"], jt.Float[torch.Tensor, "J T 3"], float]:
    """Joint ``q``/``dq`` of ``JOINT_TREE`` from per-site world orientations and body-frame gyros.

    Inputs share the mounting convention; ``names`` labels axis 0 and must cover ``JOINT_SITES``.
    With ``R_pc = R_wp^T R_wc``, ``q = quat(R_pc)`` (xyzw, sign-continuous) and
    ``dq = omega_c - R_pc^T omega_p`` (child-frame axes, rad/s). ``q`` is a quaternion because
    ``Log`` is discontinuous near 180 deg. Returns ``(q, dq, max_angle)`` with the angle in rad.
    """
    idx = {n: i for i, n in enumerate(names)}
    missing = [s for s in JOINT_SITES if s not in idx]
    assert not missing, f"relative_joints: missing sites {missing}"

    rot_ws = pp.SO3(rot_ws.tensor() if isinstance(rot_ws, pp.LieTensor) else rot_ws)
    q_all, dq_all = [], []
    for _, parent, child in JOINT_TREE:
        R_wp, R_wc = pp.SO3(rot_ws[idx[parent]]), pp.SO3(rot_ws[idx[child]])   # (T, 4)
        R_pc = R_wp.Inv() @ R_wc                                               # (T, 4)
        q    = continuous_sign(R_pc.tensor())                                  # (T, 4) xyzw
        dq   = gyro[idx[child]] - (R_pc.Inv() @ gyro[idx[parent]])             # (T, 3) child frame
        q_all.append(q)
        dq_all.append(dq)

    q  = torch.stack(q_all,  dim=0)
    dq = torch.stack(dq_all, dim=0)
    max_angle = float(2.0 * torch.atan2(q[..., :3].norm(dim=-1), q[..., 3].abs()).max())
    return q, dq, max_angle
