"""Articulated-model structures: the kinematic model, base and root traits, and point clouds."""
from __future__ import annotations

import torch
import rerun as rr
import numpy as np
import pypose as pp
import typing as tp
import jaxtyping as jt
import pytorch_kinematics as pk

from yourdfpy.urdf import URDF
from dataclasses import dataclass, field

from ..utility.urdf import log_urdf_initial


def _field_to(
    t: torch.Tensor,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    """``Tensor.to`` for one field: always move to ``device``, cast to ``dtype`` only if floating point.

    A float cast would corrupt ns timestamps, sensor ids and boolean masks.
    """
    if dtype is not None and t.is_floating_point():
        return t.to(device=device, dtype=dtype)
    return t.to(device=device)


@dataclass(frozen=True)
class ArticulatedModel:
    names   : tuple[str, ...]
    urdf    : URDF | None

    # Derived from `urdf` in __post_init__.
    chain         : pk.Chain | None                  = field(default=None, init=False, compare=False, repr=False)
    joint_perm    : torch.Tensor | None              = field(default=None, init=False, compare=False, repr=False)
    link_names    : tuple[str, ...] | None           = field(default=None, init=False, compare=False, repr=False)
    serial_chains : dict[str, pk.SerialChain] | None = field(default=None, init=False, compare=False, repr=False)
    joint_index   : dict[str, torch.Tensor] | None   = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self):
        if self.urdf is None: return

        chain = pk.build_chain_from_urdf(self.urdf.write_xml_string())
        pk_joints = chain.get_joint_parameter_names(exclude_fixed=True)

        name_to_idx = {n: i for i, n in enumerate(self.names)}
        missing = [n for n in pk_joints if n not in name_to_idx]
        assert not missing, f"PK joints not present in ArticulatedModel.names: {missing}"
        perm = torch.tensor([name_to_idx[n] for n in pk_joints], dtype=torch.long)

        link_names = tuple(self.urdf.link_map.keys())

        # Per-link SerialChain and its joint columns in `self.names`. Fixed-joint child links are
        # valid end-effectors too, so try every link and skip failures.
        serial_chains : dict[str, pk.SerialChain] = {}
        joint_index   : dict[str, torch.Tensor]   = {}
        for n in link_names:
            try:
                sc = pk.SerialChain(chain, n)
            except Exception:
                continue
            sc_joints = sc.get_joint_parameter_names(exclude_fixed=True)
            serial_chains[n] = sc
            joint_index[n]   = torch.tensor([name_to_idx[j] for j in sc_joints], dtype=torch.long)

        object.__setattr__(self, "chain",         chain)
        object.__setattr__(self, "joint_perm",    perm)
        object.__setattr__(self, "link_names",    link_names)
        object.__setattr__(self, "serial_chains", serial_chains)
        object.__setattr__(self, "joint_index",   joint_index)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ArticulatedModel):
            return self.names == other.names
        else:
            raise TypeError(f"Cannot compare {type(self)!r} with {type(other)!r}")

    @property
    def num_joint(self) -> int: return len(self.names)

    def visualize(self, path: str, vis_override=None):
        if self.urdf is None: return
        assert self.link_names is not None
        log_urdf_initial(path, self.urdf, vis_override.urdf if vis_override else None)

        vel_color  = vis_override.velocity.color  if vis_override else [255, 205, 0, 85]
        vel_radius = vis_override.velocity.radius if vis_override else -2.0

        for node in self.link_names:
            rr.log(
                f"{path}/{node}",
                rr.Boxes3D(sizes=[0.02, 0.02, 0.02], centers=[[0., 0., 0.]], quaternions=rr.Quaternion(xyzw=[0., 0., 0., 1.]), colors=(255, 205, 0, 85)),
                static=True,
            )
            rr.log(
                f"{path}/{node}/velocity",
                rr.Arrows3D.from_fields(radii=rr.Radius(vel_radius), origins=np.zeros(3,), colors=vel_color),
                static=True
            )


@dataclass
class Articulated_BaseTrait:
    time_ns : jt.Int64[torch.Tensor, "B T 1"]
    model   : ArticulatedModel
    kind    : tp.Literal["real", "sim"]

    def clone(self) -> Articulated_BaseTrait:
        """Return a deep copy of the concrete type."""
        raise NotImplementedError("clone() must be implemented by the concrete Articulated class")

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> Articulated_BaseTrait:
        """Return a copy on `device`, floating-point fields cast to `dtype`; integer/bool fields keep
        their dtype. Follows ``Tensor.to``, so this is not a deep copy.
        """
        raise NotImplementedError("to() must be implemented by the concrete Articulated class")

    @classmethod
    def collate(cls, batch: tp.Sequence[tp.Any]) -> tp.Self:
        """Merge samples along the batch axis; ``batch`` is ``Any`` so overrides can narrow it to their own class."""
        raise NotImplementedError("collate() must be implemented by the concrete Articulated class")

    def _validate_base_fields(self):
        assert self.model.urdf is not None

    @staticmethod
    def _collate_base_fields(batch: tp.Sequence[Articulated_BaseTrait]) -> dict:
        """Concatenate base fields into a kwargs dict; `model` and `kind` come from the first sample."""
        return dict(
            time_ns = torch.cat([s.time_ns for s in batch], dim=0),
            model   = batch[0].model,
            kind    = batch[0].kind,
        )

    def _clone_base_fields(self) -> dict:
        """Deep-clone base tensor fields into a kwargs dict; `model` is shared by reference."""
        return dict(
            time_ns = self.time_ns.clone(),
            model   = self.model,
            kind    = self.kind,
        )

    def _to_base_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        return dict(
            time_ns = _field_to(self.time_ns, device, dtype),
            model   = self.model,
            kind    = self.kind,
        )


@dataclass
class Articulated_RootTrait:
    root_frame: str
    root_pose : jt.Float[pp.LieTensor, "B T 7"]
    root_vel  : jt.Float[torch.Tensor, "B T 3"] # Body frame velocity (not world frame)!
    root_omega: jt.Float[torch.Tensor, "B T 3"] # Body frame angular rate (not world frame)!

    def _validate_root_fields(self, model: ArticulatedModel):
        assert model.urdf is not None
        assert self.root_frame in model.urdf.link_map

    @staticmethod
    def _collate_root_fields(batch: tp.Sequence[Articulated_RootTrait]) -> dict:
        """Concatenate root fields into a kwargs dict; `root_frame` comes from the first sample."""
        return dict(
            root_frame = batch[0].root_frame,
            root_pose  = pp.SE3(torch.cat([s.root_pose.tensor() for s in batch], dim=0)),
            root_vel   = torch.cat([s.root_vel   for s in batch], dim=0),
            root_omega = torch.cat([s.root_omega for s in batch], dim=0),
        )

    def _clone_root_fields(self) -> dict:
        return dict(
            root_frame = self.root_frame,
            root_pose  = self.root_pose.clone(),
            root_vel   = self.root_vel.clone(),
            root_omega = self.root_omega.clone(),
        )

    def _to_root_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        return dict(
            root_frame = self.root_frame,
            root_pose  = _field_to(self.root_pose,  device, dtype),
            root_vel   = _field_to(self.root_vel,   device, dtype),
            root_omega = _field_to(self.root_omega, device, dtype),
        )

    def _visualize_trajectory(self, path: str, time_ns: torch.Tensor, index: int=0):
        """Log each segment (points[t-1], points[t]) at time_ns[t] in one send_columns call."""
        root_world: jt.Float[pp.LieTensor, "T 7"] = pp.SE3(self.root_pose[index])
        points = root_world.translation().numpy()                       # (T, 3)

        # (T-1, 2, 3): row k is the segment from points[k] to points[k+1].
        segments = np.stack([points[:-1], points[1:]], axis=1)
        stamps   = time_ns[index, 1:, 0].numpy().astype("int64").view("datetime64[ns]")

        rr.send_columns(
            path,
            indexes=[rr.TimeColumn("ros_time_ns", timestamp=stamps)],
            columns=rr.LineStrips3D.columns(strips=list(segments)),
        )


@dataclass
class Pointcloud:
    time_ns: int
    pos      : jt.Float[torch.Tensor, "N 3"]
    intensity: jt.Float[torch.Tensor, "N 1"]
