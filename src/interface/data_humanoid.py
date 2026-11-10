"""Articulated joint and link-state data, with rigid-body synthesis of virtual IMUs."""
from __future__ import annotations

import torch
import rerun as rr
import pypose as pp
import typing as tp
import jaxtyping as jt

from dataclasses import dataclass
from torch.autograd.functional import jvp

from .data      import ArticulatedModel, Articulated_RootTrait, Articulated_BaseTrait, _field_to
from .data_imu  import IMUData, IMUData_w_GT, gravity_of_kind


@dataclass
class Articulated_JointTrait:
    joints  : jt.Float[torch.Tensor, "B N T 1"]
    joint_d : jt.Float[torch.Tensor, "B N T 1"]   # Measured joint velocity, rad/s

    # No joint_dd field: Unitree lowstate ddq is zero-filled, so consumers derive q_ddot from joint_d.

    def _validate_joint_fields(self, model: ArticulatedModel):
        assert self.joints.shape[1] == model.num_joint
        assert self.joint_d.shape  == self.joints.shape

    @staticmethod
    def _collate_joint_fields(batch: tp.Sequence[Articulated_JointTrait]) -> dict:
        return dict(
            joints   = torch.cat([s.joints   for s in batch], dim=0),
            joint_d  = torch.cat([s.joint_d  for s in batch], dim=0),
        )

    def _clone_joint_fields(self) -> dict:
        return dict(
            joints   = self.joints.clone(),
            joint_d  = self.joint_d.clone(),
        )

    def _to_joint_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        return dict(
            joints   = _field_to(self.joints,   device, dtype),
            joint_d  = _field_to(self.joint_d,  device, dtype),
        )


@dataclass
class Articulated_StateTrait:
    states    : jt.Float[pp.LieTensor, "B M T 7"]
    velocities: jt.Float[torch.Tensor, "B M T 3"]   # Velocity of each link represented in body frame.

    def _validate_state_fields(self, model: ArticulatedModel):
        assert model.urdf is not None
        assert self.states.shape[1] == len(model.urdf.link_map)

    @staticmethod
    def _collate_state_fields(batch: tp.Sequence[Articulated_StateTrait]) -> dict:
        return dict(
            states     = pp.SE3(torch.cat([s.states.tensor() for s in batch], dim=0)),
            velocities = torch.cat([s.velocities for s in batch], dim=0),
        )

    def _clone_state_fields(self) -> dict:
        return dict(
            states     = self.states.clone(),
            velocities = self.velocities.clone(),
        )

    def _to_state_fields(self, device: torch.device | None, dtype: torch.dtype | None) -> dict:
        return dict(
            states     = _field_to(self.states,     device, dtype),
            velocities = _field_to(self.velocities, device, dtype),
        )


def _central_time_derivative(
    x      : jt.Float[torch.Tensor, "B T C"],
    time_ns: jt.Int64[torch.Tensor, "B T 1"],
) -> jt.Float[torch.Tensor, "B T C"]:
    """d/dt by central differences, one-sided at the ends, aligned with time_ns[t]. Requires T >= 2."""
    dt  = (time_ns[..., 0].diff(dim=1).to(x.dtype) / 1e9).unsqueeze(-1)     # (B, T-1, 1)
    fwd = x.diff(dim=1) / dt                                                # (B, T-1, C)
    return torch.cat([fwd[:, :1], 0.5 * (fwd[:, :-1] + fwd[:, 1:]), fwd[:, -1:]], dim=1)


def _windowed_time_derivative(
    x            : jt.Float[torch.Tensor, "B T C"],
    time_ns      : jt.Int64[torch.Tensor, "B T 1"],
    half_window_s: float,
) -> jt.Float[torch.Tensor, "B T C"]:
    """d/dt by a symmetric finite difference over a ±half_window_s stencil (a smoothed derivative).

    ``half_window_s <= 0`` falls back to the 2-sample central difference; indices clamp at the ends.
    """
    if half_window_s <= 0:
        return _central_time_derivative(x, time_ns)

    B, T, C = x.shape
    # Difference timestamps in int64 before any float cast: epoch ns lose sub-second resolution in fp32.
    t      = time_ns[..., 0]                                                # (B, T) int64
    dt_med = float(t.diff(dim=1).median()) / 1e9
    k      = max(1, round(half_window_s / dt_med))

    idx = torch.arange(T, device=x.device)
    hi  = (idx + k).clamp(max=T - 1)
    lo  = (idx - k).clamp(min=0)
    den = ((t[:, hi] - t[:, lo]).to(x.dtype) / 1e9).unsqueeze(-1)           # (B, T, 1)
    return (x[:, hi] - x[:, lo]) / den


def _link_relative_kinematics(
    model : ArticulatedModel,
    link  : str,
    q_bt  : jt.Float[torch.Tensor, "BT N"],
    qd_bt : jt.Float[torch.Tensor, "BT N"],
    qdd_bt: jt.Float[torch.Tensor, "BT N"],
    device: torch.device,
    dtype : torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Velocity and acceleration of `link` relative to the URDF root, in URDF-root frame.

    Returns ``(r_dot, omega_rel, r_ddot, alpha_rel)``, each ``(BT, 3)``. The bias term ``J_dot q_dot``
    is exact: the forward-mode derivative of ``q -> J(q) q_dot``. Links with no actuated DOFs return zeros.
    """
    assert model.serial_chains is not None and model.joint_index is not None
    zeros = q_bt.new_zeros(q_bt.shape[0], 3)
    if link not in model.serial_chains:
        return zeros, zeros, zeros, zeros
    cols = model.joint_index[link].to(q_bt.device)
    if cols.numel() == 0:
        return zeros, zeros, zeros, zeros

    sc  = model.serial_chains[link].to(device=device, dtype=dtype)
    qc, qdc, qddc = q_bt[:, cols], qd_bt[:, cols], qdd_bt[:, cols]

    J    = tp.cast(torch.Tensor, sc.jacobian(qc))                           # (BT, 6, k)
    vel6 = (J @ qdc.unsqueeze(-1)).squeeze(-1)                              # (BT, 6)
    Jdot_qd = tp.cast(torch.Tensor, jvp(
        lambda th: (tp.cast(torch.Tensor, sc.jacobian(th)) @ qdc.unsqueeze(-1)).squeeze(-1),
        qc, qdc,
    )[1])
    acc6 = (J @ qddc.unsqueeze(-1)).squeeze(-1) + Jdot_qd                   # (BT, 6)
    return vel6[:, :3], vel6[:, 3:], acc6[:, :3], acc6[:, 3:]


def _synthesize_imu_fields(
    model        : ArticulatedModel,
    time_ns      : jt.Int64[torch.Tensor, "B T 1"],
    joints       : jt.Float[torch.Tensor, "B N T 1"],
    joint_d      : jt.Float[torch.Tensor, "B N T 1"],
    base_acc     : jt.Float[torch.Tensor, "B T 3"],
    base_gyro    : jt.Float[torch.Tensor, "B T 3"],
    base_frame   : str,
    target_frames: tp.Sequence[str],
    sensor_ids   : tp.Sequence[int],
    imu_model    : ArticulatedModel,
    kind         : tp.Literal["real", "filtered-no-g", "sim"],
    gyro_deriv_window_s: float = 0.0,
    joint_dd_window_s  : float = 0.0,
) -> dict:
    """Synthesize rigid-body IMU readings at `target_frames` from a base IMU plus joint state.

    Everything runs in the URDF-root frame ``u``, so world pose, base velocity and gravity direction
    are never needed. With ``phi_b = R_ub f_b``:

        w_u    = R_ub gyr_b - J_w,b q_dot
        al_u   = (J_w,b q_dot) x (R_ub gyr_b) + R_ub gyr_b_dot - (J_w,b q_ddot + J_dot_w,b q_dot)
        gyro_m = R_um^T (w_u + J_w,m q_dot)
        acc_m  = R_um^T (phi_b + al_u x (r_m - r_b) + w_u x (w_u x (r_m - r_b))
                         + 2 w_u x (r_dot_m - r_dot_b) + (r_ddot_m - r_ddot_b))

    `base_acc`/`base_gyro` are at `base_frame`'s origin, in its axes, time-aligned to `time_ns`.
    `gyro_deriv_window_s`/`joint_dd_window_s` are half-widths of the stencils for ``gyr_b_dot`` and
    ``q_ddot`` (from ``joint_d``); ``<= 0`` means a 2-sample central difference.
    Every output keeps `base_acc`'s gravity convention (specific force in, specific force out), so
    `kind` must match the input. Call in time-chunks for long sequences.
    """
    assert (
        model.chain             is not None
        and model.joint_perm    is not None
        and model.link_names    is not None
        and model.serial_chains is not None
    )
    assert base_frame in model.link_names, f"{base_frame} is not a link of the model"
    for f in target_frames:
        assert f in model.link_names, f"{f} is not a link of the model"
    assert len(sensor_ids) == len(target_frames) == imu_model.num_joint

    B, N, T, _ = joints.shape
    assert T >= 2, "as_imu requires at least two timesteps to differentiate the base gyro"
    device, dtype = joints.device, joints.dtype
    n_tgt = len(target_frames)

    chain = model.chain.to(device=device, dtype=dtype)
    perm  = model.joint_perm.to(device)

    q_tm  = joints.squeeze(-1).transpose(1, 2)                              # (B, T, N)
    qd_tm = joint_d.squeeze(-1).transpose(1, 2)                             # (B, T, N)

    q_bt   = q_tm.reshape(B * T, N)
    qd_bt  = qd_tm.reshape(B * T, N)
    qdd_bt = _windowed_time_derivative(qd_tm, time_ns, joint_dd_window_s).reshape(B * T, N)

    transforms = chain.forward_kinematics(q_bt[:, perm])
    def frame_matrix(name: str) -> torch.Tensor: return transforms[name].get_matrix()  # (BT, 4, 4)

    T_ub       = frame_matrix(base_frame)
    R_ub, p_ub = T_ub[:, :3, :3], T_ub[:, :3, 3]
    rdot_b, w_rel_b, rddot_b, alpha_rel_b = _link_relative_kinematics(
        model, base_frame, q_bt, qd_bt, qdd_bt, device, dtype)

    f_b        = base_acc.reshape(B * T, 3)
    gyr_b      = base_gyro.reshape(B * T, 3)
    gyr_b_dot  = _windowed_time_derivative(base_gyro, time_ns, gyro_deriv_window_s).reshape(B * T, 3)

    gyr_u = (R_ub @ gyr_b.unsqueeze(-1)).squeeze(-1)                        # base gyro in u frame
    w_u   = gyr_u - w_rel_b                                                 # URDF-root angular velocity
    al_u  = (
        torch.linalg.cross(w_rel_b, gyr_u)
        + (R_ub @ gyr_b_dot.unsqueeze(-1)).squeeze(-1)
        - alpha_rel_b
    )                                                                       # URDF-root angular acceleration
    phi_b = (R_ub @ f_b.unsqueeze(-1)).squeeze(-1)                          # base accelerometer in u frame

    accs, gyros = [], []
    for tgt in target_frames:
        T_um       = frame_matrix(tgt)
        R_um, p_um = T_um[:, :3, :3], T_um[:, :3, 3]
        rdot_m, w_rel_m, rddot_m, _ = _link_relative_kinematics(
            model, tgt, q_bt, qd_bt, qdd_bt, device, dtype)

        dr  = p_um - p_ub
        f_u = (
            phi_b
            + torch.linalg.cross(al_u, dr)
            + torch.linalg.cross(w_u, torch.linalg.cross(w_u, dr))
            + 2 * torch.linalg.cross(w_u, rdot_m - rdot_b)
            + (rddot_m - rddot_b)
        )
        gyros.append((R_um.transpose(-1, -2) @ (w_u + w_rel_m).unsqueeze(-1)).squeeze(-1))
        accs.append((R_um.transpose(-1, -2) @ f_u.unsqueeze(-1)).squeeze(-1))

    acc  = torch.stack(accs,  dim=1).view(B, T, n_tgt, 3).transpose(1, 2)   # (B, n, T, 3) device-major
    gyro = torch.stack(gyros, dim=1).view(B, T, n_tgt, 3).transpose(1, 2)   # (B, n, T, 3) device-major

    # Both acc channels inherit `base_acc`'s gravity convention, which `kind` states.
    gravity, correct_gravity = gravity_of_kind(kind, B, n_tgt)

    return dict(
        time_ns      = time_ns.unsqueeze(1).expand(B, n_tgt, T, 1).clone(),
        acc          = acc,
        gyro         = gyro,
        correct_mask = torch.zeros(B, n_tgt, T, dtype=torch.bool, device=device),
        correct_acc  = acc.clone(),
        correct_gyro = gyro.clone(),
        gravity         = gravity.to(device),
        correct_gravity = correct_gravity.to(device),
        acc_bias     = torch.zeros_like(acc),
        gyro_bias    = torch.zeros_like(gyro),
        sensor_id    = torch.tensor(list(sensor_ids), dtype=torch.uint8, device=device)
                            .view(1, n_tgt, 1, 1).expand(B, n_tgt, T, 1).clone(),
        model        = imu_model,
        kind         = kind,
    )


@dataclass
class ArticulatedJointMeasure(Articulated_BaseTrait, Articulated_JointTrait):
    def __post_init__(self):
        self._validate_base_fields()
        self._validate_joint_fields(self.model)

    @classmethod
    def collate(cls, batch: tp.Sequence[tp.Self]) -> tp.Self:
        """Merge samples into one batched ``ArticulatedJointMeasure``."""
        return cls(
            **cls._collate_base_fields(batch),
            **cls._collate_joint_fields(batch),
        )

    def clone(self) -> ArticulatedJointMeasure:
        """Return a deep copy with all tensor fields cloned."""
        return ArticulatedJointMeasure(
            **self._clone_base_fields(),
            **self._clone_joint_fields(),
        )

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> ArticulatedJointMeasure:
        """Return a copy on `device`, floating-point fields cast to `dtype`."""
        return ArticulatedJointMeasure(
            **self._to_base_fields(device, dtype),
            **self._to_joint_fields(device, dtype),
        )

    def as_imu(
        self,
        base_acc     : jt.Float[torch.Tensor, "B T 3"],
        base_gyro    : jt.Float[torch.Tensor, "B T 3"],
        base_frame   : str,
        target_frames: tp.Sequence[str],
        sensor_ids   : tp.Sequence[int],
        imu_model    : ArticulatedModel,
        kind         : tp.Literal["real", "filtered-no-g", "sim"] = "sim",
        gyro_deriv_window_s: float = 0.0,
        joint_dd_window_s  : float = 0.0,
    ) -> IMUData:
        """Synthesize virtual IMUs at `target_frames` into an unlabelled ``IMUData``.

        `base_acc`/`base_gyro` are measured at `base_frame`; `sensor_ids` are canonical device ids matching
        `target_frames` and `imu_model`. See `_synthesize_imu_fields` for the math and gravity convention.
        """
        return IMUData(**_synthesize_imu_fields(
            self.model, self.time_ns, self.joints, self.joint_d,
            base_acc, base_gyro, base_frame, target_frames, sensor_ids, imu_model, kind,
            gyro_deriv_window_s=gyro_deriv_window_s, joint_dd_window_s=joint_dd_window_s,
        ))


@dataclass
class ArticulatedJointMeasures_wGT(Articulated_BaseTrait, Articulated_RootTrait, Articulated_JointTrait):
    def __post_init__(self):
        self._validate_base_fields()
        self._validate_root_fields(self.model)
        self._validate_joint_fields(self.model)

    @classmethod
    def collate(cls, batch: tp.Sequence[tp.Self]) -> tp.Self:
        """Merge samples into one batched ``ArticulatedJointMeasures_wGT``."""
        return cls(
            **cls._collate_base_fields(batch),
            **cls._collate_root_fields(batch),
            **cls._collate_joint_fields(batch),
        )

    def clone(self) -> ArticulatedJointMeasures_wGT:
        """Return a deep copy with all tensor fields cloned."""
        return ArticulatedJointMeasures_wGT(
            **self._clone_base_fields(),
            **self._clone_root_fields(),
            **self._clone_joint_fields(),
        )

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> ArticulatedJointMeasures_wGT:
        """Return a copy on `device`, floating-point fields cast to `dtype`."""
        return ArticulatedJointMeasures_wGT(
            **self._to_base_fields(device, dtype),
            **self._to_root_fields(device, dtype),
            **self._to_joint_fields(device, dtype),
        )

    def as_state(self) -> ArticulatedState_wGT:
        model = self.model
        assert model.chain is not None and model.joint_perm is not None and model.link_names is not None

        B, N, T, _ = self.joints.shape
        device, dtype = self.joints.device, self.joints.dtype

        chain = model.chain.to(device=device, dtype=dtype)
        perm  = model.joint_perm.to(device)

        # FK runs time-major (B, T, ...); results are transposed back to device-major below.
        q_pk = self.joints.squeeze(-1).transpose(1, 2)[..., perm].reshape(B * T, -1)
        transforms = chain.forward_kinematics(q_pk)

        # states[m] = T_urdfroot_link[m]: link pose relative to the URDF root, independent of
        # `root_frame`. Consumers compose it with `root_pose`.
        T_urdfroot_link = torch.stack(
            [transforms[name].get_matrix() for name in model.link_names],
            dim=1,
        ).reshape(B, T, len(model.link_names), 4, 4)

        states_bt     = pp.from_matrix(T_urdfroot_link, ltype=pp.SE3_type, check=False)  # (B, T, M, 7)
        velocities_bt = self.derive_velocity(states_bt)                                  # (B, T, M, 3)

        return ArticulatedState_wGT(
            root_frame=self.root_frame,
            root_pose =self.root_pose,
            root_vel  =self.root_vel,
            root_omega=self.root_omega,
            states    =tp.cast(pp.LieTensor, states_bt.transpose(1, 2)),  # (B, M, T, 7)
            velocities=velocities_bt.transpose(1, 2),                     # (B, M, T, 3)
            time_ns   =self.time_ns,
            kind      =self.kind,
            model     =model,
        )

    def as_imu(
        self,
        label        : jt.UInt8[torch.Tensor, "B"],
        base_acc     : jt.Float[torch.Tensor, "B T 3"],
        base_gyro    : jt.Float[torch.Tensor, "B T 3"],
        base_frame   : str,
        target_frames: tp.Sequence[str],
        sensor_ids   : tp.Sequence[int],
        imu_model    : ArticulatedModel,
        kind         : tp.Literal["real", "filtered-no-g", "sim"] = "sim",
        gyro_deriv_window_s: float = 0.0,
        joint_dd_window_s  : float = 0.0,
    ) -> IMUData_w_GT:
        """Synthesize virtual IMUs at `target_frames` into an ``IMUData_w_GT``.

        Channels match ``ArticulatedJointMeasure.as_imu``; `gt_pose` composes ``root_pose`` with FK and
        `gt_vel` is the body-frame link velocity, both via ``as_state()``.
        """
        fields = _synthesize_imu_fields(
            self.model, self.time_ns, self.joints, self.joint_d,
            base_acc, base_gyro, base_frame, target_frames, sensor_ids, imu_model, kind,
            gyro_deriv_window_s=gyro_deriv_window_s, joint_dd_window_s=joint_dd_window_s,
        )

        model = self.model
        assert model.link_names is not None
        state    = self.as_state()
        tgt_idx  = [model.link_names.index(f) for f in target_frames]
        root_idx = model.link_names.index(self.root_frame)

        urdfroot_world = pp.SE3(self.root_pose) @ pp.SE3(state.states[:, root_idx]).Inv()  # (B, T, 7)
        gt_pose = pp.SE3(urdfroot_world.unsqueeze(1)) @ pp.SE3(state.states[:, tgt_idx])   # (B, n, T, 7)
        gt_vel  = state.velocities[:, tgt_idx]                                             # (B, n, T, 3)

        return IMUData_w_GT(
            **fields,
            labels  = label,
            gt_pose = tp.cast(pp.LieTensor, gt_pose),
            gt_vel  = gt_vel,
        )

    def derive_velocity(
        self,
        states: jt.Float[pp.LieTensor, "B T M 7"],
    ) -> jt.Float[torch.Tensor, "B T M 3"]:
        """Per-link linear velocity in each link's body frame, in time-major (B, T, ...) layout.

        Rigid-body twist transport plus the joint term from the measured ``joint_d``, then rotated into
        the link frame:

            v_world_link = v_world_root + omega_world_base x (p_world_link - p_world_root)
                         + R_world_urdfroot @ J_m(q) @ q_dot
        """
        model = self.model
        assert (
            model.chain         is not None
            and model.link_names    is not None
            and model.serial_chains is not None
            and model.joint_index   is not None
        )

        B, N, T, _ = self.joints.shape
        M          = len(model.link_names)
        device     = self.joints.device
        dtype      = self.joints.dtype
        root_idx   = model.link_names.index(self.root_frame)

        # `joint_d` shares `joints`' canonical order, matching the Jacobian columns below.
        q     = self.joints.squeeze(-1).transpose(1, 2)                         # (B, T, N)
        q_dot = self.joint_d.squeeze(-1).transpose(1, 2)                        # (B, T, N)

        # Position Jacobian in URDF-root frame, scattered into full N. pk returns [v; omega] rows.
        J_pos = q.new_zeros(B, T, M, 3, N)
        q_bt  = q.reshape(B * T, N)
        for m, link in enumerate(model.link_names):
            if link not in model.serial_chains: continue
            cols = model.joint_index[link].to(device)
            if cols.numel() == 0: continue
            sc   = model.serial_chains[link].to(device=device, dtype=dtype)
            J_v  = sc.jacobian(q_bt[:, cols])[:, :3, :]                         # (B*T, 3, k)   #type: ignore
            J_pos[:, :, m, :, cols] = J_v.reshape(B, T, 3, -1)

        v_urdfroot_link = (J_pos @ q_dot[..., None, :, None]).squeeze(-1)       # (B, T, M, 3)

        # Base-motion contribution in world frame.
        root_pose    = pp.SE3(self.root_pose)
        R_w_root     = root_pose.rotation()
        p_w_root     = root_pose.translation()                                  # (B, T, 3)
        v_w_root     = R_w_root @ self.root_vel                                 # (B, T, 3)
        w_w_base     = R_w_root @ self.root_omega                               # (B, T, 3)

        T_w_urdfroot = root_pose @ pp.SE3(states[:, :, root_idx]).Inv()         # (B, T)
        T_w_link     = pp.SE3(T_w_urdfroot.unsqueeze(-2)) @ pp.SE3(states)      # (B, T, M)
        R_w_urdfroot = T_w_urdfroot.rotation()
        R_w_link     = T_w_link.rotation()
        p_w_link     = T_w_link.translation()                                   # (B, T, M, 3)

        r        = p_w_link - p_w_root.unsqueeze(-2)                            # (B, T, M, 3)
        v_w_link = (
            v_w_root.unsqueeze(-2)
            + torch.linalg.cross(w_w_base.unsqueeze(-2).expand_as(r), r, dim=-1)
            + (R_w_urdfroot.unsqueeze(-2) @ v_urdfroot_link)
        )                                                                       # (B, T, M, 3)

        return R_w_link.Inv() @ v_w_link                                        # (B, T, M, 3)


@dataclass
class ArticulatedState_wGT(Articulated_BaseTrait, Articulated_RootTrait, Articulated_StateTrait):
    def __post_init__(self):
        self._validate_base_fields()
        self._validate_root_fields(self.model)
        self._validate_state_fields(self.model)

    @classmethod
    def collate(cls, batch: tp.Sequence[tp.Self]) -> tp.Self:
        """Merge samples into one batched ``ArticulatedState_wGT``."""
        return cls(
            **cls._collate_base_fields(batch),
            **cls._collate_root_fields(batch),
            **cls._collate_state_fields(batch),
        )

    def clone(self) -> ArticulatedState_wGT:
        """Return a deep copy with all tensor fields cloned."""
        return ArticulatedState_wGT(
            **self._clone_base_fields(),
            **self._clone_root_fields(),
            **self._clone_state_fields(),
        )

    def to(self, device: torch.device | None, dtype: torch.dtype | None) -> ArticulatedState_wGT:
        """Return a copy on `device`, floating-point fields cast to `dtype`."""
        return ArticulatedState_wGT(
            **self._to_base_fields(device, dtype),
            **self._to_root_fields(device, dtype),
            **self._to_state_fields(device, dtype),
        )

    def visualize(self, path: str, index: int=0, vis_override=None):
        assert self.model.link_names is not None
        root_idx = self.model.link_names.index(self.root_frame)

        time_ns  = self.time_ns[index, :, 0].numpy().astype("int64").view("datetime64[ns]")
        time_col = rr.TimeColumn("ros_time_ns", timestamp=time_ns)

        # `path` carries the URDF root in the world; per-link `states[m]` are relative to it, so
        # rerun's transform hierarchy produces T_world_link.
        root_world: jt.Float[pp.LieTensor, "T 7"] = pp.SE3(self.root_pose[index])
        root_in_urdfroot: jt.Float[pp.LieTensor, "T 7"] = pp.SE3(self.states[index, root_idx])
        urdfroot_world = root_world @ root_in_urdfroot.Inv()

        rr.send_columns(
            path,
            indexes=[time_col],
            columns=rr.Transform3D.columns(
                translation=urdfroot_world.translation().numpy(),
                quaternion =urdfroot_world.rotation().numpy(),
            ),
        )


        root_vel_color  = vis_override.root_velocity.color  if vis_override else [155, 0, 255, 255]
        root_vel_radius = vis_override.root_velocity.radius if vis_override else -5.0
        rr.log(
            f"{path}/{self.root_frame}/velocity",
            rr.Arrows3D.from_fields(radii=rr.Radius(root_vel_radius), colors=root_vel_color), static=True
        )

        for m, node in enumerate(self.model.link_names):
            node_pose: jt.Float[pp.LieTensor, "T 7"] = pp.SE3(self.states[index, m])

            scale_factor = 5.0 if node == self.root_frame else 15.0
            node_vel : jt.Float[torch.Tensor, "T 3"] = self.velocities[index, m] / scale_factor

            rr.send_columns(
                f"{path}/{node}/velocity",
                indexes=[time_col],
                columns=rr.Arrows3D.columns(vectors=node_vel),
            )

            rr.send_columns(
                f"{path}/{node}",
                indexes=[time_col],
                columns=rr.Transform3D.columns(
                    translation=node_pose.translation().numpy(),
                    quaternion =node_pose.rotation().numpy(),
                ),
            )

    def visualize_trajectory(self, path: str, index: int=0):
        self._visualize_trajectory(path, self.time_ns, index)
