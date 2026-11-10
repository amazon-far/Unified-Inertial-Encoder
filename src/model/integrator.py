"""IMU preintegration and trajectory integration over jagged packs."""
import torch
import pypose as pp
import jaxtyping as jt

from dataclasses import dataclass

from ..interface.data_imu import IMUData_Jagged
from ..utility.jagged    import quaternion_multiply, segmented_cumsum, segmented_so3_prefix_scan


@dataclass(kw_only=True)
class Preint_Jagged:
    """IMU preintegration prefixes in the ``IMUData_Jagged`` layout: a run of ``L`` samples becomes
    ``L-1`` rows, with offsets rebased onto the shrunk pack. All in the IMUᵢ (run-start) frame.

        * all_Dp: Δp_ik — position of IMUₖ relative to IMUᵢ, in IMUᵢ frame [m]
        * all_Dv: Δv_ik — velocity of IMUₖ relative to IMUᵢ, in IMUᵢ frame (no gravity comp.) [m/s]
        * all_Dr: Δr_ik = RᵢᵀRₖ — SO3 taking vectors from IMUₖ frame into IMUᵢ frame
        * all_Dt: Δt_ik — elapsed time from i to k [s]
    """
    all_Dp   : jt.Float[torch.Tensor, "pack_out 3"]
    all_Dv   : jt.Float[torch.Tensor, "pack_out 3"]
    all_Dr   : jt.Float[pp.LieTensor, "pack_out 4"]
    all_Dt   : jt.Float[torch.Tensor, "pack_out 1"]
    offsets_B: jt.Int64[torch.Tensor, "B+1"]
    offsets_N: jt.Int64[torch.Tensor, "N+1"]


def euler_preintegrate(data: IMUData_Jagged, dtype: torch.dtype = torch.float64) -> Preint_Jagged:
    """Direct Euler preintegration of ``correct_acc`` [m/s²] and ``correct_gyro`` [rad/s], per device run.

    Returns deltas only: no gravity compensation, bias linearization or covariance. ``dtype`` is the
    working precision: fp64 for the frozen (no-grad) teacher paths, fp32 for differentiable losses.
    """
    a    = data.correct_acc.to(dtype)
    ω    = data.correct_gyro.to(dtype)
    t    = data.time_ns
    offn = data.offsets_N
    offb = data.offsets_B
    device = a.device

    lengths = offn[1:] - offn[:-1]
    G       = int(lengths.numel())
    Tmax    = int(lengths.max()) if G > 0 else 0
    if Tmax < 2:
        raise ValueError("euler_preintegrate needs a device run with at least 2 timesteps.")

    out_len   = (lengths - 1).clamp(min=0)
    new_offn = torch.cat([offn.new_zeros(1), out_len.cumsum(0)])
    pack_out = int(new_offn[-1])

    # One packed output row per input interval, indexed by its endpoint.
    run      = torch.repeat_interleave(torch.arange(G, device=device), out_len)
    out_row  = torch.arange(pack_out, device=device)
    local    = out_row - new_offn[:-1][run]
    endpoint = offn[:-1][run] + local + 1
    start    = endpoint - 1

    a_step   = a[start]
    ω_step   = ω[start]
    Δt_kk1   = (t[endpoint] - t[start]).to(dtype) * 1e-9

    Δr = segmented_so3_prefix_scan(
        pp.Exp(pp.so3(ω_step) * Δt_kk1),
        run,
        Tmax - 1,
    )

    # State at each interval's start: the preceding prefix, or identity/zero for a run's first interval.
    first = local == 0
    prev  = (out_row - 1).clamp(min=0)
    Δr_start = pp.SO3(torch.where(
        first[:, None],
        pp.identity_SO3(pack_out, device=device, dtype=dtype).tensor(),
        Δr.tensor()[prev],
    ))

    # Velocity is not gravity compensated.
    Δv_step = Δr_start @ (a_step * Δt_kk1)
    Δv = segmented_cumsum(Δv_step, new_offn, run)
    Δv_start = torch.where(first[:, None], torch.zeros_like(Δv), Δv[prev])

    Δp_step = Δv_start * Δt_kk1 + 0.5 * (Δr_start @ a_step) * Δt_kk1.square()
    Δp = segmented_cumsum(Δp_step, new_offn, run)
    Δt = (t[endpoint] - t[offn[:-1][run]]).to(dtype) * 1e-9

    # offsets_B land on offsets_N boundaries, so map them through the same shrink.
    new_offb = new_offn[torch.searchsorted(offn, offb)]

    return Preint_Jagged(
        all_Dp    = Δp,
        all_Dv    = Δv,
        all_Dr    = Δr,
        all_Dt    = Δt,
        offsets_B = new_offb,
        offsets_N = new_offn,
    )


def rk4_preintegrate(data: IMUData_Jagged, dtype: torch.dtype = torch.float64) -> Preint_Jagged:
    """Fourth-order Runge--Kutta preintegration, with acc and gyro linearly interpolated between samples.

    Same output and ``dtype`` conventions as :func:`euler_preintegrate`.
    """
    a    = data.correct_acc.to(dtype)
    ω    = data.correct_gyro.to(dtype)
    t    = data.time_ns
    offn = data.offsets_N
    offb = data.offsets_B
    device = a.device

    lengths = offn[1:] - offn[:-1]
    G       = int(lengths.numel())
    Tmax    = int(lengths.max()) if G > 0 else 0
    if Tmax < 2:
        raise ValueError("rk4_preintegrate needs a device run with at least 2 timesteps.")

    out_len  = (lengths - 1).clamp(min=0)
    new_offn = torch.cat([offn.new_zeros(1), out_len.cumsum(0)])
    pack_out = int(new_offn[-1])

    run      = torch.repeat_interleave(torch.arange(G, device=device), out_len)
    out_row  = torch.arange(pack_out, device=device)
    local    = out_row - new_offn[:-1][run]
    endpoint = offn[:-1][run] + local + 1
    start    = endpoint - 1

    a0 = a[start]
    a1 = a[endpoint]
    ω0 = ω[start]
    ω1 = ω[endpoint]
    amid = 0.5 * (a0 + a1)
    ωmid = 0.5 * (ω0 + ω1)
    Δt_kk1 = (t[endpoint] - t[start]).to(dtype) * 1e-9

    # q̇ = 1/2 q ⊗ [ω, 0], for xyzw quaternions and body-frame angular velocity.
    # All interval stages start at identity, so they are local to the interval's start frame.
    def quaternion_rate(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        qv, qw = q[..., :3], q[..., 3:]
        return 0.5 * torch.cat([
            qw * w + torch.linalg.cross(qv, w),
            -(qv * w).sum(dim=-1, keepdim=True),
        ], dim=-1)

    q0 = pp.identity_SO3(pack_out, device=device, dtype=dtype).tensor()
    k1q = quaternion_rate(q0, ω0)
    q2  = q0 + 0.5 * Δt_kk1 * k1q
    k2q = quaternion_rate(q2, ωmid)
    q3  = q0 + 0.5 * Δt_kk1 * k2q
    k3q = quaternion_rate(q3, ωmid)
    q4  = q0 + Δt_kk1 * k3q
    k4q = quaternion_rate(q4, ω1)

    q_inc = q0 + (Δt_kk1 / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    q_inc = q_inc / q_inc.norm(dim=-1, keepdim=True)
    Δr = segmented_so3_prefix_scan(pp.SO3(q_inc), run, Tmax - 1)

    first = local == 0
    prev  = (out_row - 1).clamp(min=0)
    Δr_start = pp.SO3(torch.where(
        first[:, None],
        q0,
        Δr.tensor()[prev],
    ))

    # Normalized only for use as rotations; the derivatives above stay the classical RK4 stages.
    r2 = pp.SO3(q2 / q2.norm(dim=-1, keepdim=True))
    r3 = pp.SO3(q3 / q3.norm(dim=-1, keepdim=True))
    r4 = pp.SO3(q4 / q4.norm(dim=-1, keepdim=True))

    # Acceleration stages expressed in the run-start frame.
    A1 = Δr_start @ a0
    A2 = Δr_start @ (r2 @ amid)
    A3 = Δr_start @ (r3 @ amid)
    A4 = Δr_start @ (r4 @ a1)

    Δv_step = (Δt_kk1 / 6.0) * (A1 + 2.0 * A2 + 2.0 * A3 + A4)
    Δv = segmented_cumsum(Δv_step, new_offn, run)
    Δv_start = torch.where(first[:, None], torch.zeros_like(Δv), Δv[prev])

    # For ṗ=v, the four RK4 velocity stages reduce to this interval displacement.
    Δp_step = Δv_start * Δt_kk1 + (Δt_kk1.square() / 6.0) * (A1 + A2 + A3)
    Δp = segmented_cumsum(Δp_step, new_offn, run)
    Δt = (t[endpoint] - t[offn[:-1][run]]).to(dtype) * 1e-9

    new_offb = new_offn[torch.searchsorted(offn, offb)]

    return Preint_Jagged(
        all_Dp    = Δp,
        all_Dv    = Δv,
        all_Dr    = Δr,
        all_Dt    = Δt,
        offsets_B = new_offb,
        offsets_N = new_offn,
    )


def apply_initial_state(
    preint  : Preint_Jagged,
    R_wb_0  : jt.Float[pp.LieTensor, "N 4"],
    v_b_0   : jt.Float[torch.Tensor, "N 3"],
    gravity_w: jt.Float[torch.Tensor, "N 3"],
) -> tuple[
    jt.Float[torch.Tensor, "pack_NT 3"],
    jt.Float[pp.LieTensor, "pack_NT 4"]
]:
    """Apply per-run initial states and restore each run's omitted initial row.

    Recovers body-frame velocity from ``v_k^w = R_0^wb v_0^b + gravity_w * Δt + R_0^wb Δv_0k``; the
    returned pack has one more row per run than ``preint``. ``gravity_w`` holds one world-frame vector
    per device; pass zeros when acceleration is already gravity-free.
    """
    offsets = preint.offsets_N
    N       = offsets.numel() - 1

    device, dtype = preint.all_Dv.device, preint.all_Dv.dtype
    R_wb_0 = pp.SO3(R_wb_0.to(device=device, dtype=dtype))
    v_b_0  = v_b_0.to(device=device, dtype=dtype)
    gravity_w = gravity_w.to(device=device, dtype=dtype)

    full_offsets = offsets + torch.arange(N + 1, device=device, dtype=offsets.dtype)
    full_lengths = full_offsets[1:] - full_offsets[:-1]
    pack_full    = int(full_offsets[-1])
    run          = torch.repeat_interleave(torch.arange(N, device=device), full_lengths)
    row          = torch.arange(pack_full, device=device)
    initial      = row == full_offsets[:-1][run]
    endpoint_row = row[~initial]

    relative_q = pp.identity_SO3(pack_full, device=device, dtype=dtype).tensor()
    relative_q = relative_q.index_copy(0, endpoint_row, preint.all_Dr.tensor())
    delta_v = torch.zeros(pack_full, 3, device=device, dtype=dtype)
    delta_v = delta_v.index_copy(0, endpoint_row, preint.all_Dv)
    delta_t = torch.zeros(pack_full, 1, device=device, dtype=dtype)
    delta_t = delta_t.index_copy(0, endpoint_row, preint.all_Dt)

    R0_q = R_wb_0.tensor()[run]
    R_wb = pp.SO3(quaternion_multiply(R0_q, relative_q))

    v_w_0 = R_wb_0 @ v_b_0
    v_w = v_w_0[run] + R_wb_0[run] @ delta_v + gravity_w[run] * delta_t
    v_b = R_wb.Inv() @ v_w
    return v_b, R_wb


def integrate_trajectory(
    v_b : jt.Float[torch.Tensor, "pack_NT 3"],
    R_wb: jt.Float[pp.LieTensor, "pack_NT 4"],
    time_ns: jt.Int64[torch.Tensor, "pack_NT 1"],
    offsets_N: jt.Int64[torch.Tensor, "N+1"]
) -> jt.Float[torch.Tensor, "pack_NT 3"]:
    """Integrate jagged body velocity into world position with the trapezoidal rule; runs start at the origin."""
    pack = v_b.shape[0]

    device, dtype = v_b.device, v_b.dtype
    offsets_N = offsets_N.to(device=device)
    time_ns   = time_ns.to(device=device)
    lengths   = offsets_N[1:] - offsets_N[:-1]
    N         = lengths.numel()
    run       = torch.repeat_interleave(torch.arange(N, device=device), lengths)
    row       = torch.arange(pack, device=device)
    initial   = row == offsets_N[:-1][run]
    previous  = (row - 1).clamp(min=0)

    v_w = pp.SO3(R_wb.to(device=device, dtype=dtype)) @ v_b
    dt = (time_ns - time_ns[previous]).to(dtype) * 1e-9
    displacement = 0.5 * (v_w + v_w[previous]) * dt
    displacement = torch.where(initial[:, None], torch.zeros_like(displacement), displacement)
    return segmented_cumsum(displacement, offsets_N, run)
