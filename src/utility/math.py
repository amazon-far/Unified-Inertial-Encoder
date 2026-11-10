"""Batched linear and SE(3) interpolation along the time axis."""
import torch
import pypose as pp
import jaxtyping as jt


def linear_interpolate(
    value  : jt.Float[torch.Tensor, "*B T C"],
    time   : jt.Real [torch.Tensor, "*B T"],
    time_ev: jt.Real [torch.Tensor, "*B T2"],
) -> jt.Float[torch.Tensor, "*B T2 C"]:
    """Batched 1-D linear interpolation along time; queries outside the range of `time` produce NaN.

    `time` must be non-decreasing. It may be an integer tensor (e.g. ns): differences are taken in
    the input dtype to keep precision, then cast to float32 for the division (MPS-compatible).
    """
    T = time.size(-1)
    right_idx = torch.searchsorted(time, time_ev).clamp(1, T - 1)
    left_idx  = right_idx - 1

    t_left  = torch.gather(time, dim=-1, index=left_idx )
    t_right = torch.gather(time, dim=-1, index=right_idx)

    delta_q = (time_ev - t_left ).to(torch.float32)
    delta_r = (t_right - t_left ).to(torch.float32)
    alpha   = (delta_q / delta_r).to(value.dtype)                       # *B T2

    C = value.size(-1)
    idx_l = left_idx [..., None].expand(*left_idx .shape, C)
    idx_r = right_idx[..., None].expand(*right_idx.shape, C)
    val_left  = torch.gather(value, dim=-2, index=idx_l)                # *B T2 C
    val_right = torch.gather(value, dim=-2, index=idx_r)                # *B T2 C

    a   = alpha[..., None]                                              # *B T2 1
    out = (1.0 - a) * val_left + a * val_right                          # *B T2 C

    oor = (time_ev < time[..., :1]) | (time_ev > time[..., -1:])        # *B T2
    out = out.masked_fill(oor[..., None], float("nan"))
    return out


def se3_linear_interpolate(
    value  : jt.Float[pp.LieTensor, "*B T 7"],
    time   : jt.Real [torch.Tensor, "*B T"],
    time_ev: jt.Real [torch.Tensor, "*B T2"],
) -> jt.Float[pp.LieTensor, "*B T2 7"]:
    """Batched SE(3) interpolation along time; queries outside the range of `time` produce NaN poses.

    Computes ``Exp(alpha * Log(P_right @ P_left^-1)) @ P_left``; `time` must be non-decreasing.
    """
    T = time.size(-1)
    right_idx = torch.searchsorted(time, time_ev).clamp(1, T - 1)
    left_idx  = right_idx - 1

    t_left  = torch.gather(time, dim=-1, index=left_idx )
    t_right = torch.gather(time, dim=-1, index=right_idx)

    delta_q = (time_ev - t_left ).to(torch.float32)
    delta_r = (t_right - t_left ).to(torch.float32)
    alpha   = (delta_q / delta_r).to(value.dtype)                       # *B T2

    D = value.size(-1)                                                  # 7 for SE3
    idx_l = left_idx [..., None].expand(*left_idx .shape, D)
    idx_r = right_idx[..., None].expand(*right_idx.shape, D)
    P_left  = pp.LieTensor(torch.gather(value, dim=-2, index=idx_l), ltype=value.ltype)
    P_right = pp.LieTensor(torch.gather(value, dim=-2, index=idx_r), ltype=value.ltype)

    delta    = (P_right @ P_left.Inv()).Log()                           # *B T2 6
    scaled   = pp.LieTensor(alpha[..., None] * delta, ltype=delta.ltype)
    P_interp = scaled.Exp() @ P_left                                    # *B T2 7

    oor = (time_ev < time[..., :1]) | (time_ev > time[..., -1:])        # *B T2
    out = torch.as_tensor(P_interp).masked_fill(oor[..., None], float("nan"))
    return pp.LieTensor(out, ltype=value.ltype)
