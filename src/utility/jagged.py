"""Segmented scans over packed (jagged) tensors."""
import torch
import pypose as pp
import jaxtyping as jt


def segmented_cumsum(
    x: jt.Float[torch.Tensor, "pack C"],
    offsets: jt.Int64[torch.Tensor, "segments+1"],
    segment: jt.Int64[torch.Tensor, "pack"],
) -> jt.Float[torch.Tensor, "pack C"]:
    """Cumulative sum over contiguous segments of a packed tensor.

    ``offsets`` holds segment boundaries and ``segment`` maps rows to segments; empty segments are allowed.
    """
    if x.shape[0] == 0:
        return x.clone()

    # Scan the last dim of the transposed [C, pack] tensor: CUDA's outer-dim scan parallelises
    # only over the few C columns, so it is nearly serial on long packs. Same for its backward.
    prefix = torch.cumsum(x.transpose(0, 1).contiguous(), dim=-1).transpose(0, 1).contiguous()
    before = (offsets[:-1] - 1).clamp(min=0)
    base = torch.where(
        (offsets[:-1] > 0)[:, None],
        prefix[before],
        torch.zeros(offsets.numel() - 1, x.shape[-1], dtype=x.dtype, device=x.device),
    )
    return prefix - base[segment]


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product of xyzw quaternions using ordinary tensor operations."""
    lv, lw = left[..., :3], left[..., 3:]
    rv, rw = right[..., :3], right[..., 3:]
    return torch.cat([
        lw * rv + rw * lv + torch.linalg.cross(lv, rv),
        lw * rw - (lv * rv).sum(dim=-1, keepdim=True),
    ], dim=-1)


class SegmentedSO3PrefixScan(torch.autograd.Function):
    """Autograd implementation of the segmented Hillis--Steele SO3 scan."""

    @staticmethod
    def forward(
        ctx,
        increments: torch.Tensor,
        segment: torch.Tensor,
        max_segment_length: int,
    ) -> torch.Tensor:
        result = increments.clone()
        strides: list[int] = []

        stride = 1
        while stride < max_segment_length:
            dst = torch.arange(stride, result.shape[0], device=result.device)
            dst = dst[segment[stride:] == segment[:-stride]]

            strides.append(stride)

            updated = quaternion_multiply(result[dst - stride], result[dst])
            updated = updated / updated.norm(dim=-1, keepdim=True)
            result.index_copy_(0, dst, updated)
            stride *= 2

        ctx.save_for_backward(increments, segment)
        ctx.strides = strides
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        grad_result = grad_outputs[0]
        increments, segment = ctx.saved_tensors

        # Recompute the read state at every level; during a higher-order backward this builds the
        # graph needed to differentiate this VJP.
        result = increments
        states: list[torch.Tensor] = []
        for stride in ctx.strides:
            dst = torch.arange(stride, result.shape[0], device=result.device)
            dst = dst[segment[stride:] == segment[:-stride]]
            states.append(result)
            updated = quaternion_multiply(result[dst - stride], result[dst])
            updated = updated / updated.norm(dim=-1, keepdim=True)
            result = result.index_copy(0, dst, updated)

        grad = grad_result

        for stride, state in reversed(list(zip(ctx.strides, states))):
            dst = torch.arange(stride, state.shape[0], device=state.device)
            dst = dst[segment[stride:] == segment[:-stride]]
            src = dst - stride

            left, right = state[src], state[dst]
            product = quaternion_multiply(left, right)
            norm = product.norm(dim=-1, keepdim=True)
            normalized = product / norm

            # VJP of z = u / ||u||.
            grad_normalized = grad[dst]
            grad_product = (
                grad_normalized
                - normalized * (grad_normalized * normalized).sum(dim=-1, keepdim=True)
            ) / norm

            # VJPs of u = left ⊗ right for xyzw Hamilton quaternions.
            lv, lw = left[..., :3], left[..., 3:]
            rv, rw = right[..., :3], right[..., 3:]
            gv, gw = grad_product[..., :3], grad_product[..., 3:]

            grad_left = torch.cat([
                rw * gv + torch.linalg.cross(rv, gv) - gw * rv,
                (gv * rv).sum(dim=-1, keepdim=True) + gw * rw,
            ], dim=-1)
            grad_right = torch.cat([
                lw * gv + torch.linalg.cross(gv, lv) - gw * lv,
                (gv * lv).sum(dim=-1, keepdim=True) + gw * lw,
            ], dim=-1)

            # Rows in dst were overwritten at this level; all other output rows are
            # identity paths. Inputs may serve as both src and dst, so accumulate both.
            grad_state = grad.index_fill(0, dst, 0)
            grad_state = grad_state.index_add(0, src, grad_left)
            grad_state = grad_state.index_add(0, dst, grad_right)
            grad = grad_state

        return grad, None, None


def segmented_so3_prefix_scan(
    increments: jt.Float[pp.LieTensor, "pack 4"],
    segment: jt.Int64[torch.Tensor, "pack"],
    max_segment_length: int,
) -> jt.Float[pp.LieTensor, "pack 4"]:
    """Inclusive SO3 prefix scan over contiguous segments of a packed tensor.

    Segmented counterpart of ``pypose.cumops``: a Hillis--Steele scan with
    ``O(log(max_segment_length))`` span and no rectangular padding.
    """
    return pp.SO3(SegmentedSO3PrefixScan.apply(
        increments.tensor(),
        segment,
        max_segment_length,
    ))
