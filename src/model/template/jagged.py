"""Attention and transformer block for the jagged (packed) token layout, on flash-attn or SDPA."""
import torch
import warnings

try:
    import flash_attn
except ImportError:
    flash_attn = None

import torch.nn  as nn
import jaxtyping as jt


from . import stride as _modules


MLP         = _modules.MLP



class Attention_on_CUDA(nn.Module):
    def __init__(
        self,
        input_dim   : int,
        num_heads   : int,
        causal      : bool,
        rope_1d     : bool,
    ) -> None:
        super().__init__()
        assert input_dim % num_heads == 0

        self.causal    = causal
        self.rope1d    = rope_1d
        self.num_heads = num_heads
        self.head_dim  = input_dim // num_heads

        self.qkv       = nn.Linear(input_dim, input_dim * 3)
        self.proj      = nn.Linear(input_dim, input_dim)

        self.rope_periods: jt.Float[torch.Tensor, "D_half"]
        self.register_buffer("rope_periods", _modules.rope_1d_periods(self.head_dim), persistent=False)

    def forward(self, x: jt.Float[torch.Tensor, "pack_BNS C"], offsets: jt.Int64[torch.Tensor, "B+1"], timestamp: jt.Float[torch.Tensor, "pack_BNS"] | None) -> jt.Float[torch.Tensor, "pack_BNS C"]:
        if flash_attn is None:
            raise ImportError("flash_attn is required for Attention_on_CUDA; the SDPA-based Attention_on_OtherPlatforms is selected automatically when flash_attn is unavailable.")

        pack, C = x.shape

        cu_seqlens = offsets.to(torch.int32)
        max_seqlen = int((offsets[1:] - offsets[:-1]).max().item())

        q, k, v = _qkv_rope(self, x, timestamp)
        qkv     = torch.stack((q, k, v), dim=1)                                  # [pack, 3, H, D]

        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, causal=self.causal,
        )                                                                        # [pack, H, D]

        return self.proj(out.reshape(pack, C))


_WARNED_NO_FLASH_ATTN = False


class Attention_on_OtherPlatforms(nn.Module):
    """SDPA drop-in for ``Attention_on_CUDA`` where ``flash_attn`` is unavailable (Apple MPS / CPU).

    Same parameters and numerically equivalent: a block-diagonal mask over the packed sequence keeps
    attention within each segment. Segments are contiguous, so a global lower-triangular constraint
    coincides with per-segment causality.
    """

    def __init__(
        self,
        input_dim   : int,
        num_heads   : int,
        causal      : bool,
        rope_1d     : bool,
    ) -> None:
        super().__init__()
        assert input_dim % num_heads == 0

        global _WARNED_NO_FLASH_ATTN
        if not _WARNED_NO_FLASH_ATTN:
            _WARNED_NO_FLASH_ATTN = True
            warnings.warn(
                "flash_attn is not available on this platform; falling back to an SDPA-based "
                "jagged attention. This is functionally identical but slower and more memory "
                "hungry than the CUDA flash-attn path. Install `flash_attn` on a CUDA machine "
                "for the fast kernel.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.causal    = causal
        self.rope1d    = rope_1d
        self.num_heads = num_heads
        self.head_dim  = input_dim // num_heads

        self.qkv       = nn.Linear(input_dim, input_dim * 3)
        self.proj      = nn.Linear(input_dim, input_dim)

        self.rope_periods: jt.Float[torch.Tensor, "D_half"]
        self.register_buffer("rope_periods", _modules.rope_1d_periods(self.head_dim), persistent=False)

    def forward(self, x: jt.Float[torch.Tensor, "pack_BNS C"], offsets: jt.Int64[torch.Tensor, "B+1"], timestamp: jt.Float[torch.Tensor, "pack_BNS"] | None) -> jt.Float[torch.Tensor, "pack_BNS C"]:
        pack, C = x.shape

        q, k, v = _qkv_rope(self, x, timestamp)

        # [pack, H, D] -> [1, H, pack, D]: SDPA attends over the second-to-last dim.
        q = q.transpose(0, 1)[None]
        k = k.transpose(0, 1)[None]
        v = v.transpose(0, 1)[None]

        attn_mask = self._block_diagonal_mask(offsets, pack, x.device)
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # [1, H, pack, D]

        out = out[0].transpose(0, 1).reshape(pack, C)                                # [pack, H, D] -> [pack, C]
        return self.proj(out)

    def _block_diagonal_mask(
        self, offsets: jt.Int64[torch.Tensor, "B+1"], pack: int, device: torch.device,
    ) -> jt.Bool[torch.Tensor, "pack pack"]:
        # True = the query/key pair may attend.
        seg_lengths = (offsets[1:] - offsets[:-1]).to(device)
        seg_id      = torch.repeat_interleave(
            torch.arange(seg_lengths.numel(), device=device), seg_lengths,
        )
        mask = seg_id[:, None] == seg_id[None, :]
        if self.causal:
            idx  = torch.arange(pack, device=device)
            mask = mask & (idx[:, None] >= idx[None, :])
        return mask


def _qkv_rope(
    attn      : Attention_on_CUDA | Attention_on_OtherPlatforms,
    x         : jt.Float[torch.Tensor, "pack_BNS C"],
    timestamp : jt.Float[torch.Tensor, "pack_BNS"] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project to per-head q/k/v (``[pack, H, D]`` each) and rotate q/k by the token timestamps when
    RoPE is enabled."""
    pack    = x.shape[0]
    qkv     = attn.qkv(x).reshape(pack, 3, attn.num_heads, attn.head_dim)
    q, k, v = qkv.unbind(1)                                                       # each [pack, H, D]

    if attn.rope1d:
        assert timestamp is not None, "rope_1d attention requires per-token timestamps"
        cos, sin = _modules.rope_1d_tables(timestamp[None], attn.rope_periods)    # [1, 1, pack, D/2]
        q        = _modules.apply_rope(q.transpose(0, 1)[None], cos, sin)[0].transpose(0, 1)
        k        = _modules.apply_rope(k.transpose(0, 1)[None], cos, sin)[0].transpose(0, 1)

    return q, k, v


Attention = Attention_on_CUDA if (flash_attn is not None) else Attention_on_OtherPlatforms




class Block(nn.Module):
    """Pre-norm MLP + attention residual block over the packed layout; ``attend`` holds the sequence
    offsets for the varlen ``Attention``."""
    def __init__(
            self,
            feature_dim : int,
            is_causal   : bool,
            rope_1d     : bool,
            num_heads   : int,
            mlp_active  : MLP.Support_Act,
        ) -> None:
        super().__init__()
        hidden = feature_dim * 4
        ffn = MLP(dim_list=(feature_dim, hidden, feature_dim), activation=mlp_active)
        self.mlp  = _modules.Shortcut(nn.Sequential(
            nn.LayerNorm(feature_dim),
            ffn,
            _modules.LayerScale(feature_dim),
        ))

        self.attn_ln = nn.LayerNorm(feature_dim)
        self.attn    = Attention(input_dim=feature_dim, num_heads=num_heads, causal=is_causal, rope_1d=rope_1d)
        self.attn_ls = _modules.LayerScale(feature_dim)

    def forward(self, x: jt.Float[torch.Tensor, "pack_BNS C"], attend: jt.Int64[torch.Tensor, "B+1"], timestamp: jt.Float[torch.Tensor, "pack_BNS"] | None) -> jt.Float[torch.Tensor, "pack_BNS C"]:
        x     = self.mlp(x)
        x     = x + self.attn_ls(self.attn(self.attn_ln(x), attend, timestamp))
        return x
