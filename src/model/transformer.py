"""Transformer trunk over the jagged token layout, with per-block device-only or full attention."""
import torch
import torch.nn as nn
import jaxtyping as jt
from typing import Literal, Sequence, TypeAlias

from .template import stride as pad_module
from .template import jagged as jag_module


Layer_Type: TypeAlias = Literal["full", "device"]


class Transformer(nn.Module):
    def __init__(
        self,
        feature_dim : int,
        layer       : int,
        layer_type  : list[Layer_Type],
        is_causal   : bool,
        use_rope    : bool,
        num_heads   : int,
        mlp_active  : pad_module.MLP.Support_Act,
        moe_config  : tuple[int, int] | list[int] | None = None,
    ) -> None:
        assert len(layer_type) == layer
        super().__init__()
        if is_causal:
            raise ValueError("Transformer requires noncausal attention (is_causal=False).")
        if moe_config is not None:
            raise ValueError("Transformer uses dense feedforward layers; moe_config must be None.")

        self.layer_type: list[Layer_Type] = layer_type
        self.is_causal = is_causal

        # moe_config is only accepted (as None) so saved dense configs still load.
        self.blocks = nn.ModuleList(
            jag_module.Block(feature_dim, is_causal, use_rope, num_heads, mlp_active=mlp_active)
            for _ in layer_type
        )

    def forward(
        self,
        x           : jt.Float[torch.Tensor, "pack_BNS C"],
        b_offsets   : jt.Int64[torch.Tensor, "B+1"],
        n_offsets   : jt.Int64[torch.Tensor, "N+1"],
        timestamp   : jt.Float[torch.Tensor, "pack_BNS"],
        frame       : jt.Int64[torch.Tensor, "pack_BNS"] | None = None,
        feature_from: Sequence[int] | None = None,
    ) -> jt.Float[torch.Tensor, "pack_BNS kC"]:
        # ``feature_from``: block indices whose outputs are channel-concatenated; None returns the last block.
        taken: list[torch.Tensor] = []
        for idx, (layer_type, block) in enumerate(zip(self.layer_type, self.blocks)):
            match layer_type:
                case "device":
                    x = block(x, n_offsets, timestamp)
                case "full":
                    x = block(x, b_offsets, timestamp)
            if feature_from is not None and idx in feature_from:
                taken.append(x)

        if feature_from is None:
            return x
        return torch.cat(taken, dim=-1)
