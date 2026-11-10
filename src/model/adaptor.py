"""Identity adaptor applied to the tokens before the trunk, called as
forward(feature, b_offsets, n_offsets, time_ms, frame)."""
import torch
import typing as tp


class EmptyAdaptor(torch.nn.Identity):
    def forward(self, input: torch.Tensor, *jagged_ctx: tp.Any) -> torch.Tensor:
        return input
