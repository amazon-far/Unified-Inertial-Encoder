"""Hybrid Muon + AdamW optimizer."""
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import AdamW, Muon, Optimizer


class AdamW_Muon_Hybrid_Optimizer(Optimizer):
    """Single Optimizer wrapper: Muon on trainable 2D parameters, AdamW on everything else."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        muon_lr: float = 1e-3,
        muon_weight_decay: float = 1e-3,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_adjust_lr_fn: str | None = "match_rms_adamw",
        adamw_lr: float = 5e-3,
        adamw_weight_decay: float = 1e-3,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
    ) -> None:
        all_params = [p for p in params if p.requires_grad]
        if not all_params:
            raise ValueError("AdamW_Muon_Hybrid_Optimizer got no trainable parameters")

        muon_params = [p for p in all_params if p.ndim == 2]
        adamw_params = [p for p in all_params if p.ndim != 2]

        defaults = {
            "muon_lr": muon_lr,
            "adamw_lr": adamw_lr,
        }
        super().__init__([{"params": all_params}], defaults)

        self.muon: Muon | None = None
        self.adamw: AdamW | None = None

        if muon_params:
            self.muon = Muon(
                muon_params,
                lr=muon_lr,
                weight_decay=muon_weight_decay,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
                ns_steps=muon_ns_steps,
                adjust_lr_fn=muon_adjust_lr_fn,
            )

        if adamw_params:
            self.adamw = AdamW(
                adamw_params,
                lr=adamw_lr,
                weight_decay=adamw_weight_decay,
                betas=adamw_betas,
                eps=adamw_eps,
            )

        # Expose the real groups for Lightning's LR/progress utilities.
        self.param_groups = []
        if self.muon is not None:
            self.param_groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            self.param_groups.extend(self.adamw.param_groups)

    def step(self, closure: Any | None = None) -> Any:
        loss = closure() if closure is not None else None
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "muon": None if self.muon is None else self.muon.state_dict(),
            "adamw": None if self.adamw is None else self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        muon_state = state_dict.get("muon")
        adamw_state = state_dict.get("adamw")
        if self.muon is not None and muon_state is not None:
            self.muon.load_state_dict(muon_state)
        if self.adamw is not None and adamw_state is not None:
            self.adamw.load_state_dict(adamw_state)


MuonAdamW = AdamW_Muon_Hybrid_Optimizer
