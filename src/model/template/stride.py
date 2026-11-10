"""Dense building blocks: MLP, Conv1D, residual and scale wrappers, RoPE and the gradient probe."""
import math
import torch
import torch.nn as nn
import jaxtyping as jt

from typing import Sequence, Literal, TypeAlias


class MLP(nn.Module):
    Support_Act: TypeAlias = Literal["relu", "gelu", "silu", "tanh", "sigmoid", "leaky_relu"]

    def __init__(
        self,
        dim_list              : Sequence[int],
        activation            : Support_Act,
        linear_override_kwargs: dict | None = None
    ):
        super().__init__()
        self.dim_list      = dim_list
        self.linear_kwargs = linear_override_kwargs or dict()
        self.activation    = activation

        linear_layers = [
            nn.Linear(dim_list[i], dim_list[i+1], **self.linear_kwargs)
            for i in range(0, len(dim_list) - 1)
        ]
        layers: list[nn.Module] = []
        for linear in linear_layers:
            layers.append(linear)
            layers.append(self.act_fn(self.activation))
        
        self.net = torch.nn.Sequential(*layers[:-1])

    @staticmethod
    def act_fn(activation: MLP.Support_Act):
        match activation:
            case "relu":    return nn.ReLU()
            case "gelu":    return nn.GELU()
            case "silu":    return nn.SiLU()
            case "tanh":    return nn.Tanh()
            case "sigmoid": return nn.Sigmoid()
            case "leaky_relu": return nn.LeakyReLU()
            case _: raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: jt.Float[torch.Tensor, "*B M"]) -> jt.Float[torch.Tensor, "*B N"]:
        return self.net(x)

    def __repr__(self) -> str:
        return f"MLP(dims={self.dim_list}, act={self.activation}, lienar_kwargs={self.linear_kwargs})"


class Conv1D(nn.Module):
    Support_Act : TypeAlias = Literal["relu", "gelu", "silu", "tanh", "sigmoid"]
    Padding_Mode: TypeAlias = Literal["zeros", "reflect", "replicate", "circular"]

    def __init__(
        self,
        kernels     : list[tuple[int, int, int, Padding_Mode]], # 1D kernels of [(kernel_size, stride, padding, padding_mode)]
        dim_list    : list[int],
        activation  : Support_Act,
        transpose   : bool
    ):
        super().__init__()
        assert len(kernels) == len(dim_list) - 1, \
            f"need one kernel spec per layer: {len(dim_list) - 1} layers but {len(kernels)} kernels"

        self.kernels    = kernels
        self.dim_list   = dim_list
        self.activation = activation

        if transpose:
            conv_layers = [
                nn.ConvTranspose1d(dim_list[i], dim_list[i+1], kernel_size=size, stride=stride, padding=pad, padding_mode=pad_mode)
                for i, (size, stride, pad, pad_mode) in enumerate(kernels)
            ]
        else:
            conv_layers = [
                nn.Conv1d(dim_list[i], dim_list[i+1], kernel_size=size, stride=stride, padding=pad, padding_mode=pad_mode)
                for i, (size, stride, pad, pad_mode) in enumerate(kernels)
            ]
        layers: list[nn.Module] = []
        for conv in conv_layers:
            layers.append(conv)
            layers.append(self.act_fn())

        self.net = torch.nn.Sequential(*layers[:-1])

    def act_fn(self):
        match self.activation:
            case "relu":    return nn.ReLU()
            case "gelu":    return nn.GELU()
            case "silu":    return nn.SiLU()
            case "tanh":    return nn.Tanh()
            case "sigmoid": return nn.Sigmoid()
            case _: raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x: jt.Float[torch.Tensor, "B C_in T"]) -> jt.Float[torch.Tensor, "B C_out T_out"]:
        return self.net(x)

    def __repr__(self) -> str:
        return f"Conv1D(dims={self.dim_list}, kernels={self.kernels}, act={self.activation})"


class Shortcut(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    
    def forward(self, x: torch.Tensor):
        return x + self.module(x)


class LayerScale(nn.Module):
    def __init__(
        self,
        input_dim   : int,
        init_values : float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# RoPE rotation periods in the tokenizer's timestamp unit (ms since window start), geometrically spaced.
# Time-based positions keep the encoding consistent across sample rates and patch sizes.
ROPE_T_MIN_MS = 1.0
ROPE_T_MAX_MS = 100_000.0


def rope_1d_periods(
    head_dim: int, t_min: float = ROPE_T_MIN_MS, t_max: float = ROPE_T_MAX_MS,
) -> jt.Float[torch.Tensor, "D_half"]:
    assert head_dim % 2 == 0
    return torch.logspace(math.log10(t_min), math.log10(t_max), head_dim // 2, dtype=torch.float32)


def rope_1d_tables(
    pos    : jt.Float[torch.Tensor, "B N"],
    periods: jt.Float[torch.Tensor, "D_half"],
) -> tuple[jt.Float[torch.Tensor, "B 1 N D_half"], jt.Float[torch.Tensor, "B 1 N D_half"]]:
    # Phase math stays fp32 even under autocast: positions reach ~1e5 ms, where bf16 error is many radians.
    with torch.autocast(device_type=pos.device.type, enabled=False):
        angle = pos.float()[:, None, :, None] / periods
        return angle.cos(), angle.sin()


def apply_rope(
    x  : jt.Float[torch.Tensor, "B H N D"],
    cos: jt.Float[torch.Tensor, "B 1 N D_half"],
    sin: jt.Float[torch.Tensor, "B 1 N D_half"],
) -> jt.Float[torch.Tensor, "B H N D"]:
    # Rotate interleaved pairs; the fp32 tables promote the products, so cast back once at the end.
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((
        x_even * cos - x_odd * sin,
        x_even * sin + x_odd * cos,
    ), dim=-1).flatten(-2).type_as(x)


class GradientProbe(nn.Module):
    """Numerically-identity op exposing one stable leaf for grad-norm loss balancing."""
    weight: torch.Tensor

    def __init__(self, num_dim: torch.Size):
        super().__init__()
        self.num_dim = num_dim
        self.weight  = torch.ones(num_dim, requires_grad=True)   # fp32; not a Param/buffer (optimizer & DDP ignore it)

    def forward(self, x: jt.Float[torch.Tensor, "*B C"]) -> jt.Float[torch.Tensor, "*B C"]:
        if self.weight.device != x.device:                       # lazy one-time move (plain attr, so .to() skips it)
            self.weight = torch.ones(self.num_dim, device=x.device, requires_grad=True)
        return x * self.weight.to(x.dtype)                       # differentiable cast keeps the fp32 leaf in-graph under AMP
