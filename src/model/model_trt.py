"""Static-shape TensorRT deployment of FoundationIMU: the timestamp-dependent jagged schedule is
precomputed as dense matrices and masks, so the network exports to ONNX and builds into an engine."""
from __future__ import annotations

import hashlib
import importlib
import math
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import jaxtyping as jt

from ..interface.human import CANONICAL_NAMES
from .template.stride import ROPE_T_MAX_MS, ROPE_T_MIN_MS

if tp.TYPE_CHECKING:
    from .model import FoundationIMU


@dataclass(frozen=True)
class OursConfig:
    """The fixed hyper-parameters of the deployed configuration (continuous_foundation_imu_full)."""
    token_dim         : int   = 768
    num_heads         : int   = 8
    layer_type        : tuple = ("device", "full", "device", "full", "device", "full")
    feature_from_trunk: tuple = (1, 3, 5)
    num_register      : int   = 4
    window_span_ms    : float = 125.0
    micro_split       : int   = 4
    num_taps          : int   = 4
    num_sensors       : int   = len(CANONICAL_NAMES)
    lang_dim          : int   = 768
    ln_eps            : float = 1e-5
    canon_eps         : float = 1e-6
    is_causal         : bool  = False
    use_rope          : bool  = True

    @property
    def micro_span_ms(self) -> float: return self.window_span_ms / self.micro_split
    @property
    def head_dim(self) -> int: return self.token_dim // self.num_heads
    @property
    def feat_dim(self) -> int: return self.token_dim * len(self.feature_from_trunk)
    @property
    def mlp_hidden(self) -> int: return self.token_dim * 4


OURS = OursConfig()
MASK_NEG = -1e4   # additive attention mask value; exp(-1e4) underflows to exactly 0 in fp16/fp32


# Dense (static-shape) network; only ONNX/TensorRT-friendly ops.

class _DenseAttention(nn.Module):
    """Masked multi-head self-attention with interleaved-pair RoPE (``stride.apply_rope``)."""
    def __init__(self, cfg: OursConfig):
        super().__init__()
        self.H, self.D = cfg.num_heads, cfg.head_dim
        self.qkv  = nn.Linear(cfg.token_dim, cfg.token_dim * 3)
        self.proj = nn.Linear(cfg.token_dim, cfg.token_dim)

    @staticmethod
    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Rotates in fp32 and casts back once, exactly like ``stride.apply_rope`` under bf16-true.
        xf = x.float()
        xe, xo = xf[..., 0::2], xf[..., 1::2]
        return torch.stack((xe * cos - xo * sin, xe * sin + xo * cos), dim=-1).flatten(-2).to(x.dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        q, k, v = self.qkv(x).reshape(L, 3, self.H, self.D).permute(1, 2, 0, 3).unbind(0)   # each [H, L, D]
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        att = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.D)) + mask.to(x.dtype)
        out = torch.matmul(torch.softmax(att, dim=-1), v)
        return self.proj(out.permute(1, 0, 2).reshape(L, C))


class _DenseBlock(nn.Module):
    """``jagged.Block``: pre-norm MLP residual first, then pre-norm attention residual, both layer-scaled."""
    def __init__(self, cfg: OursConfig):
        super().__init__()
        self.mlp_ln  = nn.LayerNorm(cfg.token_dim, eps=cfg.ln_eps)
        self.mlp_fc1 = nn.Linear(cfg.token_dim, cfg.mlp_hidden)
        self.mlp_fc2 = nn.Linear(cfg.mlp_hidden, cfg.token_dim)
        self.mlp_ls  = nn.Parameter(torch.ones(cfg.token_dim))
        self.attn_ln = nn.LayerNorm(cfg.token_dim, eps=cfg.ln_eps)
        self.attn    = _DenseAttention(cfg)
        self.attn_ls = nn.Parameter(torch.ones(cfg.token_dim))

    def forward(self, x, mask, cos, sin):
        x = x + self.mlp_fc2(F.silu(self.mlp_fc1(self.mlp_ln(x)))) * self.mlp_ls
        x = x + self.attn(self.attn_ln(x), mask, cos, sin) * self.attn_ls
        return x


class OursDenseCore(nn.Module):
    """The whole "Ours" forward on fixed shapes.

    Trunk row layout (``L = 1 + R + N*S``): ``[cls, reg_0..reg_{R-1}, dev_0 tok_0..tok_{S-1}, dev_1 ...]``.
    Attention is permutation-invariant given the RoPE tables and masks, so this order may differ from
    the jagged pack order.
    """
    def __init__(self, cfg: OursConfig, num_devices: int, max_samples: int, num_tokens: int):
        super().__init__()
        self.cfg = cfg
        self.N, self.T, self.S = num_devices, max_samples, num_tokens
        self.M = num_tokens * cfg.micro_split
        self.R = cfg.num_register
        self.L = 1 + self.R + self.N * self.S
        C, m, Cf = cfg.token_dim, cfg.num_taps, cfg.feat_dim

        # Tokenizer (ContinuousKernel_Conv1DTokenizer wrapped in AddRegister)
        self.device_embed = nn.Parameter(torch.zeros(cfg.num_sensors, C))
        self.cls_embed    = nn.Parameter(torch.zeros(1, C))
        self.reg_embed    = nn.Parameter(torch.zeros(self.R, C))
        self.enc1 = nn.Linear(6 * m, C // 2)            # micro-window taps  -> C/2
        self.enc2 = nn.Linear((C // 2) * m, C)          # token taps         -> C
        self.corr_dec1 = nn.Linear(Cf, (C // 2) * m)    # token -> C/2 per tap
        self.corr_dec2 = nn.Linear(C // 2, 6 * m)       # micro -> 6 per tap
        self.vel_dec1  = nn.Linear(Cf, (C // 2) * m)
        self.vel_dec2  = nn.Linear(C // 2, 3 * m)
        self.lang_fc1  = nn.Linear(Cf, (Cf + cfg.lang_dim) // 2)
        self.lang_fc2  = nn.Linear((Cf + cfg.lang_dim) // 2, cfg.lang_dim)

        self.blocks = nn.ModuleList(_DenseBlock(cfg) for _ in cfg.layer_type)

    # Engine I/O names, in order.
    INPUTS  = ("x6", "pool1", "pool2", "up2", "up1", "sensor_id", "rope_cos", "rope_sin", "mask_dev", "mask_full")
    OUTPUTS = ("correction", "body_vel", "lang", "trunk_feature")

    def forward(
        self,
        x6       : jt.Float[torch.Tensor, "N T 6"],
        pool1    : jt.Float[torch.Tensor, "N Mm T"],     # row-normalised sample -> micro-tap pooling
        pool2    : jt.Float[torch.Tensor, "N Sm M"],     # row-normalised micro  -> token-tap pooling
        up2      : jt.Float[torch.Tensor, "N M Sm"],     # token-tap -> micro interpolation
        up1      : jt.Float[torch.Tensor, "N T Mm"],     # micro-tap -> sample interpolation
        sensor_id: jt.Int64[torch.Tensor, "N"],
        rope_cos : jt.Float[torch.Tensor, "L Dh"],
        rope_sin : jt.Float[torch.Tensor, "L Dh"],
        mask_dev : jt.Float[torch.Tensor, "L L"],        # additive, "device" layers
        mask_full: jt.Float[torch.Tensor, "L L"],        # additive, "full" layers
    ):
        cfg, N, S, M, m, C = self.cfg, self.N, self.S, self.M, self.cfg.num_taps, self.cfg.token_dim
        # Engine I/O stays fp32; compute runs in the parameters' dtype (TensorRT >= 11 is strongly typed).
        # RoPE tables stay fp32 (see _rope).
        dt = self.enc1.weight.dtype
        x6, pool1, pool2, up2, up1 = (t.to(dt) for t in (x6, pool1, pool2, up2, up1))

        # Tokenizer: two continuous convolutions (pool -> Linear), SiLU between.
        h = torch.bmm(pool1, x6).reshape(N, M, m * 6)                   # tap-major like buf.reshape(W, m*in)
        h = F.silu(self.enc1(h))
        h = torch.bmm(pool2, h).reshape(N, S, m * (C // 2))
        tok = self.enc2(h) + self.device_embed[sensor_id][:, None, :]
        x = torch.cat([self.cls_embed, self.reg_embed, tok.reshape(N * S, C)], dim=0)   # [L, C]

        # Trunk: alternating device / full attention, multi-layer readout.
        cos, sin = rope_cos[None], rope_sin[None]
        taken = []
        for i, (lt, block) in enumerate(zip(cfg.layer_type, self.blocks)):
            x = block(x, mask_dev if lt == "device" else mask_full, cos, sin)
            if i in cfg.feature_from_trunk:
                taken.append(x)
        feat = torch.cat(taken, dim=-1)

        # FoundationIMU.canonicalize_for_task: per-token_dim-chunk LayerNorm, no affine.
        task = F.layer_norm(feat.reshape(self.L, -1, C), (C,), eps=cfg.canon_eps).reshape(self.L, -1)

        # Decoders: transpose continuous convolutions (Linear -> interpolate).
        base = task[1 + self.R:].reshape(N, S, -1)
        h = self.corr_dec1(base).reshape(N, S * m, C // 2)                               # taps [N, S*m, C/2]
        h = F.silu(torch.bmm(up2, h))
        correction = torch.bmm(up1, self.corr_dec2(h).reshape(N, M * m, 6))              # [N, T, 6]
        h = self.vel_dec1(base).reshape(N, S * m, C // 2)
        h = F.silu(torch.bmm(up2, h))
        body_vel = torch.bmm(up1, self.vel_dec2(h).reshape(N, M * m, 3))                 # [N, T, 3]
        lang = self.lang_fc2(F.silu(self.lang_fc1(task[0])))
        return correction.float(), body_vel.float(), lang.float(), feat.float()


# Schedule: the timestamp-only part of the jagged model, materialised as dense matrices.

@dataclass
class Schedule:
    """Everything the network needs besides the raw acc/gyro values (all on the engine device)."""
    pool1      : torch.Tensor   # [N, M*m, T]
    pool2      : torch.Tensor   # [N, S*m, M]
    up2        : torch.Tensor   # [N, M, S*m]
    up1        : torch.Tensor   # [N, T, M*m]
    sensor_id  : torch.Tensor   # [N] int64
    rope_cos   : torch.Tensor   # [L, D/2] fp32
    rope_sin   : torch.Tensor   # [L, D/2] fp32
    mask_dev   : torch.Tensor   # [L, L] fp32 additive
    mask_full  : torch.Tensor   # [L, L] fp32 additive
    valid      : torch.Tensor   # [N, T] bool   (samples)
    token_valid: torch.Tensor   # [N, S] bool
    token_time : torch.Tensor   # [N, S] fp32 device-relative ms (RoPE position)

    def as_inputs(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, k) for k in OursDenseCore.INPUTS if k != "x6"}


def _tap_weights(u: torch.Tensor, num_taps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Triangular basis of ``ContinuousConv1D``: ``u`` in [0, 1) -> (k0, frac) with weights (1-frac, frac)
    on taps ``k0``/``k0+1``."""
    ui = u / (1.0 / (num_taps - 1))          # same arithmetic as ContinuousConv1D (u / self.width)
    k0 = ui.floor().clamp_(0, num_taps - 2).long()
    return k0, ui - k0


def _two_tap_matrix(rows_valid: torch.Tensor, col_bin: torch.Tensor, u: torch.Tensor,
                    num_bins: int, num_taps: int) -> torch.Tensor:
    """Dense ``[N, R, num_bins*num_taps]`` matrix with each valid row's two tap weights at
    ``col_bin*num_taps + k0`` / ``+1`` (rows that are invalid stay all-zero)."""
    N, Rn = col_bin.shape
    k0, frac = _tap_weights(u, num_taps)
    col0 = (col_bin * num_taps + k0).clamp_(0, num_bins * num_taps - 2)
    w0 = torch.where(rows_valid, 1.0 - frac, torch.zeros_like(frac))
    w1 = torch.where(rows_valid, frac, torch.zeros_like(frac))
    out = u.new_zeros(N, Rn, num_bins * num_taps)
    out.scatter_(2, col0[..., None], w0[..., None])
    out.scatter_(2, (col0 + 1)[..., None], w1[..., None])
    return out


def _row_normalise(a: torch.Tensor) -> torch.Tensor:
    """Basis-weighted mean as in ``ContinuousConv1D``: divide each row by its weight mass (clamped)."""
    return a / a.sum(dim=-1, keepdim=True).clamp_min(1e-6)


@dataclass
class FoundationIMU_TRT_Output:
    dec_correct_acc : jt.Float[torch.Tensor, "N T 3"]
    dec_correct_gyro: jt.Float[torch.Tensor, "N T 3"]
    dec_body_vel    : jt.Float[torch.Tensor, "N T 3"]
    dec_lang_feature: jt.Float[torch.Tensor, "C_lang"]
    trunk_feature   : jt.Float[torch.Tensor, "L Cf"]    # [cls, registers, N*S tokens] x readout channels
    valid           : jt.Bool [torch.Tensor, "N T"]
    token_valid     : jt.Bool [torch.Tensor, "N S"]


class FoundationIMU_TRT:
    """Static-shape TensorRT deployment of the "Ours" FoundationIMU.

    Call ``from_torch_model`` to copy weights and build the engine, then call with dense ``[N, T, ...]``
    tensors on the GPU.

    Args:
        max_samples    : maximum measurements per device (``T``); shorter streams are padded and masked
                         through ``valid``.
        max_timespan_ms: maximum first-to-last-sample span of one device; fixes the token grid.
        num_devices    : number of IMU devices (``N``) in every call.
        precision      : engine compute precision; RoPE phases and the pooling normalisation stay fp32
                         outside the engine.
        cache_dir      : where serialised engines are kept (keyed by config + weights + TRT version).
    """

    def __init__(
        self,
        max_samples    : int,
        max_timespan_ms: float,
        num_devices    : int,
        device         : str | torch.device = "cuda:0",
        precision      : tp.Literal["fp32", "fp16", "bf16"] = "fp16",
        cache_dir      : str | Path | None = None,
        workspace_gb   : float = 4.0,
        cfg            : OursConfig = OURS,
    ):
        assert max_samples >= 1 and num_devices >= 1 and max_timespan_ms > 0
        self.cfg = cfg
        self.T = int(max_samples)
        self.N = int(num_devices)
        self.max_timespan_ms = float(max_timespan_ms)
        self.S = int(math.ceil(self.max_timespan_ms / cfg.window_span_ms))
        self.M = self.S * cfg.micro_split
        self.R = cfg.num_register
        self.L = 1 + self.R + self.N * self.S
        self.device = torch.device(device)
        self.precision = precision
        self.workspace_gb = workspace_gb
        self.cache_dir = Path(cache_dir) if cache_dir is not None else \
            Path(os.environ.get("FIMU_TRT_CACHE", Path.home() / ".cache" / "foundation_imu_trt"))

        self.core = OursDenseCore(cfg, self.N, self.T, self.S).eval()   # fp32 master copy of the weights
        self._rope_periods = torch.logspace(math.log10(ROPE_T_MIN_MS), math.log10(ROPE_T_MAX_MS),
                                            cfg.head_dim // 2, dtype=torch.float32, device=self.device)
        self._run_id = torch.cat([torch.tensor([0]), torch.full((self.R,), 1),
                                  torch.arange(self.N).repeat_interleave(self.S) + 2]).to(self.device)
        self._engine = None
        self._context = None
        self._io: dict[str, tuple[tuple[int, ...], torch.dtype, bool]] = {}
        self._outputs: dict[str, torch.Tensor] = {}
        self._weights_loaded = False

    def from_torch_model(self, model: "FoundationIMU", build: bool = True) -> "FoundationIMU_TRT":
        """Copy the weights of a ``FoundationIMU`` and build the engine.

        Raises ``ValueError`` unless ``model`` is the exact "Ours" configuration, since any other module
        would be silently mis-mapped.
        """
        self._validate(model)
        sd = {k: v.detach().to(torch.float32) for k, v in model.state_dict().items()}
        tk = "tokenizer.base_tokenizer."
        mapping: dict[str, str] = {
            "device_embed": tk + "device_embed", "cls_embed": tk + "cls_embed", "reg_embed": "tokenizer.reg_embed",
            "enc1.weight": tk + "imu_encoder.module.0.proj.weight", "enc1.bias": tk + "imu_encoder.module.0.proj.bias",
            "enc2.weight": tk + "imu_encoder.module.2.proj.weight", "enc2.bias": tk + "imu_encoder.module.2.proj.bias",
            "corr_dec1.weight": tk + "imu_correction_decoder.module.0.proj.weight",
            "corr_dec1.bias":   tk + "imu_correction_decoder.module.0.proj.bias",
            "corr_dec2.weight": tk + "imu_correction_decoder.module.2.proj.weight",
            "corr_dec2.bias":   tk + "imu_correction_decoder.module.2.proj.bias",
            "vel_dec1.weight":  tk + "imu_velocity_decoder.module.0.proj.weight",
            "vel_dec1.bias":    tk + "imu_velocity_decoder.module.0.proj.bias",
            "vel_dec2.weight":  tk + "imu_velocity_decoder.module.2.proj.weight",
            "vel_dec2.bias":    tk + "imu_velocity_decoder.module.2.proj.bias",
            "lang_fc1.weight": tk + "lang_decoder.net.0.weight", "lang_fc1.bias": tk + "lang_decoder.net.0.bias",
            "lang_fc2.weight": tk + "lang_decoder.net.2.weight", "lang_fc2.bias": tk + "lang_decoder.net.2.bias",
        }
        for i in range(len(self.cfg.layer_type)):
            b, s = f"blocks.{i}.", f"trunk.blocks.{i}."
            mapping.update({
                b + "mlp_ln.weight": s + "mlp.module.0.weight", b + "mlp_ln.bias": s + "mlp.module.0.bias",
                b + "mlp_fc1.weight": s + "mlp.module.1.net.0.weight", b + "mlp_fc1.bias": s + "mlp.module.1.net.0.bias",
                b + "mlp_fc2.weight": s + "mlp.module.1.net.2.weight", b + "mlp_fc2.bias": s + "mlp.module.1.net.2.bias",
                b + "mlp_ls": s + "mlp.module.2.gamma",
                b + "attn_ln.weight": s + "attn_ln.weight", b + "attn_ln.bias": s + "attn_ln.bias",
                b + "attn.qkv.weight": s + "attn.qkv.weight", b + "attn.qkv.bias": s + "attn.qkv.bias",
                b + "attn.proj.weight": s + "attn.proj.weight", b + "attn.proj.bias": s + "attn.proj.bias",
                b + "attn_ls": s + "attn_ls.gamma",
            })
        new_sd = {}
        for dst, src in mapping.items():
            if src not in sd:
                raise ValueError(f"source state_dict lacks {src!r} (expected by {dst!r})")
            new_sd[dst] = sd[src].reshape(self.core.state_dict()[dst].shape)
        self.core.load_state_dict(new_sd, strict=True)
        # RoPE periods follow the module dtype: a bf16-true checkpoint attends with bf16-rounded periods,
        # so take the buffer as the source module holds it rather than recomputing it.
        periods = tp.cast(torch.Tensor, tp.cast(tp.Any, model.trunk.blocks[0]).attn.rope_periods)
        self._rope_periods = periods.detach().to(self.device, torch.float32).clone()
        self._weights_loaded = True
        self._engine = self._context = None
        if build:
            self.build()
        return self

    def _validate(self, model: "FoundationIMU") -> None:
        from .adaptor   import EmptyAdaptor
        from .model     import FoundationIMU
        from .tokenizer import AddRegister, ContinuousKernel_Conv1DTokenizer
        cfg = self.cfg
        def need(cond: object, what: str) -> None:
            if not cond:
                raise ValueError(f"from_torch_model: module is not the 'Ours' configuration: {what}")
        need(isinstance(model, FoundationIMU), f"expected FoundationIMU, got {type(model).__name__}")
        need(isinstance(model.adaptor, EmptyAdaptor), "adaptor must be the identity EmptyAdaptor")
        need(model.is_causal == cfg.is_causal, f"is_causal must be {cfg.is_causal}")
        need(list(model.feature_from_trunk) == list(cfg.feature_from_trunk), f"feature_from_trunk must be {list(cfg.feature_from_trunk)}")
        need(model.tok_config == (cfg.token_dim, cfg.window_span_ms, "ms"), f"tok_config must be {(cfg.token_dim, cfg.window_span_ms, 'ms')}")
        tok = model.tokenizer
        need(isinstance(tok, AddRegister) and tok.num_register == cfg.num_register, f"tokenizer must be AddRegister with {cfg.num_register} registers")
        need(isinstance(tok.base_tokenizer, ContinuousKernel_Conv1DTokenizer), "base tokenizer must be ContinuousKernel_Conv1DTokenizer")
        bt = tp.cast(ContinuousKernel_Conv1DTokenizer, tok.base_tokenizer)
        need(bt.output_dim == cfg.token_dim and bt.window_span_ms == cfg.window_span_ms and bt.micro_split == cfg.micro_split,
             f"tokenizer must be dim {cfg.token_dim}, window {cfg.window_span_ms} ms, micro_split {cfg.micro_split}")
        need(bt.num_trunk_features == len(cfg.feature_from_trunk), "tokenizer num_trunk_features mismatch")
        for enc in (bt.imu_encoder, bt.imu_correction_decoder, bt.imu_velocity_decoder):
            need(list(enc.timespan_ms) == [cfg.micro_span_ms, cfg.window_span_ms], "tokenizer timespans mismatch")
            need(len(enc.module) == 3 and enc.module[0].num_taps == cfg.num_taps and enc.module[2].num_taps == cfg.num_taps
                 and type(enc.module[1]).__name__ == "SiLU", f"tokenizer convs must be 2 x {cfg.num_taps}-tap with SiLU")
        need(len(bt.lang_decoder.net) == 3 and type(bt.lang_decoder.net[1]).__name__ == "SiLU", "lang decoder must be a 2-layer SiLU MLP")
        trunk = model.trunk
        need(list(trunk.layer_type) == list(cfg.layer_type), f"trunk layer_type must be {list(cfg.layer_type)}")
        need(bool(trunk.is_causal == cfg.is_causal), "trunk causality mismatch")
        for blk in tp.cast(tp.Sequence[tp.Any], trunk.blocks):
            need(blk.attn.num_heads == cfg.num_heads and blk.attn.head_dim == cfg.head_dim, f"attention must be {cfg.num_heads} heads of {cfg.head_dim}")
            need(getattr(blk.attn, "rope1d", False) == cfg.use_rope and getattr(blk.attn, "causal", cfg.is_causal) == cfg.is_causal, "attention rope/causal mismatch")
            need(blk.attn_ln.eps == cfg.ln_eps and blk.mlp.module[0].eps == cfg.ln_eps, "LayerNorm eps mismatch")
            need(type(blk.mlp.module[1]).__name__ == "MLP" and type(blk.mlp.module[1].net[1]).__name__ == "SiLU"
                 and blk.mlp.module[1].net[0].out_features == cfg.mlp_hidden, "block FFN must be a dense SiLU MLP with 4x hidden")

    @torch.no_grad()
    def schedule(
        self,
        time_ns  : jt.Int64[torch.Tensor, "N T"],
        valid    : jt.Bool [torch.Tensor, "N T"] | None,
        sensor_id: jt.Int  [torch.Tensor, "N"],
    ) -> Schedule:
        """Materialise the timestamp-dependent part of the model.

        A static-shape, sync-free re-implementation of ``ContinuousEncoder._segment_windows`` on the
        padded ``[N, T]`` grid. ``time_ns`` are per-device int64 nanoseconds (any origin).

        Contract (not checked, to stay sync-free): valid samples are time-sorted within each device and
        every device has at least one valid sample. Samples at or beyond ``max_timespan_ms`` after the
        device's first sample are treated as invalid.

        A micro-window whose fp32 mean time sits exactly on a token boundary can differ from the
        reference's ``segment_reduce`` at the ulp level; this needs exactly representable times.

        Reuse the returned ``Schedule`` across calls whose timestamps do not change.
        """
        cfg, N, T, S, M, m, L = self.cfg, self.N, self.T, self.S, self.M, self.cfg.num_taps, self.L
        dev = self.device
        assert not time_ns.is_floating_point(), "time_ns must be an integer nanosecond tensor (IMUData.time_ns)"
        assert time_ns.shape == (N, T) and sensor_id.shape == (N,), \
            f"expected time_ns [{N}, {T}] and sensor_id [{N}], got {tuple(time_ns.shape)}, {tuple(sensor_id.shape)}"
        time_ns = time_ns.to(dev, torch.int64)
        valid = torch.ones(N, T, dtype=torch.bool, device=dev) if valid is None else valid.to(dev, torch.bool)
        sensor_id = sensor_id.to(dev, torch.int64)
        inf = torch.full((), torch.finfo(torch.float32).max, device=dev)

        # Device-relative fp32 ms, bit-identical to the tokenizer's t_ms; out-of-grid samples dropped.
        first_ns = torch.where(valid, time_ns, torch.iinfo(torch.int64).max).amin(dim=1, keepdim=True)
        t_rel = (time_ns - first_ns).float() * 1e-6
        valid = valid & (t_rel < self.max_timespan_ms)
        t_rel = torch.where(valid, t_rel, 0.0)
        vf = valid.float()

        # Level 1: samples -> micro-window bins (origin = the device's first sample, t=0).
        ratio1 = t_rel / cfg.micro_span_ms
        j = ratio1.floor()
        u1 = ratio1 - j
        j = j.long().clamp_(0, M - 1)
        count1 = torch.zeros(N, M, device=dev).scatter_add_(1, j, vf)
        micro_valid = count1 > 0
        # Window time = t0 + mean(t - t0), t0 = the bin's first (= earliest) sample, as in the reference.
        t0 = torch.full((N, M), 0.0, device=dev).fill_(inf).scatter_reduce_(1, j, torch.where(valid, t_rel, inf), "amin")
        t0 = torch.where(micro_valid, t0, 0.0)
        rel = torch.where(valid, t_rel - t0.gather(1, j), 0.0)
        t_win = t0 + torch.zeros(N, M, device=dev).scatter_add_(1, j, rel) / count1.clamp_min(1)

        # Level 2: micro-windows -> token bins (origin = the device's first non-empty window).
        b_start2 = torch.where(micro_valid, t_win, inf).amin(dim=1, keepdim=True)
        ratio2 = torch.where(micro_valid, (t_win - b_start2) / cfg.window_span_ms, 0.0)
        s = ratio2.floor()
        u2 = ratio2 - s
        s = s.long().clamp_(0, S - 1)
        token_valid = torch.zeros(N, S, device=dev).scatter_add_(1, s, micro_valid.float()) > 0

        A1 = _two_tap_matrix(valid, j, u1, M, m)                                                    # [N, T, M*m]
        A2 = _two_tap_matrix(micro_valid, s, u2, S, m)                                              # [N, M, S*m]

        # RoPE position = mean *sample* time per token (tokenizer: index_add over r2t / count).
        tok_of_samp = s.gather(1, j)
        tok_cnt = torch.zeros(N, S, device=dev).scatter_add_(1, tok_of_samp, vf)
        token_time = torch.zeros(N, S, device=dev).scatter_add_(1, tok_of_samp, t_rel) / tok_cnt.clamp_min(1)

        # Trunk-side tables. Row order: [cls, registers, dev0 tokens, dev1 tokens, ...].
        time_L = torch.cat([torch.zeros(1 + self.R, device=dev), token_time.reshape(-1)])          # cls/registers at t=0
        valid_L = torch.cat([torch.ones(1 + self.R, dtype=torch.bool, device=dev), token_valid.reshape(-1)])
        angle = time_L[:, None] / self._rope_periods
        allowed_full = valid_L[None, :].expand(L, L)
        allowed_dev = allowed_full & (self._run_id[:, None] == self._run_id[None, :])
        neg = torch.full((), MASK_NEG, device=dev)
        return Schedule(
            pool1=_row_normalise(A1.transpose(1, 2)).contiguous(),
            pool2=_row_normalise(A2.transpose(1, 2)).contiguous(),
            up2=A2.contiguous(), up1=A1.contiguous(),
            sensor_id=sensor_id.contiguous(),
            rope_cos=angle.cos().contiguous(), rope_sin=angle.sin().contiguous(),
            mask_dev=torch.where(allowed_dev, 0.0, neg).contiguous(),
            mask_full=torch.where(allowed_full, 0.0, neg).contiguous(),
            valid=valid, token_valid=token_valid, token_time=token_time,
        )

    def _engine_key(self) -> str:
        trt: tp.Any = importlib.import_module("tensorrt")   # typed Any: the pip package re-exports tensorrt_cu13 without stubs
        h = hashlib.sha256()
        h.update(f"{trt.__version__}|{self.precision}|{self.N}|{self.T}|{self.S}|{self.cfg}|{torch.cuda.get_device_name(self.device)}".encode())
        for k, v in sorted(self.core.state_dict().items()):
            h.update(k.encode()); h.update(v.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().tobytes())
        return h.hexdigest()[:20]

    def _dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        N, T, S, M, m, L, Dh = self.N, self.T, self.S, self.M, self.cfg.num_taps, self.L, self.cfg.head_dim // 2
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float32)
        return (z(N, T, 6), z(N, M * m, T), z(N, S * m, M), z(N, M, S * m), z(N, T, M * m),
                torch.zeros(N, dtype=torch.int64), z(L, Dh), z(L, Dh), z(L, L), z(L, L))

    def export_onnx(self, path: Path) -> Path:
        """Export the dense core, cast to ``self.precision``, to ONNX with static shapes."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        core = OursDenseCore(self.cfg, self.N, self.T, self.S).eval()
        core.load_state_dict(self.core.state_dict())
        core.to({"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.precision])
        with torch.no_grad():
            torch.onnx.export(core, self._dummy_inputs(), str(path), input_names=list(OursDenseCore.INPUTS),
                              output_names=list(OursDenseCore.OUTPUTS), dynamo=True, external_data=False)
        return path

    def build(self, force: bool = False) -> "FoundationIMU_TRT":
        """Build (or load from cache) the TensorRT engine for the loaded weights."""
        trt: tp.Any = importlib.import_module("tensorrt")   # typed Any: the pip package re-exports tensorrt_cu13 without stubs
        assert self._weights_loaded, "call from_torch_model() before build()"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        plan = self.cache_dir / f"ours_{self._engine_key()}.plan"
        logger = trt.Logger(trt.Logger.WARNING)
        if force or not plan.exists():
            onnx_path = self.export_onnx(plan.with_suffix(".onnx"))
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(onnx_path.read_bytes()):
                raise RuntimeError("ONNX parse failed:\n" + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            # TensorRT >= 11 is strongly typed: precision comes from the ONNX dtypes (export_onnx casts the core).
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(self.workspace_gb * (1 << 30)))
            blob = builder.build_serialized_network(network, config)
            if blob is None:
                raise RuntimeError("TensorRT engine build failed")
            plan.write_bytes(bytes(blob))
            onnx_path.unlink(missing_ok=True)
        runtime = trt.Runtime(logger)
        self._engine = runtime.deserialize_cuda_engine(plan.read_bytes())
        self._context = self._engine.create_execution_context()
        self._io.clear()
        self._outputs.clear()
        trt2torch = {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16,
                     trt.int64: torch.int64, trt.int32: torch.int32, trt.bool: torch.bool}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt2torch[self._engine.get_tensor_dtype(name)]
            is_input = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self._io[name] = (shape, dtype, is_input)
            if not is_input:
                self._outputs[name] = torch.empty(shape, dtype=dtype, device=self.device)
                self._context.set_tensor_address(name, self._outputs[name].data_ptr())
        self.plan_path = plan
        return self

    @torch.no_grad()
    def run_engine(self, x6: torch.Tensor, schedule: Schedule) -> dict[str, torch.Tensor]:
        """Raw engine call on the current torch CUDA stream; returns the engine's output buffers
        (overwritten by the next call -- clone if you keep them)."""
        assert self._context is not None, "engine not built; call from_torch_model()/build() first"
        feeds = {"x6": x6, **schedule.as_inputs()}
        keep = []   # hold the converted tensors alive until the launch has been enqueued
        for name, (shape, dtype, is_input) in self._io.items():
            if not is_input:
                continue
            t = feeds[name]
            assert tuple(t.shape) == shape, f"engine input {name}: expected {shape}, got {tuple(t.shape)}"
            t = t.to(self.device, dtype).contiguous()
            keep.append(t)
            self._context.set_tensor_address(name, t.data_ptr())
        ok = self._context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self._outputs

    @torch.no_grad()
    def forward(
        self,
        acc      : jt.Float[torch.Tensor, "N T 3"],
        gyro     : jt.Float[torch.Tensor, "N T 3"],
        time_ns  : jt.Int64[torch.Tensor, "N T"] | None = None,
        valid    : jt.Bool [torch.Tensor, "N T"] | None = None,
        sensor_id: jt.Int  [torch.Tensor, "N"]   | None = None,
        schedule : Schedule | None = None,
        backend  : tp.Literal["trt", "torch"] = "trt",
    ) -> FoundationIMU_TRT_Output:
        """Run the model on one dense stream. Pass either ``(time_ns, valid, sensor_id)`` or a
        precomputed ``schedule`` (fixed-rate streams). ``backend='torch'`` runs the same dense
        graph eagerly in fp32 (reference / debugging)."""
        if schedule is None:
            assert time_ns is not None and sensor_id is not None, "need (time_ns, sensor_id) or a schedule"
            schedule = self.schedule(time_ns, valid, sensor_id)
        x6 = torch.cat([acc, gyro], dim=-1).to(self.device, torch.float32)
        assert x6.shape == (self.N, self.T, 6), f"expected acc/gyro [{self.N}, {self.T}, 3], got {tuple(acc.shape)}"
        x6 = torch.where(schedule.valid[..., None], x6, torch.zeros_like(x6))
        if backend == "trt":
            out = self.run_engine(x6, schedule)
            correction, body_vel, lang, feat = (out[k] for k in OursDenseCore.OUTPUTS)
        else:
            core = self.core.to(self.device)
            correction, body_vel, lang, feat = core(x6, **schedule.as_inputs())
        keep = schedule.valid[..., None]
        correction = torch.where(keep, correction.float(), torch.zeros_like(correction, dtype=torch.float32))
        body_vel = torch.where(keep, body_vel.float(), torch.zeros_like(body_vel, dtype=torch.float32))
        return FoundationIMU_TRT_Output(
            dec_correct_acc=correction[..., :3].clone(), dec_correct_gyro=correction[..., 3:].clone(),
            dec_body_vel=body_vel.clone(), dec_lang_feature=lang.float().clone(), trunk_feature=feat.clone(),
            valid=schedule.valid, token_valid=schedule.token_valid,
        )

    __call__ = forward

    def __repr__(self) -> str:
        return (f"FoundationIMU_TRT(N={self.N}, T={self.T}, timespan={self.max_timespan_ms} ms -> S={self.S} tokens/device, "
                f"L={self.L} rows, precision={self.precision}, engine={'built' if self._engine else 'none'})")
