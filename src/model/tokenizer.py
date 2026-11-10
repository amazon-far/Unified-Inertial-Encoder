"""Tokenizers that turn jagged IMU batches into trunk tokens, with their decoder heads."""
import math
import torch
import jaxtyping as jt
from typing import Sequence
from torch.nn import Module
from torch.nn.parameter import Parameter as Parameter

from ..interface.data_imu import IMUData_BaseTrait_Jagged
from ..interface.model    import ITokenizer, FRAME_CLS, FRAME_REGISTER
from ..interface.human    import CANONICAL_NAMES

from .template.stride     import MLP, Conv1D
from .template.continuous import ContinuousEncoder, ContinuousSchedule


def _insert_cls_token(
    cls_embed: torch.Tensor,
    feature  : jt.Float[torch.Tensor, "pack C"],
    time_ms  : jt.Float[torch.Tensor, "pack"],
    mask_tok : jt.Bool [torch.Tensor, "pack"],
    frame    : jt.Int64[torch.Tensor, "pack"],
    off_b    : jt.Int64[torch.Tensor, "B+1"],
    off_n    : jt.Int64[torch.Tensor, "N+1"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepend one [cls] token to each sample's token run, treated as a separate device.

    [cls] joins the sample-wide runs of ``off_b`` but forms a single-token run in ``off_n``, so
    device-only attention never mixes it with real devices. It sits at t=0, at frame ``FRAME_CLS``,
    and is never masked.

    Returns the augmented ``(feature, time_ms, mask, frame, off_b, off_n)`` plus ``base_rows`` (the new
    row of every input token) and ``cls_rows`` (the new row of each sample's [cls]).
    """
    pack, B = feature.shape[0], off_b.numel() - 1
    device  = feature.device

    row         = torch.arange(pack, device=device)
    samp_of_row = torch.searchsorted(off_b[1:].contiguous(), row, right=True)
    base_rows   = row + samp_of_row + 1        # each token shifts by the [cls] of samples at-or-before it
    new_off_b   = off_b + torch.arange(B + 1, device=device)
    cls_rows    = new_off_b[:-1]               # [cls] opens its sample's run

    new_feature = feature.new_empty((pack + B, feature.shape[-1]))
    new_feature[base_rows] = feature
    new_feature[cls_rows]  = cls_embed.to(feature.dtype).reshape(1, -1).expand(B, -1)

    new_time = time_ms.new_zeros(pack + B)
    new_time[base_rows] = time_ms
    new_mask = torch.zeros(pack + B, dtype=torch.bool, device=device)
    new_mask[base_rows] = mask_tok
    new_frame = frame.new_full((pack + B,), FRAME_CLS)   # non-base rows are exactly the [cls] rows
    new_frame[base_rows] = frame

    # Old device boundaries shift by the [cls] runs at-or-before them (clamp: the trailing boundary
    # counts only the B real samples); each sample gains one boundary at its [cls] run.
    shift     = torch.searchsorted(off_b, off_n, right=True).clamp_max(B)
    new_off_n = torch.sort(torch.cat([off_n + shift, cls_rows])).values

    return new_feature, new_time, new_mask, new_frame, new_off_b, new_off_n, base_rows, cls_rows


class Conv1DTokenizer(ITokenizer):
    def __init__(
        self,
        output_dim  : int,
        token_size  : int,
        activation  : Conv1D.Support_Act,
        num_trunk_features: int = 1,
    ):
        super().__init__()
        self.output_dim   = output_dim
        # The decoder heads read the channel-cat of ``model.feature_from_trunk`` trunk block outputs.
        self.num_trunk_features = num_trunk_features
        feat_dim = output_dim * num_trunk_features
        # Leading singleton dims keep these at ndim != 2, so the hybrid optimizer routes them to AdamW.
        self.device_embed = torch.nn.Parameter(torch.zeros((1, 1, len(CANONICAL_NAMES), output_dim)))
        self.mask_embed   = torch.nn.Parameter(torch.zeros((1, 1, 1, output_dim)))
        self.cls_embed    = torch.nn.Parameter(torch.zeros((1, 1, 1, output_dim)))

        # Two non-overlapping conv layers fold token_size samples into one token, so it must be a perfect square.
        side = math.isqrt(token_size)
        assert side * side == token_size, f"token_size {token_size} must be a perfect square"
        kernels: list[tuple[int, int, int, Conv1D.Padding_Mode]] = [
            (side, side, 0, "zeros"), (side, side, 0, "zeros")
        ]
        self.patch_size = 1
        for _, stride, _, _ in kernels:
            self.patch_size *= stride

        self.imu_encoder = Conv1D(
            kernels     =kernels,
            dim_list    =[6, output_dim//2, output_dim],
            activation  =activation,
            transpose   =False
        )
        
        self.imu_correction_decoder = Conv1D(
            kernels     =kernels,
            dim_list    =[feat_dim, output_dim//2, 6],
            activation  =activation,
            transpose   =True
        )

        self.imu_velocity_decoder = Conv1D(
            kernels     =kernels,
            dim_list    =[feat_dim, output_dim//2, 3],
            activation  =activation,
            transpose   =True
        )

        lang_feat_dim = 768
        self.lang_decoder = MLP(
            dim_list    =[feat_dim, (feat_dim + lang_feat_dim) // 2, lang_feat_dim],
            activation  ="silu",
        )

        # Row maps cached from the latest forward: `_base_rows` / `_cls_rows` locate the base tokens and
        # [cls] rows in the cls-augmented pack, `_pack_rows` each real sample's slot in the run-padded layout.
        self._base_rows: torch.Tensor | None = None
        self._cls_rows : torch.Tensor | None = None
        self._pack_rows: torch.Tensor | None = None

    def forward(self, data: IMUData_BaseTrait_Jagged, mask: jt.Bool[torch.Tensor, "pack_BNT"]) -> ITokenizer.Token_Output:
        imu_data = torch.cat([data.acc, data.gyro], dim=-1)

        pack    = mask.shape[0]
        patch   = self.patch_size
        device  = imu_data.device

        # Device runs are jagged, so each is zero-padded to whole patches: every patch then lies inside one
        # device and the padded pack is encoded in one batched conv (kernel == stride, so patches are independent).
        offn      = data.offsets_N
        run_len   = offn[1:] - offn[:-1]
        run_tok   = (run_len + patch - 1) // patch
        tok_offsets_N     = torch.zeros_like(offn)
        tok_offsets_N[1:] = run_tok.cumsum(0)
        pad_off   = tok_offsets_N * patch                                       # [N+1] padded-row boundaries

        row        = torch.arange(pack, device=device)
        dev_of_row = torch.searchsorted(offn[1:].contiguous(), row, right=True)
        pack_rows  = row - offn[:-1][dev_of_row] + pad_off[:-1][dev_of_row]     # [pack] slot in padded layout
        n_padded   = int(pad_off[-1])
        n_patch    = n_padded // patch

        padded = imu_data.new_zeros((n_padded, imu_data.shape[-1]))
        padded[pack_rows] = imu_data
        patched  = padded.reshape(n_patch, patch, imu_data.shape[-1]).permute(0, 2, 1)  # [n_patch, c, patch]
        features = self.imu_encoder(patched).squeeze(-1)

        # A patch is masked if any of its real samples is (pad rows stay false).
        mask_pad = torch.zeros(n_padded, dtype=torch.bool, device=device)
        mask_pad[pack_rows] = mask
        mask_tok           = mask_pad.reshape(n_patch, patch).any(dim=1)
        features[mask_tok] = self.mask_embed.to(features.dtype).reshape(-1)

        # Patch time: mean device-relative time over the patch's REAL samples.
        time_ns  = data.time_ns[:, 0]                                           # int64 ns
        ref_ns   = time_ns[offn[:-1]][dev_of_row]
        dt_pad   = torch.zeros(n_padded, dtype=torch.float32, device=device)
        dt_pad[pack_rows] = (time_ns - ref_ns).float() * 1e-6
        cnt_pad  = torch.zeros(n_padded, dtype=torch.float32, device=device)
        cnt_pad[pack_rows] = 1.0
        dt_ms    = dt_pad.reshape(n_patch, patch).sum(dim=1) / cnt_pad.reshape(n_patch, patch).sum(dim=1)

        # A patch's first row is always a real sample.
        sensor_pad = torch.zeros(n_padded, dtype=torch.long, device=device)
        sensor_pad[pack_rows] = data.sensor_id[:, 0].long()
        features   = features + self.device_embed[0, 0][sensor_pad[::patch]]

        # Sample boundaries coincide with run boundaries, so map them through the token cumsum.
        tok_offsets_B = tok_offsets_N[torch.searchsorted(offn, data.offsets_B)]

        # Frame id = local token index within its device run.
        tok_row    = torch.arange(n_patch, device=device)
        dev_of_tok = torch.searchsorted(tok_offsets_N[1:].contiguous(), tok_row, right=True)
        frame_tok  = tok_row - tok_offsets_N[:-1][dev_of_tok]
        self._pack_rows = pack_rows

        features, dt_ms, mask_tok, frame_tok, tok_offsets_B, tok_offsets_N, self._base_rows, self._cls_rows = \
            _insert_cls_token(self.cls_embed, features, dt_ms, mask_tok, frame_tok, tok_offsets_B, tok_offsets_N)

        return ITokenizer.Token_Output(
            feature =features,
            time_ms =dt_ms,
            mask    =mask_tok,
            offset_b=tok_offsets_B,
            offset_n=tok_offsets_N,
            frame   =frame_tok,
        )

    def decode_imu_correction(
        self,
        feature       : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> tuple[jt.Float[torch.Tensor, "pack_BNT 3"], jt.Float[torch.Tensor, "pack_BNT 3"]]:
        # Strip [cls] rows (no raw samples behind them), then drop run-padded rows through `_pack_rows`.
        assert self._base_rows is not None and self._pack_rows is not None, "decode_imu_correction must follow a forward() call"
        correction = self.imu_correction_decoder(feature[self._base_rows].unsqueeze(-1)) # [n_patch, 6, patch]
        imu_result = correction.permute(0, 2, 1).reshape(-1, 6)[self._pack_rows]         # [pack, 6]

        corr_acc  = imu_result[:, :3]
        corr_gyro = imu_result[:, 3:]

        return corr_acc, corr_gyro

    def decode_body_velocity(
        self,
        feature       : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> jt.Float[torch.Tensor, "pack_BNT 3"]:
        # Strip [cls] rows, then drop run-padded rows through `_pack_rows`.
        assert self._base_rows is not None and self._pack_rows is not None, "decode_body_velocity must follow a forward() call"
        velocity = self.imu_velocity_decoder(feature[self._base_rows].unsqueeze(-1))
        velocity = velocity.permute(0, 2, 1).reshape(-1, 3)[self._pack_rows]
        return velocity

    def decode_language_feature(
        self,
        feature       : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> jt.Float[torch.Tensor, "B C_lang"]:
        # One row per sample, decoded from its [cls] row.
        assert self._cls_rows is not None, "decode_language_feature must follow a forward() call"
        return self.lang_decoder(feature[self._cls_rows])

    @property
    def imu_correction_modules(self) -> Sequence[Module]:
        return [self.imu_correction_decoder]

    @property
    def body_velocity_modules(self) -> Sequence[Module]:
        return [self.imu_velocity_decoder]

    @property
    def lang_feature_modules(self) -> Sequence[Module]:
        return [self.lang_decoder]

    @property
    def mask_tokens(self) -> Sequence[Parameter]:
        return [self.mask_embed]

    @property
    def device_tokens(self) -> Sequence[Parameter]:
        return [self.device_embed]

    @property
    def cls_token(self) -> Parameter:
        return self.cls_embed


class ContinuousKernel_Conv1DTokenizer(ITokenizer):
    """Arbitrary-rate analog of ``Conv1DTokenizer``, coalescing samples into tokens by time.

    A ``ContinuousEncoder`` pools each device's samples into ``window_span_ms/micro_split`` micro-windows,
    then into one token per ``window_span_ms``; windows never cross a device boundary.

    ``micro_split`` and ``num_taps`` jointly set basis occupancy: a tap that catches no sample pools to
    exactly 0, so an under-occupied basis stamps every token with the corpus's sample rate.
    """

    def __init__(
        self,
        output_dim    : int,
        window_span_ms: float,
        activation    : Conv1D.Support_Act,
        num_taps      : int,
        micro_split   : float = 8.,
        num_trunk_features: int = 1,
    ):
        super().__init__()
        self.output_dim     = output_dim
        self.window_span_ms = window_span_ms
        self.micro_split    = micro_split
        # Decoder-head input width for multi-layer trunk readout (see Conv1DTokenizer).
        self.num_trunk_features = num_trunk_features
        feat_dim = output_dim * num_trunk_features

        # ndim=3 keeps these off the Muon optimizer (which only works on ndim=2).
        self.device_embed = torch.nn.Parameter(torch.zeros((1, len(CANONICAL_NAMES), output_dim)))
        self.mask_embed   = torch.nn.Parameter(torch.zeros((1, 1                   , output_dim)))
        self.cls_embed    = torch.nn.Parameter(torch.zeros((1, 1                   , output_dim)))

        self.imu_encoder = ContinuousEncoder(
            dim_list     =[6, output_dim // 2, output_dim],
            timespan_ms  =[window_span_ms / micro_split, window_span_ms],
            activation   =activation,
            transpose    =False,
            num_taps     =num_taps
        )

        self.imu_correction_decoder = ContinuousEncoder(
            dim_list     =[feat_dim, output_dim // 2, 6],
            timespan_ms  =[window_span_ms, window_span_ms / micro_split],
            activation   =activation,
            transpose    =True,
            num_taps     =num_taps
        )

        self.imu_velocity_decoder = ContinuousEncoder(
            dim_list     =[feat_dim, output_dim // 2, 3],
            timespan_ms  =[window_span_ms, window_span_ms / micro_split],
            activation   =activation,
            transpose    =True,
            num_taps     =num_taps
        )

        lang_feat_dim = 768
        self.lang_decoder = MLP(
            dim_list    =[feat_dim, (feat_dim + lang_feat_dim) // 2, lang_feat_dim],
            activation  ="silu",
        )

        # Cached from the latest forward; the transpose decoders replay this schedule (the ITokenizer
        # decode signatures carry offsets but not times).
        self._t_ms     : torch.Tensor | None = None
        self._b_offset : torch.Tensor | None = None
        self._schedule : ContinuousSchedule | None = None
        self._r2t      : torch.Tensor | None = None
        # Base token / [cls] rows inside the cls-augmented pack (see `_insert_cls_token`).
        self._base_rows: torch.Tensor | None = None
        self._cls_rows : torch.Tensor | None = None

    def forward(
        self,
        data: IMUData_BaseTrait_Jagged,
        mask: jt.Bool[torch.Tensor, "pack_BNT"]
    ) -> ITokenizer.Token_Output:
        imu_data = torch.cat([data.acc, data.gyro], dim=-1)

        pack   = mask.shape[0]
        offn   = data.offsets_N
        device = imu_data.device

        # Per-sample time in ms relative to its device's first sample.
        time_ns    = data.time_ns[:, 0]
        row        = torch.arange(pack, device=device)
        dev_of_row = torch.searchsorted(offn[1:].contiguous(), row, right=True)
        ref_ns     = time_ns[offn[:-1]][dev_of_row]
        t_ms       = (time_ns - ref_ns).float() * 1e-6

        schedule, r2t  = self.imu_encoder.build_schedule(t_ms, offn)
        features, _     = self.imu_encoder(imu_data, t_ms, offn, schedule=schedule, index=r2t)
        
        self._t_ms, self._b_offset = t_ms, offn
        self._schedule, self._r2t  = schedule, r2t
        n_tok          = features.shape[0]

        # All samples in a token share one device, so last-write-wins is exact. Scatters stay fp:
        # features may be a narrower autocast dtype.
        mask_tok          = t_ms.new_zeros(n_tok).index_add_(0, r2t, mask.to(t_ms.dtype)) > 0
        dev_of_tok        = dev_of_row.new_zeros(n_tok); dev_of_tok[r2t] = dev_of_row
        sensor_of_tok     = torch.zeros(n_tok, dtype=torch.long, device=device)
        sensor_of_tok[r2t] = data.sensor_id[:, 0].long()
        cnt_tok           = t_ms.new_zeros(n_tok).index_add_(0, r2t, torch.ones_like(t_ms))
        time_tok          = t_ms.new_zeros(n_tok).index_add_(0, r2t, t_ms) / cnt_tok.clamp_min(1)

        # Tokens are ordered by device, so offset_n is the per-device token count.
        N             = offn.numel() - 1
        tok_per_dev   = torch.bincount(dev_of_tok, minlength=N)
        tok_offsets_N = torch.cat([offn.new_zeros(1), torch.cumsum(tok_per_dev, 0)])
        dev_at_b      = torch.searchsorted(offn, data.offsets_B)
        tok_offsets_B = tok_offsets_N[dev_at_b]

        # Replace masked tokens, then add the device embedding.
        features[mask_tok] = self.mask_embed.to(features.dtype).reshape(-1)
        features           = features + self.device_embed[0][sensor_of_tok]

        # Frame id = absolute time-bin index, NOT the run-local index: the scheduler drops empty bins, so
        # after a dropout gap a run-local index would sit one frame early, a future leak under frame-causal
        # attention. A token's mean sample time always lies inside its own bin.
        frame_tok = (time_tok / self.window_span_ms).floor().long()

        # The transpose decoders replay the schedule over base tokens only; decoders strip [cls] via `_base_rows`.
        features, time_tok, mask_tok, frame_tok, tok_offsets_B, tok_offsets_N, self._base_rows, self._cls_rows = \
            _insert_cls_token(self.cls_embed, features, time_tok, mask_tok, frame_tok, tok_offsets_B, tok_offsets_N)

        return ITokenizer.Token_Output(
            feature =features,
            time_ms =time_tok,
            mask    =mask_tok,
            offset_b=tok_offsets_B,
            offset_n=tok_offsets_N,
            frame   =frame_tok,
        )

    def decode_imu_correction(
        self,
        feature       : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> tuple[jt.Float[torch.Tensor, "pack_BNT 3"], jt.Float[torch.Tensor, "pack_BNT 3"]]:
        # Strip [cls] rows: the schedule is built over base tokens only.
        assert self._t_ms is not None and self._b_offset is not None and self._schedule is not None \
            and self._r2t is not None and self._base_rows is not None, \
            "decode_imu_correction must follow a forward() call"
        feature         = feature[self._base_rows]
        expected_tokens = self._schedule[-1][0].numel() - 1
        assert feature.shape[0] == expected_tokens, \
            f"decoder feature rows ({feature.shape[0]}) != schedule token rows ({expected_tokens})"
        out, _ = self.imu_correction_decoder(
            feature, self._t_ms, self._b_offset, schedule=self._schedule, index=self._r2t
        )
        return out[:, :3], out[:, 3:]

    def decode_body_velocity(
        self,
        feature       : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> jt.Float[torch.Tensor, "pack_BNT 3"]:
        assert self._t_ms is not None and self._b_offset is not None and self._schedule is not None \
            and self._r2t is not None and self._base_rows is not None, \
            "decode_body_velocity must follow a forward() call"
        feature         = feature[self._base_rows]
        expected_tokens = self._schedule[-1][0].numel() - 1
        assert feature.shape[0] == expected_tokens, \
            f"decoder feature rows ({feature.shape[0]}) != schedule token rows ({expected_tokens})"
        out, _ = self.imu_velocity_decoder(
            feature, self._t_ms, self._b_offset, schedule=self._schedule, index=self._r2t
        )
        return out

    def decode_language_feature(
        self,
        feature       : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> jt.Float[torch.Tensor, "B C_lang"]:
        # One row per sample, decoded from its [cls] row.
        assert self._cls_rows is not None, "decode_language_feature must follow a forward() call"
        return self.lang_decoder(feature[self._cls_rows])

    @property
    def imu_correction_modules(self) -> Sequence[Module]:
        return [self.imu_correction_decoder]

    @property
    def body_velocity_modules(self) -> Sequence[Module]:
        return [self.imu_velocity_decoder]

    @property
    def lang_feature_modules(self) -> Sequence[Module]:
        return [self.lang_decoder]

    @property
    def mask_tokens(self) -> Sequence[Parameter]: return [self.mask_embed]

    @property
    def device_tokens(self) -> Sequence[Parameter]: return [self.device_embed]

    @property
    def cls_token(self) -> Parameter: return self.cls_embed


class AddRegister(ITokenizer):
    """Given a {base_tokenizer}, add register tokens on top of it."""
    
    def __init__(
        self,
        base_tokenizer  : ITokenizer,
        output_dim      : int,
        num_register    : int,
    ):
        super().__init__()
        self.base_tokenizer = base_tokenizer
        self.num_register   = num_register

        # ndim=3 keeps them off Muon. Random init: all registers share one RoPE time, so identical values
        # would receive identical gradients and never differentiate.
        self.reg_embed = torch.nn.Parameter(torch.randn((1, num_register, output_dim)) * 0.02)

        # Base token rows inside the register-augmented pack, cached from the latest forward.
        self._base_rows: torch.Tensor | None = None

    def forward(
        self,
        data: IMUData_BaseTrait_Jagged,
        mask: jt.Bool[torch.Tensor, "pack_BNT"]
    ) -> ITokenizer.Token_Output:
        base   = self.base_tokenizer(data, mask)
        feat   = base.feature
        off_b  = base.offset_b
        B      = off_b.numel() - 1
        R      = self.num_register
        pack   = feat.shape[0]
        device = feat.device

        # Registers join each sample as one extra "device" prepended to its token run: inside offset_b's
        # sample-wide runs but their own run in offset_n. Prepending keeps them visible under causal attention.
        row         = torch.arange(pack, device=device)
        samp_of_row = torch.searchsorted(off_b[1:].contiguous(), row, right=True)
        base_rows   = row + (samp_of_row + 1) * R

        new_pack = pack + B * R
        is_reg   = torch.ones(new_pack, dtype=torch.bool, device=device)
        is_reg[base_rows] = False

        feature = feat.new_empty((new_pack, feat.shape[-1]))
        feature[base_rows] = feat
        # Register rows come sample-major, matching expand(B, R, C) row order.
        feature[is_reg]    = self.reg_embed[0].to(feat.dtype).expand(B, R, -1).reshape(B * R, -1)

        time_ms            = base.time_ms.new_zeros(new_pack)   # registers sit at t=0 (device-relative ms)
        time_ms[base_rows] = base.time_ms
        mask_tok           = torch.zeros(new_pack, dtype=torch.bool, device=device) # registers are never masked
        mask_tok[base_rows] = base.mask
        # FRAME_REGISTER: attended by every token under frame-causal attention, while they see only each other.
        frame            = base.frame.new_full((new_pack,), FRAME_REGISTER)
        frame[base_rows] = base.frame

        samp_id  = torch.arange(B + 1, device=device)
        offset_b = off_b + samp_id * R
        # Boundaries shift as in `_insert_cls_token`, with R-row register runs.
        shift    = torch.searchsorted(off_b, base.offset_n, right=True).clamp_max(B)
        offset_n = torch.sort(torch.cat([
            base.offset_n + shift * R,
            off_b[:-1] + samp_id[:-1] * R,
        ])).values

        self._base_rows = base_rows

        return ITokenizer.Token_Output(
            feature =feature,
            time_ms =time_ms,
            mask    =mask_tok,
            offset_b=offset_b,
            offset_n=offset_n,
            frame   =frame,
        )

    def decode_imu_correction(
        self,
        feature         : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> tuple[jt.Float[torch.Tensor, "pack_BNT 3"], jt.Float[torch.Tensor, "pack_BNT 3"]]:
        # Strip register rows so the base decoder sees exactly one row per base token.
        assert self._base_rows is not None, "decode_imu_correction must follow a forward() call"
        return self.base_tokenizer.decode_imu_correction(
            feature[self._base_rows]
        )

    def decode_body_velocity(
        self,
        feature         : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> jt.Float[torch.Tensor, "pack_BNT 3"]:
        assert self._base_rows is not None, "decode_body_velocity must follow a forward() call"
        return self.base_tokenizer.decode_body_velocity(feature[self._base_rows])

    def decode_language_feature(
        self,
        feature         : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> jt.Float[torch.Tensor, "B C_lang"]:
        # The base tokenizer's cached [cls] rows index its own register-free pack.
        assert self._base_rows is not None, "decode_language_feature must follow a forward() call"
        return self.base_tokenizer.decode_language_feature(feature[self._base_rows])

    # Decoder heads and special tokens live in the wrapped base tokenizer.
    @property
    def num_trunk_features(self) -> int:
        return getattr(self.base_tokenizer, "num_trunk_features", 1)

    @property
    def imu_correction_modules(self) -> Sequence[Module]:
        return self.base_tokenizer.imu_correction_modules

    @property
    def body_velocity_modules(self) -> Sequence[Module]:
        return self.base_tokenizer.body_velocity_modules

    @property
    def lang_feature_modules(self) -> Sequence[Module]:
        return self.base_tokenizer.lang_feature_modules

    @property
    def mask_tokens(self) -> Sequence[Parameter]:
        return self.base_tokenizer.mask_tokens

    @property
    def device_tokens(self) -> Sequence[Parameter]:
        return self.base_tokenizer.device_tokens

    @property
    def cls_token(self) -> Parameter:
        return self.base_tokenizer.cls_token
