"""Model interfaces: the FoundationIMU output and the tokenizer and loss contracts."""
import torch
import jaxtyping as jt

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from .base import Base
from .data_imu import IMUData_BaseTrait_Jagged, IMUData_w_GT_Jagged, Loss_Flags


if TYPE_CHECKING:
    from ..model.model import FoundationIMU


# Frame ids drive the frame-block-causal mask: a token attends another iff same sample and
# frame_kv <= frame_q. Equal frame must mean the same wall-clock window on every device of
# the sample, monotone in time. Register tokens (FRAME_REGISTER) are attended by every token
# but see only each other; [cls] (FRAME_CLS) reads the whole sample and no real token reads it.
FRAME_REGISTER = -1
FRAME_CLS      = 2**62   # large finite sentinel; iinfo.max is avoided so frame arithmetic can't overflow


@dataclass
class FoundationIMU_Output:
    trunk_feature : jt.Float[torch.Tensor, "pack_BNS C"]
    mask_feature  : jt.Bool [torch.Tensor, "pack_BNS"]
    offsets_B_feat: jt.Int64[torch.Tensor, "B+1"]          # Start row of each sample.
    offsets_N_feat: jt.Int64[torch.Tensor, "N+1"]          # Start row of each device.

    dec_correct_acc : jt.Float[torch.Tensor, "pack_BNT 3"]
    dec_correct_gyro: jt.Float[torch.Tensor, "pack_BNT 3"]
    dec_body_vel    : jt.Float[torch.Tensor, "pack_BNT 3"]
    dec_lang_feature: jt.Float[torch.Tensor, "B T_lang C_lang"]  # T_lang is constantly 1 during training; fold_time_batch folds B into it.
    lang_time_ns    : jt.Int64[torch.Tensor, "B T_lang"]        # Beginning-of-window timestamp of each lang feature.

    time_ns         : jt.Int64[torch.Tensor, "pack_BNT 1"]
    mask_samp       : jt.Bool [torch.Tensor, "pack_BNT"]
    offsets_B_samp  : jt.Int64[torch.Tensor, "B+1"]          # Start row of each sample.
    offsets_N_samp  : jt.Int64[torch.Tensor, "N+1"]          # Start row of each device.

    def fold_time_batch(self) -> FoundationIMU_Output:
        """Fold a batch of adjacent non-overlapping windows of one sequence into a single sample.

        Only valid when the batch really is one time-batched sequence.
        """
        def fold(offsets_B: torch.Tensor, offsets_N: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Reorder runs from window-major to device-major. Every window must carry the same runs
            # in the same order ([cls]/registers included; they fold like devices).
            B, R = offsets_B.numel() - 1, offsets_N.numel() - 1
            assert R % B == 0, f"cannot fold: {R} device runs do not divide evenly over {B} windows"
            lens  = offsets_N.diff()
            order = torch.arange(R, device=lens.device).reshape(B, R // B).T.flatten()

            starts, lens = offsets_N[:-1][order], lens[order]
            new_off = torch.cat([lens.new_zeros(1), lens.cumsum(0)])
            # row -> source row: each reordered run copies from its old start.
            perm = torch.arange(int(offsets_N[-1]), device=lens.device) \
                 - torch.repeat_interleave(new_off[:-1], lens) + torch.repeat_interleave(starts, lens)

            offsets_N_fold = torch.cat([lens.new_zeros(1), lens.reshape(R // B, B).sum(dim=1).cumsum(0)])
            offsets_B_fold = offsets_N_fold[[0, -1]]
            return perm, offsets_B_fold, offsets_N_fold

        perm_feat, offB_feat, offN_feat = fold(self.offsets_B_feat, self.offsets_N_feat)
        perm_samp, offB_samp, offN_samp = fold(self.offsets_B_samp, self.offsets_N_samp)

        return FoundationIMU_Output(
            trunk_feature    = self.trunk_feature[perm_feat],
            mask_feature     = self.mask_feature[perm_feat],
            offsets_B_feat   = offB_feat,
            offsets_N_feat   = offN_feat,
            dec_correct_acc  = self.dec_correct_acc[perm_samp],
            dec_correct_gyro = self.dec_correct_gyro[perm_samp],
            dec_body_vel     = self.dec_body_vel[perm_samp],
            # Windows are adjacent in batch order, so batch-major flatten IS time order.
            dec_lang_feature = self.dec_lang_feature.reshape(1, -1, self.dec_lang_feature.shape[-1]),
            lang_time_ns     = self.lang_time_ns.reshape(1, -1),
            time_ns          = self.time_ns[perm_samp],
            mask_samp        = self.mask_samp[perm_samp],
            offsets_B_samp   = offB_samp,
            offsets_N_samp   = offN_samp,
        )

class ITokenizer(Base, torch.nn.Module):
    @dataclass
    class Token_Output:
        feature : jt.Float[torch.Tensor, "pack_BNS C" ]
        time_ms : jt.Float[torch.Tensor, "pack_BNS"   ]
        mask    : jt.Bool [torch.Tensor, "pack_BNS"]
        offset_b: jt.Int64[torch.Tensor, "B+1"]
        offset_n: jt.Int64[torch.Tensor, "N+1"]
        frame   : jt.Int64[torch.Tensor, "pack_BNS"]   # see FRAME_REGISTER / FRAME_CLS above
    
    def forward(
        self,
        data: IMUData_BaseTrait_Jagged,
        mask: jt.Bool[torch.Tensor, "pack_BNT"]
    ) -> Token_Output:
        ...

    def decode_imu_correction(
        self,
        feature       : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> tuple[jt.Float[torch.Tensor, "pack_BNT 3"], jt.Float[torch.Tensor, "pack_BNT 3"]]:
        ...

    def decode_body_velocity(
        self,
        feature       : jt.Float[torch.Tensor, "pack_BNS C"],
    ) -> jt.Float[torch.Tensor, "pack_BNT 3"]:
        ...

    def decode_language_feature(
        self, 
        feature       : jt.Float[torch.Tensor, "pack_BNS C"]
    ) -> jt.Float[torch.Tensor, "B C_lang"]:
        ...

    @property
    def imu_correction_modules(self) -> Sequence[torch.nn.Module]: ...
    @property
    def body_velocity_modules(self) -> Sequence[torch.nn.Module]: ...
    @property
    def lang_feature_modules(self) -> Sequence[torch.nn.Module]: ...
    @property
    def mask_tokens(self) -> Sequence[torch.nn.Parameter]: ...
    @property
    def device_tokens(self) -> Sequence[torch.nn.Parameter]: ...
    @property
    def cls_token(self) -> torch.nn.Parameter: ...

    if TYPE_CHECKING:
        __call__ = forward  # so that pyright knows __call__ have same signature as forward.


class ILossFunc(Base, torch.nn.Module):
    """Compute ``(loss, metric_dict)`` from a jagged batch and the FoundationIMU output."""
    
    LABEL = Loss_Flags.EMPTY
    
    def forward(self, model: FoundationIMU, data: IMUData_w_GT_Jagged, pred: FoundationIMU_Output) -> tuple[torch.Tensor, dict[str, float]]:
        ...

    if TYPE_CHECKING: __call__ = forward
