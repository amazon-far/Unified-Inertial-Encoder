"""FoundationIMU network: tokenizer, adaptor, trunk and decoder heads."""
import torch
import typing as tp
import torch.nn as nn
import jaxtyping as jt

from ..interface.model    import ITokenizer, FoundationIMU_Output
from ..interface.data_imu import IMUData_BaseTrait_Jagged, TokenUnit

from .transformer     import Transformer
from .template.stride import GradientProbe


class FoundationIMU(nn.Module):
    def __init__(
        self,
        tokenizer   : ITokenizer,
        adaptor     : nn.Module,
        trunk       : Transformer,
        token_dim   : int,
        token_size  : float,
        token_unit  : TokenUnit,
        is_causal   : bool,
        trainable   : tp.Literal["all"] | tp.Sequence[tp.Literal["correction", "vel", "adaptor"]],
        feature_from_trunk: tp.Sequence[int] = (5,),
    ):
        super().__init__()
        self.trainable  = trainable

        # Decoders consume the channel-concatenation of these trunk block outputs, in index order.
        self.feature_from_trunk = list(feature_from_trunk)
        assert self.feature_from_trunk == sorted(set(self.feature_from_trunk)), \
            f"feature_from_trunk must be strictly increasing, got {feature_from_trunk}"
        assert all(0 <= i < len(trunk.blocks) for i in self.feature_from_trunk), \
            f"feature_from_trunk {feature_from_trunk} out of range for a {len(trunk.blocks)}-block trunk"
        n_feats = getattr(tokenizer, "num_trunk_features", 1)
        assert n_feats == len(self.feature_from_trunk), \
            f"tokenizer was built for {n_feats} trunk feature(s) but feature_from_trunk selects {len(self.feature_from_trunk)}"

        assert trunk.is_causal == is_causal, \
            f"model.is_causal={is_causal} but trunk.is_causal={trunk.is_causal}; they must agree"
        assert getattr(adaptor, "is_causal", is_causal) == is_causal, \
            f"model.is_causal={is_causal} but adaptor.is_causal={adaptor.is_causal}; they must agree"
        self.is_causal  = is_causal

        self.tok_config = (token_dim, token_size, token_unit)
        self.tokenizer  = tokenizer
        self.adaptor    = adaptor
        self.trunk      = trunk
        
        self.grad_probe = GradientProbe(torch.Size((token_dim * len(self.feature_from_trunk),)))

    def configure_trainable_params(self) -> None:
        """Freeze the whole network except the heads named in ``self.trainable`` (``"all"`` trains everything)."""
        if self.trainable == "all":
            for p in self.parameters():
                p.requires_grad_(True)
            return

        head_modules: dict[str, tp.Sequence[nn.Module]] = {
            "correction": self.tokenizer.imu_correction_modules,
            "vel"       : self.tokenizer.body_velocity_modules,
            "adaptor"   : [self.adaptor],
        }
        unknown = set(self.trainable) - head_modules.keys()
        assert not unknown, f"unknown trainable head(s) {sorted(unknown)}; expected subset of {sorted(head_modules)}"

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        for name in self.trainable:
            for module in head_modules[name]:
                module.train()
                for p in module.parameters():
                    p.requires_grad_(True)

    def forward(self, x: IMUData_BaseTrait_Jagged, mask: jt.Bool[torch.Tensor, "pack_BNT"]):
        tokens: ITokenizer.Token_Output = self.tokenizer(x, mask)

        adapted = self.adaptor(tokens.feature, tokens.offset_b, tokens.offset_n, tokens.time_ms, tokens.frame)

        # ``feature`` is the channel-cat of the ``feature_from_trunk`` block outputs.
        feature = self.trunk(adapted, tokens.offset_b, tokens.offset_n, tokens.time_ms, tokens.frame,
                             feature_from=self.feature_from_trunk)
        feature = self.grad_probe(feature) # grad-norm probe (numerically identity)

        task_feature = self.canonicalize_for_task(feature, chunk=self.tok_config[0])

        corr_acc, corr_gyro = self.tokenizer.decode_imu_correction(
            task_feature
        )

        velocity = self.tokenizer.decode_body_velocity(
            task_feature
        )

        language = self.tokenizer.decode_language_feature(
            task_feature
        )

        return FoundationIMU_Output(
            trunk_feature   = feature,
            mask_feature    = tokens.mask,

            mask_samp           = mask,
            time_ns             = x.time_ns,
            dec_correct_acc     = corr_acc,
            dec_correct_gyro    = corr_gyro,
            dec_body_vel        = velocity,
            
            # One [cls] summary per window stamped with the window-start time
            dec_lang_feature    = language.unsqueeze(1),
            lang_time_ns        = x.time_ns[x.offsets_B[:-1]],

            offsets_B_feat  = tokens.offset_b,
            offsets_N_feat  = tokens.offset_n,
            offsets_B_samp  = x.offsets_B,
            offsets_N_samp  = x.offsets_N,
        )

    @staticmethod
    def canonicalize_for_task(x: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        # Each ``chunk``-wide slice (one layer's channels) normalizes independently.
        chunk = x.shape[-1] if chunk is None else chunk
        return torch.nn.functional.layer_norm(
            x.reshape(*x.shape[:-1], -1, chunk),
            normalized_shape=(chunk,),
            weight=None,
            bias=None,
            eps=1e-6,
        ).reshape(x.shape)
