"""Frozen CLIP text encoder with multi-GPU data-parallel inference."""
from __future__ import annotations

import copy

import torch
import typing as tp
import jaxtyping as jt

if tp.TYPE_CHECKING:
    import clip.model

_CLIP_MODEL_ = "ViT-L/14@336px"
_FEAT_DIM_   = 768


class _CLIP_TextEncoder(torch.nn.Module):
    """The text tower of a CLIP model, extracted so replicas don't carry the unused vision tower."""
    def __init__(self, master: clip.model.CLIP):
        super().__init__()
        self.token_embedding      = master.token_embedding
        self.positional_embedding = master.positional_embedding
        self.transformer          = master.transformer
        self.ln_final             = master.ln_final
        self.text_projection      = master.text_projection

    @property
    def dtype(self) -> torch.dtype:
        return self.text_projection.dtype

    def forward(self, tokens: jt.Int[torch.Tensor, "B L"]) -> jt.Float[torch.Tensor, "B C_lang"]:
        # Mirrors clip.model.CLIP.encode_text on the extracted submodules.
        x = self.token_embedding(tokens).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)      # BLD -> LBD
        x = self.transformer(x)
        x = x.permute(1, 0, 2)      # LBD -> BLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the EOT embedding (the highest token id in each sequence).
        return x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection


class CLIP_Model(torch.nn.Module):
    """Frozen CLIP text encoder with one replica per visible CUDA device (fp16; fp32 on CPU).

    Replicas stay out of the ``nn.Module`` registry, so they never leak into ``parameters()``,
    optimizers or checkpoints of a composing module.
    """
    def __init__(self):
        super().__init__()
        import clip
        import clip.model

        master, _ = clip.load(_CLIP_MODEL_, device="cpu")
        master.eval().requires_grad_(False)
        text_encoder = _CLIP_TextEncoder(master)

        if torch.cuda.is_available():
            self._devices  = [torch.device("cuda", idx) for idx in range(torch.cuda.device_count())]
            clip.model.convert_weights(text_encoder)    # fp16, matching clip.load on CUDA
            self._replicas = [copy.deepcopy(text_encoder).to(device) for device in self._devices]
        else:
            self._devices  = [torch.device("cpu")]
            self._replicas = [text_encoder]

    @property
    def feature_dim(self) -> int:
        return _FEAT_DIM_

    @property
    def model_name(self) -> str:
        return _CLIP_MODEL_

    @torch.no_grad()
    def forward(self, texts: tp.Sequence[str]) -> jt.Float[torch.Tensor, "B C_lang"]:
        """Encode `texts` into fp32 L2-normalized embeddings on the primary device, rows aligned with
        `texts`; texts over CLIP's 77-token context are truncated.
        """
        import clip

        if len(texts) == 0:
            return torch.empty(0, _FEAT_DIM_, device=self._devices[0])

        tokens = clip.tokenize(list(texts), truncate=True)
        if self._devices[0].type == "cuda":
            tokens = tokens.pin_memory()

        chunks  = [chunk for chunk in tokens.tensor_split(len(self._replicas)) if len(chunk) > 0]
        inputs  = [(chunk.to(device, non_blocking=True),) for chunk, device in zip(chunks, self._devices)]
        modules = self._replicas[:len(inputs)]

        if len(modules) == 1:
            outputs = [modules[0](*inputs[0])]
        else:
            outputs = torch.nn.parallel.parallel_apply(modules, inputs, devices=self._devices[:len(inputs)])

        features = torch.cat([output.to(self._devices[0]) for output in outputs], dim=0)
        return (features / features.norm(dim=-1, keepdim=True)).float()
