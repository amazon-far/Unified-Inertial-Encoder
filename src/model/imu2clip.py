"""Adapter around the IMU2CLIP encoder vendored at ``thirdparty/imu2clip``: a pelvis-only IMU encoder
aligned to the cached CLIP ViT-L/14@336px language features with a symmetric InfoNCE loss."""

import sys
import math
import importlib.util

from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import jaxtyping as jt

import typing as tp

from typing import TYPE_CHECKING

from ..interface.data_imu import IMUData_w_GT_Lang_Jagged, Loss_Flags
from ..interface.human import CANONICAL_IDX

# Fixed by design: this baseline reads a single body site.
PELVIS_SENSOR_ID = CANONICAL_IDX["pelvis"]


def _load_mw2_encoder() -> type[nn.Module]:
    """Load ``MW2StackRNNPooling`` by file location: ``thirdparty/imu2clip`` on ``sys.path`` would
    collide on its generic ``lib`` package name."""
    mod_name = "_imu2clip_imu_models"
    if mod_name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "thirdparty/imu2clip/lib/imu_models.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec is not None and spec.loader is not None, f"cannot load IMU2CLIP encoder from {path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[mod_name] = module
    return sys.modules[mod_name].MW2StackRNNPooling


@dataclass
class IMU2CLIP_Output:
    embed    : jt.Float[torch.Tensor, "B C_lang"]   # per-window IMU embedding, sensor-run mean-pooled (pre-normalize)
    win_count: jt.Int64[torch.Tensor, "B"]          # sensor runs pooled into each window (>=1)


class IMU2CLIP(nn.Module):
    def __init__(
        self,
        size_embeddings: int = 768,        # CLIP ViT-L/14@336px width
        target_len     : int = 1000,       # samples each run is resampled to
        input_dim      : int = 32,         # MW2StackRNNPooling conv width
        chunk          : int = 4096,       # max runs encoded per forward pass (a whole take is many windows)
    ):
        super().__init__()
        self.size_embeddings = int(size_embeddings)
        self.target_len      = int(target_len)
        self.chunk           = int(chunk)

        MW2StackRNNPooling = _load_mw2_encoder()
        # Annotated nn.Module: otherwise pyright widens the dynamically loaded class to ``Tensor | Module``.
        self.encoder: nn.Module = MW2StackRNNPooling(input_dim=input_dim, size_embeddings=self.size_embeddings)

    @property
    def compute_dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    def configure_trainable_params(self) -> None:
        """Everything always trains; exists only for parity with ``FoundationIMU``/``TLIO``."""

    def _resample_runs(
        self,
        imu6  : jt.Float[torch.Tensor, "pack 6"],
        starts: jt.Int64[torch.Tensor, "N"],
        lengths: jt.Int64[torch.Tensor, "N"],
    ) -> jt.Float[torch.Tensor, "N 6 L"]:
        """Linearly resample every device run to ``target_len`` samples, batched by source length."""
        n, L = lengths.numel(), self.target_len
        out  = imu6.new_empty(n, 6, L)
        for src in torch.unique(lengths).tolist():
            sel = (lengths == src).nonzero(as_tuple=False).flatten()
            idx = starts[sel][:, None] + torch.arange(src, device=imu6.device)  # [k, src] into the pack
            seg = imu6[idx].transpose(1, 2)                                   # [k, 6, src]
            out[sel] = seg if src == L else F.interpolate(seg, size=L, mode="linear", align_corners=False)
        return out

    def forward(self, batch: IMUData_w_GT_Lang_Jagged) -> IMU2CLIP_Output:
        device    = batch.acc.device
        offsets_N = batch.offsets_N
        offsets_B = batch.offsets_B
        starts    = offsets_N[:-1]
        lengths   = offsets_N.diff()
        B         = offsets_B.numel() - 1

        # Keep only the pelvis run of each window; sensor_id is constant within a run.
        keep = batch.sensor_id[starts, 0] == PELVIS_SENSOR_ID
        starts, lengths = starts[keep], lengths[keep]
        win_of_run = torch.searchsorted(offsets_B, starts, right=True) - 1    # [N] window each run belongs to
        count = torch.zeros(B, device=device, dtype=torch.int64).index_add_(
            0, win_of_run, torch.ones_like(win_of_run)
        )
        assert bool((count > 0).all()), (
            f"{int((count == 0).sum())} window(s) carry no pelvis run; the IMU2CLIP baseline is "
            f"pelvis-only and needs a pelvis IMU (canonical id {PELVIS_SENSOR_ID}) in every window"
        )

        # Body-frame signal in IMU2CLIP's [acc; gyro] order.
        imu6 = torch.cat([batch.acc, batch.gyro], dim=-1).float()
        x    = self._resample_runs(imu6, starts, lengths)                     # [N, 6, L]

        # Cast past nn.Module.__getattr__'s ``Tensor | Module`` widening so the call type-checks.
        encoder      = tp.cast(nn.Module, self.encoder)
        param_device = next(encoder.parameters()).device
        embs = []
        for i in range(0, x.shape[0], self.chunk):
            seg = encoder(x[i : i + self.chunk].to(param_device, self.compute_dtype))
            embs.append(seg.float().to(device))
        run_emb = torch.cat(embs, dim=0) if embs else x.new_zeros(0, self.size_embeddings, device=device)

        # Mean over the rare multi-run case; normally one run per window.
        emb_sum = run_emb.new_zeros(B, self.size_embeddings).index_add_(0, win_of_run, run_emb)
        embed   = emb_sum / count.clamp(min=1)[:, None].to(emb_sum.dtype)
        return IMU2CLIP_Output(embed=embed, win_count=count)


class IMU2CLIP_Loss(nn.Module):
    """Symmetric InfoNCE between the IMU embedding and its cached CLIP text embedding, with a learned
    temperature. Off-diagonal pairs sharing a ``lang_feat_idx`` have identical targets, so they are
    masked out of both softmaxes rather than treated as negatives.
    """

    def __init__(self, init_temperature: float = 0.07, max_scale: float = 100.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.max_scale   = float(max_scale)

    def forward(
        self,
        model: IMU2CLIP,
        data : IMUData_w_GT_Lang_Jagged,
        pred : IMU2CLIP_Output,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert isinstance(data, IMUData_w_GT_Lang_Jagged), \
            f"IMU2CLIP_Loss requires an IMUData_w_GT_Lang_Jagged batch, got {type(data).__name__}"

        emb    = pred.embed
        device = emb.device

        # Per-window opt-in, as in Lang_CLIP_Loss.
        first_dev = torch.searchsorted(data.offsets_N, data.offsets_B[:-1])
        opted_in  = (data.labels[first_dev].to(torch.int64) & int(Loss_Flags.Lang_CLIP)) != 0
        valid     = opted_in & (data.lang_feat_idx >= 0)
        if not bool(valid.any()):
            # Graph-connected zero so the encoder still gets zero grads under DDP.
            return emb.float().sum() * 0.0, {}

        idx  = data.lang_feat_idx[valid]
        p    = F.normalize(emb[valid].float(), dim=-1, eps=1e-6)
        t    = F.normalize(data.lang_feature[idx].to(device).float(), dim=-1, eps=1e-6)

        scale  = self.logit_scale.exp().clamp(max=self.max_scale)
        logits = scale * (p @ t.T)

        # Mask off-diagonal same-phrase pairs (false negatives).
        v      = idx.numel()
        eye    = torch.eye(v, dtype=torch.bool, device=device)
        same   = (idx[:, None] == idx[None, :]) & ~eye
        logits = logits.masked_fill(same, float("-inf"))
        labels = torch.arange(v, device=device)

        loss_i = F.cross_entropy(logits, labels)                              # IMU -> text
        loss_t = F.cross_entropy(logits.T, labels)                           # text -> IMU
        loss   = loss_i + loss_t

        with torch.no_grad():
            cos      = (p * t).sum(dim=-1).mean()
            pred_std = emb[valid].float().std(dim=0, unbiased=False).mean()
        metric = {
            "loss"      : loss.item(),
            "i2c/loss_i": loss_i.item(),
            "i2c/loss_t": loss_t.item(),
            "i2c/cos"   : cos.item(),
            "i2c/scale" : scale.item(),
            "i2c/n"     : float(v),
            "i2c/std"   : pred_std.item(),     # -> 0 if the encoder collapses to a constant
        }
        return loss, metric

    if TYPE_CHECKING: __call__ = forward  # so pyright sees forward's signature on calls
