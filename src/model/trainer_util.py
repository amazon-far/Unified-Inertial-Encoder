"""Lightning training utilities: precision mapping, checkpoint I/O, deferred W&B logging and EMA."""
import io
import os
import re
import math
import fsspec
import torch
import lightning as L
import torch.nn as nn

from typing import Any, cast
from argparse import Namespace
from torch.optim.swa_utils import AveragedModel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.io import TorchCheckpointIO
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .tokenizer import AddRegister


def precision_dtypes(precision: str | int) -> tuple[torch.dtype, torch.dtype | None]:
    """Map a Lightning ``trainer.precision`` to ``(param_dtype, autocast_dtype)`` for inference."""
    match str(precision):
        case "64" | "64-true":     return torch.float64, None
        case "32" | "32-true":     return torch.float32, None
        case "16" | "16-true":     return torch.float16, None
        case "bf16" | "bf16-true": return torch.bfloat16, None
        case "16-mixed":           return torch.float32, torch.float16
        case "bf16-mixed":         return torch.float32, torch.bfloat16
        case _: raise ValueError(f"unsupported trainer.precision {precision!r}")


class DirectCheckpointIO(TorchCheckpointIO):
    """Save checkpoints with a single direct write instead of Lightning's transactional save.

    The transactional save spools through ``/tmp`` and leaks the spool when the final path is a fuse
    mount on another filesystem. The direct write is not atomic: a crash can truncate the checkpoint.
    """

    def save_checkpoint(self, checkpoint: dict[str, Any], path, storage_options: Any | None = None) -> None:
        if storage_options is not None:
            raise TypeError(f"{type(self).__name__} does not support storage_options={storage_options!r}")
        fs, urlpath = fsspec.core.url_to_fs(str(path))
        fs.makedirs(os.path.dirname(urlpath), exist_ok=True)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        with fs.open(urlpath, "wb") as f:   # autocommit: no /tmp spool
            f.write(buffer.getvalue())


def checkpoint_artifact_ref(output_dir: str, ckpt_filename: str) -> tuple[str, str]:
    """Deterministic ``(artifact_name, alias)`` for a checkpoint, from its location alone.

    The producer and a resuming consumer both derive it from the path: the name from the last two
    segments of the run output dir, the alias from the filename stem, sanitized to W&B's charset.
    """
    def sanitize(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", s)
    tail  = "/".join(output_dir.rstrip("/").split("/")[-2:])
    name  = "ckpt-" + sanitize(tail)
    stem  = ckpt_filename[:-5] if ckpt_filename.endswith(".ckpt") else ckpt_filename
    return name, sanitize(stem)


class DeferredWandbLogger(Logger):
    """Create the real W&B run only after DataLoader workers have forked.

    ``wandb.init`` starts background threads, which makes a later fork unsafe, and the mmap-backed
    Nymeria dataset needs the ``fork`` context. Also logs each checkpoint as a W&B reference artifact
    and links a resumed run to its parent's artifact.
    """

    def __init__(self, resume_from: str | None = None, **wandb_init: Any) -> None:
        super().__init__()
        self._wandb_init = wandb_init
        self._logger: WandbLogger | None = None
        self._active = False
        self._pending_hparams: list[tuple[dict[str, Any] | Namespace, tuple[Any, ...], dict[str, Any]]] = []
        self._resume_from  = resume_from
        self._parent_used  = False
        self._logged_ckpts: set[str] = set()

    @property
    def name(self) -> str | None:
        return self._wandb_init.get("name") or self._wandb_init.get("project")

    @property
    def version(self) -> str | None:
        return self._wandb_init.get("version") or self._wandb_init.get("id")

    @property
    def save_dir(self) -> str | None:
        save_dir = self._wandb_init.get("save_dir")
        return None if save_dir is None else str(save_dir)

    @property
    def experiment(self):
        """The underlying W&B ``Run`` once active (rank zero), else ``None``."""
        logger = self._get_logger()
        return logger.experiment if logger is not None else None

    @property
    def root_dir(self) -> str | None:
        return self._logger.root_dir if self._logger is not None else self.save_dir

    @property
    def log_dir(self) -> str | None:
        return self._logger.log_dir if self._logger is not None else self.save_dir

    @staticmethod
    def _is_global_zero() -> bool:
        rank = os.environ.get("RANK")
        if rank is not None:
            return int(rank) == 0
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return int(local_rank) == 0
        return getattr(rank_zero_only, "rank", 0) == 0

    def activate(self) -> None:
        self._active = True
        self._get_logger()

    def _get_logger(self) -> WandbLogger | None:
        if not self._active or not self._is_global_zero():
            return None
        if self._logger is None:
            self._logger = WandbLogger(**self._wandb_init)
            for params, args, kwargs in self._pending_hparams:
                self._logger.log_hyperparams(params, *args, **kwargs)
            self._pending_hparams.clear()
            self._use_parent_artifact(self._logger)
        return self._logger

    def _use_parent_artifact(self, logger: WandbLogger) -> None:
        """Mark the parent checkpoint's artifact as used; a parent that was never logged only warns."""
        if self._parent_used or self._resume_from is None:
            return
        self._parent_used = True
        run = logger.experiment
        # parent output dir := strip the trailing "checkpoints/<file>.ckpt"
        parent_out = os.path.dirname(os.path.dirname(self._resume_from))
        name, alias = checkpoint_artifact_ref(parent_out, os.path.basename(self._resume_from))
        project = self._wandb_init.get("project")
        entity  = self._wandb_init.get("entity")
        qualified = f"{entity + '/' if entity else ''}{project + '/' if project else ''}{name}:{alias}"
        try:
            run.use_artifact(qualified, type="model")
        except Exception as e:  # noqa: BLE001 -- lineage is best-effort, never block training
            print(f"[lineage] could not link parent artifact {qualified!r}: {e}")

    def _log_checkpoint_artifacts(self, logger: WandbLogger, checkpoint_callback: ModelCheckpoint) -> None:
        """Log each newly written checkpoint as a *reference* artifact, aliased by its filename stem."""
        from lightning.pytorch.loggers.utilities import _scan_checkpoints
        run     = logger.experiment
        out_dir = self._wandb_init.get("save_dir") or os.path.dirname(checkpoint_callback.dirpath or "")
        name, _ = checkpoint_artifact_ref(str(out_dir), "x.ckpt")
        for _t, path, score, _tag in _scan_checkpoints(checkpoint_callback, {}):
            if path in self._logged_ckpts:
                continue
            self._logged_ckpts.add(path)
            _, alias = checkpoint_artifact_ref(str(out_dir), os.path.basename(path))
            try:
                import wandb
                art = wandb.Artifact(name=name, type="model", metadata={
                    "path": path, "score": score.item() if isinstance(score, torch.Tensor) else score,
                })
                # Reference with checksum=False: never upload or re-read the checkpoint bytes.
                art.add_reference(f"file://{os.path.abspath(path)}", checksum=False)
                run.log_artifact(art, aliases=[alias, "latest"])
            except Exception as e:  # noqa: BLE001 -- lineage is best-effort, never block training
                print(f"[lineage] could not log checkpoint artifact for {path!r}: {e}")

    def log_hyperparams(self, params: dict[str, Any] | Namespace, *args: Any, **kwargs: Any) -> None:
        logger = self._get_logger()
        if logger is None:
            self._pending_hparams.append((params, args, kwargs))
        else:
            logger.log_hyperparams(params, *args, **kwargs)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        logger = self._get_logger()
        if logger is not None:
            logger.log_metrics(metrics, step)

    def save(self) -> None:
        if self._logger is not None:
            self._logger.save()

    def finalize(self, status: str) -> None:
        if self._logger is not None:
            self._logger.finalize(status)

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if self._logger is not None:
            self._log_checkpoint_artifacts(self._logger, checkpoint_callback)


class ActivateDeferredWandb(L.Callback):
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        logger = trainer.logger
        if isinstance(logger, DeferredWandbLogger):
            logger.activate()

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        lit: Any     = pl_module                       # LitFoundationIMU; annotating risks a circular import
        tokenizer    = getattr(lit.model, "tokenizer", None)
        if tokenizer is None:
            # Baselines have no tokenizer embeds to log.
            return
        if isinstance(tokenizer, AddRegister):
            mask_embed   = cast(torch.Tensor, tokenizer.base_tokenizer.mask_embed)
            device_embed = cast(torch.Tensor, tokenizer.base_tokenizer.device_embed).reshape(-1, mask_embed.shape[-1])
            reg_embed    = tokenizer.reg_embed.reshape(-1, mask_embed.shape[-1])

            lit.log_dict({
                "device_embed/token_norm_mean": device_embed.norm(dim=-1).mean(),
                "device_embed/mask_token_norm": mask_embed.flatten().norm(),
                "device_embed/reg_token_norm" : reg_embed.norm(dim=-1).mean(),
            }, sync_dist=True)
        
        else:
            mask_embed   = tokenizer.mask_embed
            device_embed = tokenizer.device_embed.reshape(-1, mask_embed.shape[-1])
            
            lit.log_dict({
                "device_embed/token_norm_mean": device_embed.norm(dim=-1).mean(),
                "device_embed/mask_token_norm": mask_embed.flatten().norm(),
            }, sync_dist=True)



class EMACallback(L.Callback):
    """Tracks an exponential moving average of the network weights.

    Saved under the ``ema_state_dict`` checkpoint key, with ``FoundationIMU.state_dict()`` keys (no
    Lightning ``model.`` prefix). ``decay`` is a ``(start, end)`` pair, cosine-annealed over the whole
    fit; pass ``start == end`` for a constant decay.
    """

    def __init__(self, decay: tuple[float, float]) -> None:
        assert len(decay) == 2, f"ema decay must be a (start, end) 2-tuple, got {decay!r}"
        self.decay_start, self.decay_end = decay
        self._cur_decay = self.decay_start
        self.ema: AveragedModel | None = None
        # Stashed on resume and applied in on_fit_start, once the model has its training precision.
        self._pending_ema_sd: dict[str, Any] | None = None

    @staticmethod
    def _network(pl_module) -> nn.Module:
        net = pl_module.model
        assert isinstance(net, nn.Module)
        return net

    @torch.no_grad()
    def _ema_avg(self, ema_param_list, current_param_list, _) -> None:
        # torch's get_ema_multi_avg_fn, but reading the live scheduled decay; foreach lerp is float/complex only.
        decay = self._cur_decay
        if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(ema_param_list[0]):
            torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)
        else:
            for p_ema, p_model in zip(ema_param_list, current_param_list, strict=True):
                p_ema.copy_(p_ema * decay + p_model * (1 - decay))

    def _make_shadow(self, pl_module) -> AveragedModel:
        # use_buffers=True makes the shadow a complete drop-in state_dict.
        return AveragedModel(self._network(pl_module), multi_avg_fn=self._ema_avg, use_buffers=True)

    def on_fit_start(self, trainer, pl_module) -> None:
        # Built here, after Lightning's precision conversion, so the shadow's dtype matches the live network.
        if self.ema is None:
            self.ema = self._make_shadow(pl_module)
        if self._pending_ema_sd is not None:
            self.ema.module.load_state_dict(self._pending_ema_sd)
            self._pending_ema_sd = None
        self.ema.to(pl_module.device)
        lit: Any = pl_module
        # Expose the wrapped network, not the AveragedModel shell: the loss needs `.tokenizer`/`.trunk`.
        lit.attach_ema(self.ema.module)

    def on_validation_start(self, trainer, pl_module) -> None:
        # The deepcopied shadow keeps the source's mode at copy time; force eval for the valid_ema/* pass.
        if self.ema is not None:
            self.ema.eval()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # Assumes one optimizer step per batch; gradient accumulation would need a guard here.
        assert self.ema is not None

        total    = max(1, trainer.estimated_stepping_batches)
        progress = min(1.0, trainer.global_step / total)
        self._cur_decay = self.decay_end + (self.decay_start - self.decay_end) * (1 + math.cos(math.pi * progress)) / 2

        self.ema.update_parameters(self._network(pl_module))

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        assert self.ema is not None
        checkpoint["ema_state_dict"] = self.ema.module.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        # Stash only: on_fit_start loads it after the precision conversion.
        if "ema_state_dict" in checkpoint:
            self._pending_ema_sd = checkpoint["ema_state_dict"]
