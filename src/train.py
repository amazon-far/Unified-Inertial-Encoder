"""Train FoundationIMU.

Usage:
    python -m src.train experiment=<recipe>
"""

import os
import hydra
import torch
import lightning as L

from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from .interface.model       import ILossFunc, FoundationIMU_Output
from .model.model           import FoundationIMU
from .model.trainer_util    import ActivateDeferredWandb, DeferredWandbLogger, EMACallback, DirectCheckpointIO
from .data.util             import multithread_instantiate, subsample_corpus
from .interface.data_imu    import IMUData_w_GT_Jagged, collate_imu_data


class LitFoundationIMU(L.LightningModule):
    """Lightning wrapper around ``FoundationIMU``; trains the decoder heads on unmasked input."""

    def __init__(
        self,
        model       : FoundationIMU,
        loss_func   : ILossFunc,
        optim_cfg   : DictConfig,
        cooldown_steps: int | None = None,
        cooldown_from_step: int | None = None,
    ) -> None:

        super().__init__()
        self.model      = model
        self.loss       = loss_func
        self.optim_cfg  = optim_cfg
        # Optional LR cooldown: decay every param group's lr linearly to zero over `cooldown_steps`,
        # then stop. Starts at global step `cooldown_from_step`, or at the first step trained if None.
        self.cooldown_steps  = cooldown_steps
        self._cooldown_start: int | None = cooldown_from_step
        self._cooldown_base : list[float] = []

        # EMA shadow injected by EMACallback
        self._ema_box: list[torch.nn.Module] = []

    @property
    def ema_network(self) -> torch.nn.Module:
        return self._ema_box[0]

    @property
    def compute_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def attach_ema(self, network: torch.nn.Module) -> None:
        self._ema_box[:] = [network]

    def on_train_epoch_start(self) -> None:
        self.model.configure_trainable_params()

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        if not self.cooldown_steps:
            return
        opt = self.optimizers(use_pl_optimizer=False)
        assert isinstance(opt, torch.optim.Optimizer)
        if self._cooldown_start is None:
            self._cooldown_start = self.trainer.global_step
        if self.trainer.global_step < self._cooldown_start:
            return
        if not self._cooldown_base:
            self._cooldown_base = [float(g["lr"]) for g in opt.param_groups]
        done   = self.trainer.global_step - self._cooldown_start
        factor = max(0.0, 1.0 - done / self.cooldown_steps)
        for g, base in zip(opt.param_groups, self._cooldown_base):
            g["lr"] = base * factor
        self.log("train/lr_factor", factor, on_step=True, on_epoch=False)
        if done >= self.cooldown_steps:
            self.trainer.should_stop = True

    @staticmethod
    def _no_mask(batch: IMUData_w_GT_Jagged) -> torch.Tensor:
        """All-false sample-space mask; the model requires one but nothing is masked."""
        return torch.zeros(batch.acc.shape[0], dtype=torch.bool, device=batch.acc.device)

    def training_step(self, batch: IMUData_w_GT_Jagged, batch_idx: int) -> torch.Tensor:
        out: FoundationIMU_Output = self.model(batch, self._no_mask(batch))
        loss, metric              = self.loss(self.model, batch, out)

        self.log_dict(
            {f"train/{k}": v for k, v in metric.items()},
            batch_size=batch.offsets_B.numel() - 1
        )
        return loss

    def validation_step(self, batch: IMUData_w_GT_Jagged, batch_idx: int) -> torch.Tensor:
        mask = self._no_mask(batch)

        losses = []
        for stage, net in (("valid", self.model), ("valid_ema", self.ema_network)):
            out: FoundationIMU_Output = net(batch, mask)
            loss, metric              = self.loss(self.model, batch, out)

            self.log_dict(
                {f"{stage}/{k}": v for k, v in metric.items()},
                batch_size=batch.offsets_B.numel() - 1,
                sync_dist=True
            )
            losses.append(loss)

        return torch.stack(losses).mean()

    def transfer_batch_to_device(self, batch: IMUData_w_GT_Jagged, device: torch.device, dataloader_idx: int) -> IMUData_w_GT_Jagged:
        return batch.to(device, self.compute_dtype)

    def configure_optimizers(self):
        self.model.configure_trainable_params()
        return hydra.utils.instantiate(self.optim_cfg, params=self.parameters())


@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.meta.seed, workers=True)

    model     = hydra.utils.instantiate(cfg.model)
    loss_fn   = hydra.utils.instantiate(cfg.loss)
    max_workers = min(16, (os.cpu_count() or 2) // 2)
    train_corpora = multithread_instantiate(cfg.data.train, "Loading train sets", max_workers=max_workers)
    # Applied per corpus; valid is never subsampled, so metrics stay comparable across fractions.
    if cfg.data_conf.subsample < 1.0:
        counts = []
        subsampled = []
        for corpus in train_corpora:
            if not isinstance(corpus, ConcatDataset):
                raise TypeError("Sequence subsampling requires a ConcatDataset for each corpus")
            subset = subsample_corpus(corpus, cfg.data_conf.subsample)
            subsampled.append(subset)
            counts.append(f"{len(subset.datasets)}/{len(corpus.datasets)}")
        print(f"Subsampled train sequences to {cfg.data_conf.subsample}: {counts}")
        train_set = ConcatDataset(subsampled)
    else:
        train_set = ConcatDataset(train_corpora)
    valid_set = ConcatDataset(multithread_instantiate(cfg.data.valid, "Loading valid sets", max_workers=max_workers))

    # debug=true: in-process data loading so pdb works inside dataset __getitem__.
    num_workers = 0 if cfg.debug else cfg.loader.num_workers
    loader_kwargs = dict(
        batch_size              = cfg.loader.batch_size,
        num_workers             = num_workers,
        collate_fn              = collate_imu_data,
        pin_memory              = True,
        prefetch_factor         = cfg.loader.prefetch_factor if num_workers > 0 else None,
        multiprocessing_context = "fork" if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_set, shuffle=True , persistent_workers=num_workers > 0, **loader_kwargs)
    valid_loader = DataLoader(valid_set, shuffle=False, persistent_workers=False          , **loader_kwargs)

    # Resolved against the launch cwd: Hydra chdirs into the run dir.
    ckpt_path = cfg.meta.resume_from
    if ckpt_path is not None:
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
        mode = "full-state resume" if cfg.meta.resume_full_state else "finetune (weights only)"
        print(f"Resuming from checkpoint [{mode}]: {ckpt_path}")

    run_id = f"{cfg.wandb.runid}-{datetime.now():%H%M%S-%m%d}"

    if cfg.wandb.enable and (not cfg.debug):
        wandb_logger = DeferredWandbLogger(
            resume_from = ckpt_path,
            entity      = cfg.wandb.entity,
            project     = cfg.wandb.project,
            id          = run_id,
            save_dir    = cfg.meta.output,
        )
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
        logger = wandb_logger
    else:
        logger = False

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    if cfg.debug:
        # One device keeps strategy=auto single-process, so pdb can attach.
        trainer_kwargs["devices"] = 1  # type: ignore[index]
    if cfg.meta.allow_ddp_unused_param:
        # A loss may skip some decoder heads, and mask_embed is never hit: DDP must allow unused params.
        trainer_kwargs.setdefault("strategy", "ddp_find_unused_parameters_true")  # type: ignore[union-attr]

    lit_model = LitFoundationIMU(model, loss_fn, cfg.optim, cooldown_steps=cfg.meta.get("cooldown_steps"),
                                 cooldown_from_step=cfg.meta.get("cooldown_from_step"))
    trainer   = L.Trainer(
        default_root_dir=cfg.meta.output,
        logger=logger,
        plugins=[
            # Lightning's default ckpt IO leaves /tmp garbage on S3 filesystem.
            DirectCheckpointIO() 
        ],
        callbacks=[
            # W&B can deadlock when initialized normally; defer init.
            ActivateDeferredWandb(),
            # When every_n_epochs > 1, pair it with trainer.check_val_every_n_epoch.
            ModelCheckpoint(dirpath=os.path.join(cfg.meta.output, "checkpoints"), save_top_k=-1,
                            every_n_epochs=cfg.meta.ckpt_every_n_epochs),
            EMACallback(decay=tuple(cfg.meta.ema_decay)),
        ],
        **trainer_kwargs,  # type: ignore[arg-type]
    )

    # The default input cast can't rebuild our frozen ArticulatedModel; route it through IMUData.to.
    trainer.precision_plugin.convert_input = lambda batch: batch.to(None, lit_model.compute_dtype)  # type: ignore[method-assign]

    fit_ckpt_path = ckpt_path
    if ckpt_path is not None and not cfg.meta.resume_full_state:
        # Finetune: weights only (heads may differ across stages), fresh optimizer state and counters.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            model_sd = {k[len("model."):]: v for k, v in state["state_dict"].items() if k.startswith("model.")}
        else:   # Released checkpoints hold only the EMA weights.
            model_sd = state["ema_state_dict"]
        missing, unexpected = lit_model.model.load_state_dict(model_sd, strict=False)
        print(f"Loaded {len(model_sd)} weight tensors (missing={len(missing)}, unexpected={len(unexpected)}); "
              f"optimizer and step/epoch counters start fresh.")
        fit_ckpt_path = None

    trainer.fit(lit_model, train_loader, valid_loader, ckpt_path=fit_ckpt_path)


if __name__ == "__main__":
    main()
