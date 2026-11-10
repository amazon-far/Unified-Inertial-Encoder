"""Train the IMU2CLIP contrastive baseline.

Usage:
    python -m src.train_imu2clip experiment=exp11_imu2clip_lang
"""

import os
import hydra
import torch
import lightning as L

from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from .model.imu2clip       import IMU2CLIP, IMU2CLIP_Loss, IMU2CLIP_Output
from .model.trainer_util   import ActivateDeferredWandb, DeferredWandbLogger, DirectCheckpointIO
from .data.util            import multithread_instantiate
from .interface.data_imu   import IMUData_w_GT_Lang_Jagged, collate_imu_data


class LitIMU2CLIP(L.LightningModule):
    """Minimal Lightning wrapper around IMU2CLIP; the loss owns the learned temperature parameter."""

    def __init__(
        self,
        model    : IMU2CLIP,
        loss_func: IMU2CLIP_Loss,
        optim_cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.model     = model
        self.loss      = loss_func
        self.optim_cfg = optim_cfg

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.model.compute_dtype

    def training_step(self, batch: IMUData_w_GT_Lang_Jagged, batch_idx: int) -> torch.Tensor:
        out: IMU2CLIP_Output = self.model(batch)
        loss, metric         = self.loss(self.model, batch, out)
        self.log_dict(
            {f"train/{k}": v for k, v in metric.items()},
            batch_size=batch.offsets_B.numel() - 1,
        )
        return loss

    def validation_step(self, batch: IMUData_w_GT_Lang_Jagged, batch_idx: int) -> torch.Tensor:
        out: IMU2CLIP_Output = self.model(batch)
        loss, metric         = self.loss(self.model, batch, out)
        self.log_dict(
            {f"valid/{k}": v for k, v in metric.items()},
            batch_size=batch.offsets_B.numel() - 1,
            sync_dist=True,
        )
        return loss

    def transfer_batch_to_device(self, batch: IMUData_w_GT_Lang_Jagged, device: torch.device, dataloader_idx: int) -> IMUData_w_GT_Lang_Jagged:
        return batch.to(device, self.compute_dtype)

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.optim_cfg, params=self.parameters())


@hydra.main(config_path="./config", config_name="train_imu2clip", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.meta.seed, workers=True)

    model   = hydra.utils.instantiate(cfg.model)
    loss_fn = hydra.utils.instantiate(cfg.loss)
    max_workers = min(16, (os.cpu_count() or 2) // 2)
    train_set = ConcatDataset(multithread_instantiate(cfg.data.train, "Loading train sets", max_workers=max_workers))
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
        print(f"Resuming from checkpoint: {ckpt_path}")

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
        trainer_kwargs["devices"] = 1  # type: ignore[index]

    lit_model = LitIMU2CLIP(model, loss_fn, cfg.optim)
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
            ModelCheckpoint(dirpath=os.path.join(cfg.meta.output, "checkpoints"), save_top_k=-1),
        ],
        **trainer_kwargs,  # type: ignore[arg-type]
    )

    # The default input cast can't rebuild our frozen ArticulatedModel; route it through IMUData.to.
    trainer.precision_plugin.convert_input = lambda batch: batch.to(None, lit_model.compute_dtype)  # type: ignore[method-assign]

    trainer.fit(lit_model, train_loader, valid_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
