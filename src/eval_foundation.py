"""Evaluate a FoundationIMU checkpoint on all sensor sites, scoring only the pelvis stream.

Usage:
    python -m src.eval_foundation --ckpt <run>/checkpoints/<name>.ckpt --data_from <donor_run_dir> \
        --dataset Nymeria_XSens_Processed_IMU_Language [--limit N] [--json out.json]
"""

import json
import argparse
import torch
import pypose as pp

from pathlib import Path
from omegaconf import OmegaConf

from . import evaluate
from .interface.evaluate import MetricPart
from .eval import load_model, inference
from .interface.human import CANONICAL_IDX
from .interface.model import FoundationIMU_Output
from .interface.data_imu import IMUData_BaseTrait_Jagged, IMUData_LabelTrait_Jagged, IMUData_w_GT_Jagged
from .scripts.evaluate_velocity.__main__ import valid_pipelines


def pelvis_restrict(
    batch: IMUData_BaseTrait_Jagged, pred: FoundationIMU_Output, min_len: int = 32
) -> tuple[IMUData_w_GT_Jagged, FoundationIMU_Output] | None:
    """Slice a multi-device eval pack and its aligned prediction down to the time-sorted pelvis stream."""
    assert isinstance(batch, IMUData_LabelTrait_Jagged), \
        f"pelvis_restrict needs gt_pose/gt_vel alongside the base fields, got {type(batch).__name__}"

    pelvis = int(CANONICAL_IDX["pelvis"])
    sid    = batch.sensor_id[:, 0]
    rows   = (sid == pelvis).nonzero().flatten()
    if rows.numel() < min_len:
        return None
    perm = rows[batch.time_ns[rows, 0].argsort()]

    n_runs  = batch.offsets_N.numel() - 1
    row_run = torch.repeat_interleave(torch.arange(n_runs), batch.offsets_N.diff())
    run0    = row_run[perm[0]]
    offsets = torch.tensor([0, perm.numel()], dtype=torch.int64)

    sub = IMUData_w_GT_Jagged(
        time_ns      = batch.time_ns[perm],
        acc          = batch.acc[perm],
        gyro         = batch.gyro[perm],
        correct_mask = batch.correct_mask[perm],
        correct_acc  = batch.correct_acc[perm],
        correct_gyro = batch.correct_gyro[perm],
        gravity         = batch.gravity[run0 : run0 + 1],
        correct_gravity = batch.correct_gravity[run0 : run0 + 1],
        acc_bias     = batch.acc_bias[perm],
        gyro_bias    = batch.gyro_bias[perm],
        offsets_B    = offsets.clone(),
        offsets_N    = offsets.clone(),
        sensor_id    = batch.sensor_id[perm],
        model        = batch.model,
        kind         = batch.kind,
        labels       = batch.labels[run0 : run0 + 1],
        gt_pose      = pp.SE3(batch.gt_pose.tensor()[perm]),
        gt_vel       = batch.gt_vel[perm],
    )
    device = pred.dec_body_vel.device
    p_perm = perm.to(device)
    sub_pred = FoundationIMU_Output(
        trunk_feature    = torch.zeros(0, 1, device=device),
        mask_feature     = torch.zeros(0, dtype=torch.bool, device=device),
        offsets_B_feat   = offsets.to(device),
        offsets_N_feat   = offsets.to(device),
        dec_correct_acc  = pred.dec_correct_acc[p_perm],
        dec_correct_gyro = pred.dec_correct_gyro[p_perm],
        dec_body_vel     = pred.dec_body_vel[p_perm],
        dec_lang_feature = torch.zeros(1, 1, 1, device=device),
        lang_time_ns     = pred.lang_time_ns[:1, :1].reshape(1, 1),
        time_ns          = pred.time_ns[p_perm],
        mask_samp        = pred.mask_samp[p_perm],
        offsets_B_samp   = offsets.to(device),
        offsets_N_samp   = offsets.to(device),
    )
    return sub, sub_pred


if __name__ == "__main__":
    from rich.table import Table
    from rich.console import Console

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt"     , type=str, required=True, help="FoundationIMU checkpoint (src.train run)")
    parser.add_argument("--data_from", type=str, required=True, help="run dir whose .hydra config carries the dataset's valid entries")
    parser.add_argument("--dataset"  , type=str, required=True, help="reader class name key from valid_pipelines (e.g. Humanoid_SuperOdom_NPZ_Processed_Language)")
    parser.add_argument("--limit"    , type=int, default=None, help="evaluate only the first N takes (smoke test)")
    parser.add_argument("--json"     , type=str, default=None)
    # "measured" (raw gyro) is the only source comparable with baselines, which have no denoise
    # head; "denoised" uses this model's correction head; "clean" is an oracle upper bound.
    parser.add_argument("--rot_gyro" , type=str, default="measured", choices=["measured", "denoised", "clean"],
                        help="gyro channel preintegrated for ATE_Pred's attitude (default: measured, the table convention)")
    args = parser.parse_args()

    # Pin the per-fetch transform draws so pooled numbers are reproducible.
    torch.manual_seed(0)

    model   = load_model(Path(args.ckpt)).eval().cuda()
    run_cfg = OmegaConf.load(Path(args.data_from).expanduser() / ".hydra" / "config.yaml")
    groups  = valid_pipelines(run_cfg)
    assert args.dataset in groups, f"--dataset must be one of {sorted(groups)}"
    builders = groups[args.dataset][: args.limit]

    per_seq: list[dict[str, MetricPart]] = []
    skipped = 0
    for i, build in enumerate(builders):
        seq = build()
        if len(seq) == 0:
            skipped += 1
            continue
        batch = seq[:]
        pred  = inference(model, batch)
        res   = pelvis_restrict(batch, pred)
        if res is None:
            skipped += 1
            continue
        sub, sub_pred = res
        take = {}
        for metric in [evaluate.Velocity_Metric(sub, sub_pred),
                       evaluate.Trajectory_Metric(sub, sub_pred, gyro_source=args.rot_gyro)]:
            take |= metric.parts
        per_seq.append(take)

    pooled = evaluate.merge_parts(per_seq)
    table  = Table(title=f"FoundationIMU pelvis-stream eval: {args.dataset}, {len(per_seq)} takes (+{skipped} skipped)")
    table.add_column("metric"); table.add_column("value"); table.add_column("items")
    for k in sorted(pooled):
        table.add_row(k, f"{pooled[k].value:.6g}", str(pooled[k].count))
    Console().print(table)

    if args.json:
        payload = {k: {"value": pooled[k].value, "count": pooled[k].count,
                       "total": pooled[k].total, "root": pooled[k].root} for k in pooled}
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
