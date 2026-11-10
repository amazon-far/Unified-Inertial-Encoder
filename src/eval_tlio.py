"""Evaluate a trained TLIO baseline densely over each device stream with the ``src.eval`` metrics.

Usage:
    python -m src.eval_tlio --ckpt <run>/checkpoints/<name>.ckpt             # one sequence, writes a .rrd
    python -m src.eval_tlio --ckpt <run>/checkpoints/<name>.ckpt --idx all   # pooled; or --idx 0:100
"""

import argparse
import hydra
import torch
import typing as tp
import pypose as pp

from pathlib import Path
from omegaconf import OmegaConf

from . import evaluate
from .eval import load_sequence
from .model.tlio import TLIO, TLIO_Output
from .interface.model import FoundationIMU_Output
from .interface.data_imu import IMUData_BaseTrait_Jagged, IMUData_LabelTrait_Jagged, IMUData_w_GT_Jagged


def load_model(ckpt: Path) -> TLIO:
    """Recover the TLIO instance from a ``src.train_tlio`` run's config + checkpoint."""
    ckpt    = Path(ckpt).expanduser()
    run_dir = ckpt.parent.parent
    cfg     = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    model: TLIO = hydra.utils.instantiate(cfg.model)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")})
    return model


def device_major(batch: IMUData_BaseTrait_Jagged, min_len: int) -> IMUData_w_GT_Jagged | None:
    """Reorder a window-major eval pack into device-major, time-sorted streams, one run per device.

    Drops streams shorter than ``min_len`` samples; returns None if nothing survives.
    """
    assert isinstance(batch, IMUData_LabelTrait_Jagged), \
        f"TLIO eval needs gt_pose/gt_vel alongside the base fields, got {type(batch).__name__}"

    time_ns = batch.time_ns[:, 0]
    sid     = batch.sensor_id[:, 0]
    n_runs  = batch.offsets_N.numel() - 1
    row_run = torch.repeat_interleave(torch.arange(n_runs), batch.offsets_N.diff())

    parts, bounds = [], [0]
    for s in sid.unique().tolist():
        rows = (sid == s).nonzero().flatten()
        if rows.numel() < min_len:
            continue
        parts.append(rows[time_ns[rows].argsort()])
        bounds.append(bounds[-1] + rows.numel())
    if not parts:
        return None

    perm       = torch.cat(parts)
    offsets_N  = torch.tensor(bounds, dtype=torch.int64)
    run_of_new = row_run[perm[offsets_N[:-1]]]   # original run backing each stream's first row

    return IMUData_w_GT_Jagged(
        time_ns      = batch.time_ns[perm],
        acc          = batch.acc[perm],
        gyro         = batch.gyro[perm],
        correct_mask = batch.correct_mask[perm],
        correct_acc  = batch.correct_acc[perm],
        correct_gyro = batch.correct_gyro[perm],
        gravity         = batch.gravity[run_of_new],
        correct_gravity = batch.correct_gravity[run_of_new],
        acc_bias     = batch.acc_bias[perm],
        gyro_bias    = batch.gyro_bias[perm],
        offsets_B    = offsets_N[[0, -1]],
        offsets_N    = offsets_N,
        sensor_id    = batch.sensor_id[perm],
        model        = batch.model,
        kind         = batch.kind,
        labels       = batch.labels[run_of_new],
        gt_pose      = pp.SE3(batch.gt_pose.tensor()[perm]),
        gt_vel       = batch.gt_vel[perm],
    )


@torch.inference_mode()
def tlio_inference(
    model : TLIO,
    seq   : IMUData_w_GT_Jagged,
    stride: int = 10,
    chunk : int = 4096,
) -> FoundationIMU_Output:
    """Dense sliding-window TLIO -> per-sample body velocity, as a ``FoundationIMU_Output`` row-aligned with ``seq``."""
    device = next(model.parameters()).device
    W      = model.window_size
    s_gpu  = seq.to(device, torch.float32)
    assert isinstance(s_gpu, IMUData_w_GT_Jagged)
    off    = s_gpu.offsets_N

    vel_w = torch.empty(s_gpu.acc.shape[0], 3, device=device)
    for d in range(off.numel() - 1):
        s, e = int(off[d]), int(off[d + 1])
        L    = e - s
        starts = torch.arange(0, L - W + 1, stride, device=device)
        if int(starts[-1]) != L - W:   # tail window so the last samples are covered too
            starts = torch.cat([starts, starts.new_tensor([L - W])])
        n_w = starts.numel()

        disp = []
        for c0 in range(0, n_w, chunk):
            c_starts = starts[c0 : c0 + chunk]
            rows = (s + c_starts[:, None] + torch.arange(W, device=device)).flatten()
            n    = c_starts.numel()
            sub  = IMUData_w_GT_Jagged(
                time_ns      = s_gpu.time_ns[rows],
                acc          = s_gpu.acc[rows],
                gyro         = s_gpu.gyro[rows],
                correct_mask = s_gpu.correct_mask[rows],
                correct_acc  = s_gpu.correct_acc[rows],
                correct_gyro = s_gpu.correct_gyro[rows],
                gravity         = s_gpu.gravity[d : d + 1].expand(n, 3),
                correct_gravity = s_gpu.correct_gravity[d : d + 1].expand(n, 3),
                acc_bias     = s_gpu.acc_bias[rows],
                gyro_bias    = s_gpu.gyro_bias[rows],
                offsets_B    = torch.tensor([0, n * W], dtype=torch.int64, device=device),
                offsets_N    = torch.arange(n + 1, dtype=torch.int64, device=device) * W,
                sensor_id    = s_gpu.sensor_id[rows],
                model        = s_gpu.model,
                kind         = s_gpu.kind,
                labels       = torch.zeros(n, dtype=torch.uint8, device=device),
                gt_pose      = pp.SE3(s_gpu.gt_pose.tensor()[rows]),
                gt_vel       = s_gpu.gt_vel[rows],
            )
            out: TLIO_Output = model(sub)
            disp.append(out.disp.float())
        disp_d = torch.cat(disp)

        # Window displacement -> average world velocity over the window's actual duration.
        t0   = s_gpu.time_ns[s + starts, 0]
        t1   = s_gpu.time_ns[s + starts + W - 1, 0]
        vbar = disp_d / ((t1 - t0).double() * 1e-9).float().clamp_min(1e-6)[:, None]

        # Each sample takes the velocity of the nearest window center.
        centers = (starts + W // 2).contiguous()
        i       = torch.arange(L, device=device)
        right   = torch.searchsorted(centers, i, right=True).clamp(1, n_w - 1) if n_w > 1 \
                  else torch.zeros(L, dtype=torch.int64, device=device)
        left    = right - 1
        pick    = torch.where((i - centers[left]).abs() <= (centers[right] - i).abs(), left, right) if n_w > 1 \
                  else right
        vel_w[s:e] = vbar[pick]

    # World -> body frame with the same GT orientations TLIO consumed at its input.
    v_b  = tp.cast(torch.Tensor, s_gpu.gt_pose.rotation().Inv() @ vel_w)
    pack = s_gpu.acc.shape[0]
    return FoundationIMU_Output(
        trunk_feature    = torch.zeros(0, 1, device=device),
        mask_feature     = torch.zeros(0, dtype=torch.bool, device=device),
        offsets_B_feat   = s_gpu.offsets_B.clone(),
        offsets_N_feat   = s_gpu.offsets_N.clone(),
        dec_correct_acc  = s_gpu.acc.clone(),      # identity corrections: TLIO has no denoise head
        dec_correct_gyro = s_gpu.gyro.clone(),
        dec_body_vel     = v_b,
        dec_lang_feature = torch.zeros(1, 1, 1, device=device),
        lang_time_ns     = s_gpu.time_ns[:1, :1].reshape(1, 1),
        time_ns          = s_gpu.time_ns.clone(),
        mask_samp        = torch.zeros(pack, dtype=torch.bool, device=device),
        offsets_B_samp   = s_gpu.offsets_B.clone(),
        offsets_N_samp   = s_gpu.offsets_N.clone(),
    )


def _parse_indices(spec: str, total: int) -> list[int]:
    if spec == "all":
        return list(range(total))
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a or 0), min(int(b or total), total)))
    return [int(spec)]


if __name__ == "__main__":
    import rerun as rr
    import rerun.blueprint as rrb
    from rich.table import Table
    from rich.console import Console

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt"  , type=str, required=True, help="checkpoint of a src.train_tlio run")
    parser.add_argument("--data"  , type=str, default=None, help="*.yaml in ./config/data; default: the ckpt run's own data config")
    parser.add_argument("--subset", type=str, choices=["train", "valid"], default="valid")
    parser.add_argument("--override", action="append", default=[], help="Hydra override for the data composition")
    parser.add_argument("--stride", type=int, default=10, help="window stride in samples (10 == official 20 Hz updates at 200 Hz)")
    parser.add_argument("--idx"   , type=str, default="0", help="'0' | '3:50' | 'all' -- sequence(s) to evaluate; a single index also writes a .rrd")
    parser.add_argument("--json"  , type=str, default=None, help="write pooled metrics (value+count per key) to this JSON file")
    args = parser.parse_args()

    # Pin the per-fetch transform draws so pooled numbers are reproducible.
    torch.manual_seed(0)

    model = load_model(Path(args.ckpt)).eval().cuda()
    datas = load_sequence(args.data, args.subset, args.override, ckpt=Path(args.ckpt))
    idxs  = _parse_indices(args.idx, len(datas))

    per_seq: list[dict[str, evaluate.MetricPart]] = []
    for i in idxs:
        seq = device_major(datas[i][:], min_len=model.window_size)
        if seq is None:
            print(f"[{i}] skipped: no stream has >= {model.window_size} samples")
            continue
        pred     = tlio_inference(model, seq, stride=args.stride)
        metrices = [evaluate.Velocity_Metric(seq, pred), evaluate.Trajectory_Metric(seq, pred)]

        if len(idxs) == 1:
            rr.init("TLIO_Eval", spawn=False)
            rr.save("./temp/tlio_result.rrd")
            rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(contents=[
                rrb.TextLogView(),
                rrb.Tabs(contents=[x.rr_blueprint() for x in metrices]),
            ], column_shares=[1, 3]), collapse_panels=True))
            for metric in metrices:
                metric.show()
                metric.rr_log()

        take = {}
        for metric in metrices:
            take |= metric.parts
        per_seq.append(take)

    if per_seq:
        pooled = evaluate.merge_parts(per_seq)
        if len(idxs) > 1:
            table = Table(title=f"TLIO eval pooled over {len(per_seq)} sequences ({args.subset}, stride={args.stride})")
            table.add_column("metric"); table.add_column("value"); table.add_column("items")
            for k in sorted(pooled):
                table.add_row(k, f"{pooled[k].value:.6g}", str(pooled[k].count))
            Console().print(table)
        if args.json:
            import json
            # total/count (not just the reduced value) so shards pool exactly.
            payload = {k: {"value": pooled[k].value, "count": pooled[k].count,
                           "total": pooled[k].total, "root": pooled[k].root}
                       for k in pooled}
            Path(args.json).write_text(json.dumps(payload, indent=2))
            print(f"wrote {args.json}")
