"""Score the released TartanIMU model zero-shot; the model and weights come from ``src/config/model/tartanimu.yaml``.

Usage:
    python -m src.eval_tartanimu --data_from <baseline_run_dir> \
        --dataset Nymeria_XSens_Processed_IMU_Language --json out.json
    python -m src.eval_tartanimu --data nymeria_full_9imu --idx 0   # a single index also writes a .rrd
"""

import json
import argparse
import functools
import typing as tp

from pathlib import Path

import hydra
import torch

from omegaconf import OmegaConf

from . import evaluate
from .eval import load_sequence
from .eval_tlio import device_major, _parse_indices
from .interface.evaluate import MetricPart
from .interface.model import FoundationIMU_Output
from .interface.data_imu import IMUData_w_GT_Jagged
from .model.tartanimu import TartanIMU, NET_WINDOW, SEQ_LEN
from .scripts.evaluate_velocity.__main__ import valid_pipelines

MODEL_CONFIG = Path(__file__).resolve().parent / "config" / "model" / "tartanimu.yaml"

# Absolute floor in native samples; the real per-stream bound is ``TartanIMU.min_stream_len(step)``.
MIN_STREAM = SEQ_LEN * NET_WINDOW


def build_model(head: str, ckpt: str | None = None) -> TartanIMU:
    """Instantiate ``src/config/model/tartanimu.yaml`` directly, without a whole training config."""
    cfg = OmegaConf.load(MODEL_CONFIG)
    overrides: dict[str, tp.Any] = {"head": head}
    if ckpt is not None:
        overrides["ckpt"] = ckpt
    return hydra.utils.instantiate(cfg, **overrides)


@torch.inference_mode()
def tartanimu_inference(model: TartanIMU, seq: IMUData_w_GT_Jagged) -> FoundationIMU_Output | None:
    """Per-sample body velocity over each device run, as a ``FoundationIMU_Output`` row-aligned with ``seq``.

    Returns None when any run is too short for one ten-window sequence.
    """
    device = next(model.net.parameters()).device
    s_gpu  = seq.to(device, torch.float32)
    assert isinstance(s_gpu, IMUData_w_GT_Jagged)
    off    = s_gpu.offsets_N
    n_runs = off.numel() - 1

    vel_b = torch.empty(s_gpu.acc.shape[0], 3, device=device)
    for d in range(n_runs):
        s, e = int(off[d]), int(off[d + 1])
        L    = e - s
        row_keep    = torch.zeros(s_gpu.acc.shape[0], dtype=torch.bool, device=device)
        row_keep[s:e] = True
        run_keep    = torch.zeros(n_runs, dtype=torch.bool, device=device)
        run_keep[d] = True

        feat = model.body_features(s_gpu, row_keep, run_keep, L)
        step = model.decimation(s_gpu.time_ns[s:e, 0])
        if L < model.min_stream_len(step):
            return None
        out = model.stream(feat, step)
        K   = out.vel.shape[0]
        if K == 0:
            return None

        # Each sample takes the velocity of the nearest anchor; samples before the first inherit it.
        i     = torch.arange(L, device=device)
        right = torch.searchsorted(out.centers, i, right=True).clamp(1, K - 1) if K > 1 \
                else torch.zeros(L, dtype=torch.int64, device=device)
        left  = right - 1
        pick  = torch.where((i - out.centers[left]).abs() <= (out.centers[right] - i).abs(), left, right) if K > 1 \
                else right
        vel_b[s:e] = out.vel[pick].float()

    pack = s_gpu.acc.shape[0]
    return FoundationIMU_Output(
        trunk_feature    = torch.zeros(0, 1, device=device),
        mask_feature     = torch.zeros(0, dtype=torch.bool, device=device),
        offsets_B_feat   = s_gpu.offsets_B.clone(),
        offsets_N_feat   = s_gpu.offsets_N.clone(),
        dec_correct_acc  = s_gpu.acc.clone(),      # identity corrections: TartanIMU has no denoise head
        dec_correct_gyro = s_gpu.gyro.clone(),
        dec_body_vel     = vel_b,
        dec_lang_feature = torch.zeros(1, 1, 1, device=device),
        lang_time_ns     = s_gpu.time_ns[:1, :1].reshape(1, 1),
        time_ns          = s_gpu.time_ns.clone(),
        mask_samp        = torch.zeros(pack, dtype=torch.bool, device=device),
        offsets_B_samp   = s_gpu.offsets_B.clone(),
        offsets_N_samp   = s_gpu.offsets_N.clone(),
    )


if __name__ == "__main__":
    import rerun as rr
    import rerun.blueprint as rrb
    from rich.table import Table
    from rich.console import Console

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_from", type=str, default=None, help="run dir whose .hydra config supplies the valid corpora (odometry-table route)")
    parser.add_argument("--dataset"  , type=str, default=None, help="reader class name key from valid_pipelines (required with --data_from)")
    parser.add_argument("--data"     , type=str, default=None, help="*.yaml in ./config/data to compose instead (ad-hoc route)")
    parser.add_argument("--subset"   , type=str, choices=["train", "valid"], default="valid")
    parser.add_argument("--override" , action="append", default=[], help="Hydra override for the data composition (--data route)")
    parser.add_argument("--idx"      , type=str, default="all", help="'0' | '3:50' | 'all' -- sequence(s) to evaluate (--data route); a single index also writes a .rrd")
    parser.add_argument("--head"     , type=str, default="human", help="which released head reads out the trunk (car/dog/drone/human)")
    parser.add_argument("--ckpt"     , type=str, default=None, help="override the released checkpoint path from the model config")
    parser.add_argument("--limit"    , type=int, default=None, help="evaluate only the first N takes (smoke test; numbers are partial)")
    parser.add_argument("--json"     , type=str, default=None, help="write pooled metrics (value+count per key) to this JSON file")
    args = parser.parse_args()

    assert bool(args.data_from) != bool(args.data), "pass exactly one of --data_from (with --dataset) or --data"
    assert not args.data_from or args.dataset, "--data_from needs --dataset"

    # Pin the per-fetch transform draws so pooled numbers are reproducible.
    torch.manual_seed(0)

    model = build_model(args.head, args.ckpt).eval()
    model = model.cuda() if torch.cuda.is_available() else model

    if args.data_from:
        run_cfg = OmegaConf.load(Path(args.data_from).expanduser() / ".hydra" / "config.yaml")
        groups  = valid_pipelines(run_cfg)
        assert args.dataset in groups, f"--dataset must be one of {sorted(groups)}"
        builders = groups[args.dataset]
    else:
        datas    = load_sequence(args.data, args.subset, args.override)
        builders = [functools.partial(datas.__getitem__, i) for i in _parse_indices(args.idx, len(datas))]
    builders = builders[: args.limit]

    per_seq: list[dict[str, MetricPart]] = []
    short, empty = 0, 0
    for i, build in enumerate(builders):
        seq = build()
        if len(seq) == 0:
            empty += 1                              # take shorter than one pipeline window
            continue
        stream = device_major(seq[:], min_len=MIN_STREAM)
        if stream is None:
            short += 1
            continue
        pred = tartanimu_inference(model, stream)
        if pred is None:
            short += 1                              # under ten windows: no prediction exists
            continue
        metrices = [evaluate.Velocity_Metric(stream, pred), evaluate.Trajectory_Metric(stream, pred)]

        if len(builders) == 1:
            rr.init("TartanIMU_Eval", spawn=False)
            rr.save("./temp/tartanimu_result.rrd")
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

    label = args.dataset or args.data
    print(f"TartanIMU ({args.head} head) on {label}: {len(per_seq)} takes scored, "
          f"{short} skipped as shorter than {SEQ_LEN} windows, {empty} empty")
    if not per_seq:
        raise SystemExit("no take yielded a prediction; nothing to pool")

    pooled = evaluate.merge_parts(per_seq)
    if len(builders) > 1:
        table = Table(title=f"TartanIMU zero-shot ({args.head} head) pooled over {len(per_seq)} takes: {label}")
        table.add_column("metric"); table.add_column("value"); table.add_column("items")
        for k in sorted(pooled):
            table.add_row(k, f"{pooled[k].value:.6g}", str(pooled[k].count))
        Console().print(table)

    if args.json:
        # total/count (not just the reduced value) so shards pool exactly.
        payload: dict[str, tp.Any] = {
            k: {"value": pooled[k].value, "count": pooled[k].count,
                "total": pooled[k].total, "root": pooled[k].root}
            for k in pooled
        }
        payload["_coverage"] = {"takes": len(per_seq), "skipped_short": short, "skipped_empty": empty,
                                "head": args.head, "data": label}
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
