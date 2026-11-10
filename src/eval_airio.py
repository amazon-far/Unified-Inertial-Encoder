"""Evaluate a trained AirIO baseline in network mode (GT attitude input) with the ``src.eval`` metrics.

Usage:
    python -m src.eval_airio --ckpt <run>/checkpoints/<name>.ckpt             # one sequence, writes a .rrd
    python -m src.eval_airio --ckpt <run>/checkpoints/<name>.ckpt --idx all   # pooled; or --idx 0:100
"""

import argparse
import functools
import hydra
import torch
import typing as tp

from pathlib import Path
from omegaconf import OmegaConf

from . import evaluate
from .eval import load_sequence
from .eval_tlio import device_major, _parse_indices
from .model.airio import AirIO
from .interface.model import FoundationIMU_Output
from .interface.data_imu import IMUData_w_GT_Jagged


def load_model(ckpt: Path) -> AirIO:
    """Recover the AirIO instance from a ``src.train_airio`` run's config + checkpoint."""
    ckpt    = Path(ckpt).expanduser()
    run_dir = ckpt.parent.parent
    cfg     = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    model: AirIO = hydra.utils.instantiate(cfg.model)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")})
    return model


def frame_centers(model: AirIO, length: int, device: torch.device, n_frames: int | None = None) -> torch.Tensor:
    """Input-sample index supervised by each CNN output frame; ``n_frames`` defaults to the label side."""
    if n_frames is None:
        stride   = tp.cast(list[int], model.net.s_list)[0] * tp.cast(list[int], model.net.s_list)[1]
        n_frames = (length - 1 - 1) // stride + 1
    return model.frame_centers(length, n_frames, device)


@torch.inference_mode()
def airio_net_stream(
    model: AirIO,
    acc  : torch.Tensor, gyro: torch.Tensor, xi: torch.Tensor,   # [L 3] each, fp32
    chunk: int = 200_000,
    pad  : int = 2_400,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the motion network over one stream -> ``(net_vel [K 3], cov [K 3], centers [K])``.

    cuDNN rejects GRU inputs beyond ~60k timesteps, so the stream runs in padded chunks; each
    keeps only the output frames whose supervised sample lies inside its own span.
    """
    device = acc.device
    L      = acc.shape[0]
    vels, covs, cents = [], [], []
    for c0 in range(0, L, chunk):
        c1 = min(c0 + chunk, L)
        p0 = max(c0 - pad, 0)
        p1 = min(c1 + pad, L)
        Ls = p1 - p0
        out  = model.net({"acc": acc[None, p0:p1], "gyro": gyro[None, p0:p1]}, xi[None, p0:p1])
        vel  = out["net_vel"][0].float()
        cov  = out["cov"][0].float()
        cen  = model.frame_centers(Ls, vel.shape[0], device) + p0
        keep = (cen >= c0) & (cen < c1)
        vels.append(vel[keep]); covs.append(cov[keep]); cents.append(cen[keep])
    return torch.cat(vels), torch.cat(covs), torch.cat(cents)


@torch.inference_mode()
def airio_inference(
    model: AirIO,
    seq  : IMUData_w_GT_Jagged,
) -> FoundationIMU_Output:
    """Per-sample body velocity over each device run, as a ``FoundationIMU_Output`` row-aligned with ``seq``."""
    device = next(model.parameters()).device
    s_gpu  = seq.to(device, torch.float32)
    assert isinstance(s_gpu, IMUData_w_GT_Jagged)
    off    = s_gpu.offsets_N
    n_runs = off.numel() - 1

    vel_b = torch.empty(s_gpu.acc.shape[0], 3, device=device)
    for d in range(n_runs):
        s, e = int(off[d]), int(off[d + 1])
        L    = e - s
        row_keep = torch.zeros(s_gpu.acc.shape[0], dtype=torch.bool, device=device)
        row_keep[s:e] = True
        run_keep = torch.zeros(n_runs, dtype=torch.bool, device=device)
        run_keep[d] = True

        acc, gyro, xi = model._body_features(s_gpu, row_keep, run_keep, L)
        net_vel, _, centers = airio_net_stream(
            model, acc.reshape(L, 3), gyro.reshape(L, 3), xi.reshape(L, 3))
        K = net_vel.shape[0]

        # Each sample takes the velocity of the nearest output frame.
        i     = torch.arange(L, device=device)
        right = torch.searchsorted(centers, i, right=True).clamp(1, K - 1) if K > 1 \
                else torch.zeros(L, dtype=torch.int64, device=device)
        left  = right - 1
        pick  = torch.where((i - centers[left]).abs() <= (centers[right] - i).abs(), left, right) if K > 1 \
                else right
        vel_b[s:e] = net_vel[pick]

    pack = s_gpu.acc.shape[0]
    return FoundationIMU_Output(
        trunk_feature    = torch.zeros(0, 1, device=device),
        mask_feature     = torch.zeros(0, dtype=torch.bool, device=device),
        offsets_B_feat   = s_gpu.offsets_B.clone(),
        offsets_N_feat   = s_gpu.offsets_N.clone(),
        dec_correct_acc  = s_gpu.acc.clone(),      # identity corrections: AirIO has no denoise head
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
    parser.add_argument("--ckpt"  , type=str, required=True, help="checkpoint of a src.train_airio run")
    parser.add_argument("--data"  , type=str, default=None, help="*.yaml in ./config/data; default: the ckpt run's own data config")
    parser.add_argument("--subset", type=str, choices=["train", "valid"], default="valid")
    parser.add_argument("--override", action="append", default=[], help="Hydra override for the data composition")
    parser.add_argument("--idx"   , type=str, default="0", help="'0' | '3:50' | 'all' -- sequence(s) to evaluate; a single index also writes a .rrd")
    parser.add_argument("--json"  , type=str, default=None, help="write pooled metrics (value+count per key) to this JSON file")
    parser.add_argument("--per-seq-json", type=str, default=None,
                        help="also write every sequence's own metrics, keyed by its stem (build_corpus "
                             "entries) or config index, to this JSON file")
    args = parser.parse_args()

    # Pin the per-fetch transform draws so pooled numbers are reproducible.
    torch.manual_seed(0)

    model = load_model(Path(args.ckpt)).eval().cuda()
    datas = load_sequence(args.data, args.subset, args.override, ckpt=Path(args.ckpt))
    idxs  = _parse_indices(args.idx, len(datas))

    min_len = 32

    per_seq: list[dict[str, evaluate.MetricPart]] = []
    per_seq_named: dict[str, dict[str, float]] = {}
    for i in idxs:
        seq = device_major(datas[i][:], min_len=min_len)
        if seq is None:
            print(f"[{i}] skipped: no stream has >= {min_len} samples")
            continue
        pred     = airio_inference(model, seq)
        metrices = [evaluate.Velocity_Metric(seq, pred), evaluate.Trajectory_Metric(seq, pred)]

        if len(idxs) == 1:
            rr.init("AirIO_Eval", spawn=False)
            rr.save("./temp/airio_result.rrd")
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
        spec = datas.xs[i]
        name = str(spec.args[-1]) if isinstance(spec, functools.partial) else f"[{i}]"   # nested partial: stem is the LAST positional
        per_seq_named[name] = {k: float(v.value) for k, v in take.items()}
        print(f"[{i}] {name}: vel_l1 {take['vel_l1'].value:.4f}  ate/pred_w_gt_rot {take['ate/pred_w_gt_rot'].value:.3f}")

    if per_seq:
        pooled = evaluate.merge_parts(per_seq)
        if len(idxs) > 1:
            table = Table(title=f"AirIO eval pooled over {len(per_seq)} sequences ({args.subset})")
            table.add_column("metric"); table.add_column("value"); table.add_column("items")
            for k in sorted(pooled):
                table.add_row(k, f"{pooled[k].value:.6g}", str(pooled[k].count))
            Console().print(table)
        if args.per_seq_json:
            import json
            Path(args.per_seq_json).write_text(json.dumps(per_seq_named, indent=2))
            print(f"wrote {args.per_seq_json}")
        if args.json:
            import json
            # total/count (not just the reduced value) so shards pool exactly.
            payload = {k: {"value": pooled[k].value, "count": pooled[k].count,
                           "total": pooled[k].total, "root": pooled[k].root}
                       for k in pooled}
            Path(args.json).write_text(json.dumps(payload, indent=2))
            print(f"wrote {args.json}")
