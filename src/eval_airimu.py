"""Evaluate a trained AirIMU model as a pure-preintegration odometry baseline.

Usage:
    python -m src.eval_airimu --ckpt <airimu_run>/checkpoints/<name>.ckpt --idx all [--json out.json]
"""

import json
import argparse
import torch
import pypose as pp
import typing as tp

from pathlib import Path

from . import evaluate
from .interface.evaluate import MetricPart
from .eval import load_sequence
from .eval_tlio import device_major, _parse_indices
from .eval_airio_ekf import load_airimu
from .model.airimu import AirIMU
from .interface.data_imu import IMUData_w_GT_Jagged, GRAVITY_STANDARD


@torch.inference_mode()
def integrate_stream(
    airimu: AirIMU, seq: IMUData_w_GT_Jagged, device: torch.device, chunk: int = 20_000
) -> list[dict[str, torch.Tensor]]:
    """Correct + preintegrate every device run of one take, in both attitude modes."""
    s_gpu = seq.to(device, torch.float32)
    assert isinstance(s_gpu, IMUData_w_GT_Jagged)
    off, n_runs = s_gpu.offsets_N, s_gpu.offsets_N.numel() - 1
    out = []
    for d in range(n_runs):
        s, e = int(off[d]), int(off[d + 1])
        L    = e - s
        row_keep = torch.zeros(s_gpu.acc.shape[0], dtype=torch.bool, device=device); row_keep[s:e] = True
        run_keep = torch.zeros(n_runs, dtype=torch.bool, device=device); run_keep[d] = True

        acc, gyro = airimu.body_features(s_gpu, row_keep, run_keep, L)
        corr   = airimu.correct_stream(acc.reshape(L, 3), gyro.reshape(L, 3))
        acc_c  = torch.cat([acc.reshape(L, 3)[: airimu.INTERVAL], corr["corrected_acc" ]], dim=0).double()
        gyro_c = torch.cat([gyro.reshape(L, 3)[: airimu.INTERVAL], corr["corrected_gyro"]], dim=0).double()

        t_ns = s_gpu.time_ns[s:e, 0]
        # 1 us floor: some caches carry duplicate/backward timestamps.
        dt   = (t_ns.diff().double() * 1e-9).clamp_min(1e-6).unsqueeze(0).unsqueeze(-1)  # [1 L-1 1]
        gt_R = pp.SO3(s_gpu.gt_pose.rotation().tensor()[s:e].double())
        gt_p = s_gpu.gt_pose.translation()[s:e].double()
        gt_v_w = tp.cast(torch.Tensor, gt_R @ s_gpu.gt_vel[s:e].double())

        est = {}
        for mode in ("inte_rot", "gt_rot"):
            integrator = pp.module.IMUPreintegrator(
                gravity=GRAVITY_STANDARD, prop_cov=False, reset=True
            ).to(device).double()
            init: dict = {
                "pos": gt_p[None, 0:1],
                "rot": pp.SO3(gt_R.tensor()[None, 0:1]),
                "vel": gt_v_w[None, 0:1],
            }
            pos, vel = [gt_p[None, 0:1]], [gt_v_w[None, 0:1]]
            state = None
            for c0 in range(0, L - 1, chunk):
                sel = slice(c0, min(c0 + chunk, L - 1))
                if state is not None:
                    init = {"pos": state["pos"][:, -1:], "vel": state["vel"][:, -1:], "rot": state["rot"][:, -1:]}
                state = integrator(
                    init_state = init,
                    dt         = dt[:, sel],
                    gyro       = gyro_c[None, sel],
                    acc        = acc_c [None, sel],
                    rot        = pp.SO3(gt_R.tensor()[None, sel.start + 1 : sel.stop + 1]) if mode == "gt_rot" else None,
                )
                pos.append(state["pos"][0][None])
                vel.append(state["vel"][0][None])
            est[mode] = {"pos": torch.cat(pos, dim=1)[0], "vel": torch.cat(vel, dim=1)[0]}

        out.append({
            "gt_p": gt_p.cpu(), "gt_v_w": gt_v_w.cpu(),
            **{f"{m}_{k}": est[m][k].cpu() for m in est for k in est[m]},
        })
    return out


if __name__ == "__main__":
    from rich.table import Table
    from rich.console import Console

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt"  , type=str, required=True, help="checkpoint of a src.train_airimu run")
    parser.add_argument("--data"  , type=str, default=None)
    parser.add_argument("--subset", type=str, choices=["train", "valid"], default="valid")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--idx"   , type=str, default="all")
    parser.add_argument("--json"  , type=str, default=None)
    args = parser.parse_args()

    # Pin the per-fetch transform draws so pooled numbers are reproducible.
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    airimu = load_airimu(Path(args.ckpt)).eval().to(device)

    datas = load_sequence(args.data, args.subset, args.override, ckpt=Path(args.ckpt))
    idxs  = _parse_indices(args.idx, len(datas))

    per_seq: list[dict[str, MetricPart]] = []
    for i in idxs:
        seq = device_major(datas[i][:], min_len=32)
        if seq is None:
            continue
        for run in integrate_stream(airimu, seq, device):
            parts: dict[str, MetricPart] = {}
            for mode, ate_key in [("inte_rot", "ate/pred_w_inte_rot"), ("gt_rot", "ate/pred_w_gt_rot")]:
                pos_err = (run[f"{mode}_pos"] - run["gt_p"]).norm(dim=-1)
                vel_err = (run[f"{mode}_vel"] - run["gt_v_w"]).norm(dim=-1)
                parts[ate_key]            = MetricPart(float(pos_err.square().sum()), pos_err.numel(), root=True)
                parts[f"vel_l1/{mode}"]   = MetricPart(float(vel_err.sum()),          vel_err.numel(), root=False)
                parts[f"vel_rmse/{mode}"] = MetricPart(float(vel_err.square().sum()), vel_err.numel(), root=True)
            per_seq.append(parts)

    pooled = evaluate.merge_parts(per_seq)
    table  = Table(title=f"AirIMU preintegration eval pooled over {len(per_seq)} streams ({args.subset})")
    table.add_column("metric"); table.add_column("value"); table.add_column("items")
    for k in sorted(pooled):
        table.add_row(k, f"{pooled[k].value:.6g}", str(pooled[k].count))
    Console().print(table)

    if args.json:
        payload = {k: {"value": pooled[k].value, "count": pooled[k].count,
                       "total": pooled[k].total, "root": pooled[k].root} for k in pooled}
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
