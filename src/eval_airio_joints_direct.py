"""Evaluate the AirIO + joints baseline open-loop, lifting its velocity by AirIMU-integrated attitude.

Usage:
    python -m src.eval_airio_joints_direct --ckpt <joints_run>/checkpoints/<name>.ckpt \
        --airimu_ckpt <airimu_run>/checkpoints/<name>.ckpt --idx all [--json out.json]
"""

import json
import argparse
import torch
import pypose as pp

from pathlib import Path

from . import evaluate
from .interface.evaluate import MetricPart
from .eval import load_sequence
from .eval_tlio import _parse_indices
from .eval_airio_ekf import load_airimu, run_direct
from .eval_airio_joints import load_model, single_stream, airio_joints_stream
from .model.airio_joints import AirIOJoints
from .model.airimu import AirIMU
from .interface.data_joint import IMUJointData_w_GT_Jagged


@torch.inference_mode()
def prepare_stream(
    model: AirIOJoints, airimu: AirIMU, seq: IMUJointData_w_GT_Jagged, device: torch.device,
) -> dict[str, torch.Tensor]:
    """One prepared run in the ``eval_airio_ekf`` format (fp64, cpu)."""
    s_gpu = seq.to(device, torch.float32)
    assert isinstance(s_gpu, IMUJointData_w_GT_Jagged)
    L = s_gpu.acc.shape[0]
    row_keep = torch.ones(L, dtype=torch.bool, device=device)
    run_keep = torch.ones(1, dtype=torch.bool, device=device)

    # AirIMU-corrected gyro -> attitude dead reckoning, as in eval_airio_ekf.prepare_sequence.
    acc, gyro = airimu.body_features(s_gpu, row_keep, run_keep, L)
    corr   = airimu.correct_stream(acc.reshape(L, 3), gyro.reshape(L, 3))
    gyro_c = torch.cat([gyro.reshape(L, 3)[: airimu.INTERVAL], corr["corrected_gyro"]], dim=0)
    t_ns   = s_gpu.time_ns[:, 0]
    dt     = (t_ns.diff().double() * 1e-9).clamp_min(1e-6)
    R0     = s_gpu.gt_pose.rotation().tensor()[0].double()
    ident  = pp.identity_SO3(1, dtype=torch.float64, device=device)
    dR     = pp.so3(gyro_c[:-1].double() * dt[:, None].to(device)).Exp()
    R_rel  = pp.cumprod(pp.SO3(torch.cat([ident.tensor(), dR.tensor()], dim=0)), dim=0)
    R_int  = pp.SO3(R0).unsqueeze(0) @ R_rel                                  # [L 4]

    net_vel, centers = airio_joints_stream(model, s_gpu)
    return {
        "dt"     : dt.cpu(),
        "net_vel": net_vel.double().cpu(),
        "centers": centers.cpu(),
        "R_int"  : R_int.tensor().cpu(),
        "gt_R"   : s_gpu.gt_pose.rotation().tensor().double().cpu(),
        "gt_p"   : s_gpu.gt_pose.translation().double().cpu(),
        "gt_v_b" : s_gpu.gt_vel.double().cpu(),
    }


if __name__ == "__main__":
    from rich.table import Table
    from rich.console import Console

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt"       , type=str, required=True, help="checkpoint of a train_airio_joints run")
    parser.add_argument("--airimu_ckpt", type=str, required=True, help="checkpoint of a src.train_airimu run (same one as the AirIO row)")
    parser.add_argument("--data"  , type=str, default=None)
    parser.add_argument("--subset", type=str, choices=["train", "valid"], default="valid")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--idx"   , type=str, default="all")
    parser.add_argument("--json"  , type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model (Path(args.ckpt)).eval().to(device)
    airimu = load_airimu(Path(args.airimu_ckpt)).eval().to(device)

    datas = load_sequence(args.data, args.subset, args.override, ckpt=Path(args.ckpt))
    idxs  = _parse_indices(args.idx, len(datas))

    per_seq: list[dict[str, MetricPart]] = []
    for i in idxs:
        seq = single_stream(datas[i][:], min_len=32)
        if seq is None:
            continue
        res     = run_direct([prepare_stream(model, airimu, seq, device)])[0]
        pos_err = (res["est_p"] - res["gt_p"]).norm(dim=-1)
        gt_v_w  = pp.SO3(res["gt_R"]) @ res["gt_v_b"]
        vel_err = (res["est_v"] - gt_v_w).norm(dim=-1)
        per_seq.append({
            "ate/direct"     : MetricPart(float(pos_err.square().sum()), pos_err.numel(), root=True),
            "vel_rmse/direct": MetricPart(float(vel_err.square().sum()), vel_err.numel(), root=True),
        })

    pooled = evaluate.merge_parts(per_seq)
    table  = Table(title=f"AirIO+joints direct eval pooled over {len(per_seq)} streams ({args.subset})")
    table.add_column("metric"); table.add_column("value"); table.add_column("items")
    for k in sorted(pooled):
        table.add_row(k, f"{pooled[k].value:.6g}", str(pooled[k].count))
    Console().print(table)
    if args.json:
        payload = {k: {"value": pooled[k].value, "count": pooled[k].count,
                       "total": pooled[k].total, "root": pooled[k].root} for k in pooled}
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
