"""Evaluate a trained AirIO + joints baseline with the ``src.eval`` metrics.

Usage:
    python -m src.eval_airio_joints --ckpt <run>/checkpoints/<name>.ckpt [--idx all] [--json out.json]
"""

import argparse
import hydra
import torch
import pypose as pp

from pathlib import Path
from omegaconf import OmegaConf

from . import evaluate
from .eval import load_sequence
from .eval_tlio import _parse_indices
from .model.airio_joints import AirIOJoints
from .interface.model import FoundationIMU_Output
from .interface.data_imu import IMUData_BaseTrait_Jagged
from .interface.data_joint import IMUJointData_w_GT_Jagged


def load_model(ckpt: Path) -> AirIOJoints:
    """Recover the ``AirIOJoints`` instance from a run's config + checkpoint."""
    ckpt    = Path(ckpt).expanduser()
    run_dir = ckpt.parent.parent
    cfg     = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    model: AirIOJoints = hydra.utils.instantiate(cfg.model)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")})
    return model


def single_stream(batch: IMUData_BaseTrait_Jagged, min_len: int) -> IMUJointData_w_GT_Jagged | None:
    """Reorder a window-major one-device (pelvis) pack into one time-sorted stream with its joints.

    Returns None for streams shorter than ``min_len``.
    """
    assert isinstance(batch, IMUJointData_w_GT_Jagged), \
        f"AirIOJoints eval needs IMUJointData_w_GT_Jagged packs, got {type(batch).__name__}"
    assert torch.equal(batch.offsets_N, batch.offsets_B), "expected exactly one device per window"
    assert batch.sensor_id[:, 0].unique().numel() == 1, "expected a single (pelvis) device"

    L = batch.acc.shape[0]
    if L < min_len:
        return None
    perm = batch.time_ns[:, 0].argsort()
    offs = torch.tensor([0, L], dtype=torch.int64)
    return IMUJointData_w_GT_Jagged(
        time_ns      = batch.time_ns[perm],
        acc          = batch.acc[perm],
        gyro         = batch.gyro[perm],
        correct_mask = batch.correct_mask[perm],
        correct_acc  = batch.correct_acc[perm],
        correct_gyro = batch.correct_gyro[perm],
        gravity         = batch.gravity[:1],
        correct_gravity = batch.correct_gravity[:1],
        acc_bias     = batch.acc_bias[perm],
        gyro_bias    = batch.gyro_bias[perm],
        offsets_B    = offs,
        offsets_N    = offs.clone(),
        sensor_id    = batch.sensor_id[perm],
        model        = batch.model,
        kind         = batch.kind,
        labels       = batch.labels[:1],
        gt_pose      = pp.SE3(batch.gt_pose.tensor()[perm]),
        gt_vel       = batch.gt_vel[perm],
        joint_q      = batch.joint_q[perm],
        joint_dq     = batch.joint_dq[perm],
        offsets_J    = offs.clone(),
        joint_names  = batch.joint_names,
        joint_kind   = batch.joint_kind,
    )


@torch.inference_mode()
def airio_joints_stream(
    model: AirIOJoints, seq: IMUJointData_w_GT_Jagged,
    chunk: int = 200_000, pad: int = 2_400,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the network over one stream in overlapping chunks -> ``(net_vel [M 3], anchors [M])``."""
    device = seq.acc.device
    L      = seq.acc.shape[0]
    all_rows = torch.ones(L, dtype=torch.bool, device=device)
    acc, gyro = model.body_features(seq, all_rows, torch.ones(1, dtype=torch.bool, device=device), L)
    vels, cents = [], []
    for c0 in range(0, L, chunk):
        c1 = min(c0 + chunk, L)
        p0 = max(c0 - pad, 0)
        p1 = min(c1 + pad, L)
        sl = slice(p0, p1)
        x  = model.net_input(
            acc[sl][None], gyro[sl][None], seq.time_ns[sl][None],
            seq.joint_q[sl][None], seq.joint_dq[sl][None], seq.joint_kind,
        )
        vel  = model.net(x).vel_b[0].float()
        cen  = model.anchors(p1 - p0, device) + p0
        keep = (cen >= c0) & (cen < c1)
        vels.append(vel[keep]); cents.append(cen[keep])
    return torch.cat(vels), torch.cat(cents)


@torch.inference_mode()
def airio_joints_inference(model: AirIOJoints, seq: IMUJointData_w_GT_Jagged) -> FoundationIMU_Output:
    device = next(model.parameters()).device
    s_gpu  = seq.to(device, torch.float32)
    assert isinstance(s_gpu, IMUJointData_w_GT_Jagged)
    L = s_gpu.acc.shape[0]

    net_vel, centers = airio_joints_stream(model, s_gpu)
    K = net_vel.shape[0]
    i     = torch.arange(L, device=device)
    right = torch.searchsorted(centers, i, right=True).clamp(1, K - 1) if K > 1 \
            else torch.zeros(L, dtype=torch.int64, device=device)
    left  = right - 1
    pick  = torch.where((i - centers[left]).abs() <= (centers[right] - i).abs(), left, right) if K > 1 else right
    vel_b = net_vel[pick]

    return FoundationIMU_Output(
        trunk_feature    = torch.zeros(0, 1, device=device),
        mask_feature     = torch.zeros(0, dtype=torch.bool, device=device),
        offsets_B_feat   = s_gpu.offsets_B.clone(),
        offsets_N_feat   = s_gpu.offsets_N.clone(),
        dec_correct_acc  = s_gpu.acc.clone(),
        dec_correct_gyro = s_gpu.gyro.clone(),
        dec_body_vel     = vel_b,
        dec_lang_feature = torch.zeros(1, 1, 1, device=device),
        lang_time_ns     = s_gpu.time_ns[:1, :1].reshape(1, 1),
        time_ns          = s_gpu.time_ns.clone(),
        mask_samp        = torch.zeros(L, dtype=torch.bool, device=device),
        offsets_B_samp   = s_gpu.offsets_B.clone(),
        offsets_N_samp   = s_gpu.offsets_N.clone(),
    )


if __name__ == "__main__":
    import rerun as rr
    import rerun.blueprint as rrb
    from rich.table import Table
    from rich.console import Console

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt"  , type=str, required=True, help="checkpoint of a train_airio_joints run")
    parser.add_argument("--data"  , type=str, default=None, help="*.yaml in ./config/data; default: the ckpt run's own data config")
    parser.add_argument("--subset", type=str, choices=["train", "valid"], default="valid")
    parser.add_argument("--override", action="append", default=[], help="Hydra override for the data composition")
    parser.add_argument("--idx"   , type=str, default="0", help="'0' | '3:50' | 'all' -- sequence(s) to evaluate; a single index also writes a .rrd")
    parser.add_argument("--json"  , type=str, default=None, help="write pooled metrics (value+count per key) to this JSON file")
    args = parser.parse_args()

    torch.manual_seed(0)   # pins the per-fetch corruption draws

    model = load_model(Path(args.ckpt)).eval().cuda()
    datas = load_sequence(args.data, args.subset, args.override, ckpt=Path(args.ckpt))
    idxs  = _parse_indices(args.idx, len(datas))
    min_len = 32

    per_seq: list[dict[str, evaluate.MetricPart]] = []
    for i in idxs:
        seq = single_stream(datas[i][:], min_len=min_len)
        if seq is None:
            print(f"[{i}] skipped: stream shorter than {min_len} samples")
            continue
        pred     = airio_joints_inference(model, seq)
        metrices = [evaluate.Velocity_Metric(seq, pred), evaluate.Trajectory_Metric(seq, pred)]

        if len(idxs) == 1:
            rr.init("AirIOJoints_Eval", spawn=False)
            rr.save("./temp/airio_joints_result.rrd")
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
            table = Table(title=f"AirIO+joints eval pooled over {len(per_seq)} sequences ({args.subset})")
            table.add_column("metric"); table.add_column("value"); table.add_column("items")
            for k in sorted(pooled):
                table.add_row(k, f"{pooled[k].value:.6g}", str(pooled[k].count))
            Console().print(table)
        if args.json:
            import json
            payload = {k: {"value": pooled[k].value, "count": pooled[k].count,
                           "total": pooled[k].total, "root": pooled[k].root} for k in pooled}
            Path(args.json).write_text(json.dumps(payload, indent=2))
            print(f"wrote {args.json}")
