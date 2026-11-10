"""Evaluate AirIO in EKF mode: a batched port of the official offline EKF on AirIMU-corrected signals.

Usage:
    python -m src.eval_airio_ekf --ckpt <airio_run>/checkpoints/<name>.ckpt \
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
from .eval_tlio import device_major, _parse_indices
from .eval_airio import load_model as load_airio, frame_centers
from .model.airio import AirIO
from .model.airimu import AirIMU
from .interface.data_imu import IMUData_w_GT_Jagged

from omegaconf import OmegaConf
import hydra

# Official fusion weights (EKF/IMUofflinerunner.py) and raw-signal covariances (datasets/dataset.py).
BIAS_WEIGHT   = 1e-12
INPUT_WEIGHT  = 1e2
OBS_WEIGHT    = 1e-1
RAW_ACC_COV   = 0.08
RAW_GYRO_COV  = 0.006


def load_airimu(ckpt: Path) -> AirIMU:
    ckpt    = Path(ckpt).expanduser()
    cfg     = OmegaConf.load(ckpt.parent.parent / ".hydra" / "config.yaml")
    model: AirIMU = hydra.utils.instantiate(cfg.model)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")})
    return model


# SO(3) helpers in pure torch: jacrev/vmap-compatible, so no data-dependent branching.

def so3_hat(v: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(v[..., 0])
    return torch.stack([
        zero, -v[..., 2], v[..., 1],
        v[..., 2], zero, -v[..., 0],
        -v[..., 1], v[..., 0], zero,
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """Rodrigues: so(3) vector -> rotation matrix, Taylor-guarded near zero."""
    theta = phi.norm(dim=-1, keepdim=True).clamp_min(1e-12).unsqueeze(-1)   # [..,1,1]
    K     = so3_hat(phi / theta.squeeze(-1))
    eye   = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    small = (theta < 1e-7)
    s, c  = torch.sin(theta), torch.cos(theta)
    R     = eye + s * K + (1 - c) * (K @ K)
    R_lin = eye + so3_hat(phi)              # first-order fallback for tiny angles
    return torch.where(small, R_lin, R)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> so(3) vector (angle in [0, pi]), Taylor-guarded."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos   = ((trace - 1) / 2).clamp(-1 + 1e-12, 1 - 1e-12)
    theta = torch.acos(cos)[..., None]                                     # [..,1]
    w     = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    small = theta < 1e-6
    scale = torch.where(small, 0.5 + theta**2 / 12, theta / (2 * torch.sin(theta).clamp_min(1e-12)))
    return scale * w


def make_dynamics(gravity: float):
    """Official ``SingleIMU`` dynamics/observation: state [15] = (Log R, v_w, p_w, b_g, b_a),
    input [12] = (gyro, acc, b_g, b_a), all fp64."""
    g_w = torch.tensor([0.0, 0.0, gravity], dtype=torch.float64)

    def f(x: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        R  = so3_exp(x[:3])
        bg, ba = u[6:9], u[9:12]
        w  = u[0:3] - bg
        a  = u[3:6] - R.mT @ g_w.to(x.device) - ba
        Dr = so3_exp(w * dt)
        Dv = (Dr @ a) * dt
        Dp = Dv * dt + (Dr @ a) * 0.5 * dt**2
        xi = so3_log(R @ Dr)
        V  = x[3:6] + R @ Dv
        P  = x[6:9] + x[3:6] * dt + R @ Dp
        return torch.cat([xi, V, P, bg, ba], dim=-1)

    def h(x: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        nx = f(x, u, dt)
        return so3_exp(nx[:3]).mT @ nx[3:6]

    return f, h


class BatchedEKF:
    """The official ``IMUEKF`` step (thirdparty/airio/EKF/ekf.py), vmapped over sequences."""

    def __init__(self, gravity: float, device: torch.device):
        f, h = make_dynamics(gravity)
        self.f, self.h = f, h
        self.jac_f = torch.func.vmap(torch.func.jacrev(f, argnums=(0, 1)))
        self.jac_h = torch.func.vmap(torch.func.jacrev(h, argnums=0))
        self.vf    = torch.func.vmap(f)
        self.vh    = torch.func.vmap(h)
        self.eye15 = torch.eye(15, dtype=torch.float64, device=device)

    def step(
        self,
        x: torch.Tensor, P: torch.Tensor,                 # [S 15], [S 15 15]
        u: torch.Tensor, dt: torch.Tensor,                # [S 12], [S]
        Q: torch.Tensor,                                  # [S 12] (diagonal)
        obs: torch.Tensor | None,                         # [S 3] or None
        R_obs: torch.Tensor | None,                       # [S 3] (diagonal) or None
        upd: torch.Tensor | None,                         # [S] bool: which rows update
        active: torch.Tensor,                             # [S] bool: still inside sequence
    ) -> tuple[torch.Tensor, torch.Tensor]:
        A, B = self.jac_f(x, u, dt)
        xp   = self.vf(x, u, dt)
        Pp   = A @ P @ A.mT + B @ torch.diag_embed(Q) @ B.mT

        if obs is not None:
            assert R_obs is not None and upd is not None
            C  = self.jac_h(x, u, dt)
            Rm = torch.diag_embed(R_obs)
            K  = Pp @ C.mT @ torch.linalg.pinv(C @ Pp @ C.mT + Rm)
            e  = obs - self.vh(x, u, dt)
            x_upd = xp + (K @ e.unsqueeze(-1)).squeeze(-1)
            IKC   = self.eye15 - K @ C
            P_upd = IKC @ Pp @ IKC.mT + K @ Rm @ K.mT
            m3 = upd.view(-1, 1)
            xp = torch.where(m3, x_upd, xp)
            Pp = torch.where(upd.view(-1, 1, 1), P_upd, Pp)

        keep = active.view(-1, 1)
        x = torch.where(keep, xp, x)
        P = torch.where(active.view(-1, 1, 1), Pp, P)
        return x, P


@torch.inference_mode()
def prepare_sequence(
    airio: AirIO, airimu: AirIMU, seq: IMUData_w_GT_Jagged, device: torch.device
) -> list[dict[str, torch.Tensor]]:
    """Network passes for every device run of one take: AirIMU-corrected signals + covs,
    integrated attitude, AirIO velocities + covs, GT states. One dict per run, fp64/cpu."""
    from .interface.data_imu import GRAVITY_STANDARD
    s_gpu = seq.to(device, torch.float32)
    assert isinstance(s_gpu, IMUData_w_GT_Jagged)
    off, n_runs = s_gpu.offsets_N, s_gpu.offsets_N.numel() - 1
    out = []
    for d in range(n_runs):
        s, e = int(off[d]), int(off[d + 1])
        L    = e - s
        row_keep = torch.zeros(s_gpu.acc.shape[0], dtype=torch.bool, device=device); row_keep[s:e] = True
        run_keep = torch.zeros(n_runs, dtype=torch.bool, device=device); run_keep[d] = True

        # Specific force with gravity, shared by both networks; GT orientation only rebuilds the signal.
        acc, gyro = airimu.body_features(s_gpu, row_keep, run_keep, L)
        corr = airimu.correct_stream(acc.reshape(L, 3), gyro.reshape(L, 3))
        # The first INTERVAL samples have no correction: raw signals, default covariances.
        acc_c  = torch.cat([acc.reshape(L, 3)[: airimu.INTERVAL], corr["corrected_acc" ]], dim=0)
        gyro_c = torch.cat([gyro.reshape(L, 3)[: airimu.INTERVAL], corr["corrected_gyro"]], dim=0)
        acc_cov  = torch.cat([torch.full((airimu.INTERVAL, 3), RAW_ACC_COV , device=device), corr["acc_cov" ]], dim=0)
        gyro_cov = torch.cat([torch.full((airimu.INTERVAL, 3), RAW_GYRO_COV, device=device), corr["gyro_cov"]], dim=0)

        # Attitude: corrected-gyro dead-reckoning from the GT initial rotation, not GT attitude.
        t_ns = s_gpu.time_ns[s:e, 0]
        # 1 us floor: some caches carry duplicate/backward timestamps.
        dt   = (t_ns.diff().double() * 1e-9).clamp_min(1e-6)                  # [L-1]
        R0   = s_gpu.gt_pose.rotation().tensor()[s].double()
        # R_{k+1} = R_k * Exp(w_k dt_k), seeded with identity so R_int[0] == R0.
        ident = pp.identity_SO3(1, dtype=torch.float64, device=device)
        dR    = pp.so3(gyro_c[:-1].double() * dt[:, None].to(device)).Exp()
        R_rel = pp.cumprod(pp.SO3(torch.cat([ident.tensor(), dR.tensor()], dim=0)), dim=0)
        R_int = pp.SO3(R0).unsqueeze(0) @ R_rel                               # [L 4]

        # Net inputs stay fp32.
        from .eval_airio import airio_net_stream
        xi = R_int.Log().tensor().float()
        net_vel, net_cov, centers = airio_net_stream(
            airio, acc.reshape(L, 3), gyro.reshape(L, 3), xi.reshape(L, 3))

        gt_R = s_gpu.gt_pose.rotation().tensor()[s:e].double()
        gt_p = s_gpu.gt_pose.translation()[s:e].double()
        gt_v_b = s_gpu.gt_vel[s:e].double()
        out.append({
            "gyro"    : gyro_c.double().cpu(), "acc": acc_c.double().cpu(),
            "gyro_cov": gyro_cov.double().cpu(), "acc_cov": acc_cov.double().cpu(),
            "dt"      : dt.cpu(),
            "net_vel" : net_vel.double().cpu(),
            "net_cov" : net_cov.double().cpu(),
            "centers" : centers.cpu(),
            "R_int"   : R_int.tensor().cpu(),
            "gt_R"    : gt_R.cpu(), "gt_p": gt_p.cpu(), "gt_v_b": gt_v_b.cpu(),
        })
    return out


def run_direct(runs: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Open-loop dead reckoning: nearest-frame velocity rotated by the integrated attitude and
    trapezoid-integrated from the GT initial position. Same result format as ``run_ekf_batch``."""
    results = []
    for r in runs:
        K, L    = r["net_vel"].shape[0], r["gt_p"].shape[0]
        centers = r["centers"]
        i     = torch.arange(L)
        right = torch.searchsorted(centers, i, right=True).clamp(1, K - 1) if K > 1 \
                else torch.zeros(L, dtype=torch.int64)
        left  = right - 1
        pick  = torch.where((i - centers[left]).abs() <= (centers[right] - i).abs(), left, right) if K > 1 \
                else right
        v_w   = pp.SO3(r["R_int"]) @ r["net_vel"][pick]                     # [L 3]
        disp  = 0.5 * (v_w[1:] + v_w[:-1]) * r["dt"][:, None]
        est_p = r["gt_p"][0] + torch.cat([torch.zeros(1, 3, dtype=torch.float64), disp.cumsum(0)])
        results.append({"est_p": est_p, "est_v": v_w,
                        "gt_p": r["gt_p"], "gt_R": r["gt_R"], "gt_v_b": r["gt_v_b"]})
    return results


def run_ekf_batch(
    runs: list[dict[str, torch.Tensor]], device: torch.device, gravity: float
) -> list[dict[str, torch.Tensor]]:
    """Batched EKF over a list of prepared runs; returns per-run estimated trajectories."""
    S    = len(runs)
    Lmax = max(r["dt"].shape[0] for r in runs)          # steps = L-1 per run

    def pad(key: str, width: int, fill: float = 0.0) -> torch.Tensor:
        t = torch.full((S, Lmax, width), fill, dtype=torch.float64)
        for i, r in enumerate(runs):
            n = min(r[key].shape[0], Lmax)
            t[i, :n] = r[key][:n]
        return t

    gyro, acc   = pad("gyro", 3), pad("acc", 3)
    g_cov, a_cov = pad("gyro_cov", 3, RAW_GYRO_COV), pad("acc_cov", 3, RAW_ACC_COV)
    dts  = torch.zeros(S, Lmax, dtype=torch.float64)
    for i, r in enumerate(runs):
        dts[i, : r["dt"].shape[0]] = r["dt"]

    # At step k (propagating sample k -> k+1), update when an AirIO output lands on sample k.
    obs      = torch.zeros(S, Lmax, 3, dtype=torch.float64)
    obs_cov  = torch.full((S, Lmax, 3), 1e-3, dtype=torch.float64)
    has_obs  = torch.zeros(S, Lmax, dtype=torch.bool)
    n_steps  = torch.tensor([r["dt"].shape[0] for r in runs])
    for i, r in enumerate(runs):
        c = r["centers"].clamp_max(n_steps[i] - 1)
        has_obs[i, c] = True
        obs[i, c]     = r["net_vel"]
        obs_cov[i, c] = (r["net_cov"] * OBS_WEIGHT).clamp_min(1e-12)

    # Initial state from GT, P = 0.
    x = torch.zeros(S, 15, dtype=torch.float64)
    for i, r in enumerate(runs):
        R0 = pp.SO3(r["gt_R"][0])
        x[i, 0:3] = R0.Log().tensor()
        x[i, 3:6] = R0 @ r["gt_v_b"][0]
        x[i, 6:9] = r["gt_p"][0]
    P = torch.zeros(S, 15, 15, dtype=torch.float64)

    to = lambda t: t.to(device)
    x, P = to(x), to(P)
    gyro, acc, g_cov, a_cov, dts = to(gyro), to(acc), to(g_cov), to(a_cov), to(dts)
    obs, obs_cov, has_obs = to(obs), to(obs_cov), to(has_obs)
    n_steps = n_steps.to(device)

    ekf   = BatchedEKF(gravity, device)
    est_p = torch.zeros(S, Lmax + 1, 3, dtype=torch.float64, device=device)
    est_v = torch.zeros(S, Lmax + 1, 3, dtype=torch.float64, device=device)
    est_p[:, 0], est_v[:, 0] = x[:, 6:9], x[:, 3:6]

    for k in range(Lmax):
        active = k < n_steps
        u  = torch.cat([gyro[:, k], acc[:, k], x[:, 9:12], x[:, 12:15]], dim=-1)
        Q  = torch.cat([
            g_cov[:, k],
            a_cov[:, k] * INPUT_WEIGHT,
            torch.full((S, 6), BIAS_WEIGHT, dtype=torch.float64, device=device),
        ], dim=-1)
        upd = has_obs[:, k] & active
        x, P = ekf.step(
            x, P, u, dts[:, k], Q,
            obs=obs[:, k] if bool(upd.any()) else None,
            R_obs=obs_cov[:, k], upd=upd, active=active,
        )
        est_p[:, k + 1], est_v[:, k + 1] = x[:, 6:9], x[:, 3:6]

    results = []
    for i, r in enumerate(runs):
        n = int(n_steps[i]) + 1
        results.append({"est_p": est_p[i, :n].cpu(), "est_v": est_v[i, :n].cpu(),
                        "gt_p": r["gt_p"][:n], "gt_R": r["gt_R"][:n], "gt_v_b": r["gt_v_b"][:n]})
    return results


if __name__ == "__main__":
    from rich.table import Table
    from rich.console import Console
    from .interface.data_imu import GRAVITY_STANDARD

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt"       , type=str, required=True, help="checkpoint of a src.train_airio run")
    parser.add_argument("--airimu_ckpt", type=str, required=True, help="checkpoint of a src.train_airimu run")
    parser.add_argument("--data"  , type=str, default=None)
    parser.add_argument("--subset", type=str, choices=["train", "valid"], default="valid")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--idx"   , type=str, default="all")
    parser.add_argument("--batch" , type=int, default=256, help="EKF sequences per batched filter pass")
    parser.add_argument("--mode"  , type=str, choices=["ekf", "direct"], default="ekf",
                        help="'ekf' = the official filter; 'direct' = open-loop trapezoid integration of the predicted velocity with the same integrated attitude")
    parser.add_argument("--json"  , type=str, default=None, help="write pooled metrics to this JSON file")
    args = parser.parse_args()

    # Pin the per-fetch transform draws so pooled numbers are reproducible.
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    airio  = load_airio (Path(args.ckpt)).eval().to(device)
    airimu = load_airimu(Path(args.airimu_ckpt)).eval().to(device)

    datas = load_sequence(args.data, args.subset, args.override, ckpt=Path(args.ckpt))
    idxs  = _parse_indices(args.idx, len(datas))

    runs: list[dict[str, torch.Tensor]] = []
    for i in idxs:
        seq = device_major(datas[i][:], min_len=32)
        if seq is None:
            continue
        runs += prepare_sequence(airio, airimu, seq, device)

    per_seq: list[dict[str, MetricPart]] = []
    for c0 in range(0, len(runs), args.batch):
        chunk = runs[c0 : c0 + args.batch]
        ress  = run_ekf_batch(chunk, device, GRAVITY_STANDARD) if args.mode == "ekf" else run_direct(chunk)
        for res in ress:
            pos_err = (res["est_p"] - res["gt_p"]).norm(dim=-1)
            gt_v_w  = pp.SO3(res["gt_R"]) @ res["gt_v_b"]
            vel_err = (res["est_v"] - gt_v_w).norm(dim=-1)
            per_seq.append({
                f"ate/{args.mode}"     : MetricPart(float(pos_err.square().sum()), pos_err.numel(), root=True),
                f"vel_rmse/{args.mode}": MetricPart(float(vel_err.square().sum()), vel_err.numel(), root=True),
            })

    pooled = evaluate.merge_parts(per_seq)
    table  = Table(title=f"AirIO {args.mode} eval pooled over {len(per_seq)} streams ({args.subset})")
    table.add_column("metric"); table.add_column("value"); table.add_column("items")
    for k in sorted(pooled):
        table.add_row(k, f"{pooled[k].value:.6g}", str(pooled[k].count))
    Console().print(table)

    if args.json:
        payload = {k: {"value": pooled[k].value, "count": pooled[k].count,
                       "total": pooled[k].total, "root": pooled[k].root} for k in pooled}
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
