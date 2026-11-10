"""Verify the batched EKF in ``src.eval_airio_ekf`` against the vendored AirIO reference."""

import sys
import importlib.util
from pathlib import Path

import torch
import pypose as pp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_airio_ekf import BatchedEKF, make_dynamics, so3_exp, so3_log  # noqa: E402


def _load_by_path(mod_name: str, rel: str):
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[mod_name] = module
    return module


_ekf_mod   = _load_by_path("_airio_ekf_ref", "thirdparty/airio/EKF/ekf.py")
_state_mod = _load_by_path("_airio_imustate_ref", "thirdparty/airio/EKF/IMUstate.py")


class SingleIMU(_state_mod.IMUstate):
    """Copied verbatim from thirdparty/airio/EKF/IMUofflinerunner.py:29-73."""

    def state_transition(self, state, input, dt, t=None):
        init_rot = pp.so3(state[..., :3]).Exp()
        bg = input[..., 6:9]
        ba = input[..., 9:12]

        w = (input[..., 0:3] - bg)
        a = (input[..., 3:6] - init_rot.Inv() @ self.gravity.double() - ba)
        Dr = pp.so3(w * dt).Exp()
        Dv = Dr @ a * dt
        Dp = Dv * dt + Dr @ a * 0.5 * dt**2
        R = (init_rot @ Dr).Log()
        V = state[..., 3:6] + init_rot @ Dv
        P = state[..., 6:9] + state[..., 3:6] * dt + init_rot @ Dp

        return torch.cat([R, V, P, bg, ba], dim=-1).tensor()  # type: ignore[union-attr]  # verbatim reference code

    def observation(self, state, input, dt, t=None):
        nstate = self.state_transition(state, input, dt)
        rot = pp.so3(nstate[..., :3]).Exp()
        velo = rot.Inv() @ nstate[..., 3:6]
        return velo


GRAVITY = 9.80665


def _synthetic_stream(T: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    dt   = torch.full((T,), 0.005, dtype=torch.float64)
    gyro = 0.4 * torch.randn(T, 3, generator=g, dtype=torch.float64)
    # hovering-ish specific force: gravity reaction + noise
    acc  = torch.tensor([0.0, 0.0, GRAVITY]).expand(T, 3) + 0.5 * torch.randn(T, 3, generator=g, dtype=torch.float64)
    obs  = 0.3 * torch.randn(T, 3, generator=g, dtype=torch.float64)
    g_cov = (0.01 + 0.005 * torch.rand(T, 3, generator=g, dtype=torch.float64))
    a_cov = (0.05 + 0.02  * torch.rand(T, 3, generator=g, dtype=torch.float64))
    o_cov = (0.002 + 0.001 * torch.rand(T, 3, generator=g, dtype=torch.float64))
    return dt, gyro, acc, obs, g_cov, a_cov, o_cov


def _run_reference(T: int):
    dt, gyro, acc, obs, g_cov, a_cov, o_cov = _synthetic_stream(T)
    model = SingleIMU().double()
    model.gravity = torch.tensor([0.0, 0.0, GRAVITY], dtype=torch.float64)
    runner = _ekf_mod.IMUEKF.__new__(_ekf_mod.IMUEKF)   # bypass __init__ (needs Q/R buffers)
    torch.nn.Module.__init__(runner)
    runner.model = model

    x = torch.zeros(15, dtype=torch.float64)
    x[3:6] = torch.tensor([0.1, -0.2, 0.05])
    P = torch.zeros(15, 15, dtype=torch.float64)
    xs, Ps = [], []
    for k in range(T):
        q = torch.cat([g_cov[k], a_cov[k] * 1e2, torch.full((6,), 1e-12, dtype=torch.float64)])
        Q = torch.eye(12, dtype=torch.float64) * q
        u = torch.cat([gyro[k], acc[k], x[9:12], x[12:15]])
        if k % 9 == 4:
            R = torch.eye(3, dtype=torch.float64) * (o_cov[k] * 1e-1)
            x, P = runner.forward(state=x, obs=obs[k], input=u, P=P, dt=dt[k], Q=Q, R=R)
        else:
            x, P = runner.state_propogate(state=x, input=u, P=P, dt=dt[k], Q=Q)
        xs.append(x.clone()); Ps.append(P.clone())
    return torch.stack(xs), torch.stack(Ps)


def _run_batched(T: int, S: int = 3):
    dt, gyro, acc, obs, g_cov, a_cov, o_cov = _synthetic_stream(T)
    ekf = BatchedEKF(GRAVITY, torch.device("cpu"))
    x = torch.zeros(S, 15, dtype=torch.float64)
    x[:, 3:6] = torch.tensor([0.1, -0.2, 0.05])
    P = torch.zeros(S, 15, 15, dtype=torch.float64)
    active = torch.ones(S, dtype=torch.bool)
    xs, Ps = [], []
    for k in range(T):
        u = torch.cat([gyro[k].expand(S, 3), acc[k].expand(S, 3), x[:, 9:12], x[:, 12:15]], dim=-1)
        Q = torch.cat([g_cov[k].expand(S, 3), a_cov[k].expand(S, 3) * 1e2,
                       torch.full((S, 6), 1e-12, dtype=torch.float64)], dim=-1)
        upd = torch.full((S,), k % 9 == 4, dtype=torch.bool)
        x, P = ekf.step(
            x, P, u, dt[k].expand(S), Q,
            obs=obs[k].expand(S, 3) if bool(upd.any()) else None,
            R_obs=(o_cov[k] * 1e-1).expand(S, 3), upd=upd, active=active,
        )
        xs.append(x[0].clone()); Ps.append(P[0].clone())
    return torch.stack(xs), torch.stack(Ps)


def test_so3_roundtrip():
    g = torch.Generator().manual_seed(1)
    v = torch.randn(64, 3, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True) * (torch.rand(64, 1, generator=g, dtype=torch.float64) * 3.0)
    err = (so3_log(so3_exp(v)) - v).norm(dim=-1)
    assert float(err.max()) < 1e-8

    ref = pp.so3(v).Exp().matrix()
    assert float((so3_exp(v) - ref).abs().max()) < 1e-10


def test_dynamics_match_reference():
    f, h = make_dynamics(GRAVITY)
    model = SingleIMU().double()
    model.gravity = torch.tensor([0.0, 0.0, GRAVITY], dtype=torch.float64)
    g = torch.Generator().manual_seed(2)
    for _ in range(16):
        x  = torch.randn(15, generator=g, dtype=torch.float64) * 0.5
        u  = torch.randn(12, generator=g, dtype=torch.float64)
        dt = torch.tensor(0.005, dtype=torch.float64)
        assert float((f(x, u, dt) - model.state_transition(x, u, dt)).abs().max()) < 1e-9
        assert float((h(x, u, dt) - model.observation(x, u, dt)).abs().max()) < 1e-9


def test_filter_matches_reference():
    T = 120
    x_ref, P_ref = _run_reference(T)
    x_bat, P_bat = _run_batched(T)
    x_err = (x_ref - x_bat).abs().max()
    P_err = (P_ref - P_bat).abs().max()
    assert float(x_err) < 1e-7, f"state divergence {x_err}"
    assert float(P_err) < 1e-7, f"covariance divergence {P_err}"
