"""Fit a Chinchilla-style scaling law L(N, D) = E + A / N^alpha + B / D^beta to a model-size sweep.

Usage:
    python -m src.scripts.fit_scaling_law --runs temp/exp22/runs.json   # writes temp/exp22/fit/
    python -m src.scripts.fit_scaling_law --synthetic                   # self-test on a known law
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import minimize

STEPS_PER_EPOCH   = 321
BATCH_PER_GPU     = 128
WORLD_SIZE        = 8
BATCH             = BATCH_PER_GPU * WORLD_SIZE
WINDOW_S          = 4.096
TOKENS_PER_WINDOW = 12 * 33 + 5          # 12 sites x 33 tokens of 125 ms + 4 registers + 1 global
TOKENS_PER_S      = TOKENS_PER_WINDOW / WINDOW_S
C_ANCHOR          = 6 * 55.56e6 * 6420 * BATCH * TOKENS_PER_WINDOW   # reference compute


def load_points(registry: dict, metric: str, min_epoch: int, max_passes: float, unique_hours: float):
    """registry: {name: {"N": params, "wandb": run path or null, "csv": path or null, "epochs": max}}."""
    import pandas as pd
    pts = []
    for name, r in registry.items():
        if r.get("csv"):
            h = pd.read_csv(r["csv"]).set_index("epoch")
        else:
            import wandb
            api = wandb.Api(timeout=120)
            run = api.run(r["wandb"])
            h = pd.DataFrame(run.history(keys=["epoch", "trainer/global_step", metric], samples=50000, pandas=True))
            h = h.dropna(subset=[metric]).groupby("epoch").last()
            os.makedirs("temp/exp22/hist", exist_ok=True)
            h.to_csv(f"temp/exp22/hist/{name}.csv")
        for ep, loss in zip(h.index.to_numpy(dtype=float), h[metric].to_numpy(dtype=float)):
            steps  = (ep + 1) * STEPS_PER_EPOCH
            hours  = steps * BATCH * WINDOW_S / 3600
            if ep < min_epoch or hours / unique_hours > max_passes:
                continue
            pts.append((name, float(r["N"]), steps * BATCH * TOKENS_PER_WINDOW, float(loss)))
    return pts


def fit_law(N, D, L, huber_delta=1e-3, seed=0, init=None):
    """init: a previous fit dict -> start from it plus a few perturbations (fast bootstrap)."""
    lN, lD, lL = np.log(N), np.log(D), np.log(L)

    def model(p):
        a, b, e, al, be = p   # log A, log B, log E, alpha, beta
        return np.logaddexp(np.logaddexp(a - al * lN, b - be * lD), e)

    def huber(r):
        return np.where(np.abs(r) <= huber_delta, 0.5 * r * r, huber_delta * (np.abs(r) - 0.5 * huber_delta))

    def obj(p):
        return huber(model(p) - lL).sum()

    best = None
    rng = np.random.default_rng(seed)
    if init is not None:
        p = np.array([np.log(init["A"]), np.log(init["B"]), np.log(init["E"]), init["alpha"], init["beta"]])
        grid = [p] + [p + rng.normal(0, [1, 1, 0.3, 0.1, 0.1]) for _ in range(3)]
    else:
        grid = [(a, b, e, al, be) for a in (0, 6) for b in (0, 8)
                for e in (np.log(L.min()) - 0.7, np.log(L.min()) - 0.1)
                for al in (0.15, 0.4, 0.8) for be in (0.15, 0.4, 0.8)]
    for p0 in grid:
        res = minimize(obj, np.array(p0, float), method="L-BFGS-B",
                       bounds=[(-30, 30), (-30, 30), (None, np.log(L.min())), (1e-3, 3), (1e-3, 3)])
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    a, b, e, al, be = best.x
    return dict(A=float(np.exp(a)), B=float(np.exp(b)), E=float(np.exp(e)), alpha=float(al), beta=float(be), obj=float(best.fun))


def optimum(fit, C):
    al, be, A, B = fit["alpha"], fit["beta"], fit["A"], fit["B"]
    a, b = be / (al + be), al / (al + be)
    G = (al * A / (be * B)) ** (1 / (al + be))
    N_opt = G * (C / 6) ** a
    D_opt = (C / 6) ** b / G
    return dict(a=a, b=b, G=G, N_opt=N_opt, D_opt=D_opt,
                tokens_per_param=D_opt / N_opt, seconds_per_param=D_opt / N_opt / TOKENS_PER_S,
                params_per_second=N_opt / (D_opt / TOKENS_PER_S))


def isoflop(pts, levels):
    out = []
    for C in levels:
        rows = {}
        for name, N, D, L in pts:
            c = 6 * N * D
            if abs(np.log(c / C)) < np.log(1.25):    # within +-25% of the level
                if name not in rows or abs(np.log(c / C)) < rows[name][2]:
                    rows[name] = (N, L, abs(np.log(c / C)))
        if len(rows) >= 3:
            x = np.log([v[0] for v in rows.values()]); y = np.array([v[1] for v in rows.values()])
            q = np.polyfit(x, y, 2)
            n_opt = float(np.exp(-q[1] / (2 * q[0]))) if q[0] > 0 else float("nan")
            out.append(dict(C=C, n_runs=len(rows), N_opt=n_opt, D_opt=C / 6 / n_opt if n_opt == n_opt else float("nan")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="temp/exp22/runs.json")
    ap.add_argument("--metric", default="valid_ema/copy/vel_loss")
    ap.add_argument("--min-epoch", type=int, default=2)
    ap.add_argument("--max-passes", type=float, default=32.0,
                    help="Maximum processed / unique corpus duration, across all GPUs")
    ap.add_argument("--unique-hours", type=float, default=629.0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="temp/exp22/fit")
    args = ap.parse_args()

    if args.synthetic:
        rng = np.random.default_rng(0)
        true = dict(A=40.0, B=30.0, E=0.02, alpha=0.4, beta=0.3)
        pts = []
        for N in (2.8e6, 4.7e6, 14.6e6, 25e6, 55.6e6, 123e6, 274e6):
            for ep in range(2, 20):
                D = (ep + 1) * STEPS_PER_EPOCH * BATCH * TOKENS_PER_WINDOW
                L = true["E"] + true["A"] / N ** true["alpha"] + true["B"] / D ** true["beta"]
                pts.append((f"N{N:.0f}", N, D, L * np.exp(rng.normal(0, 0.01))))
        print("synthetic truth:", true, "->", {k: round(v, 3) for k, v in optimum(true, C_ANCHOR).items() if k in ("a", "b", "tokens_per_param")})
    else:
        registry = json.load(open(args.runs))
        pts = load_points(registry, args.metric, args.min_epoch, args.max_passes, args.unique_hours)

    names = np.array([p[0] for p in pts]); N = np.array([p[1] for p in pts]); D = np.array([p[2] for p in pts]); L = np.array([p[3] for p in pts])
    print(f"{len(pts)} points from {len(set(names))} runs")
    fit = fit_law(N, D, L)
    opt = optimum(fit, C_ANCHOR)
    print("fit:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fit.items()})
    print("at C_anchor:", {k: (f"{v:.4g}") for k, v in opt.items()})

    boots = []
    uniq = sorted(set(names))
    rng = np.random.default_rng(0)
    for _ in range(args.bootstrap):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        m = np.concatenate([np.where(names == u)[0] for u in pick])
        f = fit_law(N[m], D[m], L[m], init=fit); boots.append(optimum(f, C_ANCHOR) | f)
    summary = {}
    for k in ("alpha", "beta", "E", "a", "b", "tokens_per_param", "seconds_per_param", "params_per_second"):
        v = np.array([b[k] for b in boots]); summary[k] = dict(point=(fit | opt)[k], lo=float(np.percentile(v, 5)), hi=float(np.percentile(v, 95)))
    for k, v in summary.items():
        print(f"  {k:18s} {v['point']:10.4g}   [{v['lo']:.4g}, {v['hi']:.4g}]")

    iso = isoflop(pts, [C_ANCHOR * f for f in (1/8, 1/4, 1/2, 1, 2)])
    for r in iso:
        print(f"  isoflop C={r['C']:.3g}: {r['n_runs']} runs, N_opt={r['N_opt']:.3g}, D_opt/N_opt={r['D_opt']/r['N_opt'] if r['N_opt']==r['N_opt'] else float('nan'):.3g} tok/param")

    os.makedirs(args.out, exist_ok=True)
    accounting = dict(batch_per_gpu=BATCH_PER_GPU, world_size=WORLD_SIZE, global_batch=BATCH,
                      tokens_per_nominal_window=TOKENS_PER_WINDOW, nominal_window_seconds=WINDOW_S,
                      steps_per_epoch=STEPS_PER_EPOCH, compute_proxy="6*N*D_tokens")
    selection = dict(min_epoch_zero_based=args.min_epoch, max_processed_corpus_passes=args.max_passes,
                     nominal_corpus_hours=args.unique_hours)
    json.dump(dict(fit=fit, optimum_at_anchor=opt, bootstrap=summary, isoflop=iso,
                   n_points=len(pts), metric=args.metric, accounting=accounting, fit_selection=selection),
              open(os.path.join(args.out, "fit.json"), "w"), indent=2)
    np.savetxt(os.path.join(args.out, "points.csv"), np.c_[N, D, L], delimiter=",", header="N,D_tokens,loss", comments="")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
