"""Profile AirIO (CodeNetMotionwithRot) inference stage by stage with ``torch.utils.benchmark``.

Usage:
    python -m src.scripts.airio_profiling [--ckpt <run>/checkpoints/<name>.ckpt] [--device cuda:0] [--out file.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import typing as tp

from ...model.airio import AirIO
from ..effiency_profiling.__main__ import GRAVITY, Stat, time_fn


def make_inputs(rate_hz: float, duration_s: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(acc, gyro, xi)`` each ``[1, T, 3]`` fp32: gravity-carrying accelerometer, gait-like gyro, small attitude log."""
    gen = torch.Generator().manual_seed(seed)
    T = int(round(rate_hz * duration_s))
    t = torch.arange(T, dtype=torch.float32) / rate_hz
    acc = torch.zeros(T, 3); acc[:, 2] = GRAVITY
    acc[:, 0] += 1.5 * torch.sin(2 * torch.pi * 1.8 * t)
    acc += 0.05 * torch.randn(T, 3, generator=gen)
    gyro = 0.3 * torch.sin(2 * torch.pi * 1.8 * t)[:, None].expand(T, 3).clone() + 0.01 * torch.randn(T, 3, generator=gen)
    xi = 0.1 * torch.sin(2 * torch.pi * 0.5 * t)[:, None].expand(T, 3).clone()
    return acc[None], gyro[None], xi[None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, default=None, help="src.train_airio run checkpoint (optional; timing is weight-independent)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate-hz", type=float, default=200.0)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--min-run-time", type=float, default=5.0, help="seconds of samples per stage")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON with the numbers below")
    args = parser.parse_args()

    device = torch.device(args.device)
    T = int(round(args.rate_hz * args.duration_s))
    if args.ckpt is not None:
        from ...eval_airio import load_model
        model = load_model(args.ckpt)
    else:
        model = AirIO(window_size=T)
    model = model.eval().to(device)
    net: tp.Any = model.net   # vendored CodeNetMotionwithRot, untyped
    dtype = model.compute_dtype
    acc, gyro, xi = (x.to(device, dtype) for x in make_inputs(args.rate_hz, args.duration_s))
    data = {"acc": acc, "gyro": gyro}

    # Reference pass: the stage tensors the isolated benchmarks feed on, and a warmup.
    with torch.inference_mode():
        for _ in range(2):
            net(data, xi)
        out = net(data, xi)
        feature = torch.cat([acc, gyro], dim=-1)
        x1 = net.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        x2 = net.ori_encoder(xi.transpose(-1, -2)).transpose(-1, -2)
        codes = torch.cat([x1, x2], dim=-1)
        fused = net.gelu(net.batchnorm2(net.fcn2(codes).transpose(-1, -2)).transpose(-1, -2))
        for g in (net.gru1, net.gru2):
            g.flatten_parameters()
        enc = net.gru2(net.gru1(fused)[0])[0]
    n_frames = int(out["net_vel"].shape[1])

    def cnn():
        a = net.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        b = net.ori_encoder(xi.transpose(-1, -2)).transpose(-1, -2)
        return torch.cat([a, b], dim=-1)

    stages = {
        "cnn encoders (imu + attitude)": cnn,
        "fuse (fcn2 + bn + gelu)":       lambda: net.gelu(net.batchnorm2(net.fcn2(codes).transpose(-1, -2)).transpose(-1, -2)),
        "gru1 + gru2 (bidirectional)":   lambda: net.gru2(net.gru1(fused)[0])[0],
        "decoders (vel + cov)":          lambda: (net.veldecoder(enc), net.velcov_decoder(enc)),
        "end-to-end net forward":        lambda: net(data, xi),
    }
    print(f"model    : AirIO CodeNetMotionwithRot ({'random init' if args.ckpt is None else args.ckpt})")
    print(f"device   : {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}  dtype={dtype}"
          f"  threads={torch.get_num_threads()}  torch={torch.__version__}")
    print(f"input    : 1 IMU x {args.rate_hz:g} Hz x {args.duration_s:g} s = {T} samples -> {n_frames} output frames (stride 9)")
    n_cnn = sum(p.numel() for m in (net.feature_encoder, net.ori_encoder) for p in m.parameters())
    n_gru = sum(p.numel() for g in (net.gru1, net.gru2) for p in g.parameters())
    n_dec = sum(p.numel() for m in (net.veldecoder, net.velcov_decoder) for p in m.parameters())
    n_all = sum(p.numel() for p in net.parameters())
    print(f"params   : {n_all / 1e6:.3f} M (cnn {n_cnn / 1e6:.3f} M, gru {n_gru / 1e6:.3f} M, decoders {n_dec / 1e6:.3f} M, "
          f"other {(n_all - n_cnn - n_gru - n_dec) / 1e6:.3f} M)")
    print()
    header = f"{'stage':36s} {'median ms':>10s} {'mean ms':>9s} {'iqr ms':>8s} {'min ms':>8s} {'max ms':>8s} {'blocks':>6s}"
    print(header)
    print("-" * len(header))
    stats: list[Stat] = []
    for name, fn in stages.items():
        s = time_fn(name, fn, args.min_run_time)
        stats.append(s)
        print(f"{s.name:36s} {s.median_ms:10.3f} {s.mean_ms:9.3f} {s.iqr_ms:8.3f} {s.min_ms:8.3f} {s.max_ms:8.3f} {s.blocks:6d}")
    by = {s.name: s for s in stats}
    print("-" * len(header))
    parts = sum(by[k].median_ms for k in list(stages)[:4])
    e2e = by["end-to-end net forward"].median_ms
    print(f"{'cnn + fuse + gru + decoders':36s} {parts:10.3f}   (end-to-end {e2e:.3f})")
    print(f"real-time factor: {args.duration_s * 1e3 / e2e:.0f}x faster than real time")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "model": "airio-codenet-motion-with-rot", "ckpt": None if args.ckpt is None else str(args.ckpt),
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "dtype": str(dtype), "torch": torch.__version__, "num_threads": torch.get_num_threads(),
            "rate_hz": args.rate_hz, "duration_s": args.duration_s, "num_samples": T, "num_output_frames": n_frames,
            "num_params": n_all, "stages": [asdict(s) for s in stats],
        }, args.out.open("w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
