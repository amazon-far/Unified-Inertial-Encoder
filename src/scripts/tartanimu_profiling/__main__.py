"""Profile the released TartanIMU model stage by stage with ``torch.utils.benchmark``.

Usage:
    python -m src.scripts.tartanimu_profiling [--device cuda:0] [--out file.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ...eval_tartanimu import build_model
from ...model.tartanimu import NET_WINDOW, SEQ_LEN, TartanIMU, _attr
from ..effiency_profiling.__main__ import GRAVITY, Stat, time_fn


def make_feat(rate_hz: float, duration_s: float, seed: int = 0) -> torch.Tensor:
    """Body-frame ``[gyro; acc]`` (official channel order), gravity kept in the accelerometer."""
    gen = torch.Generator().manual_seed(seed)
    T = int(round(rate_hz * duration_s))
    t = torch.arange(T, dtype=torch.float32) / rate_hz
    gyro = 0.3 * torch.sin(2 * torch.pi * 1.8 * t)[:, None].expand(T, 3).clone() + 0.01 * torch.randn(T, 3, generator=gen)
    acc = torch.zeros(T, 3)
    acc[:, 2] = GRAVITY
    acc[:, 0] += 1.5 * torch.sin(2 * torch.pi * 1.8 * t)
    acc += 0.05 * torch.randn(T, 3, generator=gen)
    return torch.cat([gyro, acc], dim=-1)   # [T, 6]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head", default="human", choices=["car", "dog", "drone", "human"])
    parser.add_argument("--rate-hz", type=float, default=200.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--min-run-time", type=float, default=5.0, help="seconds of samples per stage")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON with the numbers below")
    args = parser.parse_args()

    device = torch.device(args.device)
    model: TartanIMU = build_model(args.head).eval().to(device)
    dtype = model.compute_dtype
    feat = make_feat(args.rate_hz, args.duration_s).to(device)
    T = feat.shape[0]

    # Reference pass: the stage tensors the isolated benchmarks feed on, and a warmup.
    fake_ns = (torch.arange(T, dtype=torch.float64) * (1e9 / args.rate_hz)).round().long().to(device)
    step = model.decimation(fake_ns)
    trunk, head = _attr(model.net, "model"), _attr(model.net, "heads")[args.head]
    with torch.inference_mode():
        y = feat[::step].to(dtype)
        n_win = y.shape[0] - NET_WINDOW + 1
        n_seq = n_win - (SEQ_LEN - 1) * NET_WINDOW
        assert n_seq >= 1, f"{args.duration_s}s @ {args.rate_hz} Hz is below TartanIMU's {SEQ_LEN}-window warm-up"
        windows = y.unfold(0, NET_WINDOW, 1).contiguous()                      # [n_win, 6, NET_WINDOW]
        embed = model._encode(windows)                                          # [n_win, C]
        trunk.lstm.flatten_parameters()
        off = torch.arange(SEQ_LEN, device=device) * NET_WINDOW
        seq_embed = embed[torch.arange(n_seq, device=device)[:, None] + off[None, :]]   # [n_seq, SEQ_LEN, C]
        last = trunk.lstm(seq_embed)[0][:, -1].contiguous()
        one_window = windows[-1:].contiguous()
        for _ in range(3):
            model.stream(feat, step)

    def incremental():
        # One new prediction in a warm stream: encode the newest window, LSTM over its sequence, head.
        e = model._encode(one_window)
        seq = torch.cat([embed[-(SEQ_LEN - 1) * NET_WINDOW::NET_WINDOW][: SEQ_LEN - 1], e])[None]   # [1, SEQ_LEN, C]
        o = trunk.lstm(seq)[0][:, -1].contiguous()
        return head(o, 1, 1)

    stages = {
        "decimate + unfold":                 lambda: feat[::step].to(dtype).unfold(0, NET_WINDOW, 1).contiguous(),
        "resnet encode (all windows)":       lambda: model._encode(windows),
        "resnet encode (1 window)":          lambda: model._encode(one_window),
        "lstm (1 sequence x 10 windows)":    lambda: trunk.lstm(seq_embed),
        "head (human)":                      lambda: head(last, last.shape[0], 1),
        "end-to-end stream()":               lambda: model.stream(feat, step),
        "incremental per-prediction":        incremental,
    }
    print(f"model    : TartanIMU unified.pt, head={args.head}")
    print(f"device   : {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}  dtype={dtype}"
          f"  threads={torch.get_num_threads()}  torch={torch.__version__}")
    print(f"input    : 1 IMU x {args.rate_hz:g} Hz x {args.duration_s:g} s = {T} samples -> step {step} -> {y.shape[0]} net samples"
          f" -> {n_win} windows of {NET_WINDOW} -> {n_seq} sequence(s) of {SEQ_LEN} -> {n_seq} prediction(s)")
    print(f"params   : {sum(p.numel() for p in model.net.parameters()) / 1e6:.2f} M "
          f"(resnet {sum(p.numel() for n, p in trunk.named_parameters() if not n.startswith('lstm')) / 1e6:.2f} M, "
          f"lstm {sum(p.numel() for p in trunk.lstm.parameters()) / 1e6:.2f} M, heads x4 {sum(p.numel() for p in _attr(model.net, 'heads').parameters()) / 1e6:.2f} M)")
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
    e2e = by["end-to-end stream()"].median_ms
    print(f"{'encode + lstm + head':36s} {by['resnet encode (all windows)'].median_ms + by['lstm (1 sequence x 10 windows)'].median_ms + by['head (human)'].median_ms:10.3f}"
          f"   (end-to-end {e2e:.3f})")
    print(f"real-time factor: {args.duration_s * 1e3 / e2e:.0f}x faster than real time for the 10 s warm-up; "
          f"steady state one prediction per {step / args.rate_hz * 1e3:.0f} ms costs {by['incremental per-prediction'].median_ms:.3f} ms")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "model": "tartanimu-unified", "head": args.head, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "dtype": str(dtype), "torch": torch.__version__, "num_threads": torch.get_num_threads(),
            "rate_hz": args.rate_hz, "duration_s": args.duration_s, "num_samples": T, "step": step,
            "num_windows": n_win, "num_sequences": n_seq, "stages": [asdict(s) for s in stats],
        }, args.out.open("w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
