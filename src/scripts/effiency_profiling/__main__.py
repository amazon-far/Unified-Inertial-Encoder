"""Profile FoundationIMU inference stage by stage on a synthetic multi-IMU stream with ``torch.utils.benchmark``.

Usage:
    python -m src.scripts.effiency_profiling --ckpt <run>/checkpoints/<name>.ckpt [--device cuda:0] [--trt] [--out file.json]
"""
from __future__ import annotations

from src.runtime_prologue import machine_conf

import argparse
import json
import math
import platform
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark

from ...eval import load_model
from ...interface.data import ArticulatedModel
from ...interface.data_imu import IMUData_Jagged
from ...interface.human import CANONICAL_IDX
from ...interface.joint import JOINT_SITES
from ...model.model import FoundationIMU

PAPER_CKPT = Path(machine_conf('exp_root')) / 'unified-inertial-encoder/checkpoints/unified_encoder.ckpt'
GRAVITY = 9.80665


def make_stream(num_imu: int, rate_hz: float, duration_s: float, seed: int = 0) -> IMUData_Jagged:
    """One sample (B=1) of ``num_imu`` devices, each ``rate_hz * duration_s`` samples long."""
    assert num_imu <= len(JOINT_SITES), f"at most {len(JOINT_SITES)} canonical sites available"
    gen = torch.Generator().manual_seed(seed)
    T = int(round(rate_hz * duration_s))
    N = num_imu
    pack = N * T
    dt_ns = int(round(1e9 / rate_hz))

    time_ns = (torch.arange(T, dtype=torch.int64) * dt_ns).repeat(N).view(pack, 1)
    t = torch.arange(T, dtype=torch.float32) / rate_hz
    acc = torch.zeros(N, T, 3)
    acc[..., 2] = -GRAVITY
    acc[..., 0] += 1.5 * torch.sin(2 * math.pi * 1.8 * t)
    acc += 0.05 * torch.randn(N, T, 3, generator=gen)
    gyro = 0.3 * torch.sin(2 * math.pi * 1.8 * t)[None, :, None].expand(N, T, 3).clone()
    gyro += 0.01 * torch.randn(N, T, 3, generator=gen)
    acc, gyro = acc.reshape(pack, 3), gyro.reshape(pack, 3)

    names = JOINT_SITES[:N]
    sensor_id = torch.tensor([CANONICAL_IDX[n] for n in names], dtype=torch.uint8).repeat_interleave(T).view(pack, 1)
    gravity = torch.tensor([0.0, 0.0, -GRAVITY]).expand(N, 3).clone()
    return IMUData_Jagged(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        correct_mask=torch.zeros(pack, dtype=torch.bool),
        correct_acc=acc.clone(),
        correct_gyro=gyro.clone(),
        gravity=gravity,
        correct_gravity=gravity.clone(),
        acc_bias=torch.zeros(pack, 3),
        gyro_bias=torch.zeros(pack, 3),
        offsets_B=torch.tensor([0, pack], dtype=torch.int64),
        offsets_N=torch.arange(N + 1, dtype=torch.int64) * T,
        sensor_id=sensor_id,
        model=ArticulatedModel(names=names, urdf=None),
        kind="sim",
    )


@dataclass
class Stat:
    name: str
    median_ms: float
    mean_ms: float
    iqr_ms: float
    min_ms: float
    max_ms: float
    blocks: int   # blocked_autorange measurement blocks; each time is a per-call value

    @classmethod
    def of(cls, name: str, m: benchmark.Measurement) -> "Stat":
        return cls(name, m.median * 1e3, m.mean * 1e3, m.iqr * 1e3,
                   min(m.times) * 1e3, max(m.times) * 1e3, len(m.times))


def time_fn(name: str, fn, min_run_time: float) -> Stat:
    """``Timer`` synchronizes CUDA around every call; ``fn`` runs under inference_mode."""
    def run():
        with torch.inference_mode():
            return fn()
    timer = benchmark.Timer(stmt="run()", globals={"run": run}, label=name, num_threads=torch.get_num_threads())
    return Stat.of(name, timer.blocked_autorange(min_run_time=min_run_time))


def bench_trt(model: FoundationIMU, x: IMUData_Jagged, args, stages: dict[str, Stat]) -> list[Stat]:
    """Static-shape TensorRT engine of the same network on the same stream (see model_trt.py)."""
    from ...model.model_trt import FoundationIMU_TRT
    N, T = args.num_imu, int(round(args.rate_hz * args.duration_s))
    trt = FoundationIMU_TRT(max_samples=T, max_timespan_ms=args.duration_s * 1e3, num_devices=N,
                            device=args.device, precision=args.trt_precision)
    trt.from_torch_model(model)
    # make_stream is a full regular grid: the dense layout is a plain reshape of the pack.
    acc, gyro = x.acc.float().reshape(N, T, 3), x.gyro.float().reshape(N, T, 3)
    time_ns, sensor = x.time_ns.reshape(N, T), x.sensor_id.reshape(N, T)[:, 0].long()
    valid = torch.ones(N, T, dtype=torch.bool, device=acc.device)
    with torch.inference_mode():
        ref = model(x, torch.zeros(x.acc.shape[0], dtype=torch.bool, device=acc.device))
        out = trt(acc, gyro, time_ns, valid, sensor)
        err = float((out.dec_body_vel.reshape(-1, 3) - ref.dec_body_vel.float()).norm() / ref.dec_body_vel.float().norm())
    sched = trt.schedule(time_ns, valid, sensor)
    x6 = torch.cat([acc, gyro], -1)
    print(f"\nTensorRT engine ({args.trt_precision}): {trt}\n  body_vel rel-L2 vs the bf16 jagged model: {err:.2e}")
    stages_trt = {
        "trt: schedule (timestamp prep)":  lambda: trt.schedule(time_ns, valid, sensor),
        "trt: engine only":                lambda: trt.run_engine(x6, sched),
        "trt: forward, cached schedule":   lambda: trt(acc, gyro, schedule=sched),
        "trt: forward incl. schedule":     lambda: trt(acc, gyro, time_ns, valid, sensor),
    }
    out_stats = []
    for name, fn in stages_trt.items():
        s = time_fn(name, fn, args.min_run_time)
        out_stats.append(s)
        print(f"{s.name:36s} {s.median_ms:10.3f} {s.mean_ms:9.3f} {s.iqr_ms:8.3f} {s.min_ms:8.3f} {s.max_ms:8.3f} {s.blocks:6d}")
    e2e = stages["end-to-end forward"].median_ms
    print(f"speedup vs jagged end-to-end: engine only {e2e / out_stats[1].median_ms:.1f}x, "
          f"forward with cached schedule {e2e / out_stats[2].median_ms:.1f}x, incl. schedule {e2e / out_stats[3].median_ms:.1f}x")
    return out_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, default=PAPER_CKPT, help="run whose .hydra config defines the model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-imu", type=int, default=12)
    parser.add_argument("--rate-hz", type=float, default=200.0)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--min-run-time", type=float, default=5.0, help="seconds of samples per stage")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON with the numbers below")
    parser.add_argument("--trt", action="store_true", help="also benchmark the TensorRT deployment (src/model/model_trt.py)")
    parser.add_argument("--trt-precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    device = torch.device(args.device)
    model: FoundationIMU = load_model(args.ckpt).eval().to(device)
    dtype = next(model.parameters()).dtype
    x = make_stream(args.num_imu, args.rate_hz, args.duration_s).to(device, dtype)
    mask = torch.zeros(x.acc.shape[0], dtype=torch.bool, device=device)

    # One reference pass: stage outputs the isolated benchmarks feed on, and a warmup.
    def reference_pass():
        with torch.inference_mode():
            tokens = model.tokenizer(x, mask)
            adapted = model.adaptor(tokens.feature, tokens.offset_b, tokens.offset_n, tokens.time_ms, tokens.frame)
            feature = model.trunk(adapted, tokens.offset_b, tokens.offset_n, tokens.time_ms, tokens.frame,
                                  feature_from=model.feature_from_trunk)
            task_feature = model.canonicalize_for_task(feature, chunk=model.tok_config[0])
            model(x, mask)
        return tokens, adapted, task_feature
    for _ in range(2):
        reference_pass()
    tokens, adapted, task_feature = reference_pass()
    n_tokens = int(tokens.feature.shape[0])
    n_base_tokens = n_tokens - 1 - getattr(model.tokenizer, "num_register", 0)   # minus [cls] and registers

    stages = {
        "tokenizer.encode":                 lambda: model.tokenizer(x, mask),
        "trunk":                            lambda: model.trunk(adapted, tokens.offset_b, tokens.offset_n,
                                                                tokens.time_ms, tokens.frame,
                                                                feature_from=model.feature_from_trunk),
        "tokenizer.decode.imu_correction":  lambda: model.tokenizer.decode_imu_correction(task_feature),
        "tokenizer.decode.body_velocity":   lambda: model.tokenizer.decode_body_velocity(task_feature),
        "tokenizer.decode.language":        lambda: model.tokenizer.decode_language_feature(task_feature),
        "tokenizer.decode (all heads)":     lambda: (model.tokenizer.decode_imu_correction(task_feature),
                                                    model.tokenizer.decode_body_velocity(task_feature),
                                                    model.tokenizer.decode_language_feature(task_feature)),
        "end-to-end forward":               lambda: model(x, mask),
    }
    print(f"model    : {args.ckpt}")
    print(f"device   : {torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor()}"
          f"  dtype={dtype}  threads={torch.get_num_threads()}  torch={torch.__version__}")
    print(f"input    : {args.num_imu} IMUs x {args.rate_hz:g} Hz x {args.duration_s:g} s = {x.acc.shape[0]} samples"
          f" -> {n_base_tokens} base tokens (+1 cls, +{n_tokens - 1 - n_base_tokens} registers) = {n_tokens} tokens")
    print(f"params   : {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M "
          f"(tokenizer {sum(p.numel() for p in model.tokenizer.parameters()) / 1e6:.2f} M, "
          f"trunk {sum(p.numel() for p in model.trunk.parameters()) / 1e6:.2f} M)")
    print()
    stats: list[Stat] = []
    header = f"{'stage':36s} {'median ms':>10s} {'mean ms':>9s} {'iqr ms':>8s} {'min ms':>8s} {'max ms':>8s} {'blocks':>6s}"
    print(header)
    print("-" * len(header))
    for name, fn in stages.items():
        s = time_fn(name, fn, args.min_run_time)
        stats.append(s)
        print(f"{s.name:36s} {s.median_ms:10.3f} {s.mean_ms:9.3f} {s.iqr_ms:8.3f} {s.min_ms:8.3f} {s.max_ms:8.3f} {s.blocks:6d}")
    by = {s.name: s for s in stats}
    parts = by["tokenizer.encode"].median_ms + by["trunk"].median_ms + by["tokenizer.decode (all heads)"].median_ms
    print("-" * len(header))
    print(f"{'encode + trunk + decode':36s} {parts:10.3f}   (end-to-end {by['end-to-end forward'].median_ms:.3f}; "
          f"remainder = adaptor, layer-norm canonicalization, output packing)")
    print(f"real-time factor: {args.duration_s * 1e3 / by['end-to-end forward'].median_ms:.0f}x faster than real time")

    trt_stats: list[Stat] = []
    if args.trt:
        trt_stats = bench_trt(model, x, args, stages=by)
        stats += trt_stats

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "ckpt": str(args.ckpt), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "dtype": str(dtype), "torch": torch.__version__, "num_threads": torch.get_num_threads(),
            "num_imu": args.num_imu, "rate_hz": args.rate_hz, "duration_s": args.duration_s,
            "num_samples": int(x.acc.shape[0]), "num_tokens": n_tokens, "num_base_tokens": n_base_tokens,
            "stages": [asdict(s) for s in stats],
        }, args.out.open("w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
