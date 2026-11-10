"""Pack raw Nymeria XSens sequences into the per-sequence IMU cache.

Usage:
    python -m src.scripts.nymeria_imu_process src/config/data/nymeria_full_9imu <save_to> [--workers N]
"""

import json
import torch
import omegaconf
import numpy as np
import multiprocessing

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress

from ...data.imu.nymeria    import Nymeria_XSens_IMU
from ...interface.human     import CANONICAL_NAMES


# Channel layout of the device-major (N, T, F) fp16 measure.npy; order is load-bearing. gt_pose is
# a separate fp32 file (it degrades in fp16); the all-zero bias fields are synthesized by the loader.
PACKED_FIELDS: tuple[tuple[str, int], ...] = (
    ("acc",    3),
    ("gyro",   3),
    ("gt_vel", 3),
)


def _worker_init() -> None:
    # One intra-op thread per worker avoids oversubscribing the host.
    torch.set_num_threads(1)


def process_one(seq_path: str, save_to: Path) -> str:
    seq_name = Path(seq_path).name
    out_dir  = save_to / seq_name

    # spec.json is written last, so its presence marks a completed sequence.
    if (out_dir / "spec.json").exists():
        return f"[skip] {seq_name}"

    # Cache every canonical segment so any target_sensors can be column-sliced from these files.
    source = Nymeria_XSens_IMU(path=seq_path, target_sensors=CANONICAL_NAMES)
    data   = source[:]
    T      = len(source)
    N      = len(CANONICAL_NAMES)
    assert data.gt_vel is not None

    tensors = {
        "acc"    : data.acc,
        "gyro"   : data.gyro,
        "gt_vel" : data.gt_vel,
    }

    fields:  dict[str, dict] = {}
    columns: list[np.ndarray] = []
    start = 0
    for name, C in PACKED_FIELDS:
        # Clamp into fp16's finite range: rare tracking glitches would become inf and NaN the loss.
        block = tensors[name].squeeze(0) \
            .clamp_(-65504.0, 65504.0).to(torch.float16).numpy()           # (N, T, C)
        fields[name] = {"col": [start, start + C], "shape": [N, C]}
        columns.append(block)
        start += C
    measure = np.concatenate(columns, axis=-1)                             # (N, T, F)
    pose    = data.gt_pose.tensor().squeeze(0).to(torch.float32).numpy()   # (N, T, 7)

    # Identical across sensors: store (T,); the loader re-expands to (1, N, T, 1).
    time_ns = data.time_ns[0, 0, :, 0].to(torch.int64).numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "measure.npy", measure)
    np.save(out_dir / "pose.npy",    pose)
    np.save(out_dir / "time_ns.npy", time_ns)
    with open(out_dir / "spec.json", "w") as f:
        json.dump({
            "format"  : 3,
            "length"  : T,
            "names"   : list(CANONICAL_NAMES),
            "kind"    : data.kind,
            "source"  : seq_path,
            "measure" : {"file": "measure.npy", "dtype": "float16", "shape": [N, T, start], "fields": fields},
            "pose"    : {"file": "pose.npy",    "dtype": "float32", "shape": [N, T, 7]},
            "time_ns" : {"file": "time_ns.npy", "dtype": "int64",   "shape": [T]},
        }, f, indent=2)

    return f"[done] {seq_name} ({T} frames, {(measure.nbytes + pose.nbytes) / 1e6:.0f} MB)"


def main(data_cfg: list[omegaconf.DictConfig], save_to: Path, workers: int = 8):
    # The source sequence is the first layer of each build_nested_seq spec.
    seq_paths = sorted({
        str(omegaconf.OmegaConf.to_container(spec.layers[0], resolve=True)["path"])  # type: ignore[index]
        for spec in data_cfg
    })

    failures: list[tuple[str, Exception]] = []
    with Progress() as pbar:
        task = pbar.add_task("Processing sequences", total=len(seq_paths))
        # Fork, not forkserver: functions in a `__main__.py` can't be re-imported by its children.
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, mp_context=ctx) as pool:
            futures = {pool.submit(process_one, p, save_to): p for p in seq_paths}
            for fut in as_completed(futures):
                try:
                    pbar.console.print(fut.result())
                except Exception as e:  # keep going; report at the end
                    failures.append((futures[fut], e))
                    pbar.console.print(f"[red][fail] {futures[fut]}: {e}[/red]")
                pbar.advance(task)

    print(f"{len(seq_paths) - len(failures)}/{len(seq_paths)} sequences processed, saved to {save_to}")
    if failures:
        print(f"{len(failures)} failed:")
        for path, e in failures:
            print(f"  {path}: {e!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("dataset_config", type=Path)    # e.g. ./src/config/data/nymeria_full_9imu
    parser.add_argument("save_to"       , type=Path)
    parser.add_argument("--workers"     , type=int, default=8)
    args   = parser.parse_args()

    cfg_path    = args.dataset_config if args.dataset_config.suffix else args.dataset_config.with_suffix(".yaml")
    cfg         = omegaconf.OmegaConf.load(cfg_path)
    all_configs = list(cfg.train) + list(cfg.valid)

    main(all_configs, args.save_to, workers=args.workers)
