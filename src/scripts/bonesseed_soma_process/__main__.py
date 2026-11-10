"""Synthesize BONES-SEED SOMA virtual-IMU and language-feature caches from the raw BVH motions.

Usage:
    python -m src.scripts.bonesseed_soma_process src/config/data/bones_seed_soma_full <save_to> [--workers N]
"""

import json
import torch
import pypose as pp
import omegaconf
import numpy as np
import ml_dtypes
import multiprocessing

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.markup import escape
from rich.progress import Progress

from ...data.imu.bones_seed_g1   import _LABELS_NAME_
from ...data.imu.bones_seed_soma import (
    Bones_Seed_SOMA_IMU, Bones_Seed_SOMA_IMU_w_GT_Lang, CANONICALIZED_BONES_SOMA,
)
from ...utility.clip             import _FEAT_DIM_


# Channel layout of the device-major (N, T, F) fp16 measure.npy; order is load-bearing. gt_pose is
# a separate fp32 file (it degrades in fp16); the all-zero bias fields are synthesized by the loader.
PACKED_FIELDS: tuple[tuple[str, int], ...] = (
    ("acc",    3),
    ("gyro",   3),
    ("gt_vel", 3),
)

# The 16 canonical sites SOMA can carry: all 17 human sites except `torso_t8`.
NAMES: tuple[str, ...] = tuple(CANONICALIZED_BONES_SOMA)

# Constant rotation M with R_ws_new = R_ws_old @ M, rebasing each SOMA BVH joint frame onto the
# XSens/Nymeria segment-frame convention. M = S_soma^-1 @ S_nymeria, where S_ds = <R_anat^T R_ws>
# is the pose-independent offset between the sensor frame and an anatomical frame.
MOUNT_CORRECTION: dict[str, list[list[float]]] = {
    "pelvis"          : [[-0.22147, -0.02501, +0.97485], [+0.97505, -0.02132, +0.22097], [+0.01526, +0.99946, +0.02911]],
    "head"            : [[-0.40900, +0.00834, +0.91250], [+0.91248, -0.00720, +0.40905], [+0.00998, +0.99994, -0.00466]],
    "left_shoulder"   : [[-0.41920, +0.89334, +0.16194], [+0.90270, +0.42917, -0.03079], [-0.09701, +0.13328, -0.98632]],
    "left_upper_arm"  : [[+0.00000, +1.00000, -0.00000], [+0.84582, -0.00000, -0.53347], [-0.53347, -0.00000, -0.84582]],
    "left_forearm"    : [[+0.00019, +1.00000, +0.00000], [+0.74760, -0.00014, -0.66415], [-0.66415, +0.00013, -0.74760]],
    "left_hand"       : [[+0.04870, +0.99792, -0.04217], [+0.69124, -0.06415, -0.71977], [-0.72098, +0.00591, -0.69293]],
    "right_shoulder"  : [[+0.42152, +0.90515, -0.05502], [-0.90535, +0.41660, -0.08243], [-0.05169, +0.08456, +0.99508]],
    "right_upper_arm" : [[+0.00000, +1.00000, +0.00000], [-0.91867, +0.00000, +0.39502], [+0.39502, -0.00000, +0.91867]],
    "right_forearm"   : [[-0.00018, +1.00000, +0.00000], [-0.88790, -0.00016, +0.46003], [+0.46003, +0.00008, +0.88790]],
    "right_hand"      : [[-0.05463, +0.99451, +0.08924], [-0.86943, -0.09133, +0.48554], [+0.49102, -0.05106, +0.86965]],
    "left_upper_leg"  : [[+0.00002, +0.00000, -1.00000], [-0.99896, -0.04557, -0.00002], [-0.04557, +0.99896, -0.00000]],
    "left_lower_leg"  : [[+0.00059, +0.00001, -1.00000], [-0.99335, -0.11514, -0.00059], [-0.11514, +0.99335, -0.00006]],
    "left_foot"       : [[+0.96962, +0.12248, -0.21174], [-0.20322, -0.07845, -0.97598], [-0.13615, +0.98937, -0.05118]],
    "right_upper_leg" : [[-0.00002, +0.00000, +1.00000], [+0.99391, -0.11021, +0.00002], [+0.11021, +0.99391, -0.00000]],
    "right_lower_leg" : [[-0.00060, +0.00001, +1.00000], [+0.99404, -0.10901, +0.00059], [+0.10901, +0.99404, +0.00006]],
    "right_foot"      : [[-0.97515, +0.11355, +0.19023], [+0.18421, -0.06144, +0.98096], [+0.12308, +0.99163, +0.03899]],
}
assert set(MOUNT_CORRECTION) == set(NAMES), \
    f"MOUNT_CORRECTION sites {sorted(MOUNT_CORRECTION)} != cached sites {sorted(NAMES)}"


def _apply_mount_correction(data, names: tuple[str, ...]):
    """Rebase every sensor frame onto the XSens/Nymeria convention (see `MOUNT_CORRECTION`).

    With R_ws_new = R_ws_old @ M, body-frame vectors transform by M^T and orientations by M.
    """
    M  = torch.tensor([MOUNT_CORRECTION[n] for n in names], dtype=torch.float32)   # (N, 3, 3)
    Mt = M.transpose(-1, -2)
    for attr in ("acc", "gyro", "gt_vel"):
        setattr(data, attr, torch.einsum("nij,bntj->bnti", Mt, getattr(data, attr)))
    qM    = pp.mat2SO3(M).tensor().view(1, len(names), 1, 4)
    new_r = data.gt_pose.rotation() @ pp.SO3(qM)                                    # R_wb_old @ M
    data.gt_pose = pp.SE3(torch.cat([data.gt_pose.translation(), new_r.tensor()], dim=-1))
    return data


# Motions per CLIP chunk; encoding is batched across motions because each has only a few phrases.
LANG_CHUNK = 512


def _worker_init() -> None:
    # One intra-op thread per worker avoids oversubscribing the host.
    torch.set_num_threads(1)


def _bf16_bytes(x: torch.Tensor) -> np.ndarray:
    """View a float tensor as an ``ml_dtypes.bfloat16`` array; readers must ``import ml_dtypes``."""
    return x.to(torch.bfloat16).view(torch.uint16).numpy().view(ml_dtypes.bfloat16)


def _labels_path_of(bvh_path: str) -> Path:
    """``<root>/metadata/<labels>.jsonl`` for a ``<root>/soma_proportional/bvh/<date>/<motion>.bvh`` source."""
    return Path(bvh_path).parents[3] / "metadata" / _LABELS_NAME_


def process_imu(bvh_path: str, save_to: Path) -> str:
    """Synthesize and cache one motion's 16 virtual IMUs.

    Reads the IMU-only class: in this fork pool, CLIP would be replicated into every worker.
    """
    stem    = Path(bvh_path).stem
    out_dir = save_to / stem

    # spec.json is written last, so its presence marks a completed sequence.
    if (out_dir / "spec.json").exists():
        return f"[skip] {stem}"

    source = Bones_Seed_SOMA_IMU(bvh_path)
    data   = source[:]
    T      = len(source)
    N      = len(NAMES)
    assert data.gt_vel is not None

    # Rebase raw BVH-joint sensor frames onto the XSens/Nymeria mounting convention.
    data = _apply_mount_correction(data, NAMES)

    tensors = {"acc": data.acc, "gyro": data.gyro, "gt_vel": data.gt_vel}

    fields:  dict[str, dict] = {}
    columns: list[np.ndarray] = []
    start = 0
    for name, C in PACKED_FIELDS:
        # Clamp into fp16's finite range: an inf would NaN the training loss.
        block = tensors[name].squeeze(0) \
            .clamp_(-65504.0, 65504.0).to(torch.float16).numpy()            # (N, T, C)
        fields[name] = {"col": [start, start + C], "shape": [N, C]}
        columns.append(block)
        start += C
    measure = np.concatenate(columns, axis=-1)                              # (N, T, F)
    pose    = data.gt_pose.tensor().squeeze(0).to(torch.float32).numpy()    # (N, T, 7)

    # Identical across devices: store (T,); the loader re-expands to (1, N, T, 1).
    time_ns = data.time_ns[0, 0, :, 0].to(torch.int64).numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "measure.npy", measure)
    np.save(out_dir / "pose.npy",    pose)
    np.save(out_dir / "time_ns.npy", time_ns)
    with open(out_dir / "spec.json", "w") as f:
        json.dump({
            "format"  : 3,
            "length"  : T,
            "names"   : list(NAMES),
            # "filtered-no-g": the cached acc is kinematic; AddGravity supplies gravity at train time.
            "kind"    : data.kind,
            "source"  : bvh_path,
            "measure" : {"file": "measure.npy", "dtype": "float16", "shape": [N, T, start], "fields": fields},
            "pose"    : {"file": "pose.npy",    "dtype": "float32", "shape": [N, T, 7]},
            "time_ns" : {"file": "time_ns.npy", "dtype": "int64",   "shape": [T]},
        }, f, indent=2)

    return f"[done] {stem} ({T} frames, {(measure.nbytes + pose.nbytes) / 1e6:.1f} MB)"


def process_lang_chunk(bvh_paths: list[str], save_to: Path, table: dict) -> list[str]:
    """Encode and cache the temporal-label features for a chunk of motions in one CLIP pass.

    lang_feature.npy: one bf16 row per unique phrase in a motion. lang_index.npy: each event's row
    (-1 = blank). lang_time_ns.npy: (start, end) ns per event, on the clock of ``time_ns.npy``.
    """
    plan, flat, out = [], [], []
    for bvh_path in bvh_paths:
        stem    = Path(bvh_path).stem
        out_dir = save_to / stem
        # lang_spec.json is written last: resumable per motion, independently of the IMU pass.
        if (out_dir / "lang_spec.json").exists():
            out.append(f"[skip] {stem} (lang)")
            continue

        events = table.get(stem, [])
        # Repeated phrases in a motion share one feature row; blank phrases get -1.
        index, row_of = [-1] * len(events), {}
        for i, (_, _, text) in enumerate(events):
            if text:
                index[i] = row_of.setdefault(text, len(row_of))
        plan.append((stem, out_dir, events, index, len(row_of), len(flat), bvh_path))
        flat.extend(row_of)

    if not plan:
        return out

    feats = Bones_Seed_SOMA_IMU_w_GT_Lang._encoder()(flat).cpu() if flat else torch.empty(0, _FEAT_DIM_)

    for stem, out_dir, events, index, n_uniq, base, bvh_path in plan:
        N       = len(events)
        feature = _bf16_bytes(feats[base:base + n_uniq])                     # (V, C) bf16
        idx     = np.asarray(index, dtype=np.int64)                          # (N,)
        time_ns = np.asarray([[s, e] for s, e, _ in events], dtype=np.int64).reshape(N, 2)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "lang_feature.npy", feature)
        np.save(out_dir / "lang_index.npy",   idx)
        np.save(out_dir / "lang_time_ns.npy", time_ns)
        with open(out_dir / "lang_spec.json", "w") as f:
            json.dump({
                "format"  : 1,
                "length"  : N,                                  # temporal events in this motion
                "n_valid" : n_uniq,                             # compact feature rows (unique phrases)
                "encoder" : Bones_Seed_SOMA_IMU_w_GT_Lang._encoder().model_name,
                "source"  : str(_labels_path_of(bvh_path)),
                "feature" : {"file": "lang_feature.npy", "dtype": "bfloat16", "shape": [n_uniq, _FEAT_DIM_]},
                "index"   : {"file": "lang_index.npy",   "dtype": "int64",    "shape": [N]},
                "time_ns" : {"file": "lang_time_ns.npy", "dtype": "int64",    "shape": [N, 2], "cols": ["start", "end"]},
            }, f, indent=2)
        out.append(f"[done] {stem} ({N} events, {n_uniq} unique)")

    return out


def corpus_paths(data_cfg: list[omegaconf.DictConfig]) -> list[str]:
    """Expand ``build_corpus`` entries' ``root`` + ``stems`` into sorted, deduped source paths.

    Touches only these fields: layer interpolations don't resolve on a plainly-loaded config.
    """
    repo_root = Path(__file__).resolve().parents[3]
    paths: set[str] = set()
    for entry in data_cfg:
        root = str(entry.root).rstrip("/")
        for line in (repo_root / str(entry.stems)).read_text().splitlines():
            if line and not line.startswith("#"):
                paths.add(f"{root}/{line}")
    return sorted(paths)


def main(data_cfg: list[omegaconf.DictConfig], save_to: Path, workers: int = 8) -> None:
    bvh_paths = corpus_paths(data_cfg)

    failures: list[tuple[str, Exception]] = []

    # IMU: CPU-bound synthesis, one motion per task.
    with Progress() as pbar:
        task = pbar.add_task("Synthesizing IMU", total=len(bvh_paths))
        # Fork, not forkserver: functions in a `__main__.py` can't be re-imported by its children.
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, mp_context=ctx) as pool:
            futures = {pool.submit(process_imu, p, save_to): p for p in bvh_paths}
            for fut in as_completed(futures):
                try:
                    # markup=False: rich would swallow "[done]"/"[skip]" as style tags.
                    pbar.console.print(fut.result(), markup=False)
                except Exception as e:  # keep going; report at the end
                    failures.append((futures[fut], e))
                    # escape(): brackets in the message would raise MarkupError inside this handler.
                    pbar.console.print(f"[red]\\[fail] {Path(futures[fut]).stem}: {escape(str(e))}[/red]")
                pbar.advance(task)

    # Language: no worker pool, as one shared CLIP tower already spans every visible GPU.
    labels_paths = {_labels_path_of(p) for p in bvh_paths}
    assert len(labels_paths) == 1, f"config mixes BONES-SEED roots: {sorted(labels_paths)}"
    table = Bones_Seed_SOMA_IMU_w_GT_Lang._label_table(labels_paths.pop())

    with Progress() as pbar:
        task = pbar.add_task("Encoding language", total=len(bvh_paths))
        for i in range(0, len(bvh_paths), LANG_CHUNK):
            chunk = bvh_paths[i:i + LANG_CHUNK]
            try:
                for line in process_lang_chunk(chunk, save_to, table):
                    pbar.console.print(line, markup=False)
            except Exception as e:  # a chunk failure loses that chunk, not the run
                failures.extend((p, e) for p in chunk)
                pbar.console.print(f"[red]\\[fail] lang chunk at {i}: {escape(str(e))}[/red]")
            pbar.advance(task, advance=len(chunk))

    failed = {p for p, _ in failures}
    print(f"{len(bvh_paths) - len(failed)}/{len(bvh_paths)} motions processed, saved to {save_to}")
    if failures:
        print(f"{len(failed)} failed:")
        for path, e in failures[:20]:
            print(f"  {Path(path).stem}: {e!r}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("dataset_config", type=Path)    # e.g. ./src/config/data/bones_seed_soma_full
    parser.add_argument("save_to"       , type=Path)
    parser.add_argument("--workers"     , type=int, default=8)
    args = parser.parse_args()

    cfg_path    = args.dataset_config if args.dataset_config.suffix else args.dataset_config.with_suffix(".yaml")
    cfg         = omegaconf.OmegaConf.load(cfg_path)
    all_configs = list(cfg.train) + list(cfg.valid)

    main(all_configs, args.save_to, workers=args.workers)
