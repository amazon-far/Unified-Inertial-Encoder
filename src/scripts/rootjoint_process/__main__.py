"""Build root (pelvis) IMU + 12-site joint-angle caches from the raw files of every corpus.

Usage:
    python -m src.scripts.rootjoint_process <corpus|all> [--out DIR] [--workers N] [--limit K] [--source-root DIR]
"""
from __future__ import annotations

from src.runtime_prologue import machine_conf

import json
import torch
import numpy as np
import pypose as pp
import multiprocessing

from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.markup import escape
from rich.progress import Progress

from ...interface.joint import JOINT_SITES, JOINT_TREE, JOINT_NAMES, relative_joints
from ...interface.human import CANONICAL_IDX
from ..bonesseed_imu_process.__main__  import MOUNT_CORRECTION as G1_MOUNT_CORRECTION
from ..bonesseed_soma_process.__main__ import MOUNT_CORRECTION as SOMA_MOUNT_CORRECTION


REPO_ROOT   = Path(__file__).resolve().parents[3]
STEMS_DIR   = REPO_ROOT / "src" / "config" / "data" / "stems"
DEFAULT_OUT = (Path(machine_conf('cache_root')) / 'rootjoint_v1')

# Default source root per corpus; --source-root overrides it.
SOURCE_ROOTS: dict[str, Path] = {
    "nymeria"      : (Path(machine_conf('data_root')) / 'nymeria_imu'),
    "naturalmotion": (Path(machine_conf('data_root')) / 'vt_natural_motion/h5-dataset'),
    "bones_soma"   : (Path(machine_conf('data_root')) / 'bones_seed'),
    "bones_g1"     : (Path(machine_conf('data_root')) / 'bones_seed'),
    "superodom_clips": (Path(machine_conf('cache_root')) / 'humanoid_superodom_clips_v2'),
}

MEASURE_FIELDS: tuple[tuple[str, int], ...] = (("acc", 3), ("gyro", 3), ("gt_vel", 3))
JOINT_FIELDS  : tuple[tuple[str, int], ...] = (("q", 4), ("dq", 3))
ENCODER_FIELDS: tuple[tuple[str, int], ...] = (("q", 1), ("dq", 1))


def _stems(*files: str) -> list[str]:
    out: list[str] = []
    for f in files:
        for line in (STEMS_DIR / f).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


# Readers are built inside the worker (see `_build`), so the parent never imports them.

def _jobs_nymeria(root: Path) -> list[tuple[str, str]]:
    return [(str(root / s), s) for s in _stems("nymeria_train.txt", "nymeria_valid.txt")]

def _jobs_naturalmotion(root: Path) -> list[tuple[str, str]]:
    return [(str(root / f"{s}.h5"), s) for s in _stems("naturalmotion_train.txt", "naturalmotion_valid.txt")]

def _jobs_bones_soma(root: Path) -> list[tuple[str, str]]:
    return [(str(root / s), Path(s).stem) for s in _stems("bones_soma_raw_train.txt", "bones_soma_raw_valid.txt")]

def _jobs_bones_g1(root: Path) -> list[tuple[str, str]]:
    return [(str(root / s), Path(s).stem) for s in _stems("bones_g1_raw_train.txt", "bones_g1_raw_valid.txt")]

def _jobs_superodom_clips(root: Path) -> list[tuple[str, str]]:
    return [(str(root / s), s) for s in _stems("superodom_clips_train.txt", "superodom_clips_eval.txt")]

CORPORA: dict[str, tuple] = {
    #  name           jobs                 workers  mount table            encoders
    "nymeria"      : (_jobs_nymeria,       16,      None,                  False),
    "naturalmotion": (_jobs_naturalmotion,  8,      None,                  False),
    "bones_soma"   : (_jobs_bones_soma,    96,      SOMA_MOUNT_CORRECTION, False),
    "bones_g1"     : (_jobs_bones_g1,      96,      G1_MOUNT_CORRECTION,   True),
    "superodom_clips": (_jobs_superodom_clips, 16,  None,                  False),  # already corrected
}


def _build(corpus: str, path: str):
    """Instantiate the raw reader for one sequence (imports deferred to the worker)."""
    match corpus:
        case "nymeria":
            from ...data.imu.nymeria import Nymeria_XSens_IMU
            return Nymeria_XSens_IMU(path, target_sensors=JOINT_SITES)
        case "naturalmotion":
            from ...data.imu.naturalmotion import NaturalMotion_XSens_IMU
            return NaturalMotion_XSens_IMU(path, target_sensors=JOINT_SITES)
        case "bones_soma":
            from ...data.imu.bones_seed_soma import Bones_Seed_SOMA_IMU
            return Bones_Seed_SOMA_IMU(path, target_sensors=JOINT_SITES)
        case "bones_g1":
            from ...data.imu.bones_seed_g1 import Bones_Seed_G1_IMU
            return Bones_Seed_G1_IMU(path)
        case "superodom_clips":
            from ...data.imu.g1_superodom_processed import Unitree_G1_SuperOdom_IMU_Processed
            return Unitree_G1_SuperOdom_IMU_Processed(path)
        case _:
            raise ValueError(corpus)


def _apply_mount_correction(data, table: dict[str, list[list[float]]], names: tuple[str, ...]):
    """Rebase every site frame onto the XSens convention: with R_ws_new = R_ws_old @ M, body-frame
    vectors rotate by M^T and pose orientations by M."""
    missing = sorted(set(names) - set(table))
    assert not missing, f"mount-correction table missing sites {missing}"
    M  = torch.tensor([table[n] for n in names], dtype=torch.float32)                # (N, 3, 3)
    Mt = M.transpose(-1, -2)
    for attr in ("acc", "gyro", "gt_vel"):
        setattr(data, attr, torch.einsum("nij,bntj->bnti", Mt, getattr(data, attr)))
    qM    = pp.mat2SO3(M).tensor().view(1, len(names), 1, 4)
    new_r = data.gt_pose.rotation() @ pp.SO3(qM)
    data.gt_pose = pp.SE3(torch.cat([data.gt_pose.translation(), new_r.tensor()], dim=-1))
    return data


def _worker_init() -> None:
    torch.set_num_threads(1)


def process_one(corpus: str, path: str, stem: str, out_root: Path) -> str:
    _, _, mount, with_encoders = CORPORA[corpus]
    out_dir = out_root / corpus / stem
    # spec.json is written last, so its presence marks a completed sequence.
    if (out_dir / "spec.json").exists():
        return f"[skip] {stem}"

    source = _build(corpus, path)
    data   = source[:]
    names  = tuple(data.model.names)
    assert data.gt_vel is not None
    if mount is not None:
        data = _apply_mount_correction(data, mount, names)

    # Joints from the 12-site tree.
    sel    = [names.index(s) for s in JOINT_SITES]
    rot_ws = pp.SO3(data.gt_pose.rotation().tensor()[0, sel].float())               # (12, T, 4)
    gyro   = data.gyro[0, sel].float()                                              # (12, T, 3)
    q, dq, max_angle = relative_joints(rot_ws, gyro, JOINT_SITES)
    joint  = torch.cat([q, dq], dim=-1).numpy().astype(np.float32)                  # (J, T, 7)

    # Pelvis root IMU.
    pel     = names.index("pelvis")
    measure = torch.cat([data.acc[0, pel], data.gyro[0, pel], data.gt_vel[0, pel]], dim=-1) \
                   .float().numpy()[None]                                           # (1, T, 9)
    pose    = data.gt_pose.tensor()[0, pel].float().numpy()[None]                   # (1, T, 7)
    time_ns = data.time_ns[0, 0, :, 0].to(torch.int64).numpy()                      # (T,)
    T = time_ns.shape[0]
    assert measure.shape[1] == pose.shape[1] == joint.shape[1] == T
    assert np.isfinite(measure).all() and np.isfinite(pose).all() and np.isfinite(joint).all(), \
        f"{stem}: non-finite values"

    # Native encoders (G1 only).
    encoder, encoder_names = None, []
    if with_encoders:
        js = source.joint_store                                                     # type: ignore[attr-defined]
        enc_q  = js.joints [0, :, :, 0].float().numpy()                             # (29, T)
        enc_dq = js.joint_d[0, :, :, 0].float().numpy()
        assert enc_q.shape[1] == T
        encoder       = np.stack([enc_q, enc_dq], axis=-1).astype(np.float32)       # (29, T, 2)
        encoder_names = list(js.model.names)

    def cols(fields):
        out, start = {}, 0
        for name, C in fields:
            out[name] = [start, start + C]; start += C
        return out, start

    m_cols, m_F = cols(MEASURE_FIELDS)
    j_cols, j_F = cols(JOINT_FIELDS)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "measure.npy", measure)
    np.save(out_dir / "pose.npy",    pose)
    np.save(out_dir / "time_ns.npy", time_ns)
    np.save(out_dir / "joint.npy",   joint)
    spec: dict = {
        "format"  : "rootjoint-1",
        "corpus"  : corpus,
        "length"  : T,
        "names"   : ["pelvis"],
        "kind"    : data.kind,
        "source"  : path,
        "mount_correction": "xsens" if mount is None else ("soma->xsens" if corpus == "bones_soma" else "g1->xsens"),
        "measure" : {"file": "measure.npy", "dtype": "float32", "shape": [1, T, m_F],
                     "fields": {k: {"col": v, "shape": [1, v[1] - v[0]]} for k, v in m_cols.items()}},
        "pose"    : {"file": "pose.npy",    "dtype": "float32", "shape": [1, T, 7]},
        "time_ns" : {"file": "time_ns.npy", "dtype": "int64",   "shape": [T]},
        "joint"   : {"file": "joint.npy",   "dtype": "float32", "shape": [len(JOINT_NAMES), T, j_F],
                     "names": list(JOINT_NAMES),
                     "tree" : [{"name": n, "parent": p, "child": c} for n, p, c in JOINT_TREE],
                     "sites": list(JOINT_SITES),
                     "site_ids": [CANONICAL_IDX[s] for s in JOINT_SITES],
                     "fields": j_cols,
                     "q"     : "unit quaternion xyzw of R_parent^T R_child, sign-continuous in time",
                     "dq"    : "relative angular velocity in the child frame, rad/s",
                     "max_angle_rad": max_angle},
    }
    if encoder is not None:
        np.save(out_dir / "encoder.npy", encoder)
        e_cols, e_F = cols(ENCODER_FIELDS)
        spec["encoder"] = {"file": "encoder.npy", "dtype": "float32", "shape": list(encoder.shape),
                           "names": encoder_names, "fields": e_cols,
                           "q": "encoder position, rad", "dq": "encoder velocity, rad/s"}
    with open(out_dir / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    return f"[done] {stem} ({T} frames, max joint angle {max_angle:.2f} rad)"


def run(corpus: str, out_root: Path, workers: int | None, limit: int | None, source_root: Path | None) -> int:
    jobs_fn, default_workers, _, _ = CORPORA[corpus]
    jobs = jobs_fn(source_root or SOURCE_ROOTS[corpus])
    assert len({s for _, s in jobs}) == len(jobs), f"{corpus}: duplicate output stems"
    if limit: jobs = jobs[:limit]
    workers = workers or default_workers

    failures: list[tuple[str, Exception]] = []
    with Progress() as pbar:
        task = pbar.add_task(f"{corpus}", total=len(jobs))
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, mp_context=ctx) as pool:
            futures = {pool.submit(process_one, corpus, p, s, out_root): (p, s) for p, s in jobs}
            for fut in as_completed(futures):
                p, s = futures[fut]
                try:
                    pbar.console.print(fut.result(), markup=False)
                except Exception as e:
                    failures.append((p, e))
                    pbar.console.print(f"[red]\\[fail] {s}: {escape(repr(e))}[/red]")
                pbar.advance(task)

    print(f"{corpus}: {len(jobs) - len(failures)}/{len(jobs)} sequences processed -> {out_root / corpus}")
    if failures:
        print(f"{len(failures)} failed:")
        for p, e in failures[:20]:
            print(f"  {p}: {e!r}")
    return len(failures)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("corpus", choices=[*CORPORA, "all"])
    parser.add_argument("--out",     type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int,  default=None)
    parser.add_argument("--limit",   type=int,  default=None, help="process only the first K sequences (smoke test)")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="read the corpus from this root instead of SOURCE_ROOTS[corpus] (e.g. an `aws s3 sync` staging copy on /data)")
    args = parser.parse_args()

    corpora = list(CORPORA) if args.corpus == "all" else [args.corpus]
    assert args.source_root is None or len(corpora) == 1, "--source-root applies to a single corpus"
    n_fail  = sum(run(c, args.out, args.workers, args.limit, args.source_root) for c in corpora)
    raise SystemExit(1 if n_fail else 0)
