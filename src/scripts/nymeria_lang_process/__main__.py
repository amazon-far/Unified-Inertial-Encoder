"""Encode transcribed Nymeria narrations with CLIP into the per-sequence language cache.

Usage:
    python -m src.scripts.nymeria_lang_process src/config/data/nymeria_full_9imu_mmap <save_to> [--variant bones]
"""

import json
import torch
import omegaconf
import numpy as np
import ml_dtypes

from pathlib import Path
from rich.progress import Progress

from ...data.lang.nymeria    import Nymeria_Language, _CSV_NAME_
from ...data.util            import listing_files
from ...utility.clip         import _FEAT_DIM_
from ...runtime_prologue     import machine_conf


# Narration CSVs live beside the raw data; a processed dir maps back to them by basename.
_NARRATION_ROOT = Path(machine_conf("data_root"), "nymeria_imu")

# variant -> (transcription CSV, sidecar filename infix); selected at train time via the reader's
# ``lang_variant`` argument.
_VARIANTS_: dict[str, tuple[str, str]] = {
    "default": (_CSV_NAME_, ""),
    "bones"  : ("transcribed_bones_style.csv", ".bones"),
}


def _bf16_bytes(x: torch.Tensor) -> np.ndarray:
    """View a float tensor as an ``ml_dtypes.bfloat16`` array; readers must ``import ml_dtypes``."""
    return x.to(torch.bfloat16).view(torch.uint16).numpy().view(ml_dtypes.bfloat16)


def process_one(seq_path: str, save_to: Path, variant: str = "default") -> str:
    csv_name, infix = _VARIANTS_[variant]
    seq_name = Path(seq_path).name
    out_dir  = save_to / seq_name

    # Sequences without the transcribed CSV stay IMU-only.
    csv_path = _NARRATION_ROOT / seq_name / "narration" / csv_name
    if not csv_path.exists():
        return f"[skip] {seq_name} (no {csv_name})"

    # source[:] dedups phrases and encodes them in one CLIP forward.
    source = Nymeria_Language(str(_NARRATION_ROOT / seq_name), csv_name=csv_name)
    data   = source[:]
    N      = len(source)
    V      = data.lang_feature.shape[0]

    # feature: one bf16 row per unique phrase; index: each annotation's row (-1 = empty description).
    feature = _bf16_bytes(data.lang_feature)                                   # (V, C) bf16
    index   = data.lang_feat_idx.to(torch.int64).numpy()                       # (N,)
    time_ns = torch.stack([data.lang_time_ns, data.lang_time_end_ns], dim=-1) \
        .to(torch.int64).numpy()                                               # (N, 2) [start, end]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"lang_feature{infix}.npy", feature)
    np.save(out_dir / f"lang_index{infix}.npy",   index)
    np.save(out_dir / f"lang_time_ns{infix}.npy", time_ns)
    with open(out_dir / f"lang_spec{infix}.json", "w") as f:
        json.dump({
            "format"  : 1,
            "variant" : variant,
            "length"  : N,                                  # annotations in this sequence
            "n_valid" : V,                                  # compact feature rows (unique phrases)
            "encoder" : Nymeria_Language._encoder().model_name,
            "source"  : str(csv_path),
            "feature" : {"file": f"lang_feature{infix}.npy", "dtype": "bfloat16", "shape": [V, _FEAT_DIM_]},
            "index"   : {"file": f"lang_index{infix}.npy",   "dtype": "int64",    "shape": [N]},
            "time_ns" : {"file": f"lang_time_ns{infix}.npy", "dtype": "int64",    "shape": [N, 2], "cols": ["start", "end"]},
        }, f, indent=2)

    return f"[done] {seq_name} ({N} annotations, {V} unique, {feature.nbytes / 1e3:.0f} KB)"


def _config_seq_paths(data_cfg: list[omegaconf.DictConfig]) -> list[str]:
    """Every sequence path a data config names, deduped.

    Resolves only the accessed nodes: a raw-loaded config cannot resolve its ``layers`` interpolations.
    """
    paths: set[str] = set()
    for entry in data_cfg:
        if str(entry.get("_target_", "")).endswith("build_corpus"):
            for listing in listing_files(entry.stems):
                with open(listing) as f:
                    stems = [line for line in f.read().splitlines() if line and not line.startswith("#")]
                paths |= {str(Path(str(entry.root)) / stem) for stem in stems}
        else:
            paths.add(str(omegaconf.OmegaConf.to_container(entry.layers[0], resolve=True)["path"]))  # type: ignore[index]
    return sorted(paths)


def main(data_cfg: list[omegaconf.DictConfig], save_to: Path, variant: str) -> None:
    # No worker pool: one shared CLIP tower already spans every visible GPU.
    seq_paths = _config_seq_paths(data_cfg)
    print(f"{len(seq_paths)} sequences from config; narration read from {_NARRATION_ROOT} (variant: {variant})")

    failures: list[tuple[str, Exception]] = []
    with Progress() as pbar:
        task = pbar.add_task("Encoding narration", total=len(seq_paths))
        for path in seq_paths:
            try:
                pbar.console.print(process_one(path, save_to, variant))
            except Exception as e:  # keep going; report at the end
                failures.append((path, e))
                pbar.console.print(f"[red][fail] {Path(path).name}: {e}[/red]")
            pbar.advance(task)

    print(f"{len(seq_paths) - len(failures)}/{len(seq_paths)} sequences processed, saved to {save_to}")
    if failures:
        print(f"{len(failures)} failed:")
        for path, e in failures:
            print(f"  {Path(path).name}: {e!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("dataset_config", type=Path)    # e.g. ./src/config/data/nymeria_full_9imu_mmap
    parser.add_argument("save_to"       , type=Path)
    parser.add_argument("--variant", choices=sorted(_VARIANTS_), default="default",
                        help="transcription CSV to encode and the sidecar names to write (see _VARIANTS_)")
    args = parser.parse_args()

    cfg_path    = args.dataset_config if args.dataset_config.suffix else args.dataset_config.with_suffix(".yaml")
    cfg         = omegaconf.OmegaConf.load(cfg_path)
    all_configs = list(cfg.train) + list(cfg.valid)

    main(all_configs, args.save_to, args.variant)
