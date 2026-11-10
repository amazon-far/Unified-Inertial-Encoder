"""Evaluate one sensor's velocity on the fixed generalization suite (the valid side of exp7_cotraining_C).

Usage:
    python -m src.scripts.evaluate_velocity_generalization <ckpt> [--sensor pelvis] [--limit N]
"""

import argparse
import json
import typing as tp

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from ... import eval_stream, evaluate
from ...data.util import corpus_pipelines
from ...eval import inference, load_model
from ...interface.data_imu import IMUData_LabelTrait_Jagged
from ...interface.human import CANONICAL_IDX, CANONICAL_NAMES
from ...runtime_prologue import machine_conf
from ...data.imu.bones_seed_processed import canonical_motion


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "src" / "config"
GROUPS_JSON = CONFIG_DIR / "data" / "bones_seed_split_groups.json"

DATASET_ORDER = (
    "Bones-Seed-Soma",
    "Bones-Seed-G1-Valid",
    "Bones-Seed-G1-Human2Robot",
    "Nymeria",
    "NaturalMotion",
)


@dataclass
class Cell:
    """One dataset's tally; ``parts`` holds one entry per forward, tagged with its first take's index."""
    takes  : int = 0
    windows: int = 0
    short  : int = 0
    missing: int = 0
    parts  : list[tuple[int, dict[str, evaluate.MetricPart]]] = field(default_factory=list)

    def merge(self, other: "Cell") -> None:
        """Fold another shard's tally for the same dataset into this one."""
        self.takes   += other.takes
        self.windows += other.windows
        self.short   += other.short
        self.missing += other.missing
        self.parts   += other.parts

    def pooled(self) -> dict[str, evaluate.MetricPart]:
        """Row-weighted pooling across every chunk, in undivided take order."""
        return evaluate.merge_parts(part for _, part in sorted(self.parts, key=lambda x: x[0]))


def fixed_valid_pipelines() -> dict[str, list[tuple[str, tp.Callable[[str], tp.Any]]]]:
    """Return the fixed validation stems and their lazy per-take builders."""
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="train", overrides=["experiment=exp7_cotraining_C"])

    with GROUPS_JSON.open() as f:
        # Keyed by canonical motion; later groups overwrite earlier ones for repeated names.
        group_of = {m: group for group, motions in json.load(f)["groups"].items() for m in motions}

    pipelines: dict[str, list[tuple[str, tp.Callable[[str], tp.Any]]]] = defaultdict(list)
    for entry in cfg.data.valid:
        spec = OmegaConf.to_container(entry, resolve=True)
        assert isinstance(spec, dict) and str(spec.get("_target_", "")).endswith("build_corpus"), spec
        kwargs = {str(k): v for k, v in spec.items() if not str(k).startswith("_")}
        stems, build = corpus_pipelines(**kwargs)
        reader = str(spec["reader"]["_target_"])

        for stem in stems:
            if "Bones_Seed_SOMA" in reader:
                dataset = "Bones-Seed-Soma"
            elif "Bones_Seed_G1" in reader:
                group = group_of[canonical_motion(stem)]
                if group == "valid":
                    dataset = "Bones-Seed-G1-Valid"
                elif group == "human2robot":
                    dataset = "Bones-Seed-G1-Human2Robot"
                else:
                    raise AssertionError(f"{stem}: unexpected G1 group {group!r} on the valid side")
            elif "Nymeria" in reader:
                dataset = "Nymeria"
            elif "NaturalMotion" in reader:
                dataset = "NaturalMotion"
            else:
                raise AssertionError(f"unexpected Exp7-C validation reader: {reader}")
            pipelines[dataset].append((str(stem), build))

    assert set(pipelines) == set(DATASET_ORDER), f"fixed suite changed: found {sorted(pipelines)}"
    return dict(pipelines)


def collect(args, shard: int, shards: int, report: eval_stream.Report) -> dict[str, Cell]:
    """Accumulate this shard's slice of every dataset; builds its own model, as CUDA cannot cross the spawn."""
    torch.manual_seed(args.seed)                # pins the corpus builders' construction seed
    model    = load_model(Path(args.ckpt)).eval().to(eval_stream.shard_device(shard))
    sensor   = CANONICAL_IDX[args.sensor]
    prefix   = f"{args.sensor}/"
    accum    = {name: Cell() for name in DATASET_ORDER}

    for dataset, items in fixed_valid_pipelines().items():
        cell = accum[dataset]
        for metas, chunks in eval_stream.stream_chunks(
            eval_stream.shard_items(items[: args.limit], shard, shards),
            seed       = args.seed,
            workers    = args.workers,
            chunk_rows = args.chunk_rows,
            report     = lambda n, dataset=dataset: report(dataset, n),
        ):
            for meta in metas:
                # Tally off the metas so it survives takes being merged.
                if meta.windows == 0:
                    cell.short += 1
                elif sensor not in meta.sensor_ids:
                    cell.missing += 1
                else:
                    cell.takes   += 1
                    cell.windows += meta.windows

            for chunk in chunks:
                batch  = tp.cast(IMUData_LabelTrait_Jagged, chunk.batch)   # the suite is all GT-labelled
                metric = evaluate.Velocity_Metric(batch, inference(model, chunk.batch))
                # Keep this sensor's entries, keyed vel_rmse / vel_l1 / vel_mse.
                part = {k.removeprefix(prefix): v for k, v in metric.parts.items() if k.startswith(prefix)}
                if part:
                    cell.parts.append((chunk.index, part))
    return accum


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("ckpt", help="checkpoint path, absolute or relative to ${machine-conf:exp_root}")
    parser.add_argument("--sensor", choices=CANONICAL_NAMES, default="pelvis", help="canonical sensor to evaluate (default: pelvis)")
    parser.add_argument("--limit", type=int, default=None, help="first N takes per dataset (smoke test; results are partial)")
    parser.add_argument("--seed", type=int, default=7, help="pin per-stem construction and transform draws")
    eval_stream.add_stream_args(parser)
    args = parser.parse_args()

    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.exists():
        ckpt = Path(machine_conf("exp_root")) / args.ckpt
    if not ckpt.exists():
        parser.error(f"checkpoint does not exist: {ckpt}")
    args.ckpt = str(ckpt)                       # resolved once here; the shards reuse it

    console = Console()
    shards  = eval_stream.device_count(args.devices)
    # Composing the pipelines touches no device, so the parent stays CUDA-free.
    totals  = {name: len(items[: args.limit]) for name, items in fixed_valid_pipelines().items()}
    console.print(eval_stream.worker_summary(shards, args.workers))

    accum = {name: Cell() for name in DATASET_ORDER}
    for partial in eval_stream.run_sharded(collect, args, shards, totals):
        for name, cell in partial.items():
            accum[name].merge(cell)

    table = Table(title=f"Fixed generalization-suite velocity error — sensor: {args.sensor}\n{ckpt}")
    table.add_column("dataset", justify="left")
    for column in ("takes", "windows", "rows", "RMSE (m/s)", "L1 (m/s)"):
        table.add_column(column, justify="right")

    for dataset in DATASET_ORDER:
        cell = accum[dataset]
        pooled = cell.pooled()
        rows = pooled["vel_l1"].count if pooled else 0
        rmse = pooled["vel_rmse"].value if pooled else float("nan")
        l1 = pooled["vel_l1"].value if pooled else float("nan")
        notes = []
        if cell.short:
            notes.append(f"{cell.short} short")
        if cell.missing:
            notes.append(f"{cell.missing} without {args.sensor}")
        takes = str(cell.takes) + (f" ({', '.join(notes)})" if notes else "")
        table.add_row(dataset, takes, str(cell.windows), str(rows), f"{rmse:.4f}", f"{l1:.4f}")

    if args.limit is not None:
        console.print(f"[yellow]--limit {args.limit}: partial results, not the full fixed suite[/yellow]")
    console.print(table)


if __name__ == "__main__":
    main()
