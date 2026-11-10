"""Velocity error over a run's validation set, overall and per device.

Usage:
    python -m src.scripts.evaluate_velocity <ckpt> [<ckpt> ...] [--limit N] [--seed 7] \
        [--data_from <run_dir>] [--json out.json]
"""

import argparse
import functools
import json
import typing as tp

from dataclasses import dataclass, field
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ... import evaluate
from ...eval import inference, load_model
from ...data.util import corpus_pipelines
from ...interface.human import CANONICAL_IDX, CANONICAL_NAMES
from ...runtime_prologue import machine_conf


def valid_pipelines(run_cfg) -> dict[str, list]:
    """Map each reader class name on the run's valid side to its lazy per-take builders.

    Stem seeding builds on the process seed, so call ``torch.manual_seed`` first.
    """
    groups: dict[str, list] = {}
    for entry in run_cfg.data.valid:
        target = str(entry["_target_"])
        if target.endswith("build_corpus"):
            reader = str(entry["reader"]["_target_"])
            spec   = OmegaConf.to_container(entry, resolve=True)
            assert isinstance(spec, dict), f"unexpected valid entry {spec!r}"
            kwargs = {str(k): v for k, v in spec.items() if not str(k).startswith("_")}
            stems, build = corpus_pipelines(**kwargs)
            builders = [functools.partial(build, stem) for stem in stems]
        elif target.endswith("build_nested_seq"):
            reader   = str(entry["layers"][0]["_target_"])
            builders = [functools.partial(hydra.utils.instantiate, entry)]
        else:
            reader   = target
            builders = [functools.partial(hydra.utils.instantiate, entry)]
        groups.setdefault(reader.rsplit(".", 1)[-1], []).extend(builders)
    return groups


@dataclass
class Cell:
    """One dataset's tally; ``parts`` holds each take's statistics, pooled row-weighted at print time."""
    takes  : int = 0
    windows: int = 0
    short  : int = 0
    parts  : list[dict[str, evaluate.MetricPart]] = field(default_factory=list)


def accumulate(cell: Cell, metric: "evaluate.Velocity_Metric", n_windows: int) -> None:
    cell.parts.append(metric.parts)
    cell.windows += n_windows
    cell.takes   += 1


def merged(cells: list[Cell]) -> Cell:
    return Cell(
        takes   = sum(c.takes   for c in cells),
        windows = sum(c.windows for c in cells),
        short   = sum(c.short   for c in cells),
        parts   = [p for c in cells for p in c.parts],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("ckpt", type=str, nargs="+", help="checkpoint path(s), absolute or relative to ${machine-conf:exp_root}; several are scored in one pass over the corpora")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N takes per dataset (smoke test; numbers are partial)")
    parser.add_argument("--seed", type=int, default=7, help="pins per-stem construction and per-fetch transform draws")
    parser.add_argument("--data_from", type=str, default=None, help="run dir whose .hydra config supplies the valid corpora (default: the checkpoint's own run)")
    parser.add_argument("--json", type=str, default=None, help="write the pooled per-dataset x per-device statistics here")
    args = parser.parse_args()

    def resolve(raw: str) -> Path:
        p = Path(raw).expanduser()
        return p if p.exists() else Path(machine_conf("exp_root")) / raw

    ckpts    = [resolve(raw) for raw in args.ckpt]
    data_run = Path(args.data_from).expanduser() if args.data_from else ckpts[0].parent.parent
    run_cfg  = OmegaConf.load(data_run / ".hydra" / "config.yaml")

    # Label each checkpoint by its run dir, disambiguated within a run.
    labels = [c.parent.parent.name for c in ckpts]
    labels = [f"{n}#{i}" if labels.count(n) > 1 else n for i, n in enumerate(labels)]

    torch.manual_seed(args.seed)
    models = []
    for label, c in zip(labels, ckpts):
        m = load_model(c).eval()
        models.append((label, m.cuda() if torch.cuda.is_available() else m))

    # Each take is built once, so every checkpoint sees byte-identical inputs.
    acc: dict[str, dict[str, Cell]] = {label: {} for label in labels}
    with Progress() as pbar:
        for dataset, builders in valid_pipelines(run_cfg).items():
            builders = builders[: args.limit]
            cells    = {label: acc[label].setdefault(dataset, Cell()) for label in labels}
            task     = pbar.add_task(f"{dataset} valid takes", total=len(builders))
            for build in builders:
                seq = build()
                if len(seq) == 0:
                    for cell in cells.values():
                        cell.short += 1         # take shorter than one window
                    pbar.advance(task)
                    continue
                # Collate once: inference and the metric must see the same batch (see `inference`).
                batch = seq[:]
                for label, model in models:
                    accumulate(cells[label], evaluate.Velocity_Metric(batch, inference(model, batch)), len(seq))
                pbar.advance(task)

    def render(label: str, ckpt: Path) -> Table:
        title = f"Velocity error on the validation set, per dataset x device\n{ckpt}"
        if args.data_from:
            title += f"\nvalid corpora from {data_run}"
        table = Table(title=title)
        for col in ("dataset", "device", "rows", "vel_mse (m/s)^2", "vel_rmse m/s", "vel_l1 m/s"):
            table.add_column(col, justify="right" if col[0].islower() and col not in ("dataset", "device") else "left")
        for dataset, cell in acc[label].items():
            add_rows(table, dataset, cell)
            table.add_section()
        add_rows(table, "ALL (entire valid set)", merged(list(acc[label].values())))
        return table

    def add_rows(table: Table, dataset: str, cell: Cell) -> None:
        pooled = evaluate.merge_parts(cell.parts)

        def fmt(prefix: str) -> tuple[str, str, str, str]:
            mse, rmse, l1 = (pooled.get(f"{prefix}vel_{k}") for k in ("mse", "rmse", "l1"))
            if mse is None or rmse is None or l1 is None:
                return "0", "-", "-", "-"
            return f"{mse.count}", f"{mse.value:.6f}", f"{rmse.value:.4f}", f"{l1.value:.4f}"

        takes = f"{cell.takes} takes, {cell.windows} windows" + (f" (+{cell.short} short)" if cell.short else "")
        table.add_row(f"{dataset}\n{takes}", "(all devices)", *fmt(""))
        # Devices are whatever the pooled keys carry; unknown ids sort last.
        devices = {key.split("/", 1)[0] for key in pooled if "/" in key}
        order   = tp.cast(dict[str, int], CANONICAL_IDX)
        for name in sorted(devices, key=lambda n: order.get(n, len(CANONICAL_NAMES))):
            table.add_row("", name, *fmt(f"{name}/"))

    if args.limit is not None:
        Console().print(f"[yellow]--limit {args.limit}: partial numbers, not the full valid set[/yellow]")
    for label, ckpt in zip(labels, ckpts):
        Console().print(render(label, ckpt))

    if args.json:
        def dump(cell: Cell) -> dict:
            pooled = evaluate.merge_parts(cell.parts)
            return {
                "takes": cell.takes, "windows": cell.windows, "short": cell.short,
                "metrics": {k: {"value": p.value, "count": p.count, "total": p.total, "root": p.root}
                            for k, p in pooled.items()},
            }

        Path(args.json).write_text(json.dumps({
            "data_from": str(data_run), "seed": args.seed, "limit": args.limit,
            "ckpt": {label: str(c) for label, c in zip(labels, ckpts)},
            "results": {label: {"datasets": {ds: dump(cell) for ds, cell in acc[label].items()},
                                "all": dump(merged(list(acc[label].values())))}
                        for label in labels},
        }, indent=2))
        Console().print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
