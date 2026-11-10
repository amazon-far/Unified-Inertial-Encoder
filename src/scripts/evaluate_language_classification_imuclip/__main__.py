"""Zero-shot language retrieval for the IMU2CLIP baseline, on the suite and metrics of ``evaluate_language_classification``.

Usage:
    python -m src.scripts.evaluate_language_classification_imuclip <ckpt> [--limit N]
"""

import argparse

from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from ... import eval_stream
from ...interface.data_imu import IMUData_w_GT_Lang_Jagged
from ...model.imu2clip import IMU2CLIP
from ...runtime_prologue import machine_conf
from ..evaluate_language_classification.__main__ import (
    DATASET_ORDER,
    _Cell,
    accumulate,
    add_suite_args,
    balanced_accuracy,
    chance_row,
    imu_to_text,
    imu_to_text_sampled,
    lang_variant,
    spherical_kmeans,
    suite_pipelines,
    tally,
    text_to_imu,
    tolerant_top1,
)


def load_model(ckpt: Path) -> IMU2CLIP:
    """Recover the IMU2CLIP encoder from a ``src.train_imu2clip`` run's config + checkpoint."""
    ckpt    = Path(ckpt).expanduser()
    run_dir = ckpt.parent.parent    # checkpoints live at <run_dir>/checkpoints/<ckpt>
    cfg     = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    model: IMU2CLIP = hydra.utils.instantiate(cfg.model)
    state   = torch.load(ckpt, map_location="cpu", weights_only=False)
    weights = {k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")}
    # Some checkpoints carry a `lang_mean` key the model does not define.
    weights.pop("lang_mean", None)
    model.load_state_dict(weights)
    return model


def collect(args, shard: int, shards: int, report: eval_stream.Report) -> dict[str, _Cell]:
    """Accumulate this shard's slice of every dataset.  Runs in its own process, one per GPU."""
    torch.manual_seed(args.seed)                # pins the corpus builders' construction seed
    device = eval_stream.shard_device(shard)
    model  = load_model(Path(args.ckpt)).eval().to(device)
    accum  = {name: _Cell() for name in DATASET_ORDER}

    for dataset, items in suite_pipelines(args).items():
        cell = accum[dataset]
        for metas, chunks in eval_stream.stream_chunks(
            eval_stream.shard_items(items[: args.limit], shard, shards),
            seed       = args.seed,
            workers    = args.workers,
            chunk_rows = args.chunk_rows,
            report     = lambda n, dataset=dataset: report(dataset, n),
        ):
            tally(cell, metas)
            for chunk in chunks:
                assert isinstance(chunk.batch, IMUData_w_GT_Lang_Jagged), type(chunk.batch).__name__
                with torch.inference_mode():
                    out = model(chunk.batch.to(device, model.compute_dtype)).embed.float().cpu()
                accumulate(cell, chunk, out)            # [B, 768], one embedding per window
    return accum


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("ckpt", help="src.train_imu2clip checkpoint, absolute or relative to ${machine-conf:exp_root}")
    parser.add_argument("--limit", type=int, default=None, help="first N takes per dataset (smoke test; results and the gallery are partial)")
    parser.add_argument("--seed", type=int, default=7, help="pin per-stem construction, transform draws, and tie-breaking order")
    add_suite_args(parser)
    parser.add_argument("--gallery", type=int, nargs="+", default=[100, 20],
                        help="fixed gallery sizes for difficulty-calibrated retrieval (correct phrase + N-1 distractors)")
    parser.add_argument("--tau", type=float, default=0.95,
                        help="cosine above which a retrieved phrase counts as the same motion, differently worded")
    parser.add_argument("--categories", type=int, nargs="+", default=[20, 50, 200],
                        help="category counts for the coarse readout; each K clusters the phrase gallery by cosine")
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
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Composing the pipelines touches no device, so the parent stays CUDA-free.
    totals  = {name: len(items[: args.limit])
               for name, items in suite_pipelines(args).items()}
    console.print(eval_stream.worker_summary(shards, args.workers))

    accum = {name: _Cell() for name in DATASET_ORDER}
    for partial in eval_stream.run_sharded(collect, args, shards, totals):
        for name, cell in partial.items():
            accum[name].merge(cell)

    # Gallery, predictions and targets, rebuilt in the undivided take order (see _Cell.resolve).
    resolved = {name: cell.resolve() for name, cell in accum.items()}

    if args.limit is not None:
        console.print(f"[yellow]--limit {args.limit}: partial results; the gallery only covers the evaluated windows[/yellow]")

    if not resolved["Nymeria"][1].numel():
        variant = lang_variant(args)
        console.print(f"[red]Nymeria produced no annotated window: lang_spec"
                      f"{'.' + variant if variant else ''}.json is missing from the "
                      f"cache, or none of its spans cover a window.[/red]")

    # (gallery, preds, targets) per dataset, shared by the phrase and category readouts.
    ranked: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    table = Table(title=f"IMU2CLIP language retrieval, per-dataset phrase gallery "
                        f"(Nymeria lang_variant={args.lang_variant})\n{ckpt}")
    for column in ("dataset", "variant", "takes", "windows", "N", "K", "cos", "R@1", "R@5", "R@10", "MRR", "t2i mAP"):
        table.add_column(column, justify="right" if column not in ("dataset", "variant") else "left")

    for dataset in DATASET_ORDER:
        cell = accum[dataset]
        preds, targets, gallery = resolved[dataset]
        n, k = targets.numel(), gallery.shape[0]
        takes = str(cell.takes) + (f" ({cell.short} short)" if cell.short else "")
        n_col = f"{n} ({n / max(cell.windows, 1):.0%})"
        if not n:
            table.add_row(dataset, "model", takes, str(cell.windows), "0", "0", *("nan",) * 6)
            table.add_section()
            continue

        gallery, preds, targets = gallery.to(device), preds.to(device), targets.to(device)

        # Break score ties without leaking dataset order.
        shuffle = torch.randperm(n, device=device)
        preds, targets = preds[shuffle], targets[shuffle]
        ranked[dataset] = (gallery, preds, targets)

        counts   = torch.bincount(targets, minlength=k)
        collapse = F.normalize(gallery[targets].mean(dim=0), dim=-1).expand(n, -1)

        rows = {
            "chance"  : chance_row(counts.cpu(), n),
            "collapse": imu_to_text(collapse, targets, gallery) | {"t2i": text_to_imu(collapse, targets, gallery)},
            "model"   : imu_to_text(preds, targets, gallery)    | {"t2i": text_to_imu(preds, targets, gallery)},
        }
        for variant, m in rows.items():
            cos = f"{cell.cos_sum / n:.4f}" if variant == "model" else "-"
            head = variant == "chance"
            table.add_row(
                dataset if head else "", variant,
                takes if head else "", str(cell.windows) if head else "",
                n_col if head else "", str(k) if head else "",
                cos, f"{m['R@1']:.4f}", f"{m['R@5']:.4f}", f"{m['R@10']:.4f}",
                f"{m['MRR']:.4f}", f"{m['t2i']:.4f}",
            )
        table.add_section()

    console.print(table)
    console.print("[dim]R@k and MRR are IMU->text over the K distinct phrases; t2i mAP is text->IMU over the N "
                  "annotated windows. Both scale with K/N, so rows are not comparable across datasets.[/dim]")

    # Same predictions, scored at several difficulties.
    table = Table(title=f"Difficulty-calibrated phrase retrieval (tau={args.tau}, "
                        f"{'/'.join(map(str, args.gallery))}-way galleries averaged over 8 draws)")
    cols = ["dataset", "variant", "K", "R@1 full", f"R@1 tau", *[f"R@1 N={n}" for n in args.gallery]]
    for column in cols:
        table.add_column(column, justify="right" if column not in ("dataset", "variant") else "left")

    for dataset in DATASET_ORDER:
        if dataset not in ranked:
            continue
        gallery, preds, targets = ranked[dataset]
        collapse = F.normalize(gallery[targets].mean(dim=0), dim=-1).expand(preds.shape[0], -1)
        for variant, p in (("collapse", collapse), ("model", preds)):
            row = [
                f"{imu_to_text(p, targets, gallery, ks=(1,))['R@1']:.4f}",
                f"{tolerant_top1(p, targets, gallery, args.tau):.4f}",
                *[f"{imu_to_text_sampled(p, targets, gallery, n, seed=args.seed)['R@1']:.4f}"
                  for n in args.gallery],
            ]
            head = variant == "collapse"
            table.add_row(dataset if head else "", variant,
                          str(gallery.shape[0]) if head else "", *row)
        table.add_section()

    console.print(table)
    console.print("[dim]'R@1 tau' keeps the full gallery but counts any phrase within tau cosine of the target; "
                  "its gap to 'R@1 full' is the share of error that is wording, not motion. 'R@1 N=' ranks the "
                  "target against N-1 random distractors, so chance is 1/N regardless of corpus.[/dim]")

    # Coarse readout: retrieve a category rather than an exact phrase.
    table = Table(title="IMU2CLIP category retrieval — spherical k-means over each dataset's phrase gallery")
    for column in ("dataset", "C", "phrases", "chance", "collapse R@1", "model R@1", "model BAcc", "model MRR"):
        table.add_column(column, justify="right" if column != "dataset" else "left")

    for dataset in DATASET_ORDER:
        if dataset not in ranked:
            continue
        gallery, preds, targets = ranked[dataset]
        k = gallery.shape[0]
        collapse = F.normalize(gallery[targets].mean(dim=0), dim=-1).expand(preds.shape[0], -1)

        for c in sorted({min(c, k) for c in args.categories}):
            label, protos = spherical_kmeans(gallery, c, seed=args.seed)
            cat_target    = label[targets]
            c_eff         = protos.shape[0]
            model_m       = imu_to_text(preds,    cat_target, protos, ks=(1,))
            coll_m        = imu_to_text(collapse, cat_target, protos, ks=(1,))
            bacc          = balanced_accuracy((preds @ protos.T).argmax(dim=1), cat_target, c_eff)
            table.add_row(
                dataset if c == min(min(x, k) for x in args.categories) else "",
                str(c_eff), str(k), f"{1.0 / c_eff:.4f}",
                f"{coll_m['R@1']:.4f}", f"{model_m['R@1']:.4f}", f"{bacc:.4f}", f"{model_m['MRR']:.4f}",
            )
        table.add_section()

    console.print(table)
    console.print("[dim]Categories are clusters of the phrase embeddings, so they need no metadata and cover "
                  "every corpus alike. BAcc is mean per-class recall (clusters are imbalanced); it shares the "
                  "1/C chance level with R@1.[/dim]")


if __name__ == "__main__":
    main()
