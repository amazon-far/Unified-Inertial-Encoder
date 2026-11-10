"""Zero-shot language retrieval for the ImageBind baseline, on the suite and metrics of ``evaluate_language_classification``.

Usage:
    python -m src.scripts.evaluate_language_classification_imagebind
    python -m src.scripts.evaluate_language_classification_imagebind --sensor head pelvis left_hand
"""

import argparse
import dataclasses
import typing as tp

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from ... import eval_stream
from ...data.lang.cache_text import embedding_key, text_by_embedding
from ...interface.data_imu import IMUData_w_GT_Lang_Jagged
from ...interface.human import CANONICAL_NAMES
from ...model.imagebind import FRAME_SPECS, WINDOW_MS, ImageBind_IMU, all_frames
from ..evaluate_language_classification.__main__ import (
    CONFIG_DIR,
    DATASET_ORDER,
    _Cell,
    _window_keys,
    add_suite_args,
    balanced_accuracy,
    chance_row,
    embed,
    fixed_valid_cache_dirs,
    imu_to_text,
    imu_to_text_sampled,
    lang_variant,
    spherical_kmeans,
    suite_pipelines,
    tally,
    text_to_imu,
    tolerant_top1,
)

MODEL_CONFIG = CONFIG_DIR / "model" / "imagebind.yaml"

# preds/targets/gallery per dataset, as `_Cell.resolve` returns them.
Resolved = dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

# Separates a swept hypothesis's sensor and frame in the label keying the shard accumulators.
_SEP = " | "


def variants(args) -> list[str]:
    """Every (sensor, frame) hypothesis this run scores, as ``"<sensor> | <frame>"`` labels."""
    return [f"{s}{_SEP}{f}" for s in args.sensor for f in args.frame]


def split_variant(label: str) -> tuple[str, str]:
    """``"head | +x+y+z"`` -> ``("head", "+x+y+z")``."""
    sensor, frame = label.split(_SEP)
    return sensor, frame


@dataclasses.dataclass
class _IBCell(_Cell):
    """``_Cell`` plus the annotated windows ImageBind excluded as shorter than its 10 s input."""
    unrepresentable: int = 0

    def merge(self, other: "_Cell") -> None:
        super().merge(other)
        self.unrepresentable += getattr(other, "unrepresentable", 0)


def build_model(args, modalities: tuple[str, ...]) -> ImageBind_IMU:
    """Instantiate ``src/config/model/imagebind.yaml`` with this run's overrides."""
    cfg = OmegaConf.load(MODEL_CONFIG)
    overrides: dict[str, tp.Any] = {
        "sensor"    : args.sensor[0],
        "frame"     : args.frame[0],
        "modalities": list(modalities),
        # What the pipeline actually delivers, so the model can validate against it.
        "window_ms" : args.window_ms,
    }
    if args.ckpt is not None:
        overrides["ckpt"] = args.ckpt
    return hydra.utils.instantiate(cfg, **overrides)


def accumulate_crossspace(
    cell : _IBCell,
    chunk: eval_stream.Chunk,
    out  : torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Fold one forward into ``cell``, keeping the phrase rows UNNORMALIZED.

    The phrase rows are the key for the gallery's ImageBind text embedding
    (``cache_text.embedding_key``), so they must stay the raw fp32 widening of the stored bf16.
    ``valid`` masks windows the encoder could not represent. ``cell.cos_sum`` stays 0: cosine
    between an ImageBind embedding and a CLIP row is meaningless.
    """
    batch   = chunk.batch
    assert isinstance(batch, IMUData_w_GT_Lang_Jagged), type(batch).__name__
    covered = batch.lang_feat_idx.cpu() >= 0                          # [B]
    cell.unrepresentable += int((covered & ~valid.cpu()).sum())
    covered &= valid.cpu()
    if not bool(covered.any()):
        return

    idx       = batch.lang_feat_idx.cpu()[covered]                    # rows of the compact storage
    uniq, inv = idx.unique(return_inverse=True)
    phrases   = batch.lang_feature.cpu()[uniq].float()                # [u, 768], as cached
    cell.blocks.append((embed(out[covered]), phrases, inv, _window_keys(chunk)[covered]))


def collect(args, shard: int, shards: int, report: eval_stream.Report) -> dict[str, dict[str, _IBCell]]:
    """Accumulate this shard's slice of every dataset, for every swept hypothesis (IMU tower only)."""
    torch.manual_seed(args.seed)                # pins the corpus builders' construction seed
    device = eval_stream.shard_device(shard)
    model  = build_model(args, ("imu",)).eval().to(device)
    accum  = {v: {name: _IBCell() for name in DATASET_ORDER} for v in variants(args)}

    for dataset, items in suite_pipelines(args).items():
        for metas, chunks in eval_stream.stream_chunks(
            eval_stream.shard_items(items[: args.limit], shard, shards),
            seed       = args.seed,
            workers    = args.workers,
            chunk_rows = args.chunk_rows,
            report     = lambda n, dataset=dataset: report(dataset, n),
        ):
            for v in variants(args):
                tally(accum[v][dataset], metas)
            for chunk in chunks:
                batch = chunk.batch.to(device, model.compute_dtype)
                for v in variants(args):
                    site, frame = split_variant(v)
                    out = model(batch, sensor=site, frame=frame)              # [B, 1024]
                    accumulate_crossspace(accum[v][dataset], chunk,
                                          out.embed.float().cpu(), out.valid)
    return accum


def rebuild_galleries(
    args,
    resolved: dict[str, Resolved],
    console : Console,
) -> dict[str, dict[str, torch.Tensor]]:
    """Replace every cached-CLIP gallery with ImageBind text embeddings of the same phrases, same row order.

    Phrase strings come from the raw label files; a missing row raises, as it would mislabel the gallery.
    """
    variant  = lang_variant(args)
    scored   = {dataset for per_dataset in resolved.values()
                for dataset, (_, targets, _) in per_dataset.items() if targets.numel()}
    needed   = [d for dataset, dirs in fixed_valid_cache_dirs(variant).items()
                if dataset in scored for d in dirs[: args.limit]]
    console.print(f"[dim]recovering phrase text for {len(needed)} takes from the raw label files "
                  f"under ${{machine-conf:data_root}}[/dim]")
    table, aliased = text_by_embedding(needed, variant)
    if aliased:
        # Already one gallery entry each; this only decides which spelling gets re-encoded.
        console.print(f"[dim]{aliased} of {len(table)} phrase rows have several source spellings "
                      f"that tokenize identically for CLIP; the lexicographically first is "
                      f"re-encoded[/dim]")

    model = build_model(args, ("text",)).eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cache: dict[str, torch.Tensor] = {}

    def encode(texts: list[str]) -> torch.Tensor:
        fresh = [t for t in dict.fromkeys(texts) if t not in cache]
        if fresh:
            cache.update(zip(fresh, F.normalize(model.encode_text(fresh).float(), dim=-1)))
            console.print(f"[dim]encoded {len(fresh)} new phrases with ImageBind's text tower[/dim]")
        return torch.stack([cache[t] for t in texts])

    galleries: dict[str, dict[str, torch.Tensor]] = {}
    for site, per_dataset in resolved.items():
        galleries[site] = {}
        for dataset, (_, targets, gallery) in per_dataset.items():
            if not targets.numel():
                continue
            texts = []
            for row in gallery:
                text = table.get(embedding_key(row))
                assert text is not None, (
                    f"{dataset}: a gallery phrase embedding has no recovered text; the processed "
                    f"cache and the label file under ${{machine-conf:data_root}} disagree"
                )
                texts.append(text)
            galleries[site][dataset] = encode(texts)
    return galleries


def rank_inputs(
    resolved : Resolved,
    galleries: dict[str, torch.Tensor],
    device   : torch.device,
    seed      : int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(gallery, preds, targets)`` per scored dataset, shuffled identically for every swept sensor."""
    out: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for dataset, (preds, targets, _) in resolved.items():
        if not targets.numel():
            continue
        gen     = torch.Generator(device=device).manual_seed(seed)
        shuffle = torch.randperm(targets.numel(), device=device, generator=gen)
        out[dataset] = (galleries[dataset].to(device),
                        preds.to(device)[shuffle], targets.to(device)[shuffle])
    return out


def print_headline(args, console: Console, accum, resolved: Resolved, ranked) -> None:
    """Per-dataset retrieval table, with cosine measured in ImageBind's space."""
    table = Table(title=f"ImageBind zero-shot language retrieval, per-dataset phrase gallery "
                        f"(sensor={args.sensor[0]}, frame={args.frame[0]}, window={args.window_ms:.0f} ms, "
                        f"Nymeria lang_variant={args.lang_variant})")
    for column in ("dataset", "variant", "takes", "windows", "short win", "N", "K", "cos", "R@1", "R@5", "R@10", "MRR", "t2i mAP"):
        table.add_column(column, justify="right" if column not in ("dataset", "variant") else "left")

    for dataset in DATASET_ORDER:
        cell  = accum[dataset]
        n     = resolved[dataset][1].numel()
        takes = str(cell.takes) + (f" ({cell.short} short)" if cell.short else "")
        n_col = f"{n} ({n / max(cell.windows, 1):.0%})"
        # Annotated windows the encoder could not represent (take shorter than its 10 s input).
        drop  = str(cell.unrepresentable)
        if dataset not in ranked:
            table.add_row(dataset, "model", takes, str(cell.windows), drop, "0", "0", *("nan",) * 6)
            table.add_section()
            continue

        gallery, preds, targets = ranked[dataset]
        k        = gallery.shape[0]
        counts   = torch.bincount(targets, minlength=k)
        collapse = F.normalize(gallery[targets].mean(dim=0), dim=-1).expand(n, -1)
        # Cosine to the matched phrase, in ImageBind's own space.
        cos_mean = float((preds * gallery[targets]).sum(dim=-1).mean())

        rows = {
            "chance"  : chance_row(counts.cpu(), n),
            "collapse": imu_to_text(collapse, targets, gallery) | {"t2i": text_to_imu(collapse, targets, gallery)},
            "model"   : imu_to_text(preds, targets, gallery)    | {"t2i": text_to_imu(preds, targets, gallery)},
        }
        for variant, m in rows.items():
            cos  = f"{cos_mean:.4f}" if variant == "model" else "-"
            head = variant == "chance"
            table.add_row(
                dataset if head else "", variant,
                takes if head else "", str(cell.windows) if head else "", drop if head else "",
                n_col if head else "", str(k) if head else "",
                cos, f"{m['R@1']:.4f}", f"{m['R@5']:.4f}", f"{m['R@10']:.4f}",
                f"{m['MRR']:.4f}", f"{m['t2i']:.4f}",
            )
        table.add_section()

    console.print(table)
    console.print("[dim]R@k and MRR are IMU->text over the K distinct phrases; t2i mAP is text->IMU over the N "
                  "annotated windows. Both scale with K/N, so rows are not comparable across datasets. "
                  "Retrieval ranks in ImageBind's 1024-d joint space, not the cached CLIP space the "
                  "sibling evaluators use. 'short win' counts annotated windows dropped because their take "
                  "is shorter than ImageBind's 10 s input; they are excluded from N, not stretched.[/dim]")


def print_difficulty(args, console: Console, ranked) -> None:
    """Difficulty-calibrated table: same predictions, several gallery sizes."""
    table = Table(title=f"Difficulty-calibrated phrase retrieval (tau={args.tau}, "
                        f"{'/'.join(map(str, args.gallery))}-way galleries averaged over 8 draws)")
    for column in ["dataset", "variant", "K", "R@1 full", "R@1 tau", *[f"R@1 N={n}" for n in args.gallery]]:
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
                  "target against N-1 random distractors, so chance is 1/N regardless of corpus. Note tau is "
                  "an ImageBind-space cosine here; its text geometry is not CLIP's, so the tau row is only "
                  "comparable across ImageBind runs.[/dim]")


def print_categories(args, console: Console, ranked) -> None:
    """Coarse readout: retrieve a phrase cluster rather than an exact phrase."""
    table = Table(title="ImageBind category retrieval — spherical k-means over each dataset's phrase gallery")
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


def print_sweep(args, console: Console, ranked_by_variant: dict[str, dict], top: int = 8) -> None:
    """One row per (dataset, sensor, frame) hypothesis, best-first on the smallest-gallery R@1, top ``top``."""
    n_way = min(args.gallery)
    table = Table(title=f"ImageBind sensor/frame sweep (window={args.window_ms:.0f} ms, "
                        f"Nymeria lang_variant={args.lang_variant})")
    for column in ("dataset", "sensor", "frame", "N", "K", f"R@1 N={n_way}", "R@1 full", "MRR", "t2i mAP"):
        table.add_column(column, justify="right" if column not in ("dataset", "sensor", "frame") else "left")

    dropped = 0
    for dataset in DATASET_ORDER:
        present = [v for v in variants(args) if dataset in ranked_by_variant[v]]
        if not present:
            continue
        gallery, _, targets = ranked_by_variant[present[0]][dataset]
        collapse = F.normalize(gallery[targets].mean(dim=0), dim=-1).expand(targets.numel(), -1)
        scored = []
        for v in present:
            _, preds, v_targets = ranked_by_variant[v][dataset]
            scored.append((v, {
                "n_way": imu_to_text_sampled(preds, v_targets, gallery, n_way, seed=args.seed)["R@1"],
                **imu_to_text(preds, v_targets, gallery),
                "t2i" : text_to_imu(preds, v_targets, gallery),
            }))
        scored.sort(key=lambda row: -row[1]["n_way"])
        dropped += max(0, len(scored) - top)
        scored = scored[:top]

        floor = {
            "n_way": imu_to_text_sampled(collapse, targets, gallery, n_way, seed=args.seed)["R@1"],
            **imu_to_text(collapse, targets, gallery),
            "t2i" : text_to_imu(collapse, targets, gallery),
        }
        for i, (v, m) in enumerate([("collapse", floor), *scored]):
            sensor, frame = split_variant(v) if _SEP in v else (v, "")
            best = i == 1 and len(scored) > 1        # the winning hypothesis, right under the floor
            table.add_row(
                dataset if i == 0 else "",
                f"[bold]{sensor}[/bold]" if best else sensor,
                f"[bold]{frame}[/bold]" if best else frame,
                str(targets.numel()) if i == 0 else "", str(gallery.shape[0]) if i == 0 else "",
                f"{m['n_way']:.4f}", f"{m['R@1']:.4f}", f"{m['MRR']:.4f}", f"{m['t2i']:.4f}",
            )
        table.add_section()

    console.print(table)
    if dropped:
        console.print(f"[dim]{dropped} lower-scoring hypothesis rows hidden (top {top} per dataset); "
                      f"every one was scored, none were skipped[/dim]")
    console.print(f"[dim]All hypotheses see the SAME windows and the same gallery, so these rows are "
                  f"directly comparable. Chance on the R@1 N={n_way} column is {1 / n_way:.1%}; a row that "
                  f"does not clearly beat 'collapse' carries no window-specific signal for ImageBind. "
                  f"Note the best row of a large sweep is an oracle over hypotheses -- report it as "
                  f"'best of {len(variants(args))}', not as a single measurement.[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--ckpt", default=None,
                        help=f"imagebind_huge.pth (default: the path in {MODEL_CONFIG.name})")
    parser.add_argument("--sensor", nargs="+", default=None, choices=CANONICAL_NAMES, metavar="SITE",
                        help=f"body site(s) the single-IMU baseline consumes (default: the committed "
                             f"'head' in {MODEL_CONFIG.name}). Several sites are scored in one pass "
                             f"and printed as a placement sweep instead of the full readout")
    parser.add_argument("--frame", nargs="+", default=None, metavar="SPEC",
                        help=f"axis convention(s) mapping our body frame (X forward, Y left, Z up) "
                             f"onto the encoder's, as '+x-z+y'-style specs or the names "
                             f"{sorted(FRAME_SPECS)}; 'all' sweeps every one of the 24 right-handed "
                             f"mountings. Default: the committed frame in {MODEL_CONFIG.name}. "
                             f"ImageBind's own convention is undocumented (Ego4D defines no axes "
                             f"and its IMU is uncalibrated), so this is meant to be measured")
    parser.add_argument("--limit", type=int, default=None, help="first N takes per dataset (smoke test; results and the gallery are partial)")
    parser.add_argument("--seed", type=int, default=7, help="pin per-stem construction, transform draws, and tie-breaking order")
    # ImageBind's input is 2000 samples at 200 Hz: results are not comparable to 4096 ms windows.
    add_suite_args(parser, window_ms=WINDOW_MS)
    parser.add_argument("--gallery", type=int, nargs="+", default=[100, 20],
                        help="fixed gallery sizes for difficulty-calibrated retrieval (correct phrase + N-1 distractors)")
    parser.add_argument("--tau", type=float, default=0.95,
                        help="cosine above which a retrieved phrase counts as the same motion, differently worded")
    parser.add_argument("--categories", type=int, nargs="+", default=[20, 50, 200],
                        help="category counts for the coarse readout; each K clusters the phrase gallery by cosine")
    eval_stream.add_stream_args(parser)
    args = parser.parse_args()

    cfg = OmegaConf.load(MODEL_CONFIG)                      # resolved once; the shards reuse it
    if args.sensor is None:
        args.sensor = [str(cfg.sensor)]
    if args.frame is None:
        args.frame = [str(OmegaConf.select(cfg, "frame", default="xsens"))]
    elif args.frame == ["all"]:
        args.frame = all_frames()
    args.frame = [FRAME_SPECS.get(f, f) for f in args.frame]        # names -> specs, once
    assert len(set(args.sensor)) == len(args.sensor), f"--sensor repeats a site: {args.sensor}"
    assert len(set(args.frame)) == len(args.frame), f"--frame repeats a convention: {args.frame}"
    assert args.window_ms > 0, f"--window-ms must be positive, got {args.window_ms}"

    console = Console()
    shards  = eval_stream.device_count(args.devices)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Composing the pipelines touches no device, so the parent stays CUDA-free.
    totals  = {name: len(items[: args.limit])
               for name, items in suite_pipelines(args).items()}
    console.print(eval_stream.worker_summary(shards, args.workers))
    warp = WINDOW_MS / args.window_ms
    console.print(f"[dim]ImageBind zero-shot: {len(variants(args))} hypothesis/es "
                  f"({', '.join(args.sensor)} IMU x {len(args.frame)} frame"
                  f"{'s' if len(args.frame) != 1 else ' ' + args.frame[0]}), "
                  f"{args.window_ms:.0f} ms windows resampled to 2000 samples[/dim]")
    console.print(f"[dim]native window[/dim]" if warp == 1.0 else
                  f"[yellow]non-native window: {args.window_ms:.0f} ms over ImageBind's 2000-sample "
                  f"input is {2000 / (args.window_ms / 1e3):.0f} Hz, not the 200 Hz it was trained "
                  f"at — the motion is rendered {warp:.2f}x "
                  f"{'slower' if warp > 1 else 'faster'} than real time[/yellow]")

    accum = {v: {name: _IBCell() for name in DATASET_ORDER} for v in variants(args)}
    for partial in eval_stream.run_sharded(collect, args, shards, totals):
        for site, per_dataset in partial.items():
            for name, cell in per_dataset.items():
                accum[site][name].merge(cell)

    # Predictions, targets and the CLIP-keyed gallery, rebuilt in the undivided take order.
    resolved  = {site: {name: cell.resolve() for name, cell in per_dataset.items()}
                 for site, per_dataset in accum.items()}
    galleries = rebuild_galleries(args, resolved, console)
    ranked    = {site: rank_inputs(resolved[site], galleries[site], device, args.seed)
                 for site in variants(args)}

    if args.limit is not None:
        console.print(f"[yellow]--limit {args.limit}: partial results; the gallery only covers the evaluated windows[/yellow]")

    if not any("Nymeria" in ranked[v] for v in variants(args)):
        variant = lang_variant(args)
        console.print(f"[red]Nymeria produced no annotated window: lang_spec"
                      f"{'.' + variant if variant else ''}.json is missing from the cache, or no "
                      f"annotation span covers a {args.window_ms:.0f} ms window.[/red]")

    if len(variants(args)) > 1:
        print_sweep(args, console, ranked)
        return

    site = variants(args)[0]
    print_headline(args, console, accum[site], resolved[site], ranked[site])
    print_difficulty(args, console, ranked[site])
    print_categories(args, console, ranked[site])


if __name__ == "__main__":
    main()
