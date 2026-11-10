"""Evaluate the language head as bidirectional retrieval on the fixed suite (the valid side of exp7_cotraining_C).

Usage:
    python -m src.scripts.evaluate_language_classification <ckpt> [--limit N] [--lang-variant v1]
"""

import argparse
import dataclasses
import json
import typing as tp

from collections import defaultdict
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from ... import eval_stream
from ...data.util import corpus_pipelines
from ...eval import inference, load_model
from ...interface.data_imu import IMUData_w_GT_Lang_Jagged
from ...runtime_prologue import machine_conf
from ...data.imu.bones_seed_processed import canonical_motion


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "src" / "config"
SPLIT_GROUPS_JSON = CONFIG_DIR / "data" / "bones_seed_split_groups.json"

DATASET_ORDER = (
    "Bones-Seed-Soma",
    "Bones-Seed-G1-Valid",
    "Bones-Seed-G1-Human2Robot",
    "Nymeria",
)

# Annotation sidecar for the Nymeria reader: lang_spec.bones.json, the BONES-style re-transcription.
NYMERIA_LANG_VARIANT = "bones"


def _fixed_valid_takes(
    lang_variant: str | None   = NYMERIA_LANG_VARIANT,
    window_ms   : float | None = None,
) -> dict[str, list[tuple[str, tp.Callable[[str], tp.Any], str]]]:
    """The fixed validation suite as ``dataset -> [(stem, builder, cache root)]``."""
    overrides = ["experiment=exp7_cotraining_C"]
    if window_ms is not None:
        # In `data_conf.window_unit` units (ms); the valid windower steps by the window size.
        overrides.append(f"data_conf.window_size={float(window_ms)}")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="train", overrides=overrides)

    with SPLIT_GROUPS_JSON.open() as f:
        # Keyed by canonical motion; later groups overwrite earlier ones for repeated names.
        group_of = {m: group for group, motions in json.load(f)["groups"].items() for m in motions}

    takes: dict[str, list[tuple[str, tp.Callable[[str], tp.Any], str]]] = defaultdict(list)
    for entry in cfg.data.valid:
        spec = OmegaConf.to_container(entry, resolve=True)
        assert isinstance(spec, dict) and str(spec.get("_target_", "")).endswith("build_corpus"), spec
        kwargs = {str(k): v for k, v in spec.items() if not str(k).startswith("_")}
        reader = str(spec["reader"]["_target_"])
        if "Nymeria" in reader:
            # corpus_pipelines forwards the reader spec's non-underscore keys as kwargs.
            kwargs["reader"]["lang_variant"] = lang_variant
        stems, build = corpus_pipelines(**kwargs)
        root = str(spec["root"])

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
                continue    # no language GT in these caches; nothing to retrieve
            else:
                raise AssertionError(f"unexpected Exp7-C validation reader: {reader}")
            takes[dataset].append((str(stem), build, root))

    assert set(takes) == set(DATASET_ORDER), f"fixed suite changed: found {sorted(takes)}"
    return dict(takes)


def fixed_valid_pipelines(
    lang_variant: str | None   = NYMERIA_LANG_VARIANT,
    window_ms   : float | None = None,
) -> dict[str, list[tuple[str, tp.Callable[[str], tp.Any]]]]:
    """Return the fixed validation stems and lazy per-take builders of the language-carrying datasets.

    ``lang_variant`` overrides the Nymeria sidecar (``None`` = the original cache) and must match
    what the checkpoint trained on. ``window_ms`` overrides the window duration (``None`` = config).
    """
    return {name: [(stem, build) for stem, build, _ in takes]
            for name, takes in _fixed_valid_takes(lang_variant, window_ms).items()}


def fixed_valid_cache_dirs(
    lang_variant: str | None = NYMERIA_LANG_VARIANT,
) -> dict[str, list[Path]]:
    """The processed-cache directory of every take in the fixed suite, per dataset."""
    return {name: [Path(root, stem) for stem, _, root in takes]
            for name, takes in _fixed_valid_takes(lang_variant).items()}


# Bits reserved for a window's position inside its take (see ``_window_keys``).
_WINDOW_BITS = 32


def _window_keys(chunk: eval_stream.Chunk) -> torch.Tensor:
    """Global ordering key per window of ``chunk``: ``take index << 32 | position in take``."""
    index  = torch.tensor([meta.index for meta in chunk.metas], dtype=torch.int64)
    counts = torch.tensor([meta.windows for meta in chunk.metas], dtype=torch.int64)
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    within = torch.arange(int(counts.sum())) - torch.repeat_interleave(starts, counts)
    return (torch.repeat_interleave(index, counts) << _WINDOW_BITS) + within


@dataclasses.dataclass
class _Cell:
    """Per-dataset accumulator, gathered per forward and mergeable across GPU shards.

    ``blocks`` holds one ``(preds, phrases, inverse, keys)`` entry per chunk with annotated windows:
    predictions, distinct phrase embeddings, each window's phrase, and its global ordering key.
    The gallery is built in ``resolve``, after sorting back into the undivided take order.
    """
    takes  : int = 0
    short  : int = 0
    windows: int = 0
    cos_sum: float = 0.0
    blocks : list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = dataclasses.field(default_factory=list)

    def merge(self, other: "_Cell") -> None:
        """Fold another shard's accumulator for the same dataset into this one."""
        self.takes   += other.takes
        self.short   += other.short
        self.windows += other.windows
        self.cos_sum += other.cos_sum
        self.blocks  += other.blocks

    def resolve(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``(preds, targets, gallery)`` in undivided take order.

        Phrases are deduped on their embedding bytes; a phrase's gallery index is the rank of its
        first window.
        """
        if not self.blocks:
            return torch.empty(0), torch.empty(0, dtype=torch.int64), torch.empty(0)

        # Dedup across blocks first, so each window can name its phrase by a single global id.
        gallery_of: dict[bytes, int] = {}
        rows      : list[torch.Tensor] = []
        ids       : list[torch.Tensor] = []
        for _, phrases, inverse, _ in self.blocks:
            local = []
            for row in phrases:
                key = row.numpy().tobytes()
                j   = gallery_of.get(key)
                if j is None:
                    j = gallery_of[key] = len(rows)
                    rows.append(row)
                local.append(j)
            ids.append(torch.tensor(local, dtype=torch.int64)[inverse])

        order   = torch.cat([keys for _, _, _, keys in self.blocks]).argsort()
        preds   = torch.cat([block_preds for block_preds, _, _, _ in self.blocks])[order]
        phrase  = torch.cat(ids)[order]                 # global phrase id per window, in take order
        n, k    = phrase.numel(), len(rows)

        # `first` is total: every phrase in `rows` came from some window, so no sentinel survives.
        first = torch.full((k,), n, dtype=torch.int64)
        first.scatter_reduce_(0, phrase, torch.arange(n), reduce="amin")
        appearance = first.argsort()                    # gallery row -> global phrase id
        rank       = torch.empty(k, dtype=torch.int64)
        rank[appearance] = torch.arange(k)              # global phrase id -> gallery row
        return preds, rank[phrase], torch.stack(rows)[appearance]


def embed(x: torch.Tensor) -> torch.Tensor:
    """Unit-norm -- the raw CLIP geometry both sides are ranked in."""
    return F.normalize(x.float(), dim=-1)


def tally(cell: _Cell, metas: tp.Sequence[eval_stream.TakeMeta]) -> None:
    """Count the takes behind one fetch, short ones included (they emit no window)."""
    for meta in metas:
        if meta.windows == 0:
            cell.short += 1
        else:
            cell.takes   += 1
            cell.windows += meta.windows


def accumulate(cell: _Cell, chunk: eval_stream.Chunk, out: torch.Tensor) -> None:
    """Fold one forward's raw ``[B, C_lang]`` CPU output into ``cell``; unannotated windows add nothing."""
    batch   = tp.cast(IMUData_w_GT_Lang_Jagged, chunk.batch)          # asserted by the caller
    covered = batch.lang_feat_idx.cpu() >= 0                          # [B]
    if not bool(covered.any()):
        return

    out        = out[covered]                                         # annotated windows only
    idx        = batch.lang_feat_idx.cpu()[covered]                   # rows of the compact storage
    uniq, inv  = idx.unique(return_inverse=True)
    phrases    = batch.lang_feature.cpu()[uniq].float()               # [u, C_lang], as cached
    # Cosine to each window's own target, summed here because `phrases` is deduped downstream.
    cell.cos_sum += float((F.normalize(out, dim=-1) * F.normalize(phrases, dim=-1)[inv]).sum())
    cell.blocks.append((embed(out), embed(phrases), inv, _window_keys(chunk)[covered]))


def imu_to_text(
    preds  : torch.Tensor,
    targets: torch.Tensor,
    gallery: torch.Tensor,
    ks     : tuple[int, ...] = (1, 5, 10),
    chunk  : int = 4096,
) -> dict[str, float]:
    """Single-positive retrieval: rank the gallery for each window, score its one correct row.

    The rank counts strictly-better gallery rows, so exact ties resolve in the target's favour.
    """
    n, hits, rr = preds.shape[0], {k: 0 for k in ks}, 0.0
    for i in range(0, n, chunk):
        sim  = preds[i : i + chunk] @ gallery.T                    # [b, K]
        gt   = sim.gather(1, targets[i : i + chunk, None])         # [b, 1]
        rank = (sim > gt).sum(dim=1) + 1                           # [b]
        for k in ks:
            hits[k] += int((rank <= k).sum())
        rr += float(rank.float().reciprocal().sum())
    return {f"R@{k}": hits[k] / n for k in ks} | {"MRR": rr / n}


def text_to_imu(
    preds  : torch.Tensor,
    targets: torch.Tensor,
    gallery: torch.Tensor,
    chunk  : int = 128,
) -> float:
    """Multi-positive retrieval: mAP over the full ranking, as every window carrying a phrase is a positive."""
    n, k   = preds.shape[0], gallery.shape[0]
    depth  = torch.arange(1, n + 1, device=preds.device, dtype=torch.float32)
    ap_sum = 0.0
    for j in range(0, k, chunk):
        q     = torch.arange(j, min(j + chunk, k), device=preds.device)
        order = (gallery[q] @ preds.T).argsort(dim=1, descending=True)   # [q, N]
        pos   = (targets[order] == q[:, None]).float()                   # [q, N]
        prec  = pos.cumsum(dim=1) / depth
        ap_sum += float(((prec * pos).sum(dim=1) / pos.sum(dim=1).clamp(min=1)).sum())
    return ap_sum / k


def imu_to_text_sampled(
    preds    : torch.Tensor,
    targets  : torch.Tensor,
    gallery  : torch.Tensor,
    n_gallery: int,
    ks       : tuple[int, ...] = (1,),
    repeats  : int = 8,
    seed     : int = 0,
    chunk    : int = 4096,
) -> dict[str, float]:
    """R@k against the correct phrase plus ``n_gallery - 1`` random distractors, over ``repeats`` draws."""
    n, k = preds.shape[0], gallery.shape[0]
    if k <= n_gallery:                       # already at or below the requested difficulty
        return imu_to_text(preds, targets, gallery, ks=ks, chunk=chunk)

    g = torch.Generator(device="cpu").manual_seed(seed)
    hits, rr = {key: 0 for key in ks}, 0.0
    for _ in range(repeats):
        for i in range(0, n, chunk):
            sim = preds[i : i + chunk] @ gallery.T                      # [b, k]
            tgt = targets[i : i + chunk]
            gt  = sim.gather(1, tgt[:, None])                           # [b, 1]
            # Draw in [0, k-2] and shift past the target, so the positive is never a distractor.
            d = torch.randint(k - 1, (tgt.numel(), n_gallery - 1), generator=g).to(preds.device)
            d = d + (d >= tgt[:, None]).long()
            rank = (sim.gather(1, d) > gt).sum(dim=1) + 1               # [b]
            for key in ks:
                hits[key] += int((rank <= key).sum())
            rr += float(rank.float().reciprocal().sum())
    total = n * repeats
    return {f"R@{key}": hits[key] / total for key in ks} | {"MRR": rr / total}


def tolerant_top1(
    preds  : torch.Tensor,
    targets: torch.Tensor,
    gallery: torch.Tensor,
    tau    : float,
    chunk  : int = 4096,
) -> float:
    """Top-1 rate where retrieving any phrase within ``tau`` cosine of the target counts."""
    n   = preds.shape[0]
    gg  = gallery @ gallery.T                                           # [K, K] phrase-phrase
    hit = 0
    for i in range(0, n, chunk):
        top = (preds[i : i + chunk] @ gallery.T).argmax(dim=1)          # [b]
        hit += int((gg[top, targets[i : i + chunk]] >= tau).sum())
    return hit / n


def spherical_kmeans(
    x     : torch.Tensor,
    k     : int,
    iters : int = 50,
    seed  : int = 0,
    n_init: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster unit-norm rows by cosine, best of ``n_init`` restarts. Returns ``(label_per_row, centroids)``."""
    best, best_obj = None, -float("inf")
    for i in range(n_init):
        label, centroids = _spherical_kmeans_once(x, k, iters, seed + i)
        obj = float((x * centroids[label]).sum())
        if obj > best_obj:
            best, best_obj = (label, centroids), obj
    assert best is not None
    return best


def _spherical_kmeans_once(
    x    : torch.Tensor,
    k    : int,
    iters: int,
    seed : int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    k = min(k, n)

    # k-means++ seeding on angular distance.
    picked = [int(torch.randint(n, (1,), generator=g))]
    while len(picked) < k:
        d2 = (1.0 - (x @ x[picked].T).max(dim=1).values).clamp_min(0.0).pow(2)
        d2[torch.tensor(picked, device=x.device)] = 0.0
        picked.append(int(torch.multinomial(d2.cpu() + 1e-12, 1, generator=g)))
    centroids = x[torch.tensor(picked, device=x.device)].clone()

    label = torch.full((n,), -1, dtype=torch.int64, device=x.device)
    for _ in range(iters):
        new_label = (x @ centroids.T).argmax(dim=1)         # [n]
        settled   = torch.equal(new_label, label)
        label     = new_label
        counts    = torch.bincount(label, minlength=k)
        # Stop only when stable AND every cluster is populated: an empty one would make chance (1/k) wrong.
        if settled and bool((counts > 0).all()):
            break
        for j in range(k):
            if int(counts[j]):
                centroids[j] = F.normalize(x[label == j].mean(dim=0), dim=-1, eps=1e-12)
            else:
                # Re-seed onto the row currently worst served by any centroid.
                centroids[j] = x[(x @ centroids.T).max(dim=1).values.argmin()]
    return label, centroids


def balanced_accuracy(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """Mean per-class recall; shares the 1/k chance level with plain accuracy."""
    recalls = [float((pred[m] == c).float().mean()) for c in range(k) if bool((m := target == c).any())]
    return sum(recalls) / len(recalls) if recalls else float("nan")


def chance_row(counts: torch.Tensor, n: int, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Analytic random-ranking reference: R@k = k/K, MRR = H_K/K, mAP = mean of |P_j|/N."""
    k = counts.numel()
    return {f"R@{key}": min(key, k) / k for key in ks} | {
        "MRR"  : sum(1.0 / r for r in range(1, k + 1)) / k,
        "t2i"  : float((counts.float() / n).mean()),
    }


def collect(args, shard: int, shards: int, report: eval_stream.Report) -> dict[str, _Cell]:
    """Accumulate this shard's slice of every dataset; builds its own model, as CUDA cannot cross the spawn."""
    torch.manual_seed(args.seed)                # pins the corpus builders' construction seed
    model = load_model(Path(args.ckpt)).eval().to(eval_stream.shard_device(shard))
    accum = {name: _Cell() for name in DATASET_ORDER}

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
                pred = inference(model, chunk.batch)
                accumulate(cell, chunk, pred.dec_lang_feature[:, 0].float().cpu())   # [B, 768]
    return accum


def lang_variant(args) -> str | None:
    """The Nymeria annotation sidecar named by ``--lang-variant`` (``None`` = the original v1)."""
    return None if args.lang_variant == "v1" else args.lang_variant


def add_suite_args(parser, *, window_ms: float | None = None) -> None:
    """Add the two knobs selecting the suite's windows; ``window_ms`` is the ``--window-ms`` default."""
    parser.add_argument("--lang-variant", default=NYMERIA_LANG_VARIANT,
                        help=f"Nymeria annotation sidecar: '{NYMERIA_LANG_VARIANT}' (lang_spec."
                             f"{NYMERIA_LANG_VARIANT}.json, the default) or 'v1' for the original "
                             "lang_spec.json; must match what the checkpoint trained on")
    parser.add_argument("--window-ms", type=float, default=window_ms,
                        help=f"window duration for the suite, overriding data_conf.window_size "
                             f"(default: {'the config value, 4096' if window_ms is None else window_ms}). "
                             "Windows are non-overlapping, so a longer window means fewer of them "
                             "and lower annotation coverage; numbers at different window sizes "
                             "are not comparable")


def suite_pipelines(args) -> dict[str, list[tuple[str, tp.Callable[[str], tp.Any]]]]:
    """The fixed suite as this run's ``--lang-variant`` / ``--window-ms`` select it."""
    return fixed_valid_pipelines(lang_variant(args), args.window_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("ckpt", help="checkpoint path, absolute or relative to ${machine-conf:exp_root}")
    parser.add_argument("--limit", type=int, default=None, help="first N takes per dataset (smoke test; results and the gallery are partial)")
    parser.add_argument("--seed", type=int, default=7, help="pin per-stem construction, transform draws, and tie-breaking order")
    add_suite_args(parser)
    parser.add_argument("--gallery", type=int, nargs="+", default=[100, 20],
                        help="fixed gallery sizes for difficulty-calibrated retrieval: the correct phrase "
                             "plus N-1 random distractors, so every dataset is asked the same question")
    parser.add_argument("--tau", type=float, default=0.95,
                        help="cosine above which a retrieved phrase counts as the same motion, differently worded")
    parser.add_argument("--categories", type=int, nargs="+", default=[20, 50, 200],
                        help="category counts for the coarse readout; each K clusters the dataset's "
                             "phrase gallery by cosine and retrieves against the K prototypes")
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

    # The reader silently emits empty language when a sidecar is absent.
    if not resolved["Nymeria"][1].numel():
        variant = lang_variant(args)
        console.print(f"[red]Nymeria produced no annotated window: lang_spec"
                      f"{'.' + variant if variant else ''}.json is missing from the "
                      f"cache, or none of its spans cover a window.[/red]")

    # (gallery, preds, targets) per dataset, shared by the phrase and category readouts.
    ranked: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    table = Table(title=f"Language retrieval, per-dataset phrase gallery "
                        f"(Nymeria lang_variant={args.lang_variant})\n{ckpt}")
    for column in ("dataset", "variant", "takes", "windows", "N", "K", "cos", "R@1", "R@5", "R@10", "MRR", "t2i mAP"):
        table.add_column(column, justify="right" if column not in ("dataset", "variant") else "left")

    for dataset in DATASET_ORDER:
        cell = accum[dataset]
        preds, targets, gallery = resolved[dataset]
        n, k = targets.numel(), gallery.shape[0]
        takes = str(cell.takes) + (f" ({cell.short} short)" if cell.short else "")
        # N is the annotated subset of `windows`; the ratio is the annotation coverage.
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

        # The best constant predictor, where a collapsed head lands under a cosine loss.
        counts   = torch.bincount(targets, minlength=k)
        collapse = F.normalize(gallery[targets].mean(dim=0), dim=-1).expand(n, -1)

        rows = {
            "chance"  : chance_row(counts.cpu(), n),
            "collapse": imu_to_text(collapse, targets, gallery) | {"t2i": text_to_imu(collapse, targets, gallery)},
            "model"   : imu_to_text(preds, targets, gallery)    | {"t2i": text_to_imu(preds, targets, gallery)},
        }
        for variant, m in rows.items():
            cos = f"{cell.cos_sum / n:.4f}" if variant == "model" else "-"
            head = variant == "chance"      # take/window counts are per dataset, printed once
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
    table = Table(title="Category retrieval — spherical k-means over each dataset's phrase gallery")
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
            cat_target    = label[targets]                       # each window's category
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
