"""t-SNE of the trunk [CLS] token over sampled human / humanoid IMU segments, plus a PCA companion view.

Usage:
    python -m src.scripts.analyze_cls_tsne --ckpt <run>/checkpoints/<name>.ckpt
    python -m src.scripts.analyze_cls_tsne --ckpt <run>/checkpoints/<name>.ckpt --from-npz   # restyle only
"""
from __future__ import annotations


import os

# OpenBLAS can segfault on many-core machines; cap threading before numpy/sklearn load.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.manifold import TSNE

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.runtime_prologue import machine_conf

import src  # noqa: E402,F401 -- registers the machine-conf OmegaConf resolver
from src.eval import load_model  # noqa: E402
from src.model.model import FoundationIMU  # noqa: E402
from src.data.util import corpus_pipelines  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent

# reader class name fragment -> (label, embodiment, marker)
DATASETS = {
    "Nymeria_XSens": ("Nymeria", "Human", "o"),
    "NaturalMotion_XSens": ("Natural Motion", "Human", "s"),
    "Bones_Seed_SOMA": ("BONES (Human)", "Human", "^"),
    "Bones_Seed_G1": ("BONES (G1)", "Humanoid", "D"),
    "Humanoid_SuperOdom": ("Real-Trajectory (G1)", "Humanoid", "v"),
}
# Embodiment hues; dataset identity is carried by marker shape.
FILL = {"Human": "#94b6d2", "Humanoid": "#d8b35c"}
# Raw layer: the same hues blended toward white.
RAW_FILL = {"Human": "#dae6ef", "Humanoid": "#f1e5c7"}
INK = "#444444"


def cls_rows_in_trunk_pack(tokenizer) -> torch.Tensor:
    """Map the innermost tokenizer's cached CLS rows into the outer packed layout."""
    mapping, current = None, tokenizer
    while hasattr(current, "base_tokenizer"):
        assert current._base_rows is not None, "tokenizer mapping requires a forward pass"
        mapping = current._base_rows if mapping is None else mapping[current._base_rows]
        current = current.base_tokenizer
    assert current._cls_rows is not None, "CLS mapping requires a forward pass"
    return current._cls_rows if mapping is None else mapping[current._cls_rows]


def dataset_spec(entry) -> tuple[str, str, str]:
    target = str(entry.reader._target_)
    for fragment, spec in DATASETS.items():
        if fragment in target:
            return spec
    raise ValueError(f"unrecognized reader {target}")


def sample_windows(build_one, stem_order, n_windows, rng, min_stems=8, max_scan=4000):
    """Scan stems until the pipelines cover ``n_windows`` windows, then sample them uniformly."""
    sequences, counts = [], []
    for stem in stem_order[:max_scan]:
        seq = build_one(stem)
        if len(seq) == 0:
            continue
        sequences.append(seq)
        counts.append(len(seq))
        if len(sequences) >= min_stems and sum(counts) >= 2 * n_windows:
            break
    available = sum(counts)
    take = min(n_windows, available)
    bounds = np.cumsum([0, *counts])
    flat = np.sort(rng.choice(available, size=take, replace=False))
    seq_idx = np.searchsorted(bounds, flat, side="right") - 1
    return [(sequences[int(s)], int(f - bounds[int(s)])) for s, f in zip(seq_idx, flat)], available


@torch.inference_mode()
def embed(model, refs, chunk):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    out = []
    for start in range(0, len(refs), chunk):
        windows = [seq[i] for seq, i in refs[start : start + chunk]]
        batch = type(windows[0]).collate(windows).to(device, dtype)
        mask = torch.zeros(batch.acc.shape[0], dtype=torch.bool, device=device)
        output = model(batch, mask)
        cls_rows = cls_rows_in_trunk_pack(model.tokenizer)
        out.append(
            FoundationIMU.canonicalize_for_task(output.trunk_feature)[cls_rows].float().cpu().numpy()
        )
    return np.concatenate(out)


def collect(args) -> dict[str, np.ndarray]:
    cfg = OmegaConf.load(Path(args.ckpt).parent.parent / ".hydra" / "config.yaml")
    model = load_model(args.ckpt).cuda().eval()
    window_layer = OmegaConf.to_container(cfg.data.window.train, resolve=True)

    feats, labels, embodiments, variants = [], [], [], []
    for entry in cfg.data.train:
        label, embodiment, _ = dataset_spec(entry)
        root = str(entry.root)
        stems, build_aug = corpus_pipelines(entry.reader, root, entry.stems, entry.layers)
        _, build_raw = corpus_pipelines(entry.reader, root, entry.stems, [window_layer])
        rng = np.random.default_rng(args.seed)
        stem_order = [stems[i] for i in rng.permutation(len(stems))]
        for variant, build in (("raw", build_raw), ("aug", build_aug)):
            torch.manual_seed(args.seed)  # per-fetch draws (noise, window starts) reproducible
            refs, available = sample_windows(build, stem_order, args.windows, rng)
            x = embed(model, refs, args.chunk)
            feats.append(x)
            labels += [label] * len(x)
            embodiments += [embodiment] * len(x)
            variants += [variant] * len(x)
            print(f"{label:>14} {variant}: {len(x)} windows (pool={available})")

    return {
        "features": np.concatenate(feats),
        "dataset": np.array(labels),
        "embodiment": np.array(embodiments),
        "variant": np.array(variants),
    }


def project(data, seed, perplexity):
    n = len(data["features"])
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, n // 8)),
        init="pca",
        random_state=seed,
    )
    return tsne.fit_transform(data["features"].astype(np.float64))


def mixing_score(points, embodiment, k=10):
    """Cross-embodiment share of k nearest neighbors over chance 2p(1-p): 1.0 = mixed, 0 = separated."""
    from sklearn.neighbors import NearestNeighbors

    is_h = embodiment == "Humanoid"
    nn = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, idx = nn.kneighbors(points)
    cross = (is_h[idx[:, 1:]] != is_h[:, None]).mean()
    p = is_h.mean()
    return float(cross / (2 * p * (1 - p)))


def sweep(data, seed, values):
    """Report the augmented points' 2-D mixing score per perplexity and the high-D reference."""
    aug = data["variant"] == "aug"
    emb = data["embodiment"][aug]
    ref = mixing_score(data["features"][aug].astype(np.float64), emb)
    print(f"high-D reference mixing (768-d, fixed): {ref:.3f}")
    best = None
    for perplexity in values:
        xy = project(data, seed, perplexity)
        score = mixing_score(xy[aug], emb)
        print(f"perplexity {perplexity:>5.1f}: 2-D mixing {score:.3f}")
        if best is None or score > best[1]:
            best = (perplexity, score)
    if best is None:
        raise ValueError("Provide at least one perplexity for the sweep")
    print(f"best: perplexity {best[0]} (2-D mixing {best[1]:.3f})")
    return best[0]


# Per-marker point sizes (pts^2) equalizing visual weight.
MARKER_S = {"o": 8.5, "s": 7.5, "^": 10.5, "D": 7.0, "v": 10.5}

# Darkened hues for the legend group headers.
HEADER_INK = {"Human": "#4f759a", "Humanoid": "#96762e"}


LEGEND_GROUPS = [
    ("Human", "Human", [("o", "Nymeria"), ("s", "Natural Motion"), ("^", "BONES")]),
    ("Humanoid (G1)", "Humanoid", [("D", "BONES"), ("v", "Real-Trajectory")]),
]


def legend_block(fig, y_top=0.118, y_bot=0.055):
    """Two-row grouped legend: embodiment names above dataset entries, one rule between the groups."""
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    marker_w = 0.026
    x0, x1 = 0.008, 0.992
    sep_pad = 0.018  # clearance on each side of the vertical rule

    headers, entry_texts = [], []  # per group
    for title, embodiment, entries in LEGEND_GROUPS:
        headers.append(fig.text(0, y_top, title, fontsize=6.2, va="center", ha="left",
                                color=HEADER_INK[embodiment], variant="small-caps"))
        entry_texts.append([fig.text(0, y_bot, label, fontsize=6.0, va="center", ha="left",
                                     color=INK) for _, label in entries])
    fig.canvas.draw()
    entry_w = [[t.get_window_extent().width / fig_w_px + marker_w for t in group]
               for group in entry_texts]

    # Leftover space feeds the in-part entry gaps, so the block fills the figure width.
    content = [sum(g) for g in entry_w]
    n_gaps = [len(g) - 1 for g in entry_w]
    leftover = (x1 - x0) - sum(content) - 2 * sep_pad
    gap = leftover / sum(n_gaps)
    part_w = [c + n * gap for c, n in zip(content, n_gaps)]

    x = x0
    for gi, (title, embodiment, entries) in enumerate(LEGEND_GROUPS):
        headers[gi].set_position((x, y_top))
        ex = x
        for (marker, _), text, width in zip(entries, entry_texts[gi], entry_w[gi]):
            fig.add_artist(Line2D([ex + marker_w * 0.3], [y_bot], linestyle="",
                                  marker=marker, markersize=4.2,
                                  markerfacecolor=FILL[embodiment],
                                  markeredgecolor="none", transform=fig.transFigure))
            text.set_position((ex + marker_w, y_bot))
            ex += width + gap
        x += part_w[gi]
        if gi == 0:
            fig.add_artist(Line2D([x + sep_pad] * 2, [y_bot - 0.028, y_top + 0.028],
                                  color="#b5b5b5", linewidth=0.8,
                                  transform=fig.transFigure))
            x += 2 * sep_pad


def plot(data, xy, out_pdf: Path, out_png: Path, note: str | None = None):
    plt.rcParams.update({"font.family": "serif", "font.size": 7.0, "text.color": INK})
    # The axes span the canvas except a bottom legend strip; no cropping, so the size IS the figsize.
    fig = plt.figure(figsize=(3.45, 2.0), dpi=300)
    ax = fig.add_axes((0.002, 0.20, 0.996, 0.79))
    if note:
        ax.text(0.005, 0.99, note, transform=ax.transAxes, fontsize=6.0,
                style="italic", color="#9a9a9a", va="top", ha="left")

    # Raw layer, only when the passed rows contain it (the PCA view), labeled on the cloud.
    raw = data["variant"] == "raw"
    if raw.any():
        for embodiment, tint in RAW_FILL.items():
            sel = raw & (data["embodiment"] == embodiment)
            ax.scatter(xy[sel, 0], xy[sel, 1], s=3.5, c=tint, marker="o",
                       linewidths=0, alpha=0.75, zorder=1)
        # Label above the cloud's right shoulder, clear of the top-left note.
        rawpts = xy[raw]
        shoulder = rawpts[rawpts[:, 0] >= np.quantile(rawpts[:, 0], 0.8)]
        anchor_x = float(np.median(shoulder[:, 0]))
        near_edge = anchor_x > xy[:, 0].min() + 0.8 * (xy[:, 0].max() - xy[:, 0].min())
        ax.annotate("raw (no transform)",
                    xy=(anchor_x, float(shoulder[:, 1].max())),
                    xytext=(0, 5), textcoords="offset points",
                    ha="right" if near_edge else "center",
                    fontsize=6.0, style="italic", color="#9a9a9a")

    for fragment, (label, embodiment, marker) in DATASETS.items():
        sel = ~raw & (data["dataset"] == label)
        ax.scatter(xy[sel, 0], xy[sel, 1], s=MARKER_S[marker], c=FILL[embodiment],
                   marker=marker, linewidths=0, alpha=0.85, zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.margins(x=0.015, y=0.055)

    legend_block(fig)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    print(f"wrote {out_pdf} and {out_png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=512, help="windows per (dataset, variant)")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=55.0,
                        help="t-SNE perplexity (55 maximized embodiment mixing in the "
                             "2026-08-26 sweep; see --sweep)")
    parser.add_argument("--npz", type=Path, default=SCRIPT_DIR / "cls_tsne_embeddings.npz")
    parser.add_argument("--from-npz", action="store_true", help="restyle from the cached npz")
    parser.add_argument("--sweep", type=str, default=None,
                        help="comma-separated perplexities: score embodiment mixing per value "
                             "(uses cached npz features; no plot)")
    parser.add_argument("--reproject", action="store_true",
                        help="with --from-npz: recompute t-SNE from the cached features at "
                             "--perplexity and update the npz's xy before plotting")
    parser.add_argument("--out", type=Path, default=REPO / "temp/figures/cls_tsne.pdf")
    args = parser.parse_args()

    DATA_KEYS = ("features", "dataset", "embodiment", "variant")

    def project_aug(data):
        aug = data["variant"] == "aug"
        return project({"features": data["features"][aug]}, args.seed, args.perplexity)

    if args.sweep or args.from_npz:
        cached = np.load(args.npz, allow_pickle=False)
        data = {k: cached[k] for k in DATA_KEYS}
        if args.sweep:
            sweep(data, args.seed, [float(v) for v in args.sweep.split(",")])
            return
        if args.reproject or "xy_aug" not in cached.files:
            xy_aug = project_aug(data)
            np.savez(args.npz, allow_pickle=False, **data, xy_aug=xy_aug)
            print(f"reprojected (aug-only) at perplexity {args.perplexity}; npz updated")
        else:
            xy_aug = cached["xy_aug"]
    else:
        data = collect(args)
        xy_aug = project_aug(data)
        np.savez(args.npz, allow_pickle=False, **data, xy_aug=xy_aug)
        print(f"cached embeddings -> {args.npz}")

    # t-SNE is fit on the augmented points only; its axes are arbitrary, so orient landscape.
    aug = data["variant"] == "aug"
    aug_data = {k: v[aug] for k, v in data.items()}
    if np.ptp(xy_aug[:, 1]) > np.ptp(xy_aug[:, 0]):
        xy_aug = xy_aug[:, ::-1]
    plot(aug_data, xy_aug, args.out, args.out.with_suffix(".png"))

    # Companion PCA view: linear, so a sanity reference against the t-SNE rendering.
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=args.seed).fit(data["features"].astype(np.float64))
    xy_pca = pca.transform(data["features"].astype(np.float64))
    var = 100 * pca.explained_variance_ratio_
    pca_out = args.out.with_name(args.out.stem.replace("tsne", "pca") + ".pdf")
    plot(data, xy_pca, pca_out, pca_out.with_suffix(".png"),
         note=f"PCA  (PC1 {var[0]:.0f}%, PC2 {var[1]:.0f}% var)")


if __name__ == "__main__":
    main()
