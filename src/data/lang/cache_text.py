"""Recover the phrase strings behind a processed cache's CLIP-only ``lang_feature`` rows."""

import csv
import json
import functools

from pathlib import Path

import numpy as np
import torch

from ...runtime_prologue import machine_conf


# Label sources resolve through machine_conf("data_root"), not the `source` path in
# lang_spec.json, which is the path on whichever machine wrote the cache.

# Nymeria: one narration CSV per take; column and sort order mirror ``Nymeria_Language``.
_NYMERIA_ROOT     = "nymeria_imu"
_NYMERIA_TEXT_COL = "Describe my activity"
# Sidecar ``variant`` key -> narration CSV, mirroring ``_VARIANTS_`` of src.scripts.nymeria_lang_process.
_NYMERIA_CSV = {
    "default": "transcribed_activity_summarization.csv",
    "bones"  : "transcribed_bones_style.csv",
}

# BONES-SEED: one JSONL of temporal events for the whole corpus, keyed by motion stem.
_BONES_LABELS = "bones_seed/metadata/seed_metadata_v002_temporal_labels.jsonl"


@functools.cache
def _bones_label_table() -> dict[str, list[str]]:
    """Motion stem -> its temporal-event descriptions, in file order; parsed once per process."""
    path = Path(machine_conf("data_root")) / _BONES_LABELS
    assert path.is_file(), (
        f"BONES-SEED temporal labels not found at {path}; the ImageBind gallery needs the raw "
        f"phrases, which the processed cache does not keep"
    )
    with path.open() as f:
        return {
            rec["filename"]: [event["description"].strip() for event in rec["events"]]
            for rec in map(json.loads, f)
        }


def _nymeria_annotations(stem: str, variant: str) -> list[str]:
    """A Nymeria take's narration phrases, sorted by span start (the cache's annotation order)."""
    path = Path(machine_conf("data_root"), _NYMERIA_ROOT, stem, "narration", _NYMERIA_CSV[variant])
    assert path.is_file(), (
        f"Nymeria narration not found at {path}; the ImageBind gallery needs the raw phrases, "
        f"which the processed cache does not keep"
    )
    with path.open(newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda row: float(row["start_time"]))
    return [row[_NYMERIA_TEXT_COL].strip() for row in rows]


def _distinct_in_order(annotations: list[str]) -> tuple[list[str], list[int]]:
    """The writers' row rule: distinct non-empty phrases in first-appearance order, plus the
    per-annotation row index (``-1`` for a blank description)."""
    row_of: dict[str, int] = {}
    index  = [-1] * len(annotations)
    for i, text in enumerate(annotations):
        if text:
            index[i] = row_of.setdefault(text, len(row_of))
    return list(row_of), index


def _lang_spec(cache_dir: Path, variant: str | None) -> dict | None:
    """``lang_spec.<variant>.json`` if the take has one, else ``lang_spec.json``, else None."""
    for name in filter(None, (f"lang_spec.{variant}.json" if variant else None, "lang_spec.json")):
        path = cache_dir / name
        if path.exists():
            with path.open() as f:
                return json.load(f)
    return None


def phrase_texts(cache_dir: Path | str, variant: str | None = None) -> list[str]:
    """The phrases behind ``cache_dir``'s ``lang_feature`` rows, row order preserved.

    Replays the writers' row rule over the raw label file and raises if the result disagrees with
    the cached index. ``variant`` matches the readers' ``lang_variant``; BONES-SEED ignores it.
    Returns ``[]`` for a take with no language cache.
    """
    cache_dir = Path(cache_dir)
    spec      = _lang_spec(cache_dir, variant)
    if spec is None:
        return []

    source = Path(spec["source"])
    if source.name.endswith(".jsonl"):
        annotations = _bones_label_table().get(cache_dir.name, [])
    else:
        annotations = _nymeria_annotations(cache_dir.name, spec.get("variant", "default"))

    texts, replayed = _distinct_in_order(annotations)
    cached = np.load(cache_dir / spec["index"]["file"]).tolist()
    assert replayed == cached, (
        f"{cache_dir}: replaying {source.name} does not reproduce {spec['index']['file']} "
        f"({len(replayed)} vs {len(cached)} annotations); the label file and the cache disagree"
    )
    assert len(texts) == int(spec["n_valid"]), (
        f"{cache_dir}: recovered {len(texts)} distinct phrases, cache holds {spec['n_valid']}"
    )
    return texts


def _decode_bf16(raw: np.ndarray) -> torch.Tensor:
    """An ``(N, C)`` ``ml_dtypes.bfloat16`` array (opaque ``|V2`` to numpy) as fp32."""
    return torch.from_numpy(raw.view(np.uint16)).view(torch.bfloat16).float()


def embedding_key(row: torch.Tensor) -> bytes:
    """Lookup key for one cached phrase embedding: the raw fp32 bytes of the row.

    Not the normalized row: a reduction over ``C`` may vary with batch size, breaking bit equality.
    """
    return row.float().contiguous().numpy().tobytes()


def _resolve_aliases(candidates: dict[bytes, set[str]]) -> tuple[dict[bytes, str], int]:
    """Collapse each embedding's candidate strings to their lexicographic minimum.

    Strings that tokenize identically share a cached row; any other sharing is a genuine
    collision and raises.
    """
    import clip                                  # the encoder the caches were written with

    table: dict[bytes, str] = {}
    aliased = 0
    for key, texts in candidates.items():
        if len(texts) == 1:
            table[key] = next(iter(texts))
            continue
        ordered = sorted(texts)
        tokens  = clip.tokenize(ordered, truncate=True)
        assert bool((tokens == tokens[:1]).all()), (
            f"two phrases share a cached CLIP row but do not tokenize alike: {ordered!r}; "
            f"this is a genuine embedding collision, not a whitespace alias"
        )
        table[key] = ordered[0]
        aliased   += 1
    return table, aliased


def text_by_embedding(
    cache_dirs: list[Path],
    variant   : str | None = None,
) -> tuple[dict[bytes, str], int]:
    """Map every cached phrase embedding to its phrase, keyed by ``embedding_key``.

    Also returns the number of rows that had more than one source string.
    """
    candidates: dict[bytes, set[str]] = {}
    for cache_dir in cache_dirs:
        texts = phrase_texts(cache_dir, variant)
        if not texts:
            continue
        spec     = _lang_spec(Path(cache_dir), variant)
        assert spec is not None                                  # phrase_texts returned rows
        features = _decode_bf16(np.load(Path(cache_dir) / spec["feature"]["file"]))
        assert features.shape[0] == len(texts), \
            f"{cache_dir}: {features.shape[0]} feature rows vs {len(texts)} recovered phrases"
        for row, text in zip(features, texts):
            candidates.setdefault(embedding_key(row), set()).add(text)
    return _resolve_aliases(candidates)
