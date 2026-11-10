"""Hydra helpers that build dataset pipelines and lazy corpora from flat layer specs."""

import os
import json
import zlib
import hashlib
import hydra
import torch
import functools
import typing as tp

from omegaconf import ListConfig, OmegaConf
from torch.utils.data import ConcatDataset, Dataset

from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress

from ..interface.sequence import ISequence


def build_nested_seq(layers: list) -> ISequence:
    """Build a nested sequence pipeline from a flat list of layer specs.

    The first spec is the data source; each later spec wraps the result through its ``source``
    keyword. An entry may itself be a list of specs, flattened in place, because YAML cannot
    splice an anchored list into a list. The calling spec must set ``_recursive_: false``, or
    Hydra instantiates the wrappers before this runs and fails on the missing ``source``.
    """
    flat: list = []
    for layer in layers:
        flat.extend(layer) if isinstance(layer, (list, ListConfig)) else flat.append(layer)
    if not flat:
        raise ValueError("build_nested_seq requires at least one layer spec")
    seq = hydra.utils.instantiate(flat[0])
    for layer in flat[1:]:
        seq = hydra.utils.instantiate(layer, source=seq)
    return seq


def listing_files(stems: str | tp.Sequence[str]) -> list[str]:
    """Normalize a ``stems`` spec value (one listing path or a list of them) to a list."""
    if OmegaConf.is_config(stems):
        stems = OmegaConf.to_container(stems, resolve=True)  # type: ignore[assignment]
    return [stems] if isinstance(stems, str) else [str(s) for s in stems]


def corpus_pipelines(reader, root: str, stems: str | tp.Sequence[str], layers, path_key: str = "path") -> tuple[list[str], tp.Callable[[str], ISequence]]:
    """Parse a ``build_corpus`` spec into its stem list and a per-stem pipeline builder.

    ``stems`` is one listing file or a list of them, concatenated in order. ``build_one(stem)``
    builds that stem's pipeline on demand, seeded by stem so construction-time draws reproduce.
    """
    stem_list = []
    for listing in listing_files(stems):
        with open(hydra.utils.to_absolute_path(listing)) as f:
            stem_list += [line for line in f.read().splitlines() if line and not line.startswith("#")]

    if OmegaConf.is_config(reader):
        reader = OmegaConf.to_container(reader, resolve=True)
    if OmegaConf.is_config(layers):
        layers = OmegaConf.to_container(layers, resolve=True)
    assert isinstance(reader, dict) and isinstance(layers, list), \
        "corpus_pipelines expects a mapping reader spec and a list of layer specs"
    flat: list[dict] = []
    for layer in layers:
        flat.extend(layer) if isinstance(layer, list) else flat.append(layer)

    def cls_and_kwargs(spec: dict) -> tuple[type, dict[str, tp.Any]]:
        return hydra.utils.get_class(spec["_target_"]), \
            {str(k): v for k, v in spec.items() if not str(k).startswith("_")}

    reader_cls, reader_kw = cls_and_kwargs(reader)
    wrappers = [cls_and_kwargs(spec) for spec in flat]

    build_one = functools.partial(
        _corpus_build_one, reader_cls, reader_kw, wrappers, root.rstrip("/"), path_key,
        torch.initial_seed() % (2**31),
    )
    return stem_list, build_one


def build_corpus(reader, root: str, stems: str | tp.Sequence[str], layers, path_key: str = "path") -> Dataset:
    """Build a ``ConcatDataset`` of one lazily-constructed pipeline per stem, in listing order."""
    stem_list, build_one = corpus_pipelines(reader, root, stems, layers, path_key)

    # Only window counts are computed here; pipelines are rebuilt per fetch, without a process
    # pool, which would deadlock forked workers. Counts are cached on disk, keyed on everything
    # a length depends on. The seed is excluded: lengths ignore it and DDP ranks differ in it.
    label = "+".join(os.path.basename(f) for f in listing_files(stems))
    lens  = _load_window_counts(root, reader, layers, path_key, stem_list)
    if lens is None:
        with Progress(transient=True, disable=os.environ.get("LOCAL_RANK", "0") != "0") as pbar:
            task = pbar.add_task(f"Corpus {label}", total=len(stem_list))
            lens = []
            for stem in stem_list:
                lens.append(len(build_one(stem)))
                pbar.advance(task)
        _save_window_counts(root, reader, layers, path_key, stem_list, lens)
    return ConcatDataset([_LazyPipeline(build_one, stem, n) for stem, n in zip(stem_list, lens)])


def _window_counts_file(root: str, reader, layers, path_key: str, stem_list: list[str]) -> str:
    def plain(x):
        return OmegaConf.to_container(x, resolve=True) if OmegaConf.is_config(x) else x
    payload = json.dumps({
        "reader"  : plain(reader),
        "layers"  : plain(layers),
        "root"    : root.rstrip("/"),
        "path_key": path_key,
        "stems"   : stem_list,
    }, sort_keys=True, default=str)
    key = hashlib.sha1(payload.encode()).hexdigest()[:16]
    return os.path.join(root.rstrip("/"), ".window_counts", f"{key}.json")


def _load_window_counts(root: str, reader, layers, path_key: str, stem_list: list[str]) -> list[int] | None:
    path = _window_counts_file(root, reader, layers, path_key, stem_list)
    try:
        with open(path) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    if saved.get("stems") != stem_list or len(saved.get("lens", [])) != len(stem_list):
        return None
    return [int(n) for n in saved["lens"]]


def _save_window_counts(root: str, reader, layers, path_key: str, stem_list: list[str], lens: list[int]) -> None:
    path = _window_counts_file(root, reader, layers, path_key, stem_list)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"stems": stem_list, "lens": lens}, f)
        os.replace(tmp, path)
    except OSError as e:   # read-only corpus root: fall back to recomputing every launch
        print(f"window-count cache not written ({e}); startup will recount next launch")


def subsample_corpus(corpus: ConcatDataset, fraction: float) -> ConcatDataset:
    """Keep ``fraction`` of a ``build_corpus`` output by dropping whole sequences.

    Sequence-level because ``RandomWindow_IMU`` redraws crops per fetch, so thinning windows
    would not reduce unique data. Selection follows the CRC32 order of stems, so it is
    independent of the seed and nested across fractions.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"subsample fraction must be in (0, 1], got {fraction}")
    seqs: list[_LazyPipeline] = list(corpus.datasets)  # type: ignore[arg-type]
    if fraction == 1.0:
        return corpus
    n_keep = max(1, round(fraction * len(seqs)))
    ranked = sorted(range(len(seqs)), key=lambda i: zlib.crc32(seqs[i]._stem.encode()))
    kept   = sorted(ranked[:n_keep])
    return ConcatDataset([seqs[i] for i in kept])


def _corpus_build_one(reader_cls, reader_kw: dict, wrappers: list, root: str, path_key: str, base_seed: int, stem: str) -> ISequence:
    # fork_rng: the stem seed pins construction-time draws without clobbering the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base_seed + zlib.crc32(stem.encode()))
        seq = reader_cls(**{path_key: f"{root}/{stem}"}, **reader_kw)
        for cls, kw in wrappers:
            seq = cls(source=seq, **kw)
    return seq


class _LazyPipeline(Dataset):
    """One corpus motion, constructed on demand at fetch time (see ``build_corpus``)."""

    def __init__(self, build_one, stem: str, length: int) -> None:
        self._build = build_one
        self._stem  = stem
        self._len   = length

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index):
        return self._build(self._stem)[index]


def multithread_instantiate(cfg_arr: list, description: str = "Loading datasets", max_workers: int = 16) -> list[ISequence]:
    """Instantiate dataset specs on a thread pool, preserving spec order.

    Assumes each target's ``__init__`` is thread-safe.
    """
    with Progress(transient=True, disable=os.environ.get("LOCAL_RANK", "0") != "0") as pbar:
        task = pbar.add_task(description, total=len(cfg_arr))

        def load_one(spec) -> ISequence:
            dataset = hydra.utils.instantiate(spec)
            pbar.advance(task)
            return dataset

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(load_one, cfg_arr))
