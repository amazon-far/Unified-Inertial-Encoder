"""Streaming take evaluation for the fixed-suite scripts: loader worker pool, grouped forwards, one process per GPU."""

import contextlib
import functools
import importlib
import os
import pickle
import queue as queue_mod
import random
import sys
import traceback
import typing as tp
import zlib

from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

from rich.progress import Progress
from torch.utils.data import DataLoader, Dataset

from .interface.data_imu import IMUData_BaseTrait_Jagged
from .interface.sequence import ISequence


Builder    = tp.Callable[[str], ISequence]
Item       = tuple[str, Builder]            # as the fixed_valid_pipelines listings hold them
TaggedItem = tuple[int, Item]               # the same, carrying its index in the undivided list
Report     = tp.Callable[[str, int], None]  # (dataset, takes finished) -> None

T = tp.TypeVar("T")

# Per-GPU loader workers: enough to keep the device fed; more only cost RAM.
DEFAULT_WORKERS    = 6
# Packed rows per forward.
DEFAULT_CHUNK_ROWS = 1_500_000


@dataclass(frozen=True)
class TakeMeta:
    """What the caller needs about a take; ``index`` is its position in the UNDIVIDED item list."""
    index     : int
    stem      : str
    windows   : int                  # 0 -> shorter than one window; the take emits no rows
    sensor_ids: tuple[int, ...]      # canonical ids present, for per-take "sensor missing" tallies


@dataclass
class Chunk:
    """One forward's worth of takes: the merged jagged pack and the takes that went into it."""
    metas: list[TakeMeta]
    batch: IMUData_BaseTrait_Jagged

    @property
    def index(self) -> int:
        """Position of the chunk's first take in the undivided item list (for ordering)."""
        return self.metas[0].index


# CPU pipeline: build takes in a worker pool, merge them into chunk-sized jagged packs.

class _TakeSource(Dataset):
    """One take per index: build its pipeline, collate its windows, report what it holds.

    Draws are keyed on ``(seed, stem)``, not the ambient RNG, so a take's corruption does not
    depend on shard count, worker count or ``--limit``.
    """

    def __init__(self, items: tp.Sequence[TaggedItem], seed: int) -> None:
        self.items = list(items)
        self.seed  = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[TakeMeta, IMUData_BaseTrait_Jagged | None]:
        index, (stem, build) = self.items[i]
        stem_seed = (self.seed + zlib.crc32(stem.encode())) % (2**31)
        # fork_rng keeps the per-stem seed out of the worker's stream; devices=[] skips the CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stem_seed)
            random.seed(stem_seed)                  # AddGravity/AddNoise draw from `random` too
            seq = build(stem)
            n   = len(seq)
            if n == 0:
                return TakeMeta(index, stem, 0, ()), None
            batch = seq[:]

        # The id is constant within a device run, so read it off the run starts.
        ids = tuple(sorted(set(batch.sensor_id[batch.offsets_N[:-1], 0].tolist())))
        return TakeMeta(index, stem, n, ids), batch


def _to_chunks(
    samples   : list[tuple[TakeMeta, IMUData_BaseTrait_Jagged | None]],
    chunk_rows: int,
) -> tuple[list[TakeMeta], list[Chunk]]:
    """Collate fn: merge fetched takes into packs of at most ``chunk_rows`` rows.

    Metas cover every take, including short ones with no pack; a larger take stays whole as its own chunk.
    """
    metas  = [meta for meta, _ in samples]
    chunks : list[Chunk] = []
    pending: list[tuple[TakeMeta, IMUData_BaseTrait_Jagged]] = []
    rows   = 0

    def flush() -> None:
        nonlocal pending, rows
        if not pending:
            return
        packs = [batch for _, batch in pending]
        merged = packs[0] if len(packs) == 1 else type(packs[0]).collate(packs)
        chunks.append(Chunk([meta for meta, _ in pending], merged))
        pending, rows = [], 0

    for meta, batch in samples:
        if batch is None:
            continue
        n = batch.acc.shape[0]
        if pending and rows + n > chunk_rows:
            flush()
        pending.append((meta, batch))
        rows += n
    flush()
    return metas, chunks


def _takes_per_fetch(items: tp.Sequence[TaggedItem], seed: int, chunk_rows: int, cap: int = 64) -> int:
    """``DataLoader`` batch size, sized from the first non-empty take to bound worker memory."""
    source = _TakeSource(items, seed)
    for i in range(min(len(items), 4)):
        _, batch = source[i]
        if batch is not None:
            return max(1, min(cap, chunk_rows // batch.acc.shape[0]))
    return 1


def _init_worker(threads: int, _worker_id: int) -> None:
    """Size a loader worker's intra-op pool; torch's whole-machine default badly oversubscribes."""
    torch.set_num_threads(threads)


def budget_threads(shards: int, workers: int) -> int:
    """Intra-op threads for a process that runs alongside ``shards`` shards of ``workers`` each."""
    return max(1, (os.cpu_count() or 1) // max(1, shards * (workers + 1)))


def stream_chunks(
    items     : tp.Sequence[TaggedItem],
    *,
    seed      : int,
    workers   : int      = DEFAULT_WORKERS,
    chunk_rows: int      = DEFAULT_CHUNK_ROWS,
    threads   : int      = 1,
    report    : tp.Callable[[int], None] | None = None,
) -> tp.Iterator[tuple[list[TakeMeta], list[Chunk]]]:
    """Yield ``(metas, chunks)`` in item order, building the takes in a worker pool.

    ``chunks`` is empty when every take in a fetch was short; ``report`` gets the takes per fetch.
    """
    if not items:
        return

    loader = DataLoader(
        _TakeSource(items, seed),
        batch_size     = _takes_per_fetch(items, seed, chunk_rows),
        num_workers    = workers,
        collate_fn     = functools.partial(_to_chunks, chunk_rows=chunk_rows),
        worker_init_fn = functools.partial(_init_worker, threads),
        prefetch_factor= 2 if workers else None,
        shuffle        = False,
    )
    for metas, chunks in loader:
        if report is not None:
            report(len(metas))
        yield metas, chunks


# GPU sharding: one process per visible device, round-robin over the takes.

def shard_items(items: tp.Sequence[Item], shard: int, shards: int) -> list[TaggedItem]:
    """This shard's round-robin slice of ``items``, tagged with each take's index in the whole list."""
    return list(enumerate(items))[shard::shards]


def device_count(requested: int | None) -> int:
    """Number of shards: ``requested``, else every visible CUDA device (at least 1).

    Must not create a CUDA context: the parent stays CUDA-free until the shards spawn.
    """
    available = torch.cuda.device_count()
    if requested is not None:
        assert requested >= 1, f"--devices must be >= 1, got {requested}"
        return min(requested, available) if available else 1
    return max(available, 1)


def shard_device(shard: int) -> torch.device:
    """The device this shard runs on (``cuda:<shard>`` when there are GPUs, else CPU)."""
    return torch.device(f"cuda:{shard}") if torch.cuda.is_available() else torch.device("cpu")


def add_stream_args(parser, *, default_workers: int = DEFAULT_WORKERS) -> None:
    """Add the three knobs every suite evaluator shares."""
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"per-GPU data-loading workers building takes ahead of the forward "
                             f"(default: {default_workers}; 0 loads in-process, for debugging)")
    parser.add_argument("--devices", type=int, default=None,
                        help="GPUs to shard the takes over (default: every device visible to "
                             "CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                        help=f"packed rows merged into one forward (default: {DEFAULT_CHUNK_ROWS}); "
                             f"lower it if a shard runs out of GPU memory")


@contextlib.contextmanager
def _bars(totals: dict[str, int]) -> tp.Generator[Report]:
    """One progress bar per dataset, advanced by whichever shard finished the takes."""
    with Progress() as progress:
        tasks = {name: progress.add_task(name, total=total) for name, total in totals.items()}
        yield lambda dataset, n: progress.advance(tasks[dataset], n)


def _entry_point(fn: tp.Callable) -> str:
    """``"module:qualname"`` for a function, resolvable by import in a spawned process.

    Spawn does not rebuild a ``.__main__`` module, so resolve through the spec's real dotted name.
    """
    module = fn.__module__
    if module == "__main__":
        spec = getattr(sys.modules["__main__"], "__spec__", None)
        assert spec is not None, \
            "run_sharded needs an importable entry point; run the evaluator as `python -m ...`"
        module = spec.name
    return f"{module}:{fn.__qualname__}"


def _shard_main(entry: str, args, shard: int, shards: int, queue) -> None:
    """Shard entry point: run ``entry`` on ``cuda:<shard>`` and post the accumulator back."""
    try:
        module, qualname = entry.split(":")
        collect = getattr(importlib.import_module(module), qualname)
        if torch.cuda.is_available():
            torch.cuda.set_device(shard)
        # Torch's whole-machine default makes a sharded run slower than a serial one.
        torch.set_num_threads(budget_threads(shards, getattr(args, "workers", DEFAULT_WORKERS)))
        report: Report = lambda dataset, n: queue.put(("tick", shard, (dataset, n)))
        # Pickled by value: torch's shared-memory reducer is served by this process, which is exiting.
        queue.put(("done", shard, pickle.dumps(collect(args, shard, shards, report))))
    except BaseException:
        queue.put(("fail", shard, traceback.format_exc()))


def run_sharded(
    collect: tp.Callable[[tp.Any, int, int, Report], T],
    args   : tp.Any,
    shards : int,
    totals : dict[str, int],
) -> list[T]:
    """Run ``collect(args, shard, shards, report)`` once per GPU; return results in shard order.

    ``collect`` must be module-level and build its own model and pipelines (a CUDA context cannot
    cross the spawn). A single shard runs in-process.
    """
    if shards <= 1:
        with _bars(totals) as report:
            return [collect(args, 0, 1, report)]

    entry = _entry_point(collect)
    ctx   = mp.get_context("spawn")
    queue = ctx.Queue()
    # Not daemonic: shards fork DataLoader workers. The `finally` below tears them down instead.
    procs = [ctx.Process(target=_shard_main, args=(entry, args, i, shards, queue), daemon=False)
             for i in range(shards)]

    results: dict[int, T] = {}
    try:
        for proc in procs:
            proc.start()
        with _bars(totals) as report:
            while len(results) < shards:
                try:
                    kind, shard, payload = queue.get(timeout=30)
                except queue_mod.Empty:
                    # A quiet 30 s is normal, but a shard that died without posting would hang the run.
                    dead = [i for i, p in enumerate(procs)
                            if not p.is_alive() and i not in results]
                    if dead:
                        raise RuntimeError(
                            f"shard(s) {dead} exited without reporting "
                            f"(exit codes {[procs[i].exitcode for i in dead]})"
                        )
                    continue
                if kind == "tick":
                    report(*payload)
                elif kind == "done":
                    results[shard] = pickle.loads(payload)
                else:
                    raise RuntimeError(f"shard {shard} (cuda:{shard}) failed:\n{payload}")
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=30)
    return [results[i] for i in range(shards)]


def worker_summary(shards: int, workers: int) -> str:
    """One line describing how the run is being split, printed before the bars."""
    where = f"{shards} GPU{'s' if shards > 1 else ''}" if torch.cuda.is_available() else "CPU"
    return (f"[dim]streaming takes over {where} x {workers} loader worker"
            f"{'s' if workers != 1 else ''} ({os.cpu_count()} cores visible)[/dim]")
