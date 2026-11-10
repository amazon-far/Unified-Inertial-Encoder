"""Rewrite Nymeria activity narrations with an LLM via Bedrock, one output CSV per style.

Usage:
    python -m src.scripts.nymeria_lang_transcribe [data_root] [--workers N] [--style {bones,motion-verb}]
"""

from src.runtime_prologue import machine_conf
import csv
import threading
import typing as tp

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress

from ...utility.transcriptor import transcribe_bones_style, transcribe_extract_motion_verb

_DESCRIPTION_COL_ = "Describe my activity"
_IN_NAME_  = "activity_summarization.csv"

# style -> (transcription function, output CSV); scripts.nymeria_lang_process reads these CSV names.
_STYLES_: dict[str, tuple[tp.Callable[[str], str], str]] = {
    "motion-verb": (transcribe_extract_motion_verb, "transcribed_activity_summarization.csv"),
    "bones"      : (transcribe_bones_style,         "transcribed_bones_style.csv"),
}

# Identical descriptions share one API call; the lock only guards the dict.
_cache: dict[str, str] = {}
_cache_lock = threading.Lock()


def _transcribe_cached(raw: str, transcribe: tp.Callable[[str], str]) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    with _cache_lock:
        if raw in _cache:
            return _cache[raw]
    result = transcribe(raw)
    with _cache_lock:
        _cache[raw] = result
    return result


def process_one(csv_path: Path, pool: ThreadPoolExecutor, style: str) -> str:
    transcribe, out_name = _STYLES_[style]
    out_path = csv_path.with_name(out_name)
    if out_path.exists():
        return f"[skip] {csv_path.parent.parent.name}"

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None and _DESCRIPTION_COL_ in reader.fieldnames
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # A failed row aborts the whole sequence, so the output file is all-or-nothing.
    transcriptions = list(pool.map(
        lambda raw: _transcribe_cached(raw, transcribe), (row[_DESCRIPTION_COL_] for row in rows),
    ))
    for row, transcribed in zip(rows, transcriptions):
        row[_DESCRIPTION_COL_] = transcribed

    # Write-then-rename: a partial file must not pass the exists() check above.
    tmp_path = out_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.rename(out_path)

    return f"[done] {csv_path.parent.parent.name} ({len(rows)} rows)"


def main(data_root: Path, workers: int, style: str) -> None:
    csv_paths = sorted(data_root.glob(f"*/narration/{_IN_NAME_}"))
    print(f"{len(csv_paths)} sequences with {_IN_NAME_} under {data_root} (style: {style})")

    failures: list[tuple[Path, Exception]] = []
    # `req_pool` bounds concurrency against Bedrock; `seq_pool` overlaps the per-sequence CSV IO.
    with Progress() as pbar, \
            ThreadPoolExecutor(max_workers=workers) as req_pool, \
            ThreadPoolExecutor(max_workers=8) as seq_pool:
        task = pbar.add_task("Transcribing sequences", total=len(csv_paths))
        futures = {seq_pool.submit(process_one, p, req_pool, style): p for p in csv_paths}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                pbar.console.print(fut.result())
            except Exception as e:  # keep going; report at the end
                failures.append((path, e))
                pbar.console.print(f"[red][fail] {path.parent.parent.name}: {e}[/red]")
            pbar.advance(task)

    print(f"{len(csv_paths) - len(failures)}/{len(csv_paths)} sequences transcribed")
    if failures:
        print(f"{len(failures)} failed:")
        for path, e in failures:
            print(f"  {path.parent.parent.name}: {e!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("data_root", type=Path, nargs="?", default=(Path(machine_conf('data_root')) / 'nymeria_imu'))
    parser.add_argument("--workers", type=int, default=16, help="concurrent Bedrock requests")
    parser.add_argument("--style", choices=sorted(_STYLES_), default="motion-verb",
                        help="transcription voice; each style owns its output CSV name")
    args = parser.parse_args()

    main(args.data_root, workers=args.workers, style=args.style)
