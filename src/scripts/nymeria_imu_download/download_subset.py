#!/usr/bin/env python3
"""Download, verify and extract the first N Nymeria sequences; safe to re-run.

Usage: python src/scripts/nymeria_imu_download/download_subset.py --manifest <catalogue.json> [--sequences N]"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile

DATA_TYPES = [
    "body_motion",        # XSens skeleton: body/xdata.npz (240 Hz)
    "body_xdata_mvnx",    # raw XSens export: body/xdata.mvnx (17 IMU trackers)
    "recording_head",     # Aria head device (IMU in recording_head/data/motion.vrs)
    "recording_lwrist",   # left-wrist device IMU
    "recording_rwrist",   # right-wrist device IMU
]


def sha1sum(path, chunk=8 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def download(url, dest):
    # Resumable download with retries; -C - continues a partial file.
    cmd = [
        "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
        "--retry-all-errors", "-C", "-", "-o", dest, url,
    ]
    return subprocess.call(cmd) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Local download catalogue from the Aria Dataset Explorer.")
    parser.add_argument("--output", type=Path, default=Path(
        os.environ.get("FOUNDATIONIMU_DATA_ROOT", str(Path(__file__).resolve().parents[3] / "data"))
    ).expanduser() / "nymeria_imu")
    parser.add_argument("--sequences", type=int, default=10)
    args = parser.parse_args()
    if args.sequences < 1:
        parser.error("--sequences must be positive")
    with args.manifest.open() as source:
        cfg = json.load(source)
    out_dir = args.output
    sequences = cfg["sequences"]
    names = list(sequences)[:args.sequences]

    os.makedirs(out_dir, exist_ok=True)

    grand_ok, grand_skip, grand_fail = 0, 0, 0
    for i, name in enumerate(names, 1):
        seq_dir = os.path.join(out_dir, name)
        os.makedirs(seq_dir, exist_ok=True)
        print(f"\n=== [{i}/{len(names)}] {name} ===", flush=True)

        for dtype in DATA_TYPES:
            entry = sequences[name].get(dtype)
            done_marker = os.path.join(seq_dir, f".{dtype}.done")

            if entry is None:
                print(f"  {dtype}: not available for this sequence -- skip")
                continue
            if os.path.exists(done_marker):
                print(f"  {dtype}: already extracted -- skip")
                grand_skip += 1
                continue

            zip_path = os.path.join(seq_dir, entry["filename"])
            expect_sha = entry["sha1sum"]
            size = entry["file_size_bytes"]
            print(f"  {dtype}: downloading {human(size)} ...", flush=True)

            if not (os.path.exists(zip_path) and sha1sum(zip_path) == expect_sha):
                if not download(entry["download_url"], zip_path):
                    print(f"  {dtype}: DOWNLOAD FAILED")
                    grand_fail += 1
                    continue

            actual = sha1sum(zip_path)
            if actual != expect_sha:
                print(f"  {dtype}: SHA-1 MISMATCH (got {actual}, want {expect_sha})")
                grand_fail += 1
                continue

            if zip_path.endswith(".zip"):
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(seq_dir)
                except zipfile.BadZipFile:
                    print(f"  {dtype}: BAD ZIP")
                    grand_fail += 1
                    continue
                os.remove(zip_path)
                note = "verified + extracted"
            elif zip_path.endswith(".mvnx"):
                # Raw XSens MVNX XML -- not an archive; place in standard layout.
                body_dir = os.path.join(seq_dir, "body")
                os.makedirs(body_dir, exist_ok=True)
                os.replace(zip_path, os.path.join(body_dir, "xdata.mvnx"))
                note = "verified -> body/xdata.mvnx"
            else:
                print(f"  {dtype}: UNHANDLED FILE TYPE ({entry['filename']})")
                grand_fail += 1
                continue

            open(done_marker, "w").close()
            print(f"  {dtype}: OK ({human(size)} {note})")
            grand_ok += 1

    print(f"\nDone. ok={grand_ok} skipped={grand_skip} failed={grand_fail}")
    sys.exit(1 if grand_fail else 0)


if __name__ == "__main__":
    main()
