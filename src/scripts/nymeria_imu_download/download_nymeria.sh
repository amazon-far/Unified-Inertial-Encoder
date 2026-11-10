#!/usr/bin/env bash
# Download the Nymeria dataset, restricted to the datatypes already present in ./data/:
#   body_motion, body_xdata_mvnx, recording_head, recording_lwrist, recording_rwrist
#
# Excluded (heavy / not currently used):
#   recording_head_data_data_vrs, recording_observer, recording_observer_data_data_vrs,
#   semidense_observations, video_main_rgb, narration_*, metadata_json, LICENSE
#
# Expected on-disk total for the included types: ~4.3 TB across 1100 sequences.
#
# Env overrides:
#   NYMERIA_JSON           path to Nymeria_download_urls.json
#   NYMERIA_DATA_DIR       output root (default: $FOUNDATIONIMU_DATA_ROOT/nymeria_imu)
#   NYMERIA_PARALLEL       concurrent downloads (default: 4)
#   NYMERIA_MAX_SEQUENCES  cap on number of sequences (0 = all)
#   NYMERIA_DRY_RUN        1 = list what would be downloaded, do nothing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
JSON="${NYMERIA_JSON:?Set NYMERIA_JSON to your local Aria download catalogue}"
DATA_DIR="${NYMERIA_DATA_DIR:-${FOUNDATIONIMU_DATA_ROOT:-$ROOT/data}/nymeria_imu}"
PARALLEL="${NYMERIA_PARALLEL:-4}"
MAX_SEQUENCES="${NYMERIA_MAX_SEQUENCES:-0}"
DRY_RUN="${NYMERIA_DRY_RUN:-0}"

for cmd in jq curl unzip sha1sum awk flock; do
  command -v "$cmd" >/dev/null || { echo "missing required command: $cmd" >&2; exit 1; }
done
[[ -f "$JSON" ]] || { echo "manifest not found: $JSON" >&2; exit 1; }

TYPES=(body_motion body_xdata_mvnx recording_head recording_lwrist recording_rwrist)

# Files that must exist on disk for a (seq, datatype) pair to be considered done.
expected_files() {
  case "$1" in
    body_motion)      echo "body/xdata.npz body/xdata_blueman.glb" ;;
    body_xdata_mvnx)  echo "body/xdata.mvnx" ;;
    recording_head)   echo "recording_head/data recording_head/mps" ;;
    recording_lwrist) echo "recording_lwrist/data recording_lwrist/mps" ;;
    recording_rwrist) echo "recording_rwrist/data recording_rwrist/mps" ;;
  esac
}

# Atomic completion marker per (seq, datatype). Written only after extraction succeeds.
marker_path() { printf '%s/.%s.done' "$1" "$2"; }
is_done()     { [[ -f "$(marker_path "$1" "$2")" ]]; }
mark_done()   { : > "$(marker_path "$1" "$2")"; }

# Verifies every expected path exists AND is non-empty (catches partial extractions
# where the dir was created but unzip was killed before writing its contents).
looks_complete() {
  local seq_dir="$1" dt="$2" p ef
  for ef in $(expected_files "$dt"); do
    p="$seq_dir/$ef"
    if [[ -d "$p" ]]; then
      [[ -n "$(find "$p" -mindepth 1 -print -quit 2>/dev/null)" ]] || return 1
    elif [[ -f "$p" && -s "$p" ]]; then
      :
    else
      return 1
    fi
  done
  return 0
}

mkdir -p "$DATA_DIR"

# Refuse to run if a previous buggy version of this script left extracted content at the
# dataset root (where every sequence's zip clobbered every other sequence's).
misplaced=()
for d in body recording_head recording_lwrist recording_rwrist; do
  [[ -e "$DATA_DIR/$d" ]] && misplaced+=("$d")
done
if (( ${#misplaced[@]} )); then
  {
    echo "ERROR: leftover extractions at the dataset root: ${misplaced[*]}"
    echo "  These were created by a buggy earlier run (extracted zips into $DATA_DIR/"
    echo "  instead of $DATA_DIR/<sequence>/). Content is unattributable to any sequence."
    echo "  To clean up and re-run:"
    echo "    rm -rf $DATA_DIR/{body,recording_head,recording_lwrist,recording_rwrist}"
  } >&2
  exit 2
fi

# Single-instance lock so a second invocation cannot race against an in-progress one.
LOCKFILE="$DATA_DIR/.nymeria_download.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "ERROR: another download appears to be running (lockfile: $LOCKFILE)" >&2
  echo "  If you're sure no other instance is alive: rm $LOCKFILE" >&2
  exit 3
fi

# --- Progress tracking ------------------------------------------------------
PROG_DIR="$DATA_DIR/.nymeria_progress"
rm -rf "$PROG_DIR"
mkdir -p "$PROG_DIR/active"
: > "$PROG_DIR/done.log"

human_bytes() {
  awk -v b="$1" 'BEGIN {
    u[0]="B"; u[1]="KB"; u[2]="MB"; u[3]="GB"; u[4]="TB"
    i=0; while (b>=1024 && i<4) { b/=1024; i++ }
    printf "%.2f %s", b, u[i]
  }'
}

human_dur() {
  awk -v s="$1" 'BEGIN {
    s = int(s)
    if (s < 60)        printf "%ds", s
    else if (s < 3600) printf "%dm%02ds", int(s/60), s%60
    else               printf "%dh%02dm", int(s/3600), int((s%3600)/60)
  }'
}

# Reads $PROG_DIR/done.log and emits one summary line (no trailing newline):
#   "  234/5464  4.3% | 1.45 GB / 4.30 TB | 145.2 MB/s | ETA 7h15m"
# Rate/ETA are computed from actual downloads only (skip/seed completions
# would otherwise inflate throughput at the start of the run).
progress_summary() {
  awk -F'\t' \
      -v total_f="$(cat "$PROG_DIR/total_files" 2>/dev/null)" \
      -v total_b="$(cat "$PROG_DIR/total_bytes" 2>/dev/null)" \
      -v t_now="$(date +%s)" '
    function hb(x,   i, u, n) {
      n = split("B KB MB GB TB", u, " ")
      i = 1
      while (x >= 1024 && i < n) { x /= 1024; i++ }
      return sprintf("%.2f %s", x, u[i])
    }
    BEGIN { dl_first = 0 }
    { n++; bdone += $5
      if ($2 == "done") {
        dlbytes += $5
        if (!dl_first || $1 < dl_first) dl_first = $1
      }
    }
    END {
      pct = (total_b > 0) ? 100 * bdone / total_b : 0
      if (dl_first && t_now > dl_first && dlbytes > 0) {
        rate = dlbytes / (t_now - dl_first)
        rate_str = sprintf("%6.1f MB/s", rate / 1048576)
        if (rate > 0) {
          eta = (total_b - bdone) / rate
          if (eta < 60)          eta_str = sprintf("%ds", int(eta))
          else if (eta < 3600)   eta_str = sprintf("%dm%02ds", int(eta/60), int(eta)%60)
          else if (eta < 360000) eta_str = sprintf("%dh%02dm", int(eta/3600), int((eta%3600)/60))
          else                   eta_str = ">99h"
        } else { eta_str = "?" }
      } else {
        rate_str = "      - MB/s"; eta_str = "-"
      }
      printf "%4d/%d %5.1f%% | %s / %s | %s | ETA %s",
             n+0, total_f+0, pct, hb(bdone), hb(total_b), rate_str, eta_str
    }
  ' "$PROG_DIR/done.log"
}

# log_completion <tag> <seq> <dt> <size> [note]
log_completion() {
  local tag="$1" seq="$2" dt="$3" size="$4" note="${5:-}"
  printf '%d\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$tag" "$seq" "$dt" "$size" >> "$PROG_DIR/done.log"
  printf '[%-4s] %s  %s/%s%s\n' "$tag" "$(progress_summary)" "$seq" "$dt" "${note:+  ($note)}"
  rm -f "$PROG_DIR/active/$$" 2>/dev/null || true
}

mark_active() {
  printf '%d\t%s\t%s\n' "$(date +%s)" "$1" "$2" > "$PROG_DIR/active/$$" 2>/dev/null || true
}

heartbeat_loop() {
  while sleep 30; do
    local active
    active=$(find "$PROG_DIR/active" -mindepth 1 -maxdepth 1 -type f 2>/dev/null | wc -l)
    printf '  ····  %s  │  %d active\n' "$(progress_summary)" "$active" >&2
  done
}

MANIFEST=$(mktemp)
HEARTBEAT_PID=0
cleanup() {
  [[ "$HEARTBEAT_PID" -gt 0 ]] && kill "$HEARTBEAT_PID" 2>/dev/null || true
  rm -f "$MANIFEST" "$MANIFEST.tmp"
  rm -rf "$PROG_DIR/active"
}
trap cleanup EXIT

# Flat TSV manifest: seq \t datatype \t filename \t sha1 \t size \t url
jq -r --argjson types "$(printf '%s\n' "${TYPES[@]}" | jq -R . | jq -s .)" '
  .sequences | to_entries[] as $s |
  $types[] as $t |
  ($s.value[$t] // empty) |
  [$s.key, $t, .filename, .sha1sum, (.file_size_bytes|tostring), .download_url] | @tsv
' "$JSON" > "$MANIFEST"

if [[ "$MAX_SEQUENCES" -gt 0 ]]; then
  awk -v n="$MAX_SEQUENCES" -F'\t' '
    !seen[$1]++ { kept++ }
    kept <= n
  ' "$MANIFEST" > "$MANIFEST.tmp" && mv "$MANIFEST.tmp" "$MANIFEST"
fi

total_files=$(wc -l < "$MANIFEST")
total_seq=$(cut -f1 "$MANIFEST" | sort -u | wc -l)
total_bytes=$(awk -F'\t' '{s+=$5} END {print s+0}' "$MANIFEST")
human=$(awk -v b="$total_bytes" 'BEGIN {
  u[0]="B"; u[1]="KB"; u[2]="MB"; u[3]="GB"; u[4]="TB"
  i=0; while (b>=1024 && i<4) { b/=1024; i++ }
  printf "%.2f %s", b, u[i]
}')

echo "Manifest: $total_files files across $total_seq sequences (~$human raw)"
echo "Types: ${TYPES[*]}"
echo "Output: $DATA_DIR"
echo "Parallel: $PARALLEL  Dry-run: $DRY_RUN"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] first 5 entries:"
  head -5 "$MANIFEST" | awk -F'\t' '{ printf "  %s\t%s\t%s\n", $1, $2, $3 }'
  exit 0
fi

# Stash totals for the progress tracker
echo "$total_files" > "$PROG_DIR/total_files"
echo "$total_bytes" > "$PROG_DIR/total_bytes"
date +%s > "$PROG_DIR/start_time"

download_one() {
  local seq="$1" dt="$2" filename="$3" sha="$4" size="$5" url="$6"
  local seq_dir="$DATA_DIR/$seq"
  mkdir -p "$seq_dir"

  # 1. Marker present -> already verified done by a prior run.
  if is_done "$seq_dir" "$dt"; then
    log_completion "skip" "$seq" "$dt" "$size"
    return 0
  fi

  # 2. No marker but all expected paths are present and non-empty -> seed the marker
  #    (handles data that was downloaded by the official Nymeria tool, not this script).
  if looks_complete "$seq_dir" "$dt"; then
    mark_done "$seq_dir" "$dt"
    log_completion "seed" "$seq" "$dt" "$size" "pre-existing"
    return 0
  fi

  mark_active "$seq" "$dt"
  local tmp="$seq_dir/.${filename}.part"
  local dl_start
  dl_start=$(date +%s)
  if ! curl -fsSL --retry 10 --retry-delay 5 --retry-all-errors -C - -o "$tmp" "$url"; then
    rm -f "$PROG_DIR/active/$$" 2>/dev/null || true
    echo "[fail] curl failed for $seq/$dt (.part preserved for resume)" >&2
    return 1
  fi

  local actual
  actual=$(sha1sum "$tmp" | awk '{print $1}')
  if [[ "$actual" != "$sha" ]]; then
    rm -f "$PROG_DIR/active/$$" 2>/dev/null || true
    echo "[fail] sha mismatch $seq/$dt: expected $sha got $actual; discarding .part" >&2
    rm -f "$tmp"
    return 1
  fi

  if [[ "$filename" == *.zip ]]; then
    # Nymeria zips contain top-level <datatype>/... (no sequence prefix) -> extract into the sequence dir.
    if ! unzip -qo "$tmp" -d "$seq_dir"; then
      rm -f "$PROG_DIR/active/$$" 2>/dev/null || true
      echo "[fail] unzip failed $seq/$dt" >&2
      return 1
    fi
    rm -f "$tmp"
  else
    case "$dt" in
      body_xdata_mvnx)
        mkdir -p "$seq_dir/body"
        mv "$tmp" "$seq_dir/body/xdata.mvnx"
        ;;
      *)
        mv "$tmp" "$seq_dir/$filename"
        ;;
    esac
  fi

  if ! looks_complete "$seq_dir" "$dt"; then
    rm -f "$PROG_DIR/active/$$" 2>/dev/null || true
    echo "[fail] post-extract verification failed for $seq/$dt (expected files missing/empty)" >&2
    return 1
  fi
  mark_done "$seq_dir" "$dt"
  local dl_elapsed=$(( $(date +%s) - dl_start ))
  log_completion "done" "$seq" "$dt" "$size" "$(human_bytes "$size") in $(human_dur "$dl_elapsed")"
}

export -f download_one expected_files is_done looks_complete mark_done marker_path \
          log_completion mark_active progress_summary human_bytes human_dur
export DATA_DIR PROG_DIR

heartbeat_loop &
HEARTBEAT_PID=$!

if [[ "$PARALLEL" -gt 1 ]]; then
  # Emit NUL-separated args via bash printf (mawk silently truncates \0 in format strings).
  while IFS=$'\t' read -r seq dt filename sha size url; do
    printf '%s\0%s\0%s\0%s\0%s\0%s\0' "$seq" "$dt" "$filename" "$sha" "$size" "$url"
  done < "$MANIFEST" |
    xargs -0 -n 6 -P "$PARALLEL" bash -c 'download_one "$@"' _
else
  while IFS=$'\t' read -r seq dt filename sha size url; do
    download_one "$seq" "$dt" "$filename" "$sha" "$size" "$url"
  done < "$MANIFEST"
fi

kill "$HEARTBEAT_PID" 2>/dev/null || true
HEARTBEAT_PID=0

echo
echo "──── FINAL ────"
printf '  %s\n' "$(progress_summary)"
fails=$(awk -F'\t' '$2=="fail" {c++} END {print c+0}' "$PROG_DIR/done.log" 2>/dev/null)
[[ "${fails:-0}" -gt 0 ]] && echo "  $fails files failed (see log above; rerun to retry)"
echo "Done."
