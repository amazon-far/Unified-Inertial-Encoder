"""Draw the runtime comparison figure from the JSON results of the profiling scripts.

Usage:
    python -m src.scripts.visualize_runtime [--results temp/efficiency_profiling] [--out temp/efficiency_profiling/runtime]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK       = "#444444"
AXIS_INK  = "#b8bfc4"
GRID      = "#e8ebed"
GOLD      = "#d8b35c"   # Ours
BLUE      = "#94b6d2"   # TartanIMU
GRAY      = "#a6a6a6"   # AirIO

# Parameter counts (millions) as printed by the profiling scripts.
PARAMS_M = [0.387, 5.70, 55.56, 55.56]   # AirIO, TartanIMU, Ours, Ours w/ TRT


def sig3(v: float) -> str:
    """Three significant digits, trailing zeros kept (0.387 / 5.70 / 55.6)."""
    return f"{v:.{max(0, 2 - math.floor(math.log10(abs(v))))}f}"


FILES = {
    "airio":     "airio_1imu_200hz_4s.json",
    "tartanimu": "tartanimu_1imu_200hz_10s.json",
    "ours":      "ours_12imu_200hz_4s_trt.json",
}


def medians(path: Path) -> dict[str, float]:
    return {s["name"]: float(s["median_ms"]) for s in json.loads(path.read_text())["stages"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("temp/efficiency_profiling"))
    parser.add_argument("--out", type=Path, default=Path("temp/efficiency_profiling/runtime"), help="stem; writes .pdf and .png")
    args = parser.parse_args()

    airio = medians(args.results / FILES["airio"])["end-to-end net forward"]
    tartan = medians(args.results / FILES["tartanimu"])["end-to-end stream()"]
    ours_all = medians(args.results / FILES["ours"])
    ours_e2e = ours_all["end-to-end forward"]
    ours_trt = ours_all["trt: forward incl. schedule"]
    device = json.loads((args.results / FILES["ours"]).read_text())["device_name"]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "mathtext.fontset": "dejavuserif",
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS_INK,
        "xtick.color": INK, "ytick.color": INK, "axes.labelsize": 6.2, "xtick.labelsize": 5.6, "ytick.labelsize": 5.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.55,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(1.75, 1.6))
    fig.subplots_adjust(left=0.36, right=0.99, bottom=0.21, top=0.98)

    ys = [3.75, 2.7, 1.25, 0.2]
    height = 0.62
    labels = ["AirIO\n1 IMU, 4 s", "TartanIMU\n1 IMU, 10 s", "Ours\n12 IMUs, 4 s", "Ours w/ TRT\n12 IMUs, 4 s"]
    runtimes = [airio, tartan, ours_e2e, ours_trt]
    colors = [GRAY, BLUE, GOLD, GOLD]
    ax.barh(ys, runtimes, height, color=colors, linewidth=0)
    x_max = ours_e2e * 1.9   # room for the two-line label after the longest bar
    for y, v, n_params in zip(ys, runtimes, PARAMS_M):
        ax.annotate(f"{v:.2f} ms", (v, y), xytext=(2, 3.0), textcoords="offset points", ha="left", va="center",
                    fontsize=5.6, color=INK)
        ax.annotate(f"{sig3(n_params)} M params", (v, y), xytext=(2, -3.2), textcoords="offset points", ha="left",
                    va="center", fontsize=5.0, color="#7d848b")

    divider = (ys[1] + ys[2]) / 2
    ax.axhline(divider, color=AXIS_INK, linewidth=0.7, linestyle=(0, (3.2, 2.2)), zorder=0)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(ys[-1] - 0.55, ys[0] + 0.55)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Runtime (ms)")
    ax.xaxis.grid(True, linewidth=0.45, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.tick_params(axis="x", length=2, width=0.5)


    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.01,
                    metadata={"Title": "Inference runtime per forward"} if ext == "pdf" else None)
    print(f"{device}: AirIO {airio:.2f}  TartanIMU {tartan:.2f}  Ours {ours_e2e:.2f}  Ours w/ TRT {ours_trt:.2f} ms")
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
