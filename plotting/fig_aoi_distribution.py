"""Plot the five Load-D block aggregates used in the main comparison."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs import COLORS, MARKERS, METHOD_ORDER

# ─── Style (与 generate_all_figs.py 一致, STIX font lock) ─────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["STIXGeneral"],     # Times-metric-compatible bundled font
    "mathtext.fontset":  "stix",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "legend.fontsize":   9,                   # 9pt × 0.9 scale (Fig 2 subfigure) = 8.1pt rendered ≥ IEEE 8pt min
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.titlesize":  10,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

THR_COLOR = "#888888"
THR_LS    = "--"
THR_ALPHA = 0.7
THR_LW    = 1.0

SLA_MS = 20.0

# ─── Paths ───────────────────────────────────────────────────────────
REPO        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV    = os.path.join(REPO, "reference_results", "load_D", "load_D_blocks.csv")
OUT_DIR     = os.path.join(REPO, "results", "figures")

def save_dual_format(fig, out_dir, base_name):
    for ext in [".pdf", ".png"]:
        path = os.path.join(out_dir, base_name + ext)
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"  {base_name}.pdf & .png")

def load_data(data_csv=DATA_CSV):
    """Return five block-level Safety-AoI values per method."""
    frame = pd.read_csv(data_csv)
    data = {
        method: frame[frame.method == method]["aoi_safety_ms"].tolist()
        for method in METHOD_ORDER
    }
    missing = [method for method, values in data.items() if len(values) != 5]
    if missing:
        raise ValueError(f"expected five block rows for each method; invalid: {missing}")
    return data


def fig_aoi_distribution(data_csv=DATA_CSV, out_dir=OUT_DIR):
    data = load_data(data_csv)
    fig, ax = plt.subplots(figsize=(3.5, 2.51))

    # Tier D SLA 阈值阴影
    ax.axhspan(0.5, SLA_MS, color="#2ECC71", alpha=0.12, zorder=0)   # safe zone
    ax.axhspan(SLA_MS, 300, color="#E74C3C", alpha=0.08, zorder=0)   # danger zone

    # Scatter per method, with tiny horizontal jitter for readability
    rng = np.random.default_rng(42)
    for i, m in enumerate(METHOD_ORDER):
        aois = data[m]
        jitter = rng.uniform(-0.16, 0.16, size=len(aois))
        xs = np.full(len(aois), i) + jitter
        ax.scatter(xs, aois,
                   s=60, alpha=0.7,
                   c=COLORS[m], marker=MARKERS[m],
                   edgecolors="none", linewidth=0, zorder=5)
        # Compact (n=k) caption below each column
        #ax.text(i, 0.7, f"n={len(aois)}", ha="center", va="bottom", fontsize=7, color="dimgray")

    ax.text(0.5, 0.28, "5 deployment blocks per method", transform=ax.transAxes, fontsize=8, color="dimgray", va="bottom", ha="center")

    # SLA threshold line + label (match fig_pareto style: non-bold, no "SLA" prefix)
    ax.axhline(SLA_MS, color=THR_COLOR, ls=THR_LS,
               alpha=THR_ALPHA, lw=THR_LW, zorder=2)
    ax.text(len(METHOD_ORDER) - 0.7, SLA_MS * 0.62,
            r"$\Gamma_S = 20$ ms",
            ha="right", va="bottom",
            fontsize=7, color=THR_COLOR)

    # Axes & decoration
    ax.set_title("(b)")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 300)
    ax.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)
    ax.set_xticks(range(len(METHOD_ORDER)))
    #ax.tick_params(axis='x', pad=3)
    ax.set_xticklabels(METHOD_ORDER, rotation=15)

    # Color x-axis labels to match method colors (non-bold for cleaner Fig 2 alignment)
    for tick, m in zip(ax.get_xticklabels(), METHOD_ORDER):
        tick.set_color(COLORS[m])

    ax.set_ylabel("Safety AoI [ms]")
    ax.grid(True, ls="--", alpha=0.4, axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    #os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    #fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight", pad_inches=0.02)
    #plt.close(fig)
    #print(f"  fig_aoi_distribution.pdf -> {OUT_PDF}")
    os.makedirs(out_dir, exist_ok=True)
    save_dual_format(fig, out_dir, "fig_aoi_distribution")
    plt.close(fig)


def print_summary(data_csv=DATA_CSV):
    data = load_data(data_csv)
    print("\nTier D Safety AoI distribution (deployment blocks):")
    for m in METHOD_ORDER:
        aois = data[m]
        met = sum(1 for a in aois if a < SLA_MS)
        lo, hi = min(aois), max(aois)
        print(f"  {m:10s}: n={len(aois)}, "
              f"SLA {met}/{len(aois)} ({100*met/len(aois):.0f}%), "
              f"range [{lo:.2f}, {hi:.2f}] ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", default=DATA_CSV)
    parser.add_argument("--out-dir", default=OUT_DIR)
    arguments = parser.parse_args()
    fig_aoi_distribution(arguments.data_csv, arguments.out_dir)
    print_summary(arguments.data_csv)
