"""
Tier D per-seed Safety AoI 分布图 (Fig 2(b)).

Usage: python -m plotting.fig_aoi_distribution
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
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
DATA_DIR    = os.path.join(REPO, "results", "v058_20260409_130414")
OUT_DIR     = os.path.join(REPO, "results", "figures")
OUT_PDF     = os.path.join(REPO, "results", "figures", "fig_aoi_distribution.pdf")

def save_dual_format(fig, out_dir, base_name):
    for ext in [".pdf", ".png"]:
        path = os.path.join(out_dir, base_name + ext)
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"  {base_name}.pdf & .png")

def _find_polish_5seed_csv():
    """找最新的 polish_5seed/metrics_D_5seed.csv 路径."""
    candidates = sorted(glob.glob(os.path.join(REPO, "results",
                                              "v058_polish_5seed_*",
                                              "metrics_D_5seed.csv")))
    return candidates[-1] if candidates else None


def load_data():
    """返回 {method: list[aoi_ms]}. 优先 5-seed polish CSV, fallback 到 3-seed legacy."""
    polish_csv = _find_polish_5seed_csv()
    if polish_csv is not None:
        print(f"  [load_data] 5-seed CSV: {os.path.relpath(polish_csv, REPO)}")
        df = pd.read_csv(polish_csv)
        data = {m: df[df.method == m]["aoi_safety_ms"].tolist()
                for m in METHOD_ORDER}

        # DQN-AoI 改用独立 training-seed sweep (展现训练不稳定性, 而非单 checkpoint 重放)
        stab_path = os.path.join(DATA_DIR, "dqn_stability.csv")
        if os.path.exists(stab_path):
            stab = pd.read_csv(stab_path)
            data["DQN-AoI"] = stab["aoi_mean"].tolist()
        return data

    # Fallback: 3-seed metrics_D + 5-seed DQN stability
    print(f"  [load_data] no 5-seed CSV, fallback to 3-seed legacy")
    df = pd.read_csv(os.path.join(DATA_DIR, "metrics_D.csv"))
    data = {m: df[df.method == m]["aoi_total_SAFETY"].tolist()
            for m in METHOD_ORDER}
    stab = pd.read_csv(os.path.join(DATA_DIR, "dqn_stability.csv"))
    data["DQN-AoI"] = stab["aoi_mean"].tolist()
    return data


def fig_aoi_distribution():
    data = load_data()
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

    ax.text(0.5, 0.28, "5 seeds per method", transform=ax.transAxes, fontsize=8, color="dimgray", va="bottom", ha="center")

    # SLA threshold line + label (match fig_pareto style: non-bold, no "SLA" prefix)
    ax.axhline(SLA_MS, color=THR_COLOR, ls=THR_LS,
               alpha=THR_ALPHA, lw=THR_LW, zorder=2)
    ax.text(len(METHOD_ORDER) - 0.7, SLA_MS * 0.62,
            r"$\Gamma_S = 20$ ms",
            ha="right", va="bottom",
            fontsize=7, color=THR_COLOR)

    # Axes & decoration
    ax.set_title("(b)")  # Fig 2(b) per-seed AoI distribution
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
    os.makedirs(OUT_DIR, exist_ok=True)
    save_dual_format(fig, OUT_DIR, "fig_aoi_distribution")
    plt.close(fig)


def print_summary():
    data = load_data()
    print("\nTier D Safety AoI distribution (existing CSVs):")
    for m in METHOD_ORDER:
        aois = data[m]
        met = sum(1 for a in aois if a < SLA_MS)
        lo, hi = min(aois), max(aois)
        print(f"  {m:10s}: n={len(aois)}, "
              f"SLA {met}/{len(aois)} ({100*met/len(aois):.0f}%), "
              f"range [{lo:.2f}, {hi:.2f}] ms")


if __name__ == "__main__":
    fig_aoi_distribution()
    print_summary()
