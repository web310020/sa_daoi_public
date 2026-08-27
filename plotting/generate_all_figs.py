"""
生成所有论文 figures.

Usage: python -m plotting.generate_all_figs
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs import COLORS, MARKERS, LINE_WIDTHS, METHOD_ORDER

#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42

# STIX font lock (跨 Win/Linux PDF 一致, IEEE 兼容)
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["STIXGeneral"],     # Times-metric-compatible matplotlib-bundled
    "mathtext.fontset":  "stix",              # math glyphs match body
    "pdf.fonttype":      42,                  # TrueType (IEEE PDF eXpress compliant)
    "ps.fonttype":       42,
    "font.size":         9,                   # default body
    "axes.labelsize":    9,                   # x/y axis labels
    "axes.titlesize":    10,                  # subplot titles
    "legend.fontsize":   9,                   # legend (9pt × 0.9 scale = 8.1pt rendered ≥ IEEE 8pt min)
    "xtick.labelsize":   9,                   # x tick numbers (same)
    "ytick.labelsize":   9,                   # y tick numbers (same)
    "figure.titlesize":  10,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

THR_COLOR = "#888888"
THR_LS = "--"
THR_ALPHA = 0.7
THR_LW = 1.0

TIERS = ["A", "B", "C", "D"]
TIER_LABELS = ["Tier A", "Tier B", "Tier C", "Tier D"]



def save_dual_format(fig, out_dir, base_name):
    for ext in [".pdf", ".png"]:
        path = os.path.join(out_dir, base_name + ext)
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"  {base_name}.pdf & .png")

def load_data(data_dir):
    frames = []
    for tier in TIERS:
        path = os.path.join(data_dir, f"metrics_{tier}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path); df["Load_Condition"] = f"Tier {tier}"
            frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["Load_Condition"] = pd.Categorical(
        full["Load_Condition"], categories=TIER_LABELS, ordered=True)
    full["queue_len_ES"] = full["queue_len_CE"] + full["queue_len_IOT"]
    return full


def fig2a_violation(df, out_dir):
    """Paper Fig.3(a): Safety Violation Rate across tiers. Data: violation_rate_SAFETY from metrics CSV."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    sns.lineplot(data=df, x="Load_Condition", y="violation_rate_SAFETY",
                 hue="method", hue_order=METHOD_ORDER, style="method",
                 size="method", sizes=LINE_WIDTHS, palette=COLORS,
                 markers=MARKERS, markersize=6, dashes=True,
                 errorbar="sd", alpha=0.8, err_kws={"alpha": 0.15}, ax=ax)
    ax.set_title("(a)")
    ax.set_xlabel("Traffic Scenarios")
    ax.set_ylabel(r"Safety Violation Rate ($\Phi$)")
    ax.grid(True, ls="--", alpha=0.4)

    # legend + 按 COLORS 字典着色
    leg = ax.legend(loc="upper center", frameon=True,
                    fontsize=7, framealpha=1.0, ncol=2,
                    labelspacing=0.2, handletextpad=0.2,
                    columnspacing=0.4, handlelength=1.5, borderpad=0.2)
    for text in leg.get_texts():
        method_name = text.get_text()
        if method_name in COLORS:
            text.set_color(COLORS[method_name])

    fig.savefig(os.path.join(out_dir, "fig_violation_rate.pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  fig_violation_rate.pdf")


def fig2b_safety_aoi(df, out_dir):
    """Paper Fig.3(b): Avg Safety AoI across tiers. Data: aoi_total_SAFETY from metrics CSV."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    sns.lineplot(data=df, x="Load_Condition", y="aoi_total_SAFETY",
                 hue="method", hue_order=METHOD_ORDER, style="method",
                 size="method", sizes=LINE_WIDTHS, palette=COLORS,
                 markers=MARKERS, markersize=6, dashes=True,
                 errorbar="sd", alpha=0.8, err_kws={"alpha": 0.15}, ax=ax)

    #ax.hlines(y=20, xmin=0, xmax=3.15, color=THR_COLOR, ls=THR_LS, alpha=THR_ALPHA, lw=2.0, zorder=1)
    #ax.axhline(y=20, color=THR_COLOR, ls=THR_LS, alpha=THR_ALPHA, lw=2.0, zorder=1)
    #ax.text(0.2, 22, "Threshold", ha="left", va="bottom", fontsize=9, color=THR_COLOR, fontweight="bold")
    #ax.set_xlim(-0.2, 3.41)

    # 每个 Tier 对应的 safety threshold (Gamma_S, ms): A=100 / B=60 / C=40 / D=20
    tier_gamma = {0: 100, 1: 60, 2: 40, 3: 20}

    for i, gamma in tier_gamma.items():
        x_start = i - 0.3
        x_end = i + 0.3
        # 阈值虚线
        ax.hlines(y=gamma, xmin=x_start, xmax=x_end, color=THR_COLOR, ls=THR_LS, alpha=THR_ALPHA, lw=1.0, zorder=1)
        # 安全区 (0 ~ Gamma) + 危险区 (Gamma ~ 160)
        ax.fill_between([x_start, x_end], 0, gamma, color='#2ECC71', alpha=0.12, zorder=0)
        ax.fill_between([x_start, x_end], gamma, 160, color='#E74C3C', alpha=0.08, zorder=0)
        pos = i + 0.2 if i == 3 else i
        ax.text(pos, gamma + 2, f"{gamma}ms", ha="center", va="bottom", fontsize=7, color=THR_COLOR, fontweight="bold", alpha=0.8)

    ax.text(0.4, 110, "Tier-specific Safety Limits", ha="left", va="top", fontsize=7, color=THR_COLOR, fontweight="bold", style='italic')

    ax.set_title("(b)")
    ax.set_xlabel("Traffic Scenarios")
    ax.set_ylabel("Avg. Safety AoI [ms]")
    ax.grid(True, ls="--", alpha=0.4)
    ax.get_legend().remove()
    fig.savefig(os.path.join(out_dir, "fig_safety_aoi.pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  fig_safety_aoi.pdf")


def fig2c_elastic_backlog(df, out_dir):
    """Final ES Backlog across tiers (deprecated, kept for re-enable path)."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    sns.lineplot(data=df, x="Load_Condition", y="queue_len_ES",
                 hue="method", hue_order=METHOD_ORDER, style="method",
                 size="method", sizes=LINE_WIDTHS, palette=COLORS,
                 markers=MARKERS, markersize=6, dashes=True,
                 errorbar="sd", alpha=0.8, err_kws={"alpha": 0.15}, ax=ax)
    ax.set_title("(c)")
    ax.set_xlabel("Traffic Scenarios")
    ax.set_ylabel("Final ES Backlog [pkts]")
    ax.grid(True, ls="--", alpha=0.4)
    ax.get_legend().remove()
    fig.savefig(os.path.join(out_dir, "fig_elastic_backlog.pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  fig_elastic_backlog.pdf")


def fig_alpha_evolution(out_dir, data_dir=None):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    tier_cfg = {
        "C": {"label": "Load Case C", "color": "#e377c2", "ls": "--", "lw": 2.2},
        "D": {"label": "Load Case D", "color": "#1f77b4", "ls": "-",  "lw": 1.0},
    }

    # Try to load from CSV first (generated by overnight script)
    loaded_from_csv = False
    if data_dir:
        for tier, cfg in tier_cfg.items():
            csv_path = os.path.join(data_dir, f"timeseries_{tier}.csv")
            if os.path.exists(csv_path):
                ts = pd.read_csv(csv_path)
                ax.plot(ts["alpha"].dropna().values, label=cfg["label"],
                        color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"], alpha=0.9)
                loaded_from_csv = True

    # Fallback: run simulation
    if not loaded_from_csv:
        from env.vehicular import VehicularNetworkEnv
        from agents.sa_daoi import SADAOIScheduler
        from eval.evaluator import collect_timeseries
        for tier, cfg in tier_cfg.items():
            env = VehicularNetworkEnv(load_tier=tier, max_steps=800)
            env.reset(seed=42)
            agent = SADAOIScheduler(env)
            data = collect_timeseries(agent, tier, 800, 42)
            ax.plot(data["alpha_ts"], label=cfg["label"], color=cfg["color"],
                    ls=cfg["ls"], lw=cfg["lw"], alpha=0.9)

    ax.axhline(y=1.0,  color=THR_COLOR, ls=":", alpha=0.4)
    # 注: 不画 alpha_min 阈值线; pink Load C 轨迹本身就是经验 floor
    ax.text(805, 1.0,  r"$\alpha_{max}$", va="center", fontsize=9, color=THR_COLOR)
    ax.text(805, 0.85, r"$\alpha_{min}$", va="center", fontsize=9, color=THR_COLOR)

    ax.annotate("Violation surge\n" + r"$\rightarrow$ $\alpha$ jumps to 1.0",
                xy=(373, 1.0), xytext=(450, 0.77),
                fontsize=8, color=tier_cfg["D"]["color"],
                arrowprops=dict(arrowstyle="->", color=tier_cfg["D"]["color"], lw=0.8),
                ha="center")
    ax.annotate("No violations\n" + r"$\rightarrow$ $\alpha$ stays at floor",
                xy=(99, 0.85), xytext=(150, 0.77),
                fontsize=8, color=tier_cfg["C"]["color"],
                arrowprops=dict(arrowstyle="->", color=tier_cfg["C"]["color"], lw=0.8),
                ha="center")

    ax.set_ylim(0.7, 1.02)
    ax.set_xlim(0, 800)
    ax.set_xticks(np.arange(0, 801, 100))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"Protection Bound $\alpha(t)$")
    ax.grid(True, ls=":", alpha=0.4)
    leg = ax.legend(loc="lower right", frameon=True,
              fontsize=7, framealpha=1.0, ncol=1,
              labelspacing=0.2, handletextpad=0.2,
              columnspacing=0.4, handlelength=1.5, borderpad=0.2)

    fig.tight_layout()
    # fig.savefig(os.path.join(out_dir, "fig_alpha_evolution.pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    # plt.close(fig)
    # print("  fig_alpha_evolution.pdf")
    save_dual_format(fig, out_dir, "fig_alpha_evolution")
    plt.close(fig)


def fig_ablation_bars(out_dir, data_dir=None):
    """Paper Fig.5b: Ablation bar chart (Tier D). Reads from ablation_D.csv if available."""
    from matplotlib.patches import Patch

    # Label mapping: CSV variant name → display label
    label_map = {
        "Full SA-DAoI": "Full\nSA-DAoI",
        "No ISE": "No ISE",
        "Fixed alpha=0.7": r"Fixed" + "\n" + r"$\alpha$=0.7",
        "Fixed alpha=1.0": r"Fixed" + "\n" + r"$\alpha$=1.0",
    }

    results = {}
    loaded_from_csv = False

    # Try CSV first
    if data_dir:
        csv_path = os.path.join(data_dir, "ablation_D.csv")
        if os.path.exists(csv_path):
            abl_df = pd.read_csv(csv_path)
            for _, row in abl_df.iterrows():
                display = label_map.get(row["variant"], row["variant"])
                results[display] = {
                    "aoi_m": row["aoi"], "aoi_s": 0.0,
                    "viol_m": row["viol"] / 100.0, "viol_s": 0.0,
                }
            loaded_from_csv = True

    # Fallback: run simulation
    if not loaded_from_csv:
        from env.vehicular import VehicularNetworkEnv
        from agents.ablation import AblationScheduler
        from eval.evaluator import evaluate
        modes = {
            "Full\nSA-DAoI": "full", "No ISE": "no_ise",
            r"Fixed" + "\n" + r"$\alpha$=0.7": "alpha_low",
            r"Fixed" + "\n" + r"$\alpha$=1.0": "alpha_high",
        }
        for label, mode in modes.items():
            aois, viols = [], []
            for seed in [10, 20, 30]:
                env = VehicularNetworkEnv(load_tier="D")
                env.reset(seed=seed)
                agent = AblationScheduler(env, mode=mode)
                r = evaluate(agent, "D", num_episodes=10, seed=seed)
                aois.append(r["aoi_total_SAFETY"])
                viols.append(r["violation_rate_SAFETY"])
            results[label] = {
                "aoi_m": np.mean(aois), "aoi_s": np.std(aois),
                "viol_m": np.mean(viols), "viol_s": np.std(viols),
            }

    labels = list(results.keys())

    color_map = {
        "Full\nSA-DAoI": "#E63946",
        "No ISE": "#2A9D8F",
        r"Fixed" + "\n" + r"$\alpha$=0.7": "#9467BD",
        r"Fixed" + "\n" + r"$\alpha$=1.0": "#F4A261",
    }
    edge_colors = [color_map.get(l, "black") for l in labels]
    COLOR_AOI = "#B22222"   # AoI 用 firebrick
    COLOR_VIOL = "#1D3557"  # Viol 用 navy

    x = np.arange(len(labels))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(3.5, 2.25))
    ax1.set_axisbelow(True)
    ax1.bar(x, 19, width=w * 2.6, color=edge_colors, alpha=0.3, zorder=0)

    # 主图: AoI + Viol 双轴 bar
    aoi_vals = [results[l]["aoi_m"] for l in labels]
    aoi_errs = [results[l]["aoi_s"] for l in labels]
    bars1 = ax1.bar(x - w / 2, aoi_vals, w, yerr=aoi_errs, capsize=4, color=COLOR_AOI, alpha=0.9, zorder=3)

    ax2 = ax1.twinx()
    viol_vals = [results[l]["viol_m"] * 100 for l in labels]
    viol_errs = [results[l]["viol_s"] * 100 for l in labels]
    bars2 = ax2.bar(x + w / 2, viol_vals, w, yerr=viol_errs, capsize=4,color=COLOR_VIOL, alpha=0.85, zorder=3)

    ax1.set_ylabel("Safety AoI [ms]", color=COLOR_AOI)
    ax2.set_ylabel("Violation Rate [%]", color=COLOR_VIOL)
    ax1.tick_params(axis="y", labelcolor=COLOR_AOI)
    ax2.tick_params(axis="y", labelcolor=COLOR_VIOL)

    ax1.set_ylim(0, 26)
    ax1.axhline(y=20, color=THR_COLOR, ls="--", alpha=0.7, lw=1.0, zorder=2)
    ax1.text(1.2, 20.5, r"$\Gamma_S$=20 ms", color=THR_COLOR, fontsize=7, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_AOI, label='Safety AoI [ms]'),
        Patch(facecolor=COLOR_VIOL, label='Violation Rate [%]')
    ]
    ax1.legend(handles=legend_elements, loc="upper left",
               fontsize=7, framealpha=1.0, ncol=1,
               labelspacing=0.2, handletextpad=0.2,
               columnspacing=0.4, handlelength=1.5, borderpad=0.2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax1.tick_params(axis='x', pad=3)

    for tick, color in zip(ax1.get_xticklabels(), edge_colors):
        tick.set_color(color)

    ax1.grid(True, ls=":", alpha=0.4, axis="y", zorder=1)
    fig.tight_layout()
    # fig.savefig(os.path.join(out_dir, "fig_ablation_bars.pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    # plt.close(fig)
    # print("  fig_ablation_bars.pdf")
    save_dual_format(fig, out_dir, "fig_ablation_bars")
    plt.close(fig)


def fig_pareto(df, out_dir):
    """Paper Fig.4: Pareto front (Safety AoI vs ES Backlog, Tier D). Data: aoi_total_SAFETY, queue_len_ES from metrics CSV."""
    td = df[df["Load_Condition"] == "Tier D"]
    summary = td.groupby("method").agg(
        aoi=("aoi_total_SAFETY", "mean"),
        esq=("queue_len_ES", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    for _, row in summary.iterrows():
        m = row["method"]
        ax.scatter(row.aoi, row.esq, s=60,
                   alpha=0.7,
                   c=COLORS.get(m, "gray"),
                   marker=MARKERS.get(m, "o"), zorder=5,
                   edgecolors="white", linewidth=0.6)

    offsets = {
        "SA-DAoI": (6, -3),
        "DQN-AoI": (10.2, -3),
        "T-AoI": (4, -3),
        "Whittle": (4, -2),
        "PPO-AoI": (4, -2.4),
        "C-PPO": (4, -2.0),
        "Static-RR": (0, 10),
    }
    for _, row in summary.iterrows():
        m = row["method"]
        dx, dy = offsets.get(m, (6, 0))
        ha = "right" if dx < 0 else "left"
        ax.annotate(m, (row.aoi, row.esq), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7, color=COLORS.get(m, "gray"),
                    fontweight="bold", ha=ha)

    # Tier-D SLA band (绿色 = 安全区, 红色 = 危险区, 阈值 Gamma_S=20ms)
    ax.axvspan(-100, 20,   color='#2ECC71', alpha=0.12, zorder=0)
    ax.axvspan(20,  1000,  color='#E74C3C', alpha=0.08, zorder=0)
    ax.axvline(x=20, color=THR_COLOR, ls=THR_LS, alpha=THR_ALPHA, lw=THR_LW)
    ax.text(22, 800, r"$\Gamma_S = 20$ ms", fontsize=7, color=THR_COLOR)
    ax.set_title("(a)")  # Fig 2(a) Pareto front
    ax.set_xlabel("Avg. Safety AoI [ms]")
    #ax.set_ylabel("Final ES Backlog [pkts]")
    #ax.set_ylabel("ES Backlog [pkts]")
    ax.set_ylabel("Final Backlog [pkts]")
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlim(-5, 192)
    ax.set_ylim(-50, 1220)

    # fig.savefig(os.path.join(out_dir, "fig_pareto.pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    # plt.close(fig)
    # print("  fig_pareto.pdf")
    save_dual_format(fig, out_dir, "fig_pareto")
    plt.close(fig)


def fig_sensitivity(out_dir, data_dir=None):
    """Paper Fig.6: Sensitivity analysis (3-panel). Reads from sensitivity_D.csv if available."""

    # Try to load from CSV
    sweeps = None
    if data_dir:
        csv_path = os.path.join(data_dir, "sensitivity_D.csv")
        if os.path.exists(csv_path):
            sdf = pd.read_csv(csv_path)
            sweeps = {}
            # ISE gate ratio
            ise = sdf[sdf["ise_gate_ratio"].notna()].sort_values("ise_gate_ratio")
            if len(ise) > 0:
                sweeps[r"$\epsilon_{ISE} / \Gamma_S$"] = {
                    "vals": ise["ise_gate_ratio"].tolist(),
                    "aoi": ise["aoi_total_SAFETY"].tolist(),
                    "esq": (ise["queue_len_CE"] + ise["queue_len_IOT"]).tolist(),
                    "default_idx": ise["ise_gate_ratio"].tolist().index(0.4) if 0.4 in ise["ise_gate_ratio"].values else 2,
                    "leg_loc": "center left",
                }
            # Alpha floor
            af = sdf[sdf["alpha_floor"].notna()].sort_values("alpha_floor")
            if len(af) > 0:
                sweeps[r"$\alpha_{min}$"] = {
                    "vals": af["alpha_floor"].tolist(),
                    "aoi": af["aoi_total_SAFETY"].tolist(),
                    "esq": (af["queue_len_CE"] + af["queue_len_IOT"]).tolist(),
                    "default_idx": af["alpha_floor"].tolist().index(0.85) if 0.85 in af["alpha_floor"].values else 2,
                    "leg_loc": "lower left",
                }
            # Beta
            bt = sdf[sdf["beta"].notna()].sort_values("beta")
            if len(bt) > 0:
                sweeps[r"$\beta$"] = {
                    "vals": bt["beta"].tolist(),
                    "aoi": bt["aoi_total_SAFETY"].tolist(),
                    "esq": (bt["queue_len_CE"] + bt["queue_len_IOT"]).tolist(),
                    "default_idx": bt["beta"].tolist().index(0.1) if 0.1 in bt["beta"].values else 1,
                    "leg_loc": "center right",
                }

    # Fallback: 内置默认数据
    if not sweeps:
        sweeps = {
            r"$\epsilon_{ISE} / \Gamma_S$": {
                "vals": [0.2, 0.3, 0.4, 0.5, 0.6],
                "aoi":  [13.08, 13.08, 13.63, 14.77, 12.24],
                "esq":  [480.1, 480.1, 471.4, 453.1, 451.4],
                "default_idx": 2, "leg_loc": "center left",
            },
            r"$\alpha_{min}$": {
                "vals": [0.70, 0.80, 0.85, 0.90, 0.95],
                "aoi":  [17.61, 15.28, 13.63, 7.22, 7.22],
                "esq":  [417.7, 456.0, 471.4, 577.2, 577.2],
                "default_idx": 2, "leg_loc": "center left",
            },
            r"$\beta$": {
                "vals": [0.05, 0.10, 0.15, 0.20],
                "aoi":  [17.78, 13.63, 8.74, 9.60],
                "esq":  [421.9, 471.4, 494.6, 490.4],
                "default_idx": 1, "leg_loc": "center right",
            },
        }

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.07))
    fig.subplots_adjust(wspace=0.55)

    c1, c2 = "#B22222", "Green"

    for i, (ax, (param, data)) in enumerate(zip(axes, sweeps.items())):
        x = data["vals"]
        di = data["default_idx"]

        #ln1 = ax.plot(x, data["aoi"], "o-", color=c1, lw=1.0, ms=5, label="Safety AoI")
        ln1 = ax.plot(x, data["aoi"], "o-", color=c1, lw=1.0, ms=5, label="Safety AoI", alpha=0.7)
        ax.plot(x[di], data["aoi"][di], "o", color=c1, ms=8, zorder=5,
                markerfacecolor="none", markeredgewidth=1.5)
        ax.set_xlabel(param)
        ax.set_ylabel("Safety AoI [ms]", color=c1)
        ax.tick_params(axis="y", labelcolor=c1)

        ax2 = ax.twinx()
        #ln2 = ax2.plot(x, data["esq"], "s--", color=c2, lw=1.0, ms=5, label="ES Backlog")
        #ln2 = ax2.plot(x, data["esq"], "s--", color=c2, lw=1.0, ms=5, label="Final Backlog")
        ln2 = ax2.plot(x, data["esq"], "s--", color=c2, lw=1.0, ms=5, label="Final Backlog", alpha=0.7)
        ax2.plot(x[di], data["esq"][di], "s", color=c2, ms=8, zorder=5,
                 markerfacecolor="none", markeredgewidth=1.5)
        #ax2.set_ylabel("ES Backlog [pkts]", color=c2)
        ax2.set_ylabel("Final Backlog [pkts]", color=c2)
        ax2.tick_params(axis="y", labelcolor=c2)

        #ax.axhline(y=20, color=THR_COLOR, ls=":", alpha=0.4, lw=0.7)

        if i == 1:
            lns = ln1 + ln2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc=data["leg_loc"],
                      fontsize=7, framealpha=1.0, ncol=1,
                      labelspacing=0.2, handletextpad=0.2,
                      columnspacing=0.4, handlelength=1.5, borderpad=0.2)
        ax.grid(True, ls=":", alpha=0.3)

    fig.tight_layout()
    save_dual_format(fig, out_dir, "fig_sensitivity")
    plt.close(fig)

    #print("  fig_sensitivity.pdf")


def print_table3(df):
    td = df[df["Load_Condition"] == "Tier D"]
    summary = td.groupby("method").agg(
        aoi_m=("aoi_total_SAFETY", "mean"), aoi_s=("aoi_total_SAFETY", "std"),
        viol_m=("violation_rate_SAFETY", "mean"), viol_s=("violation_rate_SAFETY", "std"),
        ceq=("queue_len_CE", "mean"), iotq=("queue_len_IOT", "mean"),
    )
    summary["esq"] = summary.ceq + summary.iotq
    summary = summary.sort_values("aoi_m")
    print("\n% ─── LaTeX Table III ─────────────────────────────────")
    for m, r in summary.iterrows():
        note = ""
        if m == "Static-RR":   note = r"$^\dagger$"
        elif m == "DQN-AoI":   note = r"$^\ddagger$"
        elif m == "SA-DAoI":   note = " (Ours)"
        print(f"  {m}{note} & ${r.aoi_m:.2f} \\pm {r.aoi_s:.2f}$ "
              f"& ${r.viol_m:.3f} \\pm {r.viol_s:.3f}$ & ${r.esq:.1f}$ \\\\")
    print("% ────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing the experiment CSV files")
    parser.add_argument("--out_dir",  default="results/figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = load_data(args.data_dir)
    print(f"\nGenerating figures -> {args.out_dir}/")
    print(f"  Data: {len(df)} rows, methods: {sorted(df.method.unique())}\n")

    # 这 3 个不生成 (数据已并入 Table III)
    #fig2a_violation(df, args.out_dir)
    #fig2b_safety_aoi(df, args.out_dir)
    # fig2c_elastic_backlog(df, args.out_dir)

    try:
        fig_alpha_evolution(args.out_dir, data_dir=args.data_dir)  # Paper Fig.5a
    except Exception as e:
        print(f"  [WARN] fig3a skipped: {e}")

    try:
        fig_ablation_bars(args.out_dir, data_dir=args.data_dir)  # Paper Fig.5b
    except Exception as e:
        print(f"  [WARN] fig3b skipped: {e}")

    fig_pareto(df, args.out_dir)          # Paper Fig.4
    fig_sensitivity(args.out_dir, data_dir=args.data_dir)  # Paper Fig.6
    #print_table3(df)

    print(f"All figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
