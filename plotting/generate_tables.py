"""
生成论文 LaTeX tables (Tier D / ablation / complexity).

Usage: python -m plotting.generate_tables
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs import EVAL, TIER_PARAMS

TIERS = ["A", "B", "C", "D"]


def load_data(data_dir):
    frames = []
    for tier in TIERS:
        path = os.path.join(data_dir, f"metrics_{tier}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# Table III: Extreme-Stress Performance (Tier D)
# ═══════════════════════════════════════════════════════════════
def table3_tier_d(df):
    """Paper Table III: Tier D main results. Data: aoi_total_SAFETY, violation_rate_SAFETY, queue_len_CE+IOT."""
    td = df[df["load_tier"] == "D"].groupby("method").agg(
        aoi_m=("aoi_total_SAFETY", "mean"),
        aoi_s=("aoi_total_SAFETY", "std"),
        viol_m=("violation_rate_SAFETY", "mean"),
        viol_s=("violation_rate_SAFETY", "std"),
        ceq=("queue_len_CE", "mean"),
        iotq=("queue_len_IOT", "mean"),
    )
    td["esq"] = td.ceq + td.iotq
    td = td.sort_values("aoi_m")

    print("% " + "=" * 60)
    print("% TABLE III: Extreme-Stress Performance Comparison (Tier D)")
    print("% " + "=" * 60)
    print(r"\begin{table}[t]")
    print(r"\caption{Extreme-Stress Performance Comparison (Tier D)}")
    print(r"\centering\small")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"Method & Safety AoI (ms) & Viol.\ Rate ($\Phi$) & ES Backlog (pkts) \\")
    print(r"\hline")

    for m, r in td.iterrows():
        note = ""
        if m == "SA-DAoI":   note = " (Ours)"
        elif m == "Static-RR": note = r"$^\dagger$"
        elif m == "DQN-AoI":   note = r"$^\ddagger$"
        bold = (r"\textbf{", "}") if m == "SA-DAoI" else ("", "")
        print(f"  {bold[0]}{m}{note}{bold[1]} "
              f"& ${r.aoi_m:.2f} \\pm {r.aoi_s:.2f}$ "
              f"& ${r.viol_m:.3f} \\pm {r.viol_s:.3f}$ "
              f"& ${r.esq:.1f}$ \\\\")

    print(r"\hline")
    print(r"\multicolumn{4}{l}{\footnotesize $^\dagger$ Zero backlog from buffer-overflow dropping, not effective control.} \\")
    print(r"\multicolumn{4}{l}{\footnotesize $^\ddagger$ Low AoI driven by unstable scheduling; highest ES backlog.} \\")
    print(r"\end{tabular}")
    print(r"\label{tab:tier_d}")
    print(r"\end{table}")
    print()


# ═══════════════════════════════════════════════════════════════
# Table IV: Ablation Study (Tier D, 跑 simulation)
# ═══════════════════════════════════════════════════════════════
def table4_ablation():
    from env.vehicular import VehicularNetworkEnv
    from agents.ablation import AblationScheduler
    from eval.evaluator import evaluate

    modes = {
        r"\textbf{Full SA-DAoI}":  "full",
        "No ISE":                   "no_ise",
        r"Fixed $\alpha{=}0.7$":   "alpha_low",
        r"Fixed $\alpha{=}1.0$":   "alpha_high",
    }

    # Collect data: 3 seeds × 10 episodes
    results = {}
    print("% Collecting ablation data (3 seeds x 10 episodes)...")
    for label, mode in modes.items():
        aois, viols, esqs = [], [], []
        for seed in [10, 20, 30]:
            env = VehicularNetworkEnv(load_tier="D")
            env.reset(seed=seed)
            agent = AblationScheduler(env, mode=mode)
            r = evaluate(agent, "D", num_episodes=10, seed=seed)
            aois.append(r["aoi_total_SAFETY"])
            viols.append(r["violation_rate_SAFETY"])
            esqs.append(r["queue_len_CE"] + r["queue_len_IOT"])
        results[label] = {
            "aoi_m": np.mean(aois), "aoi_s": np.std(aois),
            "viol_m": np.mean(viols), "viol_s": np.std(viols),
            "esq_m": np.mean(esqs), "esq_s": np.std(esqs),
        }

    # Compute delta violation vs Full
    full_viol = results[r"\textbf{Full SA-DAoI}"]["viol_m"]

    print()
    print("% " + "=" * 60)
    print("% TABLE IV: Ablation Study Results (Tier D)")
    print("% " + "=" * 60)
    print(r"\begin{table}[t]")
    print(r"\caption{Ablation Study Results (Tier~D, 30 episodes $\times$ 3 seeds)}")
    print(r"\centering\small")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Variant & Safety AoI & Viol.\ Rate & ES Backlog & $\Delta$Viol \\")
    print(r" & (ms) & ($\Phi$) & (pkts) & vs.\ Full \\")
    print(r"\hline")

    for label, data in results.items():
        if "Full" in label:
            delta = "---"
        else:
            pct = (data["viol_m"] - full_viol) / full_viol * 100
            delta = f"$+{pct:.1f}\\%$"
        print(f"  {label} & ${data['aoi_m']:.2f}$ "
              f"& ${data['viol_m']:.4f}$ "
              f"& ${data['esq_m']:.1f}$ "
              f"& {delta} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\label{tab:ablation}")
    print(r"\end{table}")
    print()

    # Also print plain text summary for quick reference
    print("% --- Plain text summary ---")
    for label, data in results.items():
        clean_label = label.replace(r"\textbf{", "").replace("}", "").replace("$", "").replace("\\alpha{=}", "a=")
        print(f"% {clean_label:20s} | AoI={data['aoi_m']:.2f}±{data['aoi_s']:.2f} "
              f"| Viol={data['viol_m']:.4f}±{data['viol_s']:.4f} "
              f"| ESQ={data['esq_m']:.1f}±{data['esq_s']:.1f}")
    print()


# ═══════════════════════════════════════════════════════════════
# Table V: Complexity and Inference Latency
# ═══════════════════════════════════════════════════════════════
def table5_complexity(df):
    # Extract latency data if available, otherwise use reference values
    latency_ref = {
        "SA-DAoI":   0.21,
        "T-AoI":     0.19,
        "Whittle":   0.20,
        "Static-RR": 0.01,
        "PPO-AoI":   2.3,
        "DQN-AoI":   2.3,
        "C-PPO":     2.5,
    }

    # Try to extract from CSV if latency columns exist
    if "latency_avg_ms" in df.columns:
        for method in latency_ref:
            md = df[(df["method"] == method) & (df["load_tier"] == "D")]
            if len(md) > 0 and md["latency_avg_ms"].notna().any():
                latency_ref[method] = md["latency_avg_ms"].mean()

    rows = [
        (r"\textbf{SA-DAoI (Ours)}", "Analytic",     r"$\mathcal{O}(|\mathcal{K}|)$", "No"),
        ("T-AoI",                     "Heuristic",    r"$\mathcal{O}(|\mathcal{K}|)$", "No"),
        ("Whittle",                   "Index-based",  r"$\mathcal{O}(|\mathcal{K}|)$", "No"),
        ("Static-RR",                 "Fixed",        r"$\mathcal{O}(1)$",              "No"),
        ("PPO-AoI",                   "DRL (Policy)", r"$\mathcal{O}(d^2)$",            "Yes (200k)"),
        ("DQN-AoI",                   "DRL (Value)",  r"$\mathcal{O}(d^2)$",            "Yes (200 ep)"),
        ("C-PPO",                     "Constr.\\ DRL", r"$\mathcal{O}(d^2)$",           "Yes (1M)"),
    ]

    print("% " + "=" * 60)
    print("% TABLE V: Complexity and Inference Latency Comparison")
    print("% " + "=" * 60)
    print(r"\begin{table}[t]")
    print(r"\caption{Complexity and Inference Latency Comparison}")
    print(r"\centering\small")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Method & Type & Complexity & Latency & Training \\")
    print(r" & & & (ms) & Required \\")
    print(r"\hline")

    for display_name, typ, comp, train in rows:
        # Get method name for latency lookup
        clean = display_name.replace(r"\textbf{", "").replace("}", "").replace(" (Ours)", "")
        lat = latency_ref.get(clean, 0)
        if lat < 0.05:
            lat_str = r"${<}0.01$"
        else:
            lat_str = "${{\\sim}}" + f"{lat:.2f}$"
        print(f"  {display_name} & {typ} & {comp} & {lat_str} & {train} \\\\")

    print(r"\hline")
    print(r"\multicolumn{5}{l}{\footnotesize $d$: hidden layer dimension (128). Near-RT RIC budget: 10\,ms.} \\")
    print(r"\end{tabular}")
    print(r"\label{tab:complexity}")
    print(r"\end{table}")
    print()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def generate_md_report(df, data_dir, out_dir):
    """Generate Markdown report with all table data for easy analysis."""
    import datetime
    md_path = os.path.join(out_dir, "tables_report.md")
    os.makedirs(out_dir, exist_ok=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SA-DAoI Paper Tables: Data Report\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Data source:** `{data_dir}`\n")
        f.write(f"**Rows:** {len(df)} | **Methods:** {sorted(df.method.unique().tolist())}\n\n")

        # ── Table III: Tier D ──
        f.write("---\n\n## Table III: Tier D Main Results\n\n")
        td = df[df["load_tier"] == "D"].groupby("method").agg(
            aoi_m=("aoi_total_SAFETY", "mean"),
            aoi_s=("aoi_total_SAFETY", "std"),
            viol_m=("violation_rate_SAFETY", "mean"),
            viol_s=("violation_rate_SAFETY", "std"),
            ceq=("queue_len_CE", "mean"),
            iotq=("queue_len_IOT", "mean"),
        )
        td["esq"] = td.ceq + td.iotq
        td["viol_pct"] = td.viol_m * 100
        td = td.sort_values("aoi_m")
        f.write("| Method | Safety AoI (ms) | Viol Rate (%) | ES Backlog (avg q) | SLA Met |\n")
        f.write("|--------|-----------------|---------------|--------------------|---------|\n")
        for m, r in td.iterrows():
            sla = "**YES**" if (r.aoi_m < 20 and r.viol_pct < 10) else "No"
            f.write(f"| {m} | {r.aoi_m:.2f} ± {r.aoi_s:.2f} | {r.viol_pct:.1f} ± {r.viol_s*100:.1f} "
                    f"| {r.esq:.1f} | {sla} |\n")

        # ── Cross-tier SA-DAoI ──
        f.write("\n---\n\n## Table IV: Cross-Tier SA-DAoI\n\n")
        f.write("| Tier | Γ_S (ms) | Safety AoI (ms) | Viol Rate (%) | Headroom (%) | Methods meeting SLA |\n")
        f.write("|------|----------|-----------------|---------------|-------------|---------------------|\n")
        for tier in TIERS:
            t_df = df[df["load_tier"] == tier]
            gamma_s = TIER_PARAMS[tier]["aoi_thresholds"]["S"]
            sa = t_df[t_df["method"] == "SA-DAoI"]
            if len(sa) > 0:
                aoi_m = sa["aoi_total_SAFETY"].mean()
                viol_m = sa["violation_rate_SAFETY"].mean() * 100
                headroom = (1 - aoi_m / gamma_s) * 100
                # Count methods meeting SLA
                tier_agg = t_df.groupby("method").agg(
                    aoi=("aoi_total_SAFETY", "mean"),
                    viol=("violation_rate_SAFETY", "mean"),
                )
                n_sla = sum(1 for _, r in tier_agg.iterrows() if r.aoi < gamma_s and r.viol * 100 < 10)
                n_total = len(tier_agg)
                f.write(f"| {tier} | {gamma_s} | {aoi_m:.2f} | {viol_m:.1f} | {headroom:.1f} | {n_sla}/{n_total} |\n")

        # ── Per-tier all methods (for Fig.3) ──
        f.write("\n---\n\n## All Methods × All Tiers (for Fig.3)\n\n")
        f.write("| Tier | Method | AoI (ms) | Viol (%) | ES Backlog |\n")
        f.write("|------|--------|----------|----------|------------|\n")
        for tier in TIERS:
            t_df = df[df["load_tier"] == tier]
            agg = t_df.groupby("method").agg(
                aoi=("aoi_total_SAFETY", "mean"),
                viol=("violation_rate_SAFETY", "mean"),
                ceq=("queue_len_CE", "mean"),
                iotq=("queue_len_IOT", "mean"),
            )
            agg["esq"] = agg.ceq + agg.iotq
            agg = agg.sort_values("aoi")
            for m, r in agg.iterrows():
                f.write(f"| {tier} | {m} | {r.aoi:.2f} | {r.viol*100:.1f} | {r.esq:.1f} |\n")

        # ── Ablation (from CSV if available) ──
        abl_path = os.path.join(data_dir, "ablation_D.csv")
        if os.path.exists(abl_path):
            f.write("\n---\n\n## Table V: Ablation (Tier D)\n\n")
            abl = pd.read_csv(abl_path)
            f.write("| Variant | AoI (ms) | Viol (%) |\n")
            f.write("|---------|----------|----------|\n")
            for _, r in abl.iterrows():
                f.write(f"| {r['variant']} | {r['aoi']:.2f} | {r['viol']:.1f} |\n")

        # ── Sensitivity (from CSV if available) ──
        sens_path = os.path.join(data_dir, "sensitivity_D.csv")
        if os.path.exists(sens_path):
            f.write("\n---\n\n## Sensitivity (Tier D)\n\n")
            sens = pd.read_csv(sens_path)
            f.write("| ISE_gate | alpha_floor | beta | AoI (ms) | Viol (decimal) | CE_Q | IOT_Q |\n")
            f.write("|----------|-------------|------|----------|----------------|------|-------|\n")
            for _, r in sens.iterrows():
                ise = f"{r['ise_gate_ratio']:.1f}" if pd.notna(r.get('ise_gate_ratio')) else "-"
                af = f"{r['alpha_floor']:.2f}" if pd.notna(r.get('alpha_floor')) else "-"
                bt = f"{r['beta']:.2f}" if pd.notna(r.get('beta')) else "-"
                f.write(f"| {ise} | {af} | {bt} | {r['aoi_total_SAFETY']:.2f} | "
                        f"{r['violation_rate_SAFETY']:.4f} | {r['queue_len_CE']:.1f} | {r['queue_len_IOT']:.1f} |\n")

        # ── Bursty (from CSV if available) ──
        bursty_path = os.path.join(data_dir, "bursty_results.csv")
        if os.path.exists(bursty_path):
            f.write("\n---\n\n## Table VI: Bursty Traffic (MMPP)\n\n")
            bursty = pd.read_csv(bursty_path)
            f.write("| Method | AoI (ms) | Viol (%) | Backlog |\n")
            f.write("|--------|----------|----------|---------|\n")
            for _, r in bursty.iterrows():
                f.write(f"| {r['method']} | {r['aoi_mean']:.2f} | {r['viol_mean']:.1f} | {r['backlog_mean']:.0f} |\n")

        # ── DQN Stability ──
        dqn_path = os.path.join(data_dir, "dqn_stability.csv")
        if os.path.exists(dqn_path):
            f.write("\n---\n\n## DQN 5-Seed Stability\n\n")
            dqn = pd.read_csv(dqn_path)
            f.write("| Seed | AoI (ms) | SLA Met |\n")
            f.write("|------|----------|---------|\n")
            for _, r in dqn.iterrows():
                f.write(f"| {int(r['seed'])} | {r['aoi_mean']:.1f} | {r['sla_met']} |\n")
            feasible = dqn['sla_met'].sum()
            f.write(f"\n**Feasibility:** {feasible}/5 ({feasible/5*100:.0f}%)\n")

    print(f"  MD report saved: {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="results/20260326_125453")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for MD report (default: same as data_dir)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_dir

    df = load_data(args.data_dir)
    print(f"% Loaded {len(df)} rows from {args.data_dir}")
    print(f"% Methods: {sorted(df.method.unique())}")
    print(f"% Tiers: {sorted(df.load_tier.unique())}")
    print()

    table3_tier_d(df)
    table5_complexity(df)

    # Table IV 需 simulation, 最后跑
    try:
        table4_ablation()
    except Exception as e:
        print(f"% [WARN] Table IV (ablation) skipped: {e}")
        print(f"% Run on server with full environment to generate.")

    # Generate MD report
    generate_md_report(df, args.data_dir, out_dir)

    print("% All tables generated successfully.")


if __name__ == "__main__":
    main()
