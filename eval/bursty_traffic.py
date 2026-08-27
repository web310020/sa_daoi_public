"""
MMPP 突发到达实验 (Tier D), 2-state Markov-modulated 到达.

Usage: python -m eval.bursty_traffic
"""

import sys, os, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) or ".")

from env.vehicular import VehicularNetworkEnv, SliceType
from agents.sa_daoi import SADAOIScheduler
from agents.heuristics import StaticRRPolicy, TAoIPolicy, WhittlePolicy
from agents.dqn_agent import DQNPolicy
from agents.ppo_cppo import PPOPolicy, CPPOPolicy


# ── MMPP parameters ──────────────────────────────────────────────────
BURST_FACTOR = 3.0        # arrival rate multiplier in burst state
Q_NORMAL_TO_BURST = 0.02  # transition prob per TTI: normal -> burst
Q_BURST_TO_NORMAL = 0.10  # transition prob per TTI: burst -> normal
# Mean sojourn: normal ~50 TTIs, burst ~10 TTIs -> ~17% time in burst

# ── Experiment config ────────────────────────────────────────────────
TIER = "D"
NUM_EPISODES = 30
EVAL_SEEDS = [10, 20, 30]
RESULTS_DIR = "results/bursty_traffic"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "load_D")

# Poisson baseline data from Table III (for comparison)
POISSON_REF = {
    "SA-DAoI":   {"aoi": 14.17, "viol": 8.5,  "backlog": 475.3},
    "DQN-AoI":   {"aoi": 17.13, "viol": 6.2,  "backlog": 877.2},
    "T-AoI":     {"aoi": 27.80, "viol": 27.1, "backlog": 321.9},
    "Whittle":   {"aoi": 50.57, "viol": 49.2, "backlog": 99.1},
    "PPO-AoI":   {"aoi": 108.12,"viol": 66.5, "backlog": 658.5},
    "C-PPO":     {"aoi": 155.09,"viol": 81.4, "backlog": 0.1},
    "Static-RR": {"aoi": 159.33,"viol": 83.8, "backlog": 0.0},
}

METHOD_ORDER = ["SA-DAoI", "DQN-AoI", "T-AoI", "Whittle",
                "PPO-AoI", "C-PPO", "Static-RR"]


def make_agent(name, env):
    """创建 agent (与 eval/run_all.py 工厂一致)."""
    if name == "SA-DAoI":
        return SADAOIScheduler(env)
    elif name == "Static-RR":
        return StaticRRPolicy(env)
    elif name == "T-AoI":
        return TAoIPolicy()
    elif name == "Whittle":
        return WhittlePolicy()
    elif name == "DQN-AoI":
        return DQNPolicy(env, MODELS_DIR, TIER)
    elif name == "PPO-AoI":
        return PPOPolicy(env, MODELS_DIR, TIER)
    elif name == "C-PPO":
        return CPPOPolicy(env, MODELS_DIR, TIER)
    else:
        raise ValueError(f"Unknown method: {name}")


def evaluate_mmpp(name, tier, num_episodes, seeds):
    """Evaluate one method under MMPP bursty arrivals."""
    all_aoi, all_viol, all_esq = [], [], []

    for seed in seeds:
        for ep in range(num_episodes):
            env = VehicularNetworkEnv(load_tier=tier)
            obs, _ = env.reset(seed=seed * 1000 + ep)
            agent = make_agent(name, env)

            # MMPP state
            burst_state = False
            original_p = env.param_p_arr

            done = False
            step_viols = []  # binary: 1 if slice-avg AoI > Gamma_S
            gamma_s = env.aoi_thresholds[SliceType.SAFETY]
            while not done:
                # MMPP state transition
                if burst_state:
                    if env.np_random.random() < Q_BURST_TO_NORMAL:
                        burst_state = False
                else:
                    if env.np_random.random() < Q_NORMAL_TO_BURST:
                        burst_state = True

                # Override arrival probability for this step
                if burst_state:
                    env.param_p_arr = min(original_p * BURST_FACTOR, 0.98)
                else:
                    env.param_p_arr = original_p

                action = agent.select_action(env)
                obs, reward, terminated, truncated, info = env.step(action)
                agent.on_step(info)

                # Violation per Definition 1: slice-average AoI > Gamma_S
                safety_vehs_step = [v for v in env.vehicles
                                    if v.slice_type == SliceType.SAFETY]
                avg_aoi_step = np.mean([v.aoi_comm for v in safety_vehs_step])
                step_viols.append(1.0 if avg_aoi_step > gamma_s else 0.0)

                done = terminated or truncated

            # Restore and collect metrics
            env.param_p_arr = original_p
            safety_vehs = [v for v in env.vehicles
                           if v.slice_type == SliceType.SAFETY]
            mean_aoi = np.mean([v.aoi_comm for v in safety_vehs])
            viol_rate = np.mean(step_viols) * 100  # percentage
            es_backlog = sum(v.queue for v in env.vehicles
                            if v.slice_type != SliceType.SAFETY)

            all_aoi.append(mean_aoi)
            all_viol.append(viol_rate)  # already in %
            all_esq.append(es_backlog)

    return {
        "aoi_mean": np.mean(all_aoi),  "aoi_std": np.std(all_aoi),
        "viol_mean": np.mean(all_viol), "viol_std": np.std(all_viol),
        "backlog_mean": np.mean(all_esq), "backlog_std": np.std(all_esq),
    }


def generate_md_report(results, md_path):
    """Generate markdown report for easy analysis."""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Bursty Traffic (MMPP) Experiment Report\n\n")
        f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Tier:** {TIER} | **Burst factor:** {BURST_FACTOR}x\n")
        f.write(f"**MMPP params:** q01={Q_NORMAL_TO_BURST}, q10={Q_BURST_TO_NORMAL} ")
        f.write(f"(mean burst duration: {1/Q_BURST_TO_NORMAL:.0f} TTIs, "
                f"~{Q_NORMAL_TO_BURST/(Q_NORMAL_TO_BURST+Q_BURST_TO_NORMAL)*100:.0f}% "
                f"time in burst)\n")
        f.write(f"**Evaluation:** {NUM_EPISODES} ep x {len(EVAL_SEEDS)} seeds\n\n")
        f.write("---\n\n")

        # Results table
        f.write("## Results (MMPP Bursty Traffic, Tier D)\n\n")
        f.write("| Method | Safety AoI (ms) | Viol. Rate (%) | ES Backlog (pkts) | SLA Met |\n")
        f.write("|--------|-----------------|----------------|-------------------|---------|\n")
        for name in METHOD_ORDER:
            if name in results:
                r = results[name]
                meets = "Yes" if (r["aoi_mean"] < 20 and r["viol_mean"] < 10) else "No"
                f.write(f"| {name} | {r['aoi_mean']:.2f} +/- {r['aoi_std']:.2f} "
                        f"| {r['viol_mean']:.1f} | {r['backlog_mean']:.1f} | {meets} |\n")
        f.write("\n---\n\n")

        # Comparison table
        f.write("## Comparison: Poisson (Table III) vs MMPP Bursty\n\n")
        f.write("| Method | Poisson AoI | MMPP AoI | Delta | Poisson Viol | MMPP Viol |\n")
        f.write("|--------|-------------|----------|-------|--------------|----------|\n")
        for name in METHOD_ORDER:
            if name in results and name in POISSON_REF:
                r = results[name]
                p = POISSON_REF[name]
                delta = ((r["aoi_mean"] / p["aoi"]) - 1) * 100
                f.write(f"| {name} | {p['aoi']:.2f} | {r['aoi_mean']:.2f} | "
                        f"{delta:+.1f}% | {p['viol']:.1f}% | {r['viol_mean']:.1f}% |\n")

    print(f"Markdown report saved to {md_path}")


def main():
    print("=" * 65)
    print("MMPP Bursty Traffic Experiment")
    print(f"Tier: {TIER} | Burst factor: {BURST_FACTOR}x")
    print(f"MMPP: q01={Q_NORMAL_TO_BURST}, q10={Q_BURST_TO_NORMAL}")
    print(f"Episodes: {NUM_EPISODES} x {len(EVAL_SEEDS)} seeds")
    print(f"Models dir: {MODELS_DIR}")
    print("=" * 65)

    results = {}
    for name in METHOD_ORDER:
        print(f"\n  Evaluating {name}...", end=" ", flush=True)
        try:
            r = evaluate_mmpp(name, TIER, NUM_EPISODES, EVAL_SEEDS)
            results[name] = r
            meets = "SLA MET" if (r["aoi_mean"] < 20 and r["viol_mean"] < 10) else "SLA FAIL"
            print(f"AoI={r['aoi_mean']:.2f}ms, Viol={r['viol_mean']:.1f}%, "
                  f"Backlog={r['backlog_mean']:.0f} [{meets}]")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()

    # Save CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "bursty_results.csv")
    with open(csv_path, "w") as f:
        f.write("method,aoi_mean,aoi_std,viol_mean,viol_std,backlog_mean,backlog_std\n")
        for name in METHOD_ORDER:
            if name in results:
                r = results[name]
                f.write(f"{name},{r['aoi_mean']:.4f},{r['aoi_std']:.4f},"
                        f"{r['viol_mean']:.4f},{r['viol_std']:.4f},"
                        f"{r['backlog_mean']:.4f},{r['backlog_std']:.4f}\n")
    print(f"\nCSV saved to {csv_path}")

    # Generate MD report
    md_path = os.path.join(RESULTS_DIR, "bursty_traffic_report.md")
    generate_md_report(results, md_path)

    # Console summary
    if "SA-DAoI" in results:
        sa = results["SA-DAoI"]
        ref_aoi = POISSON_REF["SA-DAoI"]["aoi"]
        delta = ((sa["aoi_mean"] / ref_aoi) - 1) * 100
        print(f"\n--- Summary ---")
        print(f"Under MMPP bursty arrivals ({BURST_FACTOR}x), SA-DAoI achieves")
        print(f"{sa['aoi_mean']:.2f}ms safety AoI ({delta:+.1f}% from Bernoulli)")
        print(f"with {sa['viol_mean']:.1f}% violation rate (SLA target: <10%).")


if __name__ == "__main__":
    main()
