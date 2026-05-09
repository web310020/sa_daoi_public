"""
参数敏感性分析 sweep (ISE gate / alpha_min / beta).

Usage: python -m eval.run_sensitivity
"""
import os
import sys
import csv
import numpy as np
import random
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs import SA_DAOI, EVAL
from env.vehicular import VehicularNetworkEnv, SliceType, SLICE_TYPES
from agents.sa_daoi import SADAOIScheduler
from eval.evaluator import evaluate


def make_modified_agent(env, overrides: dict):
    """Create SA-DAoI agent with specific parameter overrides."""
    agent = SADAOIScheduler(env)
    # Apply overrides after init
    for key, val in overrides.items():
        if key == "ise_gate_ratio":
            gamma_s = env.aoi_thresholds[SliceType.SAFETY]
            agent.eps_ise = val * gamma_s
        elif key == "alpha_floor":
            agent.alpha_floor = val
            agent.alpha = max(agent.alpha, val)  # ensure alpha >= floor
        elif key == "beta":
            agent.beta = val
    return agent


def sweep_parameter(param_name, values, load_tier="D", seed=42, num_episodes=None):
    """Run SA-DAoI with different values of one parameter."""
    num_episodes = num_episodes or EVAL["num_episodes"]
    results = []

    for val in values:
        print(f"  {param_name} = {val:.2f} ...", end=" ", flush=True)
        env = VehicularNetworkEnv(load_tier=load_tier)
        env.reset(seed=seed)
        agent = make_modified_agent(env, {param_name: val})
        res = evaluate(agent, load_tier, num_episodes=num_episodes, seed=seed)
        res[param_name] = val
        res["load_tier"] = load_tier
        es_q = res["queue_len_CE"] + res["queue_len_IOT"]
        print(f"AoI={res['aoi_total_SAFETY']:.2f}  "
              f"Viol={res['violation_rate_SAFETY']:.4f}  ES-Q={es_q:.1f}")
        results.append(res)

    return results


def run_all_sweeps(load_tier="D", results_dir="results", seed=42, num_episodes=None):
    os.makedirs(results_dir, exist_ok=True)

    sweeps = {
        "ise_gate_ratio": [0.2, 0.3, 0.4, 0.5, 0.6],
        "alpha_floor":    [0.70, 0.80, 0.85, 0.90, 0.95],
        "beta":           [0.05, 0.10, 0.15, 0.20],
    }

    all_results = []
    for param, values in sweeps.items():
        print(f"\n--- Sweeping {param} on Tier {load_tier} ---")
        results = sweep_parameter(param, values, load_tier=load_tier,
                                  seed=seed, num_episodes=num_episodes)
        all_results.extend(results)

    # Save to CSV
    fields = ["load_tier", "ise_gate_ratio", "alpha_floor", "beta",
              "aoi_total_SAFETY", "violation_rate_SAFETY",
              "queue_len_CE", "queue_len_IOT",
              "aoi_total_CE", "aoi_total_IOT"]
    csv_path = os.path.join(results_dir, f"sensitivity_{load_tier}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved: {csv_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Param':<18} {'Value':>6} | {'S-AoI':>8} {'Viol':>8} {'CE-Q':>8} {'IOT-Q':>8} {'ES-Q':>8}")
    print(f"-"*90)
    for r in all_results:
        p = next((k for k in sweeps if k in r and r[k] is not None), "?")
        es_q = r["queue_len_CE"] + r["queue_len_IOT"]
        print(f"{p:<18} {r.get(p, '?'):>6.2f} | "
              f"{r['aoi_total_SAFETY']:>8.2f} {r['violation_rate_SAFETY']:>8.4f} "
              f"{r['queue_len_CE']:>8.1f} {r['queue_len_IOT']:>8.1f} {es_q:>8.1f}")
    print(f"{'='*90}")

    return all_results


if __name__ == "__main__":
    run_all_sweeps(load_tier="D", results_dir="results", num_episodes=10)
