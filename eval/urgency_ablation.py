"""
Urgency-mapping ablation：在 Tier D 上扫描 exponent ``p`` ∈ {1, 2, 3, 4}。

Run with ``python -m eval.urgency_ablation``.
"""
import os
import sys
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.vehicular import VehicularNetworkEnv
from agents.sa_daoi import SADAOIScheduler
from eval.evaluator import evaluate
from configs import EVAL

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV  = os.path.join(REPO_DIR, "results", "urgency_ablation",
                        "urgency_ablation_D.csv")

TIER     = "D"
EXPONENTS = [1, 2, 3, 4]
SEEDS    = EVAL["seeds"]          # [10, 20, 30], 与主 Tier D 一致
NUM_EP   = EVAL["num_episodes"]   # 30


def main():
    print(f"Urgency-mapping ablation on Tier {TIER}")
    print(f"  exponents = {EXPONENTS}, seeds = {SEEDS}, ep/seed = {NUM_EP}")
    print()

    rows = []
    t0 = time.time()
    for p in EXPONENTS:
        for seed in SEEDS:
            env = VehicularNetworkEnv(load_tier=TIER)
            env.reset(seed=seed)               # creates env.vehicles
            agent = SADAOIScheduler(env, urgency_exponent=p)
            r = evaluate(agent, TIER, num_episodes=NUM_EP, seed=seed)

            q_es = r["queue_len_CE"] + r["queue_len_IOT"]
            rows.append({
                "urgency_exponent":     p,
                "seed":                 seed,
                "aoi_total_SAFETY":     r["aoi_total_SAFETY"],
                "violation_rate_SAFETY":r["violation_rate_SAFETY"],
                "queue_len_CE":         r["queue_len_CE"],
                "queue_len_IOT":        r["queue_len_IOT"],
                "queue_len_ES":         q_es,
                "starvation_rate_SAFETY": r.get("starvation_rate_SAFETY", 0.0),
            })
            print(f"  p={p} seed={seed:3d}  "
                  f"AoI={r['aoi_total_SAFETY']:6.2f} ms  "
                  f"Viol={100*r['violation_rate_SAFETY']:5.2f}%  "
                  f"ES_backlog={q_es:7.1f}")
        print()

    elapsed = time.time() - t0
    print(f"Total runtime: {elapsed:.1f} s")

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {OUT_CSV}")

    # Per-exponent summary
    print("\n=== Per-exponent summary (mean across seeds) ===")
    for p in EXPONENTS:
        subset = [r for r in rows if r["urgency_exponent"] == p]
        n = len(subset)
        aoi_mean  = sum(r["aoi_total_SAFETY"]     for r in subset) / n
        viol_mean = sum(r["violation_rate_SAFETY"] for r in subset) / n
        es_mean   = sum(r["queue_len_ES"]          for r in subset) / n
        meets_sla = (aoi_mean < 20) and (viol_mean < 0.10)
        print(f"  p={p}: AoI={aoi_mean:6.2f} ms, "
              f"Viol={100*viol_mean:5.2f}%, "
              f"ES={es_mean:7.1f} pkts, "
              f"SLA {'MET' if meets_sla else 'FAIL'}")


if __name__ == "__main__":
    main()
