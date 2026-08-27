"""
Tier-D 多 cell preliminary: 210 辆车分到 2 个 cell (各 105 辆), 各跑独立 SA-DAoI controller.

Usage: python -m eval.multicell_preliminary
"""
import os
import sys
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
from env.vehicular import VehicularNetworkEnv, SliceType, Vehicle
from agents.sa_daoi import SADAOIScheduler
from eval.evaluator import evaluate
from configs import EVAL

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV  = os.path.join(REPO_DIR, "results", "multicell_preliminary",
                        "multicell_preliminary_D.csv")

PER_CELL_VEHICLES = 105   # 2 cells x 105 = 210 (Tier-D total)
N_SAFETY = 80             # same ratio as Tier-D (160/210 = 0.762)
N_CE     = 20             # (40/210 = 0.190)
N_IOT    = 5              # (10/210 = 0.048)  -> sums to 105


class HalfTierDEnv(VehicularNetworkEnv):
    """单 cell, 105 辆车 (Tier-D 一半), safety-heavy 80/20/5 split."""

    def __init__(self, max_steps=None):
        super().__init__(load_tier="D", max_steps=max_steps)
        self.param_n_vehicles = PER_CELL_VEHICLES

    def reset(self, seed=None, options=None):
        # 跳过父类 reset 的车辆创建 (它用 160/40/10), 直接按 80/20/5 自己 build
        gym.Env.reset(self, seed=seed)
        self.current_step = 0
        self.vehicles = [
            Vehicle(i, SliceType.SAFETY if i < N_SAFETY
                    else (SliceType.CE if i < N_SAFETY + N_CE else SliceType.IOT))
            for i in range(PER_CELL_VEHICLES)
        ]
        return self._get_obs(), {}


def run_cell(seed: int) -> dict:
    """Run SA-DAoI on one 105-vehicle cell for num_episodes x T."""
    env = HalfTierDEnv()
    env.reset(seed=seed)              # creates env.vehicles before agent init
    agent = SADAOIScheduler(env)
    return evaluate(agent, "D", num_episodes=EVAL["num_episodes"], seed=seed)


def aggregate(cell1: dict, cell2: dict) -> dict:
    """Combine metrics from two cells serving 105 vehicles each."""
    return {
        # AoI: mean across cells (network-wide average safety AoI)
        "aoi_total_SAFETY":     (cell1["aoi_total_SAFETY"] + cell2["aoi_total_SAFETY"]) / 2,
        # Violation rate: mean across cells
        "violation_rate_SAFETY":(cell1["violation_rate_SAFETY"] + cell2["violation_rate_SAFETY"]) / 2,
        # ES backlog: SUM (total elastic backlog in the network)
        "queue_len_CE":         cell1["queue_len_CE"] + cell2["queue_len_CE"],
        "queue_len_IOT":        cell1["queue_len_IOT"] + cell2["queue_len_IOT"],
        "queue_len_ES":         (cell1["queue_len_CE"] + cell1["queue_len_IOT"]
                                 + cell2["queue_len_CE"] + cell2["queue_len_IOT"]),
    }


def main():
    print("Preliminary multi-cell experiment (Tier-D split across 2 cells)")
    print(f"  per cell: {PER_CELL_VEHICLES} vehicles "
          f"({N_SAFETY} Safety / {N_CE} CE / {N_IOT} IoT), "
          f"10 PRBs, Gamma_S=20ms, p_arr=0.93")
    print(f"  seeds: {EVAL['seeds']}, episodes/seed: {EVAL['num_episodes']}")
    print()

    rows = []
    t0 = time.time()
    for seed in EVAL["seeds"]:
        # Independent seeds per cell for independence (seed, seed+1000)
        c1 = run_cell(seed)
        c2 = run_cell(seed + 1000)
        agg = aggregate(c1, c2)

        print(f"  seed={seed:3d}  "
              f"AoI_c1={c1['aoi_total_SAFETY']:5.2f}/"
              f"c2={c2['aoi_total_SAFETY']:5.2f} "
              f"-> agg={agg['aoi_total_SAFETY']:5.2f} ms  "
              f"Viol_agg={100*agg['violation_rate_SAFETY']:5.2f}%  "
              f"ES_sum={agg['queue_len_ES']:7.1f}")

        rows.append({
            "seed":                    seed,
            "aoi_cell1":               c1["aoi_total_SAFETY"],
            "aoi_cell2":               c2["aoi_total_SAFETY"],
            "aoi_network_avg":         agg["aoi_total_SAFETY"],
            "viol_cell1":              c1["violation_rate_SAFETY"],
            "viol_cell2":              c2["violation_rate_SAFETY"],
            "viol_network_avg":        agg["violation_rate_SAFETY"],
            "es_backlog_cell1":        c1["queue_len_CE"] + c1["queue_len_IOT"],
            "es_backlog_cell2":        c2["queue_len_CE"] + c2["queue_len_IOT"],
            "es_backlog_network_sum":  agg["queue_len_ES"],
        })

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f} s")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {OUT_CSV}")

    # Summary: mean across seeds
    n = len(rows)
    mean_aoi  = sum(r["aoi_network_avg"]        for r in rows) / n
    mean_viol = sum(r["viol_network_avg"]       for r in rows) / n
    mean_es   = sum(r["es_backlog_network_sum"] for r in rows) / n
    sla_met   = (mean_aoi < 20) and (mean_viol < 0.10)
    print("\n=== 2-cell Summary (Tier-D split) ===")
    print(f"  Network-avg Safety AoI:  {mean_aoi:.2f} ms")
    print(f"  Network-avg Viol rate:   {100*mean_viol:.2f}%")
    print(f"  Network total ES backlog:{mean_es:.1f} pkts")
    print(f"  SLA (AoI<20 & Viol<10%): {'MET' if sla_met else 'FAIL'}")


if __name__ == "__main__":
    main()
