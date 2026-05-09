"""
Tier D 主结果 (5-seed n=150 per method) + 多 cell preliminary.

Usage: python -m eval.polish_5seed_and_multicell
"""
import os
import sys
import csv
import time
import datetime
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from env.vehicular import VehicularNetworkEnv
from agents.sa_daoi import SADAOIScheduler
from agents.heuristics import StaticRRPolicy, TAoIPolicy, WhittlePolicy
from agents.dqn_agent import DQNPolicy
from agents.ppo_cppo import PPOPolicy, CPPOPolicy
from eval.evaluator import evaluate
from eval.multicell_preliminary import HalfTierDEnv, aggregate, PER_CELL_VEHICLES, N_SAFETY, N_CE, N_IOT


SEEDS_5  = [10, 20, 30, 40, 50]
NUM_EP   = 30
TIER     = "D"
MODELS   = os.path.join(PROJ, "results", "v058_20260409_130414", "models")

# 多 cell network-aware Theta = total |V| / 2 = 105
THETA_NETWORK_AWARE = 105.0

_ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = os.path.join(PROJ, "results", f"v058_polish_5seed_{_ts}")


def banner(s: str):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


# 7 个 method, 5-seed 平均
def make_method(name: str, env):
    """根据 name 构建 method instance, 必要时加载预训练 DRL 模型."""
    if name == "SA-DAoI":
        return SADAOIScheduler(env)
    if name == "Static-RR":
        return StaticRRPolicy(env)
    if name == "T-AoI":
        return TAoIPolicy()
    if name == "Whittle":
        return WhittlePolicy()
    if name == "DQN-AoI":
        return DQNPolicy(env, models_dir=MODELS, load_tier=TIER)
    if name == "PPO-AoI":
        return PPOPolicy(env, models_dir=MODELS, load_tier=TIER)
    if name == "C-PPO":
        return CPPOPolicy(env, models_dir=MODELS, load_tier=TIER)
    raise ValueError(f"Unknown method: {name}")


def run_5seed_parity(out_csv: str):
    """7 个 method, 5 seeds × 30 ep, Tier D."""
    banner("5-seed parity for 7 methods on Tier D (n=150 per method)")
    methods = ["SA-DAoI", "Static-RR", "T-AoI", "Whittle",
               "DQN-AoI", "PPO-AoI", "C-PPO"]
    print(f"  Methods: {methods}")
    print(f"  Seeds:   {SEEDS_5}")
    print(f"  Episodes/seed: {NUM_EP}  =>  n={len(SEEDS_5)*NUM_EP} per method")
    print()

    rows = []
    t0 = time.time()
    for m in methods:
        per_seed_aoi = []
        per_seed_viol = []
        per_seed_es = []
        for seed in SEEDS_5:
            env = VehicularNetworkEnv(load_tier=TIER)
            env.reset(seed=seed)
            agent = make_method(m, env)
            res = evaluate(agent, TIER, num_episodes=NUM_EP, seed=seed)

            aoi  = res["aoi_total_SAFETY"]
            viol = res["violation_rate_SAFETY"]
            es   = res["queue_len_CE"] + res["queue_len_IOT"]
            per_seed_aoi.append(aoi)
            per_seed_viol.append(viol)
            per_seed_es.append(es)

            rows.append({
                "method": m,
                "seed":   seed,
                "aoi_safety_ms":   aoi,
                "viol_rate":       viol,
                "es_backlog":      es,
                "queue_ce":        res["queue_len_CE"],
                "queue_iot":       res["queue_len_IOT"],
            })
            print(f"  {m:11s} seed={seed:3d}  AoI={aoi:6.2f}  Viol={100*viol:5.2f}%  ES={es:7.1f}")

        n = len(SEEDS_5)
        m_aoi  = statistics.mean(per_seed_aoi)
        s_aoi  = statistics.stdev(per_seed_aoi) if n >= 2 else 0.0
        m_viol = statistics.mean(per_seed_viol)
        s_viol = statistics.stdev(per_seed_viol) if n >= 2 else 0.0
        m_es   = statistics.mean(per_seed_es)
        s_es   = statistics.stdev(per_seed_es) if n >= 2 else 0.0
        print(f"  {m:11s} 5-seed:  AoI={m_aoi:.2f}±{s_aoi:.2f}  "
              f"Viol={100*m_viol:.2f}±{100*s_viol:.2f}%  "
              f"ES={m_es:.1f}±{s_es:.1f}")
        print()

    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed/60:.1f} min")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out_csv}")
    return rows


# 多 cell preliminary (network-aware Theta override)
def run_one_cell_theta_override(seed: int, theta_value: float) -> dict:
    """单 105-vehicle cell, 手工 override agent.Theta."""
    env = HalfTierDEnv()
    env.reset(seed=seed)
    agent = SADAOIScheduler(env)
    agent.Theta = theta_value
    return evaluate(agent, TIER, num_episodes=NUM_EP, seed=seed)


def run_multicell_eval(out_csv: str, seeds=None):
    """多 cell preliminary, network-aware Theta=105."""
    banner("Multi-cell preliminary with network-aware Theta=105")
    if seeds is None:
        seeds = [10, 20, 30]
    print(f"  Per cell: {PER_CELL_VEHICLES} vehicles "
          f"({N_SAFETY}S/{N_CE}C/{N_IOT}I), 10 PRBs, Gamma_S=20ms, p=0.93")
    print(f"  Theta override: {THETA_NETWORK_AWARE}")
    print(f"  Seeds: {seeds}, episodes/seed: {NUM_EP}")
    print()

    rows = []
    t0 = time.time()
    for seed in seeds:
        c1 = run_one_cell_theta_override(seed,        THETA_NETWORK_AWARE)
        c2 = run_one_cell_theta_override(seed + 1000, THETA_NETWORK_AWARE)
        agg = aggregate(c1, c2)
        print(f"  seed={seed:3d}  "
              f"AoI_c1={c1['aoi_total_SAFETY']:5.2f}/"
              f"c2={c2['aoi_total_SAFETY']:5.2f} "
              f"-> agg={agg['aoi_total_SAFETY']:5.2f} ms  "
              f"Viol_agg={100*agg['violation_rate_SAFETY']:5.2f}%  "
              f"ES_sum={agg['queue_len_ES']:7.1f}")
        rows.append({
            "seed":                    seed,
            "theta_override":          THETA_NETWORK_AWARE,
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
    print(f"Elapsed: {elapsed:.1f} s")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out_csv}")
    return rows


def write_summary(seed_rows, multicell_rows, summary_path):
    """汇总写 summary.txt."""
    methods_order = ["SA-DAoI", "Static-RR", "T-AoI", "Whittle",
                     "DQN-AoI", "PPO-AoI", "C-PPO"]
    by_method = {}
    for r in seed_rows:
        by_method.setdefault(r["method"], []).append(r)

    lines = []
    lines.append("=" * 78)
    lines.append("SA-DAoI: Tier D 5-seed + Multi-cell Results")
    lines.append("=" * 78)
    lines.append("")
    lines.append("5-seed parity (Tier D, 30 ep x 5 seeds = n=150 per method)")
    lines.append("-" * 78)
    lines.append(f"{'Method':<12}{'AoI mean':>12}{'AoI std':>10}"
                 f"{'Viol mean':>12}{'Viol std':>10}{'ES mean':>10}{'ES std':>10}")
    for m in methods_order:
        if m not in by_method:
            continue
        rs = by_method[m]
        aois = [r["aoi_safety_ms"] for r in rs]
        vis  = [r["viol_rate"]     for r in rs]
        ess  = [r["es_backlog"]    for r in rs]
        n = len(aois)
        if n < 2:
            continue
        lines.append(
            f"{m:<12}"
            f"{statistics.mean(aois):>12.3f}"
            f"{statistics.stdev(aois):>10.3f}"
            f"{100*statistics.mean(vis):>11.2f}%"
            f"{100*statistics.stdev(vis):>9.2f}%"
            f"{statistics.mean(ess):>10.1f}"
            f"{statistics.stdev(ess):>10.1f}"
        )
    lines.append("")
    lines.append("=" * 78)
    lines.append("Multi-cell with network-aware Theta=105")
    lines.append("=" * 78)
    if multicell_rows:
        n = len(multicell_rows)
        m_aoi = statistics.mean(r["aoi_network_avg"]        for r in multicell_rows)
        m_viol = statistics.mean(r["viol_network_avg"]      for r in multicell_rows)
        m_es = statistics.mean(r["es_backlog_network_sum"]  for r in multicell_rows)
        sla = (m_aoi < 20) and (m_viol < 0.10)
        lines.append(f"  Network-avg Safety AoI:  {m_aoi:.3f} ms")
        lines.append(f"  Network-avg Viol rate:   {100*m_viol:.3f} %")
        lines.append(f"  Network total ES backlog:{m_es:.1f} pkts")
        lines.append(f"  SLA (AoI<20 ms & Viol<10%): {'MET' if sla else 'FAIL'}")
    lines.append("")
    lines.append("=" * 78)
    out = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(out)
    print()
    print(out)
    print()
    print(f"Summary saved: {summary_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_csv      = os.path.join(OUT_DIR, "metrics_D_5seed.csv")
    multicell_csv = os.path.join(OUT_DIR, "multicell_network_aware.csv")
    summary       = os.path.join(OUT_DIR, "summary.txt")

    print(f"Output dir: {OUT_DIR}")
    print()

    print("Step 1/2: 5-seed parity ...")
    seed_rows = run_5seed_parity(seed_csv)

    print()
    print("Step 2/2: multi-cell network-aware Theta ...")
    multicell_rows = run_multicell_eval(multicell_csv, seeds=[10, 20, 30])

    print()
    write_summary(seed_rows, multicell_rows, summary)


if __name__ == "__main__":
    main()
