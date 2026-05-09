"""
DQN 30-seed training stability sweep (并行).

Usage: python -m eval.dqn_stability_30seed [--workers N] [--train-episodes N]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# 每 worker 锁单 CPU (在 torch/numpy import 前 set)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) or ".")


# 30 个固定 seed (random.Random(0).sample(range(1, 100000), 25) + 5 baseline)
TRAIN_SEEDS_30 = [
    42, 123, 256, 777, 999,
    1318, 5847, 9234, 2461, 7632,
    18043, 24571, 31896, 47213, 52768,
    61204, 68945, 73512, 81007, 89346,
    14728, 27093, 35624, 41892, 56237,
    63415, 71286, 84529, 92708, 99151,
]


# =============================================================================
# Wilson 95% CI (manual implementation; no statsmodels dependency)
# =============================================================================
def wilson_95ci(successes: int, n: int) -> tuple[float, float]:
    """Wilson 95% CI (Brown, Cai & DasGupta 2001). 返回 (lower, upper) in [0, 1]."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054  # 95% z-score
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# =============================================================================
# Worker 函数 (独立进程, 隔离 torch/numpy state)
# =============================================================================
def _train_eval_worker(args):
    """
    Worker: train DQN with given seed, return (seed, aoi_mean, aoi_std, viol_mean, sla_met).

    Each worker forces single-CPU torch (env vars already set above before any import).
    """
    seed, train_episodes, eval_episodes, eval_seeds = args

    # Defer imports until inside the worker so each gets a clean torch state.
    import random as _random
    import numpy as _np
    import torch as _torch
    from env.vehicular import VehicularNetworkEnv, SliceType
    from agents.dqn_agent import DQNAgent

    _torch.set_num_threads(1)

    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)

    env = VehicularNetworkEnv(load_tier="D")
    obs, _ = env.reset(seed=seed)
    agent = DQNAgent(obs.shape[0], env.n_actions)

    # Train
    for ep in range(train_episodes):
        state, _ = env.reset(seed=seed * 1000 + ep)
        done = False
        eps = max(0.01, 1.0 - (ep / train_episodes))
        while not done:
            action = agent.act(state, eps=eps)
            ns, r, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.buffer.push(state, action, r, ns, done)
            state = ns
            agent.update()

    # Evaluate (deterministic policy)
    all_aoi, all_viol = [], []
    for eval_seed in eval_seeds:
        for ep in range(eval_episodes):
            env2 = VehicularNetworkEnv(load_tier="D")
            obs, _ = env2.reset(seed=eval_seed * 1000 + ep)
            done = False
            step_viols = []
            while not done:
                action = agent.act(env2._get_obs(), eps=0.0)
                obs, _, term, trunc, info = env2.step(action)
                step_viols.append(info.get("viol_safety", 0.0))
                done = term or trunc

            safety_vehs = [v for v in env2.vehicles if v.slice_type == SliceType.SAFETY]
            all_aoi.append(_np.mean([v.aoi_comm for v in safety_vehs]))
            all_viol.append(_np.mean([1.0 if sv > 0 else 0.0 for sv in step_viols]))

    aoi_mean = float(_np.mean(all_aoi))
    aoi_std = float(_np.std(all_aoi))
    viol_mean = float(_np.mean(all_viol)) * 100  # percentage
    sla_met = bool(aoi_mean < 20.0 and viol_mean < 10.0)
    return (seed, aoi_mean, aoi_std, viol_mean, sla_met)


# =============================================================================
# Driver
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="DQN 30-seed stability sweep")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1 = sequential)")
    parser.add_argument("--n-seeds", type=int, default=30,
                        help="Number of training seeds to use (max 30; default 30)")
    parser.add_argument("--train-episodes", type=int, default=200,
                        help="Training episodes per seed (default 200)")
    parser.add_argument("--eval-episodes", type=int, default=30,
                        help="Evaluation episodes per eval-seed (default 30)")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[10, 20, 30],
                        help="Evaluation seeds (default 10 20 30)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/dqn_stability_30seed_<timestamp>/)")
    args = parser.parse_args()

    if args.n_seeds > 30:
        raise ValueError("n_seeds capped at 30 (extend TRAIN_SEEDS_30 list to go higher)")

    seeds = TRAIN_SEEDS_30[: args.n_seeds]
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/dqn_stability_30seed_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DQN 30-seed Training Stability Sweep (Tier D)")
    print("=" * 70)
    print(f"  seeds         : {len(seeds)} ({seeds[0]}, {seeds[1]}, ..., {seeds[-1]})")
    print(f"  train_episodes: {args.train_episodes}")
    print(f"  eval_episodes : {args.eval_episodes}")
    print(f"  eval_seeds    : {args.eval_seeds}")
    print(f"  workers       : {args.workers}")
    print(f"  output_dir    : {out_dir}")
    print("=" * 70)

    t0 = time.time()
    work = [(s, args.train_episodes, args.eval_episodes, args.eval_seeds) for s in seeds]

    if args.workers == 1:
        results = []
        for i, w in enumerate(work):
            print(f"\n[{i+1}/{len(work)}] seed={w[0]} ...", end=" ", flush=True)
            r = _train_eval_worker(w)
            results.append(r)
            elapsed = (time.time() - t0) / 60
            eta = elapsed / (i + 1) * (len(work) - i - 1)
            print(f"AoI={r[1]:.1f}±{r[2]:.1f}ms Viol={r[3]:.1f}% SLA={'✓' if r[4] else '✗'}  "
                  f"[{elapsed:.0f}min, ETA {eta:.0f}min]")
    else:
        # 并行 pool (spawn context 跨平台兼容)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_train_eval_worker, work)):
                results.append(r)
                elapsed = (time.time() - t0) / 60
                eta = elapsed / (i + 1) * (len(work) - i - 1)
                print(f"  [{i+1}/{len(work)}] seed={r[0]:6d}  "
                      f"AoI={r[1]:.1f}±{r[2]:.1f}ms  Viol={r[3]:.1f}%  "
                      f"SLA={'✓' if r[4] else '✗'}  "
                      f"[{elapsed:.0f}min elapsed, ETA {eta:.0f}min]")
        # Sort by seed for stable output ordering
        results.sort(key=lambda r: r[0])

    # ============================================================
    # Aggregate
    # ============================================================
    aois = [r[1] for r in results]
    viols = [r[3] for r in results]
    feasible = sum(1 for r in results if r[4])
    n = len(results)
    p_hat = feasible / n
    ci_lo, ci_hi = wilson_95ci(feasible, n)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"  Seeds             : {n}")
    print(f"  Mean AoI          : {np.mean(aois):.2f} +/- {np.std(aois):.2f} ms")
    print(f"  Mean Viol Rate    : {np.mean(viols):.2f} +/- {np.std(viols):.2f} %")
    print(f"  Feasible (SLA met): {feasible}/{n} = {p_hat*100:.1f}%")
    print(f"  Wilson 95% CI     : [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
    print("=" * 70)

    # ============================================================
    # Save outputs
    # ============================================================
    # CSV
    csv_path = out_dir / "seeds.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("seed,aoi_mean_ms,aoi_std_ms,viol_mean_pct,sla_met\n")
        for r in results:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{int(r[4])}\n")

    # JSON summary
    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_seeds": n,
            "train_episodes": args.train_episodes,
            "eval_episodes_per_seed": args.eval_episodes,
            "eval_seeds": args.eval_seeds,
            "feasible_count": feasible,
            "feasibility_p_hat": p_hat,
            "wilson_95ci_lo": ci_lo,
            "wilson_95ci_hi": ci_hi,
            "aoi_mean_ms": float(np.mean(aois)),
            "aoi_std_ms": float(np.std(aois)),
            "viol_mean_pct": float(np.mean(viols)),
            "viol_std_pct": float(np.std(viols)),
            "elapsed_minutes": (time.time() - t0) / 60,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Markdown summary
    md_path = out_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# DQN 30-seed Stability Sweep (Tier D)\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Seeds**: {n}\n")
        f.write(f"**Training**: {args.train_episodes} ep/seed\n")
        f.write(f"**Evaluation**: {args.eval_episodes} ep x {len(args.eval_seeds)} eval-seeds per training-seed\n\n")
        f.write(f"## Headline\n\n")
        f.write(f"- Feasibility: {feasible}/{n} = {p_hat*100:.1f}%\n")
        f.write(f"- Wilson 95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]\n\n")
        f.write(f"## Per-seed\n\n")
        f.write("| Seed | AoI (ms) | Viol (%) | SLA |\n|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r[0]} | {r[1]:.2f} +/- {r[2]:.2f} | {r[3]:.2f} | "
                    f"{'pass' if r[4] else 'fail'} |\n")

    print(f"\nOutputs:\n  CSV : {csv_path}\n  JSON: {json_path}\n  MD  : {md_path}\n")


if __name__ == "__main__":
    main()
