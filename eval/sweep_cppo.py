"""Tier-D C-PPO Lagrangian sweep；检查不同 ``lambda_lr`` 下的训练与约束表现。

Run with ``python -m eval.sweep_cppo``.
"""
import sys, os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) or ".")

from env.vehicular import VehicularNetworkEnv, SliceType
from agents.ppo_cppo import LagrangianWrapper

try:
    from stable_baselines3 import PPO as SB3_PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("ERROR: stable-baselines3 not installed")
    sys.exit(1)


TIER = "D"
SWEEP_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
TRAIN_TIMESTEPS = 800_000   # same as original C-PPO training
EVAL_EPISODES = 10
EVAL_SEEDS = [10, 20, 30]


def train_and_eval(lambda_lr):
    """Train C-PPO with given lambda_lr, then evaluate on Tier D."""
    print(f"\n  [Train] lambda_lr={lambda_lr}, {TRAIN_TIMESTEPS} steps...", end=" ", flush=True)

    # Train
    base_env = VehicularNetworkEnv(load_tier=TIER)
    wrapped = LagrangianWrapper(base_env, lambda_lr=lambda_lr)
    model = SB3_PPO("MlpPolicy", wrapped, verbose=0,
                     learning_rate=3e-4, gamma=0.99, device="cpu")
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    final_lambda = wrapped.lam
    print(f"done. Final λ={final_lambda:.4f}")

    # Evaluate
    all_aoi, all_viol, all_esq = [], [], []
    for seed in EVAL_SEEDS:
        for ep in range(EVAL_EPISODES):
            env = VehicularNetworkEnv(load_tier=TIER)
            obs, _ = env.reset(seed=seed * 1000 + ep)

            done = False
            step_viols = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                step_viols.append(info.get("viol_safety", 0.0))
                done = terminated or truncated

            safety_vehs = [v for v in env.vehicles if v.slice_type == SliceType.SAFETY]
            mean_aoi = np.mean([v.aoi_comm for v in safety_vehs])
            viol_rate = np.mean([1.0 if sv > 0 else 0.0 for sv in step_viols])
            es_backlog = sum(v.queue for v in env.vehicles if v.slice_type != SliceType.SAFETY)

            all_aoi.append(mean_aoi)
            all_viol.append(viol_rate)
            all_esq.append(es_backlog)

    return {
        "lambda_lr": lambda_lr,
        "final_lam": final_lambda,
        "aoi_m": np.mean(all_aoi), "aoi_s": np.std(all_aoi),
        "viol_m": np.mean(all_viol), "viol_s": np.std(all_viol),
        "esq_m": np.mean(all_esq), "esq_s": np.std(all_esq),
    }


def main():
    print("=" * 70)
    print("C-PPO Lagrangian Hyperparameter Sweep (Tier D)")
    print(f"Sweep: lambda_lr = {SWEEP_VALUES}")
    print(f"Training: {TRAIN_TIMESTEPS} timesteps per config")
    print(f"Evaluation: {len(EVAL_SEEDS)} seeds × {EVAL_EPISODES} episodes")
    print("=" * 70)

    results = []
    for lr in SWEEP_VALUES:
        try:
            r = train_and_eval(lr)
            results.append(r)
            print(f"  [Result] λ_lr={lr}: AoI={r['aoi_m']:.1f}±{r['aoi_s']:.1f}, "
                  f"Viol={r['viol_m']:.3f}±{r['viol_s']:.3f}, "
                  f"ESQ={r['esq_m']:.0f}, Final λ={r['final_lam']:.3f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: C-PPO Sweep Results (Tier D)")
    print("=" * 70)
    print(f"{'λ_lr':>8s}  {'Final λ':>8s}  {'AoI (ms)':>10s}  {'Viol Rate':>10s}  {'ESQ':>8s}  {'< 20ms?':>8s}  {'< 10% viol?':>12s}")
    print("-" * 80)
    for r in results:
        meets_aoi = "✓" if r["aoi_m"] < 20 else "✗"
        meets_viol = "✓" if r["viol_m"] < 0.10 else "✗"
        print(f"{r['lambda_lr']:8.3f}  {r['final_lam']:8.3f}  "
              f"{r['aoi_m']:7.1f}±{r['aoi_s']:<4.1f}  "
              f"{r['viol_m']:7.3f}±{r['viol_s']:<5.3f}  "
              f"{r['esq_m']:8.0f}  {meets_aoi:>8s}  {meets_viol:>12s}")

    # SA-DAoI reference
    print("-" * 80)
    print(f"{'SA-DAoI':>8s}  {'N/A':>8s}  {'14.2±0.1':>10s}  {'0.085±0.002':>10s}  {'475':>8s}  {'✓':>8s}  {'✓':>12s}")

    # Check if ANY config meets both constraints
    any_feasible = any(r["aoi_m"] < 20 and r["viol_m"] < 0.10 for r in results)
    print(f"\nAny feasible C-PPO config found: {'YES' if any_feasible else 'NO'}")

    # LaTeX footnote text
    print("\n% ═══ Paper footnote text ═══")
    if not any_feasible:
        lrs = ", ".join(str(v) for v in SWEEP_VALUES)
        print(f"% We swept the Lagrangian step size η_λ ∈ {{{lrs}}}; ")
        print(f"% none of the {len(SWEEP_VALUES)} configurations achieved a feasible")
        print(f"% solution (Φ < 10% and AoI < 20ms) under Tier D conditions.")
    print("\nDone!")


if __name__ == "__main__":
    main()
