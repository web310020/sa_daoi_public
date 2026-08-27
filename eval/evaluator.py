"""
统一的 evaluation loop；各方法共享相同的 episode 与 metrics aggregation。
Every method only needs to provide:
  - select_action(env) → int
  - on_step(info)  [optional]
  - on_reset(env)  [optional]
"""
import time
import numpy as np
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.vehicular import VehicularNetworkEnv, SliceType, SLICE_TYPES
from configs import EVAL


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def evaluate(agent, load_tier, num_episodes=None, seed=42):
    """
    Run `num_episodes` episodes, return aggregated metrics dict.

    Parameters
    ----------
    agent : object with select_action(env), on_step(info), on_reset(env)
    load_tier : str in {"A","B","C","D"}
    num_episodes : int
    seed : int

    Returns
    -------
    dict  with keys matching configs.CSV_FIELDS
    """
    env = VehicularNetworkEnv(load_tier=load_tier)
    return evaluate_env(agent, env, num_episodes=num_episodes, seed=seed)


def evaluate_env(agent, env, num_episodes=None, seed=42):
    """Evaluate an agent in the exact caller-supplied environment object.

    该入口用于需要固定环境身份的实验（例如 half-cell）。它不会根据 load tier
    重新创建或替换环境，因此结果中的 census 与实际执行对象保持一致。
    """

    set_seed(seed)
    num_episodes = num_episodes or EVAL["num_episodes"]
    environment_object_id = id(env)

    rewards  = []
    history  = {"aoi": [], "vio_rate": [], "stv": [], "pend": [], "ql": []}
    latencies = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)

        if hasattr(agent, "on_reset"):
            agent.on_reset(env)

        done   = False
        ep_r   = 0.0
        v_sum  = 0.0
        s_count = 0

        while not done:
            t0 = time.perf_counter()
            action = agent.select_action(env)
            latencies.append((time.perf_counter() - t0) * 1000)

            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_r += r
            v_sum  += info.get("viol_safety", 0)
            s_count += 1

            if hasattr(agent, "on_step"):
                agent.on_step(info)

        rewards.append(ep_r)
        history["aoi"].append(env.get_slice_aoi_stats())
        history["vio_rate"].append(v_sum / max(s_count, 1))
        history["stv"].append(env.get_slice_starvation_rate())
        history["pend"].append(env.get_slice_pending_ratio())
        history["ql"].append(env.get_slice_avg_queue_len())

    # Aggregate
    res = {"avg_reward": float(np.mean(rewards))}
    for s in SLICE_TYPES:
        n = s.name.upper()
        res[f"aoi_total_{n}"]       = float(np.mean([h[s] for h in history["aoi"]]))
        res[f"violation_rate_{n}"]   = (float(np.mean(history["vio_rate"]))
                                        if s == SliceType.SAFETY else 0.0)
        res[f"starvation_rate_{n}"]  = float(np.mean([h[s] for h in history["stv"]]))
        res[f"pending_ratio_{n}"]    = float(np.mean([h[s] for h in history["pend"]]))
        res[f"queue_len_{n}"]        = float(np.mean([h[s] for h in history["ql"]]))

    if latencies:
        res["latency_avg_ms"] = float(np.mean(latencies))
        res["latency_p99_ms"] = float(np.percentile(latencies, 99))

    census = {
        slice_type.name.upper(): sum(v.slice_type == slice_type for v in env.vehicles)
        for slice_type in SLICE_TYPES
    }
    res["environment_object_id"] = environment_object_id
    res["vehicle_count"] = sum(census.values())
    res["slice_census"] = census

    return res


# ═══ Ablation-specific data collector ═══════════════════════════════

def collect_timeseries(agent, load_tier="D", max_steps=800, seed=42):
    """
    Run a single episode, return per-step timeseries for ablation plots.
    """
    set_seed(seed)
    env = VehicularNetworkEnv(load_tier=load_tier, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)

    if hasattr(agent, "on_reset"):
        agent.on_reset(env)

    data = {
        "aoi_ts": [], "queue_ts": [],
        "aoi_samples": [], "queue_samples": [],
        "alpha_ts": [],
    }

    done = False
    while not done:
        action = agent.select_action(env)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc

        if hasattr(agent, "on_step"):
            agent.on_step(info)

        # Record per-step stats
        s_aois   = [v.aoi_comm for v in env.vehicles if v.slice_type == SliceType.SAFETY]
        ce_q     = [v.queue for v in env.vehicles if v.slice_type == SliceType.CE]
        iot_q    = [v.queue for v in env.vehicles if v.slice_type == SliceType.IOT]

        avg_aoi = float(np.mean(s_aois))
        avg_esq = float(np.mean(ce_q) + np.mean(iot_q))

        data["aoi_ts"].append(avg_aoi)
        data["queue_ts"].append(avg_esq)
        data["aoi_samples"].extend(s_aois)
        data["queue_samples"].extend(ce_q + iot_q)

        if hasattr(agent, "alpha"):
            data["alpha_ts"].append(agent.alpha)

    return data
