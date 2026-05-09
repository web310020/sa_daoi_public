"""
V2X 切片环境 (Gymnasium-compatible).
3 切片: Safety / CE / IoT.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
from dataclasses import dataclass, field
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs import TIER_PARAMS


class SliceType(Enum):
    SAFETY = 0
    CE     = 1
    IOT    = 2

SLICE_TYPES = list(SliceType)


@dataclass
class Vehicle:
    vid: int
    slice_type: SliceType
    aoi_comm: float = 1.0
    queue: int = 0


class VehicularNetworkEnv(gym.Env):

    def __init__(self, load_tier: str, max_steps: int | None = None):
        super().__init__()
        if load_tier not in TIER_PARAMS:
            raise ValueError(f"Unknown tier: {load_tier}")

        self.load_tier = load_tier
        p = TIER_PARAMS[load_tier]

        self.param_n_bs       = p["n_bs"]
        self.param_n_rbs      = p["n_rbs"]
        self.param_n_vehicles = p["n_vehicles"]
        self.param_p_arr      = p["p_arr"]
        self.packets_per_rb   = p["packets_per_rb"]
        self.max_steps        = max_steps or p["max_steps"]

        self.aoi_thresholds = {
            SliceType.SAFETY: p["aoi_thresholds"]["S"],
            SliceType.CE:     p["aoi_thresholds"]["C"],
            SliceType.IOT:    p["aoi_thresholds"]["I"],
        }

        self.total_rbs = self.param_n_bs * self.param_n_rbs
        self.action_table = self._generate_action_table()
        self.n_actions = len(self.action_table)

        self.action_space      = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(6,), dtype=np.float32)

        self.TTI_MS       = 1
        self.RIC_LOOP_MS  = 10
        self.theta_q      = 10   # starvation threshold Θ (queue length)

    # ── Action table ────────────────────────────────────────────────
    def _generate_action_table(self):
        actions = []
        for s in range(self.total_rbs + 1):
            for c in range(self.total_rbs - s + 1):
                actions.append((s, c, self.total_rbs - s - c))
        return actions

    # ── Reset ───────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if self.load_tier == "D":
            # Safety-heavy distribution: 160 / 40 / 10
            self.vehicles = [
                Vehicle(i, SliceType.SAFETY if i < 160
                        else (SliceType.CE if i < 200 else SliceType.IOT))
                for i in range(self.param_n_vehicles)
            ]
        else:
            n = self.param_n_vehicles
            self.vehicles = [
                Vehicle(i, SliceType.SAFETY if i < n // 3
                        else (SliceType.CE if i < 2 * n // 3 else SliceType.IOT))
                for i in range(n)
            ]
        return self._get_obs(), {}

    # ── Observation ─────────────────────────────────────────────────
    def _get_obs(self):
        aoi = self.get_slice_aoi_stats()
        ql  = self.get_slice_avg_queue_len()
        return np.array(
            [aoi[s] for s in SLICE_TYPES] + [ql[s] for s in SLICE_TYPES],
            dtype=np.float32,
        )

    # ── Step ────────────────────────────────────────────────────────
    def step(self, action_idx):
        alloc = self.action_table[action_idx]
        slice_caps = [alloc[i] * self.packets_per_rb for i in range(3)]

        # 1. Traffic arrival
        t = self.current_step
        base_p = self.param_p_arr
        if self.load_tier == "D":
            seasonal = 0.1 * np.sin(2 * np.pi * t / 100)
            burst = 0.25 if (300 <= t < 350) else 0.0
            current_p = np.clip(base_p + seasonal + burst, 0.4, 0.98)
        else:
            current_p = base_p

        for v in self.vehicles:
            if self.np_random.random() < current_p:
                v.queue += 1

        # 2. Service & AoI update (per-slice fair scheduling)
        actual_usage = [0.0, 0.0, 0.0]
        for s_type in SLICE_TYPES:
            idx = s_type.value
            v_in_slice = [v for v in self.vehicles if v.slice_type == s_type]
            self.np_random.shuffle(v_in_slice)

            cap0 = slice_caps[idx]
            cap = cap0
            served = 0
            for v in v_in_slice:
                if cap > 0 and v.queue > 0:
                    s = min(v.queue, cap)
                    v.queue -= s
                    cap -= s
                    served += s
                    v.aoi_comm = 1.0
                else:
                    v.aoi_comm += 1.0
            actual_usage[idx] = (served / cap0) if cap0 > 0 else 0.0

        # 3. Compute violation, starvation & reward (paper Eq. 5)
        safety_vehs = [v for v in self.vehicles if v.slice_type == SliceType.SAFETY]
        viol_s = (np.mean([float(v.aoi_comm > self.aoi_thresholds[SliceType.SAFETY])
                           for v in safety_vehs]) if safety_vehs else 0.0)
        V_S = 1.0 if viol_s > 0 else 0.0

        # Elastic urgency terms (Ψ_k for CE/IoT)
        ql = self.get_slice_avg_queue_len()
        Q_max = 500.0
        psi_ce  = np.exp(ql[SliceType.CE]  / Q_max)
        psi_iot = np.exp(ql[SliceType.IOT] / Q_max)

        # Starvation indicator
        es_q = sum(v.queue for v in self.vehicles if v.slice_type != SliceType.SAFETY)
        S_starve = 1.0 if es_q > (self.param_n_vehicles * 0.5) else 0.0

        # Paper Eq.(5): r_t = -λ₁V_S - λ₂ΣΨ_k - λ₃S_starve
        reward = -100.0 * V_S - 1.0 * (psi_ce + psi_iot) - 5.0 * S_starve

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        # ES backlog for metrics collection
        es_backlog = sum(v.queue for v in self.vehicles if v.slice_type != SliceType.SAFETY)

        info = {
            "viol_safety":   float(viol_s),
            "usage_safety":  actual_usage[0],
            "current_p":     current_p,
            "es_backlog":    float(es_backlog),
        }
        return self._get_obs(), reward, terminated, False, info

    # ── Helpers ─────────────────────────────────────────────────────
    def get_slice_aoi_stats(self):
        return {s: np.mean([v.aoi_comm for v in self.vehicles if v.slice_type == s])
                for s in SLICE_TYPES}

    def get_slice_avg_queue_len(self):
        return {s: np.mean([v.queue for v in self.vehicles if v.slice_type == s])
                for s in SLICE_TYPES}

    def get_slice_starvation_rate(self):
        return {s: np.mean([1 if v.queue > self.theta_q else 0
                            for v in self.vehicles if v.slice_type == s])
                for s in SLICE_TYPES}

    def get_slice_pending_ratio(self):
        return {s: np.mean([1 if v.queue > 0 else 0
                            for v in self.vehicles if v.slice_type == s])
                for s in SLICE_TYPES}

    def get_safety_threshold(self):
        return int(self.aoi_thresholds[SliceType.SAFETY])
