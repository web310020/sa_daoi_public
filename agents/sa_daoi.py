"""
SA-DAoI scheduler.
Cubic AoI urgency scoring + adaptive alpha guardrail + ISE redistribution.
"""
import numpy as np
from collections import deque
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs import SA_DAOI as CFG
from env.vehicular import SliceType, SLICE_TYPES


class SADAOIScheduler:
    """Deterministic SA-DAoI scheduler. Training-free, O(|K|) per decision."""

    def __init__(self, env, urgency_exponent: int = 3):
        """
        Args:
          env: VehicularNetworkEnv 实例
          urgency_exponent: 紧迫度指数 p, 默认 3 (cubic). 仅 ablation 用其他值.
        """
        self.env = env
        self.urgency_exponent = int(urgency_exponent)
        gamma_s  = env.aoi_thresholds[SliceType.SAFETY]
        v_total  = len(env.vehicles)
        n_rb     = env.param_n_rbs
        n_safety = sum(1 for v in env.vehicles if v.slice_type == SliceType.SAFETY)

        # alpha 初始化, 按 safety 占比
        self.alpha       = min(0.95, (n_safety / v_total) * 1.5)
        self.beta        = CFG["beta"]
        self.alpha_floor = CFG["alpha_floor"]
        self.epsilon     = 0.05 + 0.1 * (v_total / n_rb / 30)

        # safety 权重 (公式 6): 按队列压力调节
        pressure = v_total / (n_rb * gamma_s)
        lo, hi = CFG["s_weight_range"]
        self.s_weight = np.clip(
            CFG["s_weight_base"] * (pressure / CFG["s_weight_ref"]), lo, hi
        )

        # ISE 参数 (公式 12-13)
        self.eps_ise     = CFG["ise_gate_ratio"] * gamma_s
        self.usage_cap   = CFG["ise_usage_cap"]
        self.harvest_n   = max(1, int(n_rb * CFG["ise_harvest_ratio"]))
        self.w_floor     = max(1, int(n_rb * CFG["ise_floor_ratio"]))
        self.Theta       = v_total * CFG["theta_ratio"]

        # 弹性队列 EWMA, ISE 重分配用
        self.rho_q   = 0.3
        self.Q_max   = 500.0
        self._ewma_q = {}

        self.violation_win = deque(maxlen=CFG["W_beta"])
        self.usage_win     = deque(maxlen=CFG["usage_win_len"])

    # ── ISE 安全门 (公式 12) ────────────────────────────────────────
    def _g_ise(self, s_aoi: float) -> bool:
        u_avg = np.mean(self.usage_win) if self.usage_win else 1.0
        return (s_aoi < self.eps_ise) and (u_avg < self.usage_cap)

    # ── alpha 更新 (公式 8) ────────────────────────────────────────
    def _update_alpha(self):
        r_s = np.mean(self.violation_win) if self.violation_win else 0.0
        if r_s > self.beta:
            self.alpha = min(self.alpha + self.epsilon, 1.0)
        else:
            self.alpha = max(self.alpha - self.epsilon, self.alpha_floor)

    # ── 评分 (公式 6-7, 立方紧迫度) ────────────────────────────────
    def _compute_score(self, alloc, urg, omega):
        score = (alloc[0] * urg[SliceType.SAFETY] * self.s_weight
                 + alloc[1] * urg[SliceType.CE]
                 + alloc[2] * urg[SliceType.IOT])
        # 超 omega 上限的保守惩罚
        if alloc[0] > omega and urg[SliceType.SAFETY] < CFG["cap_urg_threshold"]:
            score -= (alloc[0] - omega) * CFG["cap_penalty"]
        return score

    # ── ISE 队列感知重分配 (公式 13) ────────────────────────────────
    def _apply_ise(self, best_idx, s_aoi):
        if not self._g_ise(s_aoi):
            return best_idx

        q_ce  = sum(v.queue for v in self.env.vehicles if v.slice_type == SliceType.CE)
        q_iot = sum(v.queue for v in self.env.vehicles if v.slice_type == SliceType.IOT)
        if q_ce + q_iot <= self.Theta:
            return best_idx

        alloc = list(self.env.action_table[best_idx])
        if alloc[0] <= self.w_floor:
            return best_idx

        # 按 EWMA 指数紧迫度做 CE/IOT 比例分配
        ql = self.env.get_slice_avg_queue_len()
        psi_ce  = np.exp(self._ewma_q.get(SliceType.CE, ql[SliceType.CE]) / self.Q_max)
        psi_iot = np.exp(self._ewma_q.get(SliceType.IOT, ql[SliceType.IOT]) / self.Q_max)
        total_psi = psi_ce + psi_iot

        harvest = self.harvest_n
        alloc[0] -= harvest
        ce_share = int(harvest * psi_ce / total_psi)
        alloc[1] += ce_share
        alloc[2] += harvest - ce_share

        new_idx = self._find_nearest(alloc)
        return new_idx if new_idx != best_idx else max(0, best_idx - 1)

    # ── 主入口 ──────────────────────────────────────────────────────
    def select_action(self, env=None):
        """选 PRB 分配 action."""
        if env is not None:
            self.env = env

        aoi = self.env.get_slice_aoi_stats()
        thr = self.env.aoi_thresholds
        ql  = self.env.get_slice_avg_queue_len()
        s_aoi = aoi[SliceType.SAFETY]

        # EWMA 队列 (ISE 用)
        for s in [SliceType.CE, SliceType.IOT]:
            self._ewma_q[s] = ((1 - self.rho_q) * self._ewma_q.get(s, ql[s])
                               + self.rho_q * ql[s])

        # 紧迫度 (公式 6, 默认 p=3)
        p = self.urgency_exponent
        urg = {s: (aoi[s] / thr[s]) ** p for s in SLICE_TYPES}

        self._update_alpha()
        omega = self.alpha * self.env.total_rbs

        best_idx, max_score = 0, -1e9
        for idx, alloc in enumerate(self.env.action_table):
            score = self._compute_score(alloc, urg, omega)
            if score > max_score:
                max_score, best_idx = score, idx

        return self._apply_ise(best_idx, s_aoi)

    # ── 每步更新 (evaluator 调用) ────────────────────────────────────
    def on_step(self, info: dict):
        self.violation_win.append(info["viol_safety"])
        self.usage_win.append(info["usage_safety"])

    def on_reset(self, env):
        """每集重置 env 与 EWMA state."""
        if env is not None:
            self.env = env
        self._ewma_q = {}

    # ── 工具方法 ────────────────────────────────────────────────────
    def _find_nearest(self, target):
        diffs = [np.linalg.norm(np.array(target) - np.array(a))
                 for a in self.env.action_table]
        return int(np.argmin(diffs))
