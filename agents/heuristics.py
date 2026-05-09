"""
Baseline 启发式策略 (Static-RR / T-AoI / Whittle).
每个类暴露 select_action(env) -> int.
"""
import numpy as np
from env.vehicular import SliceType, SLICE_TYPES


# ═══ Static Round-Robin ═════════════════════════════════════════════
class StaticRRPolicy:
    """1/3 等分, 不自适应 (lower bound baseline)."""

    def __init__(self, env):
        target = np.array([1/3, 1/3, 1/3]) * env.total_rbs
        diffs = [np.linalg.norm(np.array(a) - target) for a in env.action_table]
        self._action = int(np.argmin(diffs))

    def select_action(self, env=None):
        return self._action

    def on_step(self, info):
        pass

    def on_reset(self, env):
        pass


# ═══ Threshold-Normalized AoI (T-AoI) ══════════════════════════════
class TAoIPolicy:
    """阈值归一线性紧迫度."""

    def select_action(self, env):
        stats = env.get_slice_aoi_stats()
        thr   = env.aoi_thresholds
        urg   = {s: stats[s] / thr[s] for s in SLICE_TYPES}

        best_idx, max_score = 0, -1e9
        for idx, alloc in enumerate(env.action_table):
            score = sum(alloc[s.value] * urg[s] for s in SLICE_TYPES)
            if score > max_score:
                max_score, best_idx = score, idx
        return best_idx

    def on_step(self, info):
        pass

    def on_reset(self, env):
        pass


# ═══ Whittle Index ══════════════════════════════════════════════════
class WhittlePolicy:
    """Whittle Index: W(d) = 0.5 * d * (d + 2)."""

    def select_action(self, env):
        stats = env.get_slice_aoi_stats()
        indices = {s: 0.5 * stats[s] * (stats[s] + 2) for s in SLICE_TYPES}

        best_idx, max_gain = 0, -1e9
        for idx, alloc in enumerate(env.action_table):
            gain = sum(alloc[s.value] * indices[s] for s in SLICE_TYPES)
            if gain > max_gain:
                max_gain, best_idx = gain, idx
        return best_idx

    def on_step(self, info):
        pass

    def on_reset(self, env):
        pass
