"""
SA-DAoI ablation 变体.
mode="full" 与父类 SADAOIScheduler 完全一致.
"""
from agents.sa_daoi import SADAOIScheduler
from env.vehicular import SliceType


class AblationScheduler(SADAOIScheduler):
    """
    Modes:
      "full"       -> 父类不变 (= SA-DAoI)
      "no_ise"     -> ISE 门永远关闭
      "alpha_low"  -> alpha 锁 0.70 (不自适应)
      "alpha_high" -> alpha 锁 1.00 (full safety monopoly)
    """

    def __init__(self, env, mode="full"):
        super().__init__(env)
        self.mode = mode
        if mode not in ("full", "no_ise", "alpha_low", "alpha_high"):
            raise ValueError(f"Unknown ablation mode: {mode}")

    # ── alpha 更新 override ────────────────────────────────────────
    def _update_alpha(self):
        if self.mode == "alpha_low":
            self.alpha = 0.70
        elif self.mode == "alpha_high":
            self.alpha = 1.00
        else:
            super()._update_alpha()

    # ── ISE gate override ──────────────────────────────────────────
    def _g_ise(self, s_aoi: float) -> bool:
        if self.mode == "no_ise":
            return False
        return super()._g_ise(s_aoi)
