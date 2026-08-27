"""
PPO-AoI 与 C-PPO (Lagrangian PPO) baselines.
C-PPO 加 Lagrangian 罚项处理约束违反.
"""
import os
import numpy as np
import gymnasium as gym
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs import DRL as CFG

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    from stable_baselines3 import PPO as SB3_PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


# ═══ Lagrangian Wrapper (for C-PPO) ════════════════════════════════
class LagrangianWrapper(gym.Wrapper):
    """
    在 reward 加 Lagrangian 罚项的 env wrapper.
    lambda 在每集结尾按违反量更新.
    """

    def __init__(self, env, cost_limit=None, lambda_lr=None):
        super().__init__(env)
        self.lam = CFG.get("cppo_lambda_init", 1.0)
        self.cost_limit = cost_limit or CFG["cppo_cost_limit"]
        self.lambda_lr  = lambda_lr  or CFG["cppo_lambda_lr"]
        self._ep_costs  = []

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        cost = info.get("viol_safety", 0.0)
        self._ep_costs.append(cost)

        # Lagrangian 修正 reward: r' = r - lambda * c (×100 scale)
        reward_mod = reward - self.lam * cost * 100

        if term or trunc:
            avg_cost = np.mean(self._ep_costs) if self._ep_costs else 0.0
            # lambda 对偶上升
            self.lam = max(0.0, self.lam + self.lambda_lr * (avg_cost - self.cost_limit))
            self._ep_costs = []

        return obs, reward_mod, term, trunc, info

    def reset(self, **kwargs):
        self._ep_costs = []
        return self.env.reset(**kwargs)


# ═══ Training ══════════════════════════════════════════════════════

def train_ppo(load_tier, models_dir, total_timesteps=None):
    if not HAS_SB3:
        raise RuntimeError("stable-baselines3 not installed")
    from env.vehicular import VehicularNetworkEnv
    PPO = SB3_PPO

    total_timesteps = total_timesteps or CFG["ppo_timesteps"]
    env = VehicularNetworkEnv(load_tier=load_tier)
    model = PPO("MlpPolicy", env, verbose=0,
                learning_rate=CFG["ppo_lr"], gamma=CFG["ppo_gamma"], device="cpu")
    print(f"[PPO] Training Tier {load_tier} for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, f"ppo_{load_tier}"))
    print(f"[PPO] Saved model for Tier {load_tier}")


def train_cppo(load_tier, models_dir, total_timesteps=None):
    if not HAS_SB3:
        raise RuntimeError("stable-baselines3 not installed")
    from env.vehicular import VehicularNetworkEnv
    PPO = SB3_PPO

    total_timesteps = total_timesteps or CFG["cppo_timesteps"]
    base_env = VehicularNetworkEnv(load_tier=load_tier)
    env = LagrangianWrapper(base_env)

    model = PPO("MlpPolicy", env, verbose=0,
                learning_rate=CFG["cppo_lr"], gamma=CFG["cppo_gamma"], device="cpu")
    print(f"[C-PPO] Training Tier {load_tier} for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, f"cppo_{load_tier}"))
    print(f"[C-PPO] Saved model for Tier {load_tier}")


# ═══ Evaluation policies ═══════════════════════════════════════════

class PPOPolicy:
    def __init__(self, env, models_dir, load_tier):
        if not HAS_SB3:
            raise RuntimeError("stable-baselines3 is required for PPO evaluation")
        path = os.path.join(models_dir, f"ppo_{load_tier}")
        if not os.path.isfile(path + ".zip"):
            raise FileNotFoundError(f"PPO checkpoint not found: {path}.zip")
        self.model = SB3_PPO.load(path)

    def select_action(self, env):
        obs = env._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def on_step(self, info):
        pass

    def on_reset(self, env):
        pass


class CPPOPolicy:
    def __init__(self, env, models_dir, load_tier):
        if not HAS_SB3:
            raise RuntimeError("stable-baselines3 is required for C-PPO evaluation")
        path = os.path.join(models_dir, f"cppo_{load_tier}")
        if not os.path.isfile(path + ".zip"):
            raise FileNotFoundError(f"C-PPO checkpoint not found: {path}.zip")
        self.model = SB3_PPO.load(path)

    def select_action(self, env):
        obs = env._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def on_step(self, info):
        pass

    def on_reset(self, env):
        pass
