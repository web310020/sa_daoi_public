"""
DQN baseline (AoI-aware slicing).
"""
import os
import random
from collections import deque
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs import DRL as CFG

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a),
                np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, obs_dim, n_actions):
        self.device = torch.device("cpu")
        self.n_actions  = n_actions
        self.gamma      = CFG["dqn_gamma"]
        self.batch_size = CFG["dqn_batch_size"]

        self.q_net      = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=CFG["dqn_lr"])
        self.buffer     = ReplayBuffer()
        self.train_steps = 0

    def act(self, state, eps=0.0):
        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(t).argmax(1).item())

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s_t  = torch.FloatTensor(s).to(self.device)
        a_t  = torch.LongTensor(a).to(self.device)
        r_t  = torch.FloatTensor(r).to(self.device)
        ns_t = torch.FloatTensor(ns).to(self.device)
        d_t  = torch.FloatTensor(d).to(self.device)

        q_sa = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_nq = self.target_net(ns_t).max(1)[0]
            target = r_t + self.gamma * (1.0 - d_t) * max_nq

        loss = nn.MSELoss()(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % CFG["dqn_target_update"] == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())


# ═══ Train / Eval interface ═════════════════════════════════════════

def train_dqn(load_tier, models_dir, num_episodes=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from env.vehicular import VehicularNetworkEnv

    num_episodes = num_episodes or CFG["dqn_episodes"]
    env = VehicularNetworkEnv(load_tier=load_tier)
    obs, _ = env.reset()
    agent = DQNAgent(obs.shape[0], env.n_actions)

    print(f"[DQN] Training Tier {load_tier} for {num_episodes} episodes...")
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        eps = max(CFG["dqn_eps_end"],
                  CFG["dqn_eps_start"] - (ep / num_episodes) * CFG["dqn_eps_start"])
        while not done:
            action = agent.act(state, eps=eps)
            ns, r, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.buffer.push(state, action, r, ns, done)
            state = ns
            agent.update()
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{num_episodes}")

    os.makedirs(models_dir, exist_ok=True)
    torch.save(agent.q_net.state_dict(), os.path.join(models_dir, f"dqn_{load_tier}.pth"))
    print(f"[DQN] Saved model for Tier {load_tier}")


class DQNPolicy:
    """加载训练好的 DQN 用于 evaluation."""

    def __init__(self, env, models_dir, load_tier):
        obs, _ = env.reset()
        self.agent = DQNAgent(obs.shape[0], env.n_actions)
        path = os.path.join(models_dir, f"dqn_{load_tier}.pth")
        if os.path.exists(path):
            self.agent.q_net.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True))

    def select_action(self, env=None):
        if env is not None:
            obs = env._get_obs()
        else:
            obs = np.zeros(6)
        return self.agent.act(obs, eps=0.0)

    def on_step(self, info):
        pass

    def on_reset(self, env):
        pass
