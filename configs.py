"""论文与复现实验共用的 SA-DAoI configuration；集中维护 hyperparameters。"""

# ─── Tier definitions (Table II in paper) ───────────────────────────
TIER_PARAMS = {
    "A": dict(n_bs=1, n_rbs=18, n_vehicles=60,  p_arr=0.50,
              aoi_thresholds={"S": 100, "C": 150, "I": 250},
              packets_per_rb=20, max_steps=800),
    "B": dict(n_bs=1, n_rbs=14, n_vehicles=120, p_arr=0.65,
              aoi_thresholds={"S": 60,  "C": 120, "I": 200},
              packets_per_rb=20, max_steps=800),
    "C": dict(n_bs=1, n_rbs=10, n_vehicles=200, p_arr=0.89,
              aoi_thresholds={"S": 40,  "C": 90,  "I": 160},
              packets_per_rb=20, max_steps=800),
    "D": dict(n_bs=1, n_rbs=10, n_vehicles=210, p_arr=0.93,
              aoi_thresholds={"S": 20,  "C": 60,  "I": 120},
              packets_per_rb=18, max_steps=800),
}

# ─── SA-DAoI algorithm parameters (Table I in paper) ────────────────
SA_DAOI = dict(
    beta            = 0.1,      # target violation rate (SLA)
    alpha_floor     = 0.85,     # minimum protection bound
    W_beta          = 8,        # violation sliding window length
    usage_win_len   = 20,       # utilization sliding window length
    ise_gate_ratio  = 0.4,      # ε_ISE = 0.4 × ΓS (tuned via sensitivity analysis)
    ise_usage_cap   = 0.95,     # η: utilization cap for ISE gate
    ise_harvest_ratio = 0.2,    # λ_ise: fraction of NRB to harvest
    ise_floor_ratio = 0.2,      # w_floor = ⌈0.2 × NRB⌉
    theta_ratio     = 0.5,      # Θ = |V| × 0.5
    s_weight_base   = 5.0,      # ω₀ base safety weight
    s_weight_ref    = 0.7,      # ζ_ref reference pressure
    s_weight_range  = (2.0, 15.0),  # [ω_min, ω_max]
    cap_urg_threshold = 0.2,    # urgency below which capping kicks in
    cap_penalty     = 2.0,      # score penalty multiplier for capping
)

# ─── Evaluation parameters ──────────────────────────────────────────
EVAL = dict(
    num_episodes = 30,
    seeds        = [10, 20, 30],
)

# ─── DRL training parameters ───────────────────────────────────────
DRL = dict(
    dqn_episodes     = 200,
    dqn_gamma        = 0.95,
    dqn_lr           = 1e-3,
    dqn_batch_size   = 64,
    dqn_eps_start    = 0.5,
    dqn_eps_end      = 0.01,
    dqn_target_update = 500,
    ppo_timesteps    = 200_000,
    ppo_lr           = 3e-4,
    ppo_gamma        = 0.95,
    cppo_timesteps   = 1_000_000,
    cppo_lr          = 3e-4,
    cppo_gamma       = 0.95,
    cppo_cost_limit  = 0.1,     # target violation rate constraint
    cppo_lambda_lr   = 0.01,    # slower Lagrangian update for stability
    cppo_lambda_init = 0.1,     # low init so policy can learn before constraint tightens
)

# ─── Baseline method registry ───────────────────────────────────────
# Order determines plotting legend order
METHOD_ORDER = [
    "SA-DAoI", "Static-RR", "DQN-AoI", "PPO-AoI",
    "C-PPO", "T-AoI", "Whittle",
]

# ─── Plotting style ─────────────────────────────────────────────────
COLORS = {
    "SA-DAoI":   "#d62728",
    "Static-RR": "#bcbd22",
    "DQN-AoI":   "#8c564b",
    "PPO-AoI":   "#e377c2",
    "C-PPO":     "#1f77b4",
    "T-AoI":     "#9467bd",
    "Whittle":   "#2ca02c",
}
MARKERS = {
    "SA-DAoI": "o", "Static-RR": "s", "DQN-AoI": "D",
    "PPO-AoI": "X", "C-PPO": "P", "T-AoI": "^", "Whittle": "p",
}
# 线宽 (pt). figsize=(3.5, 2.25) IEEE 2-col 子图用. SA-DAoI 加粗 2x 突出.
LINE_WIDTHS = {m: 1.0 for m in COLORS}
LINE_WIDTHS["SA-DAoI"] = 2.0

# ─── CSV output schema ──────────────────────────────────────────────
CSV_FIELDS = [
    "seed", "method", "load_tier", "avg_reward",
    "aoi_total_SAFETY", "aoi_total_CE", "aoi_total_IOT",
    "violation_rate_SAFETY",
    "starvation_rate_SAFETY", "starvation_rate_CE", "starvation_rate_IOT",
    "pending_ratio_SAFETY", "pending_ratio_CE", "pending_ratio_IOT",
    "queue_len_SAFETY", "queue_len_CE", "queue_len_IOT",
]
