# SA-DAoI

首先感谢您阅读这篇论文 First of all, thank you for reading this paper



V2X O-RAN slicing 调度的 reference implementation. 支持 SA-DAoI (主算法) 与 6 个 baseline (Static-RR, T-AoI, Whittle, DQN-AoI, PPO-AoI, C-PPO).

## 结构

```
sa_daoi_public/
├── configs.py             # hyperparameters (single source of truth)
├── env/
│   └── vehicular.py       # V2X 切片 Gymnasium 环境
├── agents/
│   ├── sa_daoi.py         # SA-DAoI 主算法
│   ├── ablation.py        # ablation 变体 (继承 sa_daoi)
│   ├── heuristics.py      # Static-RR / T-AoI / Whittle
│   ├── dqn_agent.py       # DQN-AoI baseline
│   └── ppo_cppo.py        # PPO-AoI + C-PPO (Lagrangian)
├── eval/
│   ├── evaluator.py                    # 统一 evaluation loop
│   ├── polish_5seed_and_multicell.py   # 主结果 (Tier D, n=150)
│   ├── dqn_stability_30seed.py         # 30-seed DQN sweep
│   ├── urgency_ablation.py             # 紧迫度指数 ablation
│   ├── bursty_traffic.py               # MMPP burst
│   ├── multicell_preliminary.py        # 多 cell preliminary
│   ├── run_sensitivity.py              # 敏感性 sweep
│   └── sweep_cppo.py                   # C-PPO eta_lambda sweep
└── plotting/
    ├── generate_all_figs.py
    ├── generate_tables.py
    └── fig_aoi_distribution.py
```

## 安装

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.x. CPU 即可 (SA-DAoI 不需 GPU; DRL baseline GPU 加速可选).

## Reproduction

主结果 (Tier D, 5 seeds × 30 ep, n=150):

```bash
python -m eval.polish_5seed_and_multicell
```

30-seed DQN stability sweep:

```bash
python -m eval.dqn_stability_30seed --workers 4
```

紧迫度 ablation (p ∈ {1,2,3,4}):

```bash
python -m eval.urgency_ablation
```

MMPP burst:

```bash
python -m eval.bursty_traffic
```

绘图:

```bash
python -m plotting.generate_all_figs --data-dir results/<timestamp>/
```

## 设计要点

1. `agents/sa_daoi.py` 中的 `SADAOIScheduler` 是算法唯一实现, ablation 变体通过继承 + override 关键方法.
2. `eval/evaluator.py` 提供统一 evaluation loop, 所有 method 共用.
3. `configs.py` 集中所有 hyperparameter, 避免 magic number.

## Authors

Zhiqiang Shen, Jitae Shin (Sungkyunkwan University)

## License

MIT (见 LICENSE 文件)
