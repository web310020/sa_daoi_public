# SA-DAoI

这是 **Starvation-Aware Deterministic Age of Information (SA-DAoI)** 的论文配套代码。
The repository provides the reference implementation and evaluation scripts for this
training-free, slice-level V2X O-RAN scheduler.

Repository: <https://github.com/00-Shen/sa_daoi_public>

## What is included / 仓库内容

核心 scheduler、baselines、simulator、checkpoints 与 plotting scripts 均按用途分开：

- `agents/sa_daoi.py`: exact paper-configured SA-DAoI scheduler. Its ISE threshold
  controls whether redistribution activates; it is not a guaranteed post-transfer floor.
- `agents/heuristics.py`: Static-RR, threshold-normalized AoI, and Whittle baselines.
- `agents/dqn_agent.py` and `agents/ppo_cppo.py`: learned baseline definitions.
- `env/vehicular.py`: Gymnasium-compatible V2X slicing simulator.
- `eval/reproduce_main.py`: main Load-D block-level comparison.
- `models/load_D/`: the exact DQN, PPO, and C-PPO checkpoints used by the main runner.
- `reference_results/load_D_paper_configured/`: exact frozen result carrier for the paper's Load-D values.
- `reference_results/load_D/`: later non-overlapping-block verification output and hashes.
- `reference_results/dqn_checkpoint_stability/`: per-training-seed rows for the separate
  200- and 500-episode DQN checkpoint study reported in the paper.
- `plotting/`: figure and table generators for recorded experiment outputs.

The learned-policy loaders fail closed: evaluation stops if a required checkpoint is
missing or unreadable。这样可以避免把随机初始化的策略误当成论文中的 learned baseline。

## Installation / 安装

Python 3.10 is recommended. Create an isolated environment and install:

```bash
python -m pip install -r requirements.txt
```

本代码在 CPU 环境完成验证。The verified environment used Python 3.10.19, NumPy 2.2.6, PyTorch 2.5.1,
Gymnasium 1.2.3, Stable-Baselines3 2.7.1, pandas 2.3.3, Matplotlib 3.10.7, and
Seaborn 0.13.2.

The bundled PPO and C-PPO archives retain their original Stable-Baselines3 2.8.0
producer metadata. 本 release 另外验证了这些 frozen checkpoints 可由上述 2.7.1
consumer environment 加载并执行；producer version 与 verified consumer version
因此分别记录，不能把两者误写成同一次训练环境。

## Quick verification / 快速检查

先运行 release tests：

```bash
python -m unittest -v tests.test_release_contract
```

再加载三个 learned checkpoints，并执行 one-episode smoke cell：

```bash
python -m eval.reproduce_main --smoke --output-dir results/smoke
```

## Main Load-D reproduction / 主实验复现

```bash
python -m eval.reproduce_main --output-dir results/load_D_reproduction
```

该命令运行后续的 block-level verification protocol。Its independent statistical unit is a
deployment block: `n=5` non-overlapping blocks, each containing 30 sequential episodes.
Each method therefore uses 150 distinct environment seeds, and a fresh policy instance is
constructed for every method and deployment block.

Reported values are descriptive means and sample standard deviations across the five
block aggregates。主 runner 使用 1000、2000、3000、4000 和 5000 作为 block starts。

论文配置对应的 frozen carrier 位于 `reference_results/load_D_paper_configured/`。Its five
base-seed aggregates start at 10, 20, 30, 40, and 50, so their episode-seed ranges overlap.
The accompanying provenance file records 70 distinct environment seeds across 150 episode
executions per method. **这两个 reference directories 对应不同统计口径，不能互换。**

## Auxiliary analyses / 辅助分析

`eval/` 与 `plotting/` 下的其他脚本用于 ablation、sensitivity、burst traffic、multi-cell
和 visualization。They remain separate from the main block-level comparison above; do not
combine their statistical units.

## Authors / 作者

Zhiqiang Shen and Jitae Shin, Sungkyunkwan University.

## License / 许可

MIT; see [LICENSE](LICENSE).
