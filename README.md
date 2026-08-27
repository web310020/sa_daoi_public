# SA-DAoI

This repository contains the reference implementation and evaluation code for
**Starvation-Aware Deterministic Age of Information (SA-DAoI)**, a training-free
slice-level scheduler for V2X O-RAN resource allocation.

Repository: <https://github.com/00-Shen/sa_daoi_public>

## What is included

- `agents/sa_daoi.py`: SA-DAoI scheduler with floor-preserving ISE redistribution.
- `agents/heuristics.py`: Static-RR, threshold-normalized AoI, and Whittle baselines.
- `agents/dqn_agent.py` and `agents/ppo_cppo.py`: learned baseline definitions.
- `env/vehicular.py`: Gymnasium-compatible V2X slicing simulator.
- `eval/reproduce_main.py`: main Load-D block-level comparison.
- `models/load_D/`: the exact DQN, PPO, and C-PPO checkpoints used by the main runner.
- `reference_results/load_D/`: frozen block-level reference output and hashes.
- `plotting/`: figure and table generators for recorded experiment outputs.

The learned-policy loaders fail closed: evaluation stops if a required checkpoint is
missing or unreadable.

## Installation

Python 3.10 is recommended. Create an isolated environment and install:

```bash
python -m pip install -r requirements.txt
```

The release was verified on CPU with Python 3.10.19, NumPy 2.2.6, PyTorch 2.5.1,
Gymnasium 1.2.3, Stable-Baselines3 2.7.1, pandas 2.3.3, Matplotlib 3.10.7, and
Seaborn 0.13.2.

## Quick verification

Run the release tests:

```bash
python -m unittest -v tests.test_release_contract
```

Load all three learned checkpoints and execute a one-episode smoke cell:

```bash
python -m eval.reproduce_main --smoke --output-dir results/smoke
```

## Main Load-D reproduction

```bash
python -m eval.reproduce_main --output-dir results/load_D_reproduction
```

The independent statistical unit is a deployment block: `n=5` blocks, each containing
30 sequential episodes. The protocol executes 150 distinct environment seeds per method,
but those episode executions are not treated as 150 independent inferential replicates.
A fresh policy instance is constructed for every method and deployment block.

Reported values are descriptive means and sample standard deviations across the five
block aggregates. The main runner uses block starts 1000, 2000, 3000, 4000, and 5000.

## Auxiliary analyses

The other scripts under `eval/` and `plotting/` support the paper's descriptive ablation,
sensitivity, burst-traffic, multi-cell, and visualization carriers. They are separate from
the main block-level comparison above; do not combine their statistical units.

## Authors

Zhiqiang Shen and Jitae Shin, Sungkyunkwan University.

## License

MIT; see [LICENSE](LICENSE).
