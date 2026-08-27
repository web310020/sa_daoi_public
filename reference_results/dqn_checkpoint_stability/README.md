# DQN checkpoint-stability carriers / DQN checkpoint 稳定性结果

This directory preserves the per-training-seed rows behind the paper's separate DQN
checkpoint study. 这里的 statistical unit 是 independently trained checkpoint，不是主实验中的
deployment block，也不能与 `reference_results/load_D*` 的统计口径互换。

- `200_episodes/`: 30 training seeds; 6 checkpoints meet the configured Safety criterion.
- `500_episodes/`: 30 training seeds; 8 checkpoints meet the configured Safety criterion.
- Each `seeds.csv` records one aggregate row per training seed.
- Each `summary.json` records the training/evaluation budget and Wilson interval.

The public regression test recomputes both counts and intervals directly from the CSV rows:

```bash
python -m unittest -v \
  tests.test_release_contract.ReleaseContractTests.test_dqn_checkpoint_stability_rows_recompute_the_paper_counts
```

这些 carriers 支持论文中的 6/30 与 8/30 descriptive statements，但不封存完整 training
lineage。The 60 trained checkpoint archives, training logs, and per-checkpoint source/config hashes
are not included, so these files must not be described as a bit-exact training reproduction bundle.
