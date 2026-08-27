# Paper-configured Load-D reference

`metrics_D_5seed.csv` is the frozen result carrier behind the paper's Load-D
headline comparison and Table IV. It contains seven methods and five configured
base-seed aggregates per method. Each aggregate contains 30 sequential episodes.

The base seeds are 10, 20, 30, 40, and 50, and episode `e` uses
`base_seed + e`. Consequently, the 150 episode executions per method contain
70 distinct environment seeds; they are not 150 independent inferential
replicates. The paper therefore reports descriptive means and sample standard
deviations across the five configured base-seed aggregates.

This directory and `reference_results/load_D/` serve different purposes:

- `load_D_paper_configured/` is the exact frozen carrier for the paper values.
- `load_D/` is a later verification run using five non-overlapping 30-episode
  blocks beginning at 1000, 2000, 3000, 4000, and 5000.

Do not substitute one protocol for the other. See `PROVENANCE.json` for the
hash-bound protocol identity.
