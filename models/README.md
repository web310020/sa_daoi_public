# Load-D checkpoints

`load_D/` contains the three learned-policy checkpoints used by
`python -m eval.reproduce_main`:

- `dqn_D.pth`: DQN-AoI
- `ppo_D.zip`: PPO-AoI
- `cppo_D.zip`: constrained PPO (C-PPO)

Verify them from the repository root with the SHA-256 values in `SHA256SUMS` before a
full reproduction. The evaluation constructors intentionally fail if any checkpoint is
missing or cannot be loaded.
