"""公开 release 的 checkpoints、runner protocol 与 portable paths 回归测试。"""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "load_D"
MODEL_HASHES = {
    "dqn_D.pth": "f503c2d904acf8958d54bf22fe0d91b5eadbd893e05219c1b5b1714ab4bdcd2e",
    "ppo_D.zip": "6a218732d582d8bc31c53f71164b29650d6dbc907a5981fc3a8c7fb3722283e4",
    "cppo_D.zip": "045d89f0ea8ece103e29e13786eb9aa829635d7aea539a566b34c1f8cb0ae6bd",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReleaseContractTests(unittest.TestCase):
    def test_burst_violation_metric_preserves_definition1_fraction(self):
        from eval.bursty_traffic import definition1_violation_rate

        self.assertAlmostEqual(definition1_violation_rate([0.0, 0.25, 0.50]), 0.25)
        with self.assertRaisesRegex(ValueError, "at least one TTI"):
            definition1_violation_rate([])

    def test_evaluator_can_run_the_exact_supplied_environment(self):
        from agents.sa_daoi import SADAOIScheduler
        from eval.evaluator import evaluate_env
        from eval.multicell_preliminary import HalfTierDEnv

        environment = HalfTierDEnv(max_steps=2)
        environment.reset(seed=10)
        agent = SADAOIScheduler(environment)
        original_id = id(environment)

        result = evaluate_env(agent, environment, num_episodes=1, seed=10)

        self.assertEqual(result["environment_object_id"], original_id)
        self.assertEqual(result["vehicle_count"], 105)
        self.assertEqual(result["slice_census"], {"SAFETY": 80, "CE": 20, "IOT": 5})

    def test_multicell_runner_never_substitutes_a_full_tier_d_environment(self):
        from eval import multicell_preliminary as multicell

        observed = {}

        def fake_evaluate_env(agent, environment, num_episodes, seed):
            observed["environment"] = environment
            observed["num_episodes"] = num_episodes
            observed["seed"] = seed
            return {"vehicle_count": len(environment.vehicles)}

        with patch.object(multicell, "evaluate_env", side_effect=fake_evaluate_env):
            result = multicell.run_cell(seed=20, num_episodes=2, max_steps=1)

        self.assertIsInstance(observed["environment"], multicell.HalfTierDEnv)
        self.assertEqual(observed["num_episodes"], 2)
        self.assertEqual(observed["seed"], 20)
        self.assertEqual(result["vehicle_count"], 105)

    def test_paper_configured_ise_uses_activation_threshold_not_post_transfer_floor(self):
        from agents.sa_daoi import SADAOIScheduler
        from env.vehicular import SliceType

        environment = SimpleNamespace(
            action_table=[(3, 4, 3), (2, 4, 4), (1, 5, 4)],
            vehicles=[
                SimpleNamespace(slice_type=SliceType.CE, queue=10),
                SimpleNamespace(slice_type=SliceType.IOT, queue=10),
            ],
            get_slice_avg_queue_len=lambda: {
                SliceType.CE: 1.0,
                SliceType.IOT: 1.0,
            },
        )
        scheduler = SADAOIScheduler.__new__(SADAOIScheduler)
        scheduler.env = environment
        scheduler.Theta = 0
        scheduler.harvest_n = 2
        scheduler.w_floor = 2
        scheduler.Q_max = 500.0
        scheduler._ewma_q = {}
        scheduler._g_ise = lambda _safety_aoi: True

        selected = scheduler._apply_ise(best_idx=0, s_aoi=0.0)
        allocation = environment.action_table[selected]

        self.assertEqual(allocation, (1, 5, 4))
        self.assertLess(allocation[0], scheduler.w_floor)
        self.assertEqual(sum(allocation), sum(environment.action_table[0]))

    def test_learned_policies_fail_closed_when_checkpoint_is_missing(self):
        from agents.dqn_agent import DQNPolicy
        from agents.ppo_cppo import CPPOPolicy, PPOPolicy

        class MinimalEnvironment:
            n_actions = 3

            def reset(self):
                return np.zeros(6, dtype=np.float32), {}

        environment = MinimalEnvironment()
        with tempfile.TemporaryDirectory() as empty_models:
            with self.assertRaises(FileNotFoundError):
                DQNPolicy(environment, empty_models, "D")
            with self.assertRaises(FileNotFoundError):
                PPOPolicy(environment, empty_models, "D")
            with self.assertRaises(FileNotFoundError):
                CPPOPolicy(environment, empty_models, "D")

    def test_main_runner_freezes_nonoverlapping_block_protocol(self):
        from eval import reproduce_main as runner

        self.assertEqual(runner.BLOCK_STARTS, (1000, 2000, 3000, 4000, 5000))
        self.assertEqual(runner.EPISODES_PER_BLOCK, 30)
        flattened = [
            seed
            for start in runner.BLOCK_STARTS
            for seed in range(start, start + runner.EPISODES_PER_BLOCK)
        ]
        self.assertEqual(len(flattened), 150)
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_smoke_summary_accepts_one_block(self):
        from eval import reproduce_main as runner

        row = {
            "method": "SA-DAoI",
            "block_index": 1,
            "base_seed": 1000,
            "episode_count": 1,
            "unique_environment_seed_count": 1,
            "aoi_safety_ms": 10.0,
            "violation_rate": 0.05,
            "queue_ce": 100.0,
            "queue_iot": 200.0,
            "es_backlog": 300.0,
        }
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "smoke"
            csv_path, summary_path = runner.write_rows([row], output)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertIn("sample SD unavailable", summary_path.read_text(encoding="utf-8"))

    def test_reference_comparison_accepts_the_frozen_rows(self):
        from eval import reproduce_main as runner

        rows = runner.load_reference_rows()
        self.assertEqual(len(rows), 35)
        self.assertEqual(runner.compare_to_reference(rows), [])

    def test_exact_load_d_checkpoints_are_bundled(self):
        for name, expected in MODEL_HASHES.items():
            path = MODEL_DIR / name
            self.assertTrue(path.is_file(), f"missing checkpoint: {path}")
            self.assertEqual(sha256_file(path), expected)

    def test_readme_separates_checkpoint_producer_from_verified_consumer_version(self):
        for name in ("ppo_D.zip", "cppo_D.zip"):
            with ZipFile(MODEL_DIR / name) as archive:
                producer_version = archive.read("_stable_baselines3_version").decode().strip()
            self.assertEqual(producer_version, "2.8.0")

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Stable-Baselines3 2.8.0", readme)
        self.assertIn("Stable-Baselines3 2.7.1", readme)
        self.assertIn("producer metadata", readme)
        self.assertIn("consumer environment", readme)

    def test_dqn_checkpoint_stability_rows_recompute_the_paper_counts(self):
        root = ROOT / "reference_results" / "dqn_checkpoint_stability"
        expected = {
            "200_episodes": (200, 6, 0.0950510717728987, 0.3730569641314825),
            "500_episodes": (500, 8, 0.14182663319596317, 0.4444796169518888),
        }
        z = 1.959963984540054

        for directory, (budget, feasible, expected_lo, expected_hi) in expected.items():
            source = root / directory
            with (source / "seeds.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
            observed_feasible = sum(int(row["sla_met"]) for row in rows)
            proportion = observed_feasible / len(rows)
            denominator = 1.0 + z * z / len(rows)
            center = (proportion + z * z / (2.0 * len(rows))) / denominator
            half_width = (
                z
                * math.sqrt(
                    proportion * (1.0 - proportion) / len(rows)
                    + z * z / (4.0 * len(rows) ** 2)
                )
                / denominator
            )

            self.assertEqual(len(rows), 30)
            self.assertEqual(summary["train_episodes"], budget)
            self.assertEqual(observed_feasible, feasible)
            self.assertEqual(summary["feasible_count"], feasible)
            self.assertAlmostEqual(center - half_width, expected_lo)
            self.assertAlmostEqual(center + half_width, expected_hi)

    def test_readme_uses_block_level_statistical_language(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        lowered = readme.lower().replace(" ", "")
        self.assertIn("n=5", lowered)
        self.assertEqual(set(re.findall(r"n=(\d+)", lowered)), {"5"})
        self.assertIn("150 distinct environment seeds", readme)
        self.assertIn("https://github.com/00-Shen/sa_daoi_public", readme)

    def test_public_text_uses_portable_paths_and_expected_repository_url(self):
        expected_public_url = "https://github.com/00-Shen/sa_daoi_public"
        checked_suffixes = {".py", ".md", ".txt", ".yaml", ".yml", ".json"}
        offenders = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix.lower() not in checked_suffixes and path.name != "requirements.txt":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for url in re.findall(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text):
                if url != expected_public_url:
                    offenders.append(f"{path.relative_to(ROOT)}: unexpected repository URL")
            if re.search(r"[A-Za-z]:[\\/](?:[^\\/\s]+[\\/]){2,}", text):
                offenders.append(f"{path.relative_to(ROOT)}: absolute local path")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
