import hashlib
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

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
    def test_ise_harvest_preserves_declared_floor_and_total_quota(self):
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

        self.assertEqual(allocation, (2, 4, 4))
        self.assertGreaterEqual(allocation[0], scheduler.w_floor)
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

    def test_readme_uses_block_level_statistical_language(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        lowered = readme.lower().replace(" ", "")
        self.assertIn("n=5", lowered)
        self.assertEqual(set(re.findall(r"n=(\d+)", lowered)), {"5"})
        self.assertIn("150 distinct environment seeds", readme)
        self.assertIn("https://github.com/00-Shen/sa_daoi_public", readme)

    def test_public_files_contain_no_private_workspace_residue(self):
        forbidden = (
            "_sand" + "box/",
            "_" + "sage/",
            "SAGE " + "phase",
            "v" + "058_",
            "2026" + "0326_",
        )
        expected_public_url = "https://github.com/00-Shen/sa_daoi_public"
        checked_suffixes = {".py", ".md", ".txt", ".yaml", ".yml", ".json"}
        offenders = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            if path.suffix.lower() not in checked_suffixes and path.name != "requirements.txt":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {token}")
            for url in re.findall(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text):
                if url != expected_public_url:
                    offenders.append(f"{path.relative_to(ROOT)}: unexpected repository URL")
            if re.search(r"[A-Za-z]:[\\/](?:[^\\/\s]+[\\/]){2,}", text):
                offenders.append(f"{path.relative_to(ROOT)}: absolute local path")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
