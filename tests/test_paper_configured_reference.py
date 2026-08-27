"""核对 paper-configured reference carrier 的 bytes、provenance 与统计量。"""

import csv
import hashlib
import json
import statistics
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "reference_results" / "load_D_paper_configured"
CSV_PATH = REFERENCE_DIR / "metrics_D_5seed.csv"
PROVENANCE_PATH = REFERENCE_DIR / "PROVENANCE.json"
EXPECTED_CSV_SHA256 = "1620b9c2256482fc4be0378efb14df2e4214ab2c83b0468d1174ac86027e3a2d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PaperConfiguredReferenceTests(unittest.TestCase):
    def test_exact_carrier_and_provenance_exist(self):
        self.assertTrue(CSV_PATH.is_file())
        self.assertTrue(PROVENANCE_PATH.is_file())
        self.assertEqual(sha256(CSV_PATH), EXPECTED_CSV_SHA256)

    def test_protocol_census_is_exact(self):
        with CSV_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 35)
        by_method = defaultdict(list)
        for row in rows:
            by_method[row["method"]].append(row)
        self.assertEqual(len(by_method), 7)
        for method_rows in by_method.values():
            self.assertEqual(len(method_rows), 5)
            self.assertEqual([int(row["seed"]) for row in method_rows], [10, 20, 30, 40, 50])

    def test_table_iv_headline_cells_recompute(self):
        with CSV_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by_method = defaultdict(list)
        for row in rows:
            by_method[row["method"]].append(row)

        expected = {
            "SA-DAoI": (14.63, 0.53, 8.95, 0.39, 469.3, 6.3),
            "DQN-AoI": (9.55, 0.84, 2.46, 0.28, 564.0, 11.2),
            "C-PPO": (1.08, 0.00, 0.00, 0.00, 1111.7, 0.7),
        }
        for method, target in expected.items():
            method_rows = by_method[method]
            aoi = [float(row["aoi_safety_ms"]) for row in method_rows]
            violation_percent = [100.0 * float(row["viol_rate"]) for row in method_rows]
            backlog = [float(row["es_backlog"]) for row in method_rows]
            actual = (
                round(statistics.mean(aoi), 2),
                round(statistics.stdev(aoi), 2),
                round(statistics.mean(violation_percent), 2),
                round(statistics.stdev(violation_percent), 2),
                round(statistics.mean(backlog), 1),
                round(statistics.stdev(backlog), 1),
            )
            self.assertEqual(actual, target, method)

    def test_provenance_distinguishes_the_two_reference_protocols(self):
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(provenance["schema"], "sa-daoi-paper-configured-reference/v1")
        self.assertEqual(provenance["metrics_sha256"], EXPECTED_CSV_SHA256)
        self.assertEqual(provenance["base_seeds"], [10, 20, 30, 40, 50])
        self.assertEqual(provenance["episodes_per_base_seed"], 30)
        self.assertEqual(provenance["aggregate_count_per_method"], 5)
        self.assertEqual(provenance["relation_to_load_D_nonoverlap"], "distinct_protocol")
        self.assertFalse(provenance["independent_episode_replicates"])


if __name__ == "__main__":
    unittest.main()
