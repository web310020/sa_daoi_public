"""Reproduce the block-level Load-D comparison used in the paper."""

import argparse
import csv
import datetime
from pathlib import Path
import statistics

from agents.dqn_agent import DQNPolicy
from agents.heuristics import StaticRRPolicy, TAoIPolicy, WhittlePolicy
from agents.ppo_cppo import CPPOPolicy, PPOPolicy
from agents.sa_daoi import SADAOIScheduler
from env.vehicular import VehicularNetworkEnv
from eval.evaluator import evaluate


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "load_D"
REFERENCE_PATH = ROOT / "reference_results" / "load_D" / "load_D_blocks.csv"
BLOCK_STARTS = (1000, 2000, 3000, 4000, 5000)
EPISODES_PER_BLOCK = 30
TIER = "D"
METHOD_ORDER = (
    "SA-DAoI",
    "Static-RR",
    "T-AoI",
    "Whittle",
    "DQN-AoI",
    "PPO-AoI",
    "C-PPO",
)
INTEGER_FIELDS = (
    "block_index",
    "base_seed",
    "episode_count",
    "unique_environment_seed_count",
)
FLOAT_FIELDS = (
    "aoi_safety_ms",
    "violation_rate",
    "queue_ce",
    "queue_iot",
    "es_backlog",
)


def make_method(name, environment):
    """Construct one fresh policy for one deployment block."""
    if name == "SA-DAoI":
        return SADAOIScheduler(environment)
    if name == "Static-RR":
        return StaticRRPolicy(environment)
    if name == "T-AoI":
        return TAoIPolicy()
    if name == "Whittle":
        return WhittlePolicy()
    if name == "DQN-AoI":
        return DQNPolicy(environment, str(MODEL_DIR), TIER)
    if name == "PPO-AoI":
        return PPOPolicy(environment, str(MODEL_DIR), TIER)
    if name == "C-PPO":
        return CPPOPolicy(environment, str(MODEL_DIR), TIER)
    raise ValueError(f"unknown method: {name}")


def evaluate_block(method, block_index, base_seed, episodes_per_block):
    """Evaluate one fresh policy over one sequential environment-seed block."""
    environment = VehicularNetworkEnv(load_tier=TIER)
    environment.reset(seed=base_seed)
    policy = make_method(method, environment)
    result = evaluate(
        policy,
        TIER,
        num_episodes=episodes_per_block,
        seed=base_seed,
    )
    queue_ce = result["queue_len_CE"]
    queue_iot = result["queue_len_IOT"]
    return {
        "method": method,
        "block_index": block_index,
        "base_seed": base_seed,
        "episode_count": episodes_per_block,
        "unique_environment_seed_count": episodes_per_block,
        "aoi_safety_ms": result["aoi_total_SAFETY"],
        "violation_rate": result["violation_rate_SAFETY"],
        "queue_ce": queue_ce,
        "queue_iot": queue_iot,
        "es_backlog": queue_ce + queue_iot,
    }


def run(methods=METHOD_ORDER, block_starts=BLOCK_STARTS, episodes_per_block=EPISODES_PER_BLOCK):
    """Return one aggregate row per method and independent deployment block."""
    rows = []
    for method in methods:
        for block_index, base_seed in enumerate(block_starts, start=1):
            row = evaluate_block(method, block_index, base_seed, episodes_per_block)
            rows.append(row)
            print(
                f"{method:10s} block={block_index} seed={base_seed} "
                f"AoI={row['aoi_safety_ms']:.3f} ms "
                f"viol={100 * row['violation_rate']:.2f}% "
                f"ES={row['es_backlog']:.1f}"
            )
    return rows


def load_reference_rows(path=REFERENCE_PATH):
    """Load the frozen block carrier with explicit numeric types."""
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field in INTEGER_FIELDS:
                row[field] = int(row[field])
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
            rows.append(row)
    return rows


def compare_to_reference(rows, path=REFERENCE_PATH, absolute_tolerance=1e-12):
    """Return human-readable mismatches against the frozen block carrier."""
    reference = load_reference_rows(path)
    actual_by_key = {(row["method"], int(row["block_index"])): row for row in rows}
    reference_by_key = {
        (row["method"], int(row["block_index"])): row for row in reference
    }
    mismatches = []
    if set(actual_by_key) != set(reference_by_key):
        missing = sorted(set(reference_by_key) - set(actual_by_key))
        unexpected = sorted(set(actual_by_key) - set(reference_by_key))
        if missing:
            mismatches.append(f"missing rows: {missing}")
        if unexpected:
            mismatches.append(f"unexpected rows: {unexpected}")
    for key in sorted(set(actual_by_key) & set(reference_by_key)):
        actual = actual_by_key[key]
        expected = reference_by_key[key]
        for field in INTEGER_FIELDS:
            if int(actual[field]) != int(expected[field]):
                mismatches.append(
                    f"{key} {field}: {actual[field]} != {expected[field]}"
                )
        for field in FLOAT_FIELDS:
            difference = abs(float(actual[field]) - float(expected[field]))
            if difference > absolute_tolerance:
                mismatches.append(
                    f"{key} {field}: {actual[field]} != {expected[field]} "
                    f"(abs diff {difference})"
                )
    return mismatches


def write_rows(rows, output_dir):
    """Write block rows and a compact descriptive summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "load_D_blocks.csv"
    fieldnames = list(rows[0])
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.txt"
    lines = [
        "Load-D block-level reproduction",
        "Statistical unit: n=5 deployment blocks; 30 sequential episodes per block",
        "",
    ]
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        aois = [row["aoi_safety_ms"] for row in selected]
        violations = [100 * row["violation_rate"] for row in selected]
        backlogs = [row["es_backlog"] for row in selected]
        if len(selected) >= 2:
            lines.append(
                f"{method}: AoI {statistics.mean(aois):.2f} +/- {statistics.stdev(aois):.2f} ms; "
                f"violations {statistics.mean(violations):.2f} +/- {statistics.stdev(violations):.2f}%; "
                f"ES backlog {statistics.mean(backlogs):.1f} +/- {statistics.stdev(backlogs):.1f}"
            )
        else:
            lines.append(
                f"{method}: AoI {aois[0]:.2f} ms; violations {violations[0]:.2f}%; "
                f"ES backlog {backlogs[0]:.1f}; sample SD unavailable (one block)"
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, summary_path


def default_output_dir():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / f"load_D_reproduction_{stamp}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one episode for SA-DAoI and the three checkpoint-backed policies.",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        methods = ("SA-DAoI", "DQN-AoI", "PPO-AoI", "C-PPO")
        block_starts = (BLOCK_STARTS[0],)
        episodes_per_block = 1
    else:
        methods = METHOD_ORDER
        block_starts = BLOCK_STARTS
        episodes_per_block = EPISODES_PER_BLOCK
    rows = run(methods, block_starts, episodes_per_block)
    output_dir = args.output_dir or default_output_dir()
    csv_path, summary_path = write_rows(rows, output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    if not args.smoke:
        mismatches = compare_to_reference(rows)
        if mismatches:
            print("Reference verification failed:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
            return 1
        print("Reference verification passed: 35/35 block rows match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
