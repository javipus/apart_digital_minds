#!/usr/bin/env python3
"""Summarize 4o BT-implied preferred-direction probabilities by condition."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from fit_experiment1_bt import (
    DEFAULT_LABELS,
    DEFAULT_RAW,
    DEFAULT_TASKS,
    RQ2_DIR,
    fit_group,
    load_choices,
)


DEFAULT_OUTPUT = RQ2_DIR / "results" / "experiment1_4o_bt_pairwise_probabilities.csv"
CONDITIONS = ("stated", "consequential_tokens", "consequential_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--l2", type=float, default=1e-6)
    return parser.parse_args()


def preferred_probability(score_difference: float) -> float:
    return 1.0 / (1.0 + math.exp(-abs(score_difference)))


def main() -> int:
    args = parse_args()
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    option_ids = sorted(int(task["outcome_id"]) for task in tasks)
    task_text = {int(task["outcome_id"]): task["consequential"] for task in tasks}
    choices, load_metadata = load_choices(args.raw, args.labels)

    rows: list[dict[str, object]] = []
    fit_diagnostics: dict[str, object] = {}
    for condition in CONDITIONS:
        records = [
            record for record in choices
            if record["provider"] == "openai" and record["condition"] == condition
        ]
        scores, diagnostics = fit_group(records, option_ids, "complete_case", args.l2)
        fit_diagnostics[condition] = diagnostics
        score_by_id = dict(zip(option_ids, scores))
        for first_id, second_id in itertools.combinations(option_ids, 2):
            first_score = float(score_by_id[first_id])
            second_score = float(score_by_id[second_id])
            preferred_id = first_id if first_score >= second_score else second_id
            rows.append({
                "condition": condition,
                "task_1_id": first_id,
                "task_1": task_text[first_id],
                "task_2_id": second_id,
                "task_2": task_text[second_id],
                "preferred_task_id": preferred_id,
                "preferred_task": task_text[preferred_id],
                "absolute_bt_score_gap": abs(first_score - second_score),
                "preferred_direction_probability": preferred_probability(first_score - second_score),
            })

    summaries: dict[str, object] = {}
    for condition in CONDITIONS:
        values = np.asarray([
            float(row["preferred_direction_probability"])
            for row in rows if row["condition"] == condition
        ])
        summaries[condition] = {
            "n_pairs": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "quantiles": {
                "p10": float(np.quantile(values, 0.10)),
                "p25": float(np.quantile(values, 0.25)),
                "p75": float(np.quantile(values, 0.75)),
                "p90": float(np.quantile(values, 0.90)),
            },
            "fraction_at_least": {
                "0.60": float(np.mean(values >= 0.60)),
                "two_to_one_odds_0.6667": float(np.mean(values >= 2 / 3)),
                "three_to_one_odds_0.75": float(np.mean(values >= 0.75)),
                "0.90": float(np.mean(values >= 0.90)),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition", "task_1_id", "task_1", "task_2_id", "task_2",
        "preferred_task_id", "preferred_task", "absolute_bt_score_gap",
        "preferred_direction_probability",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "load": load_metadata,
        "fits": fit_diagnostics,
        "summaries": summaries,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "probabilities": str(args.output),
        "summary": str(summary_path),
        "conditions": summaries,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
