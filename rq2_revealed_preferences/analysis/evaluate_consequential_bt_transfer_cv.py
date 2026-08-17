#!/usr/bin/env python3
"""10-fold BT prediction within and across 4o consequential conditions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from fit_experiment1_bt import (
    DEFAULT_LABELS,
    DEFAULT_RAW,
    DEFAULT_TASKS,
    RQ2_DIR,
    fit_group,
    load_choices,
)


DEFAULT_OUTPUT = RQ2_DIR / "results" / "experiment1_consequential_bt_transfer_cv.csv"
CONDITIONS = ("consequential_tokens", "consequential_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--l2", type=float, default=1e-6)
    return parser.parse_args()


def fold_id(record: dict[str, Any]) -> int:
    return int(record["order"]) * 5 + int(record["sample"])


def probability_a(scores: np.ndarray, record: dict[str, Any], id_to_index: dict[int, int]) -> float:
    difference = scores[id_to_index[record["task_a_id"]]] - scores[id_to_index[record["task_b_id"]]]
    bounded = max(-40.0, min(40.0, float(difference)))
    return 1.0 / (1.0 + math.exp(-bounded))


def evaluate(probabilities_a: list[float], records: list[dict[str, Any]]) -> dict[str, float | int]:
    probability_correct: list[float] = []
    hard_correct: list[float] = []
    for probability, record in zip(probabilities_a, records):
        if record["choice"] == "A":
            probability_correct.append(probability)
            hard_correct.append(float(probability > 0.5) if probability != 0.5 else 0.5)
        elif record["choice"] == "B":
            probability_correct.append(1.0 - probability)
            hard_correct.append(float(probability < 0.5) if probability != 0.5 else 0.5)
        else:
            raise ValueError("Only binary choices can be evaluated")
    values = np.clip(np.asarray(probability_correct), 1e-15, 1.0)
    mean_log_score_bits = float(np.mean(np.log2(values)))
    return {
        "n_test_choices": len(records),
        "mean_probability_correct": float(np.mean(values)),
        "hard_accuracy": float(np.mean(hard_correct)),
        "mean_log_score_bits": mean_log_score_bits,
        "log_score_bits_above_chance": mean_log_score_bits + 1.0,
    }


def cross_validate(
    source_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    option_ids: list[int],
    id_to_index: dict[int, int],
    l2: float,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    probabilities: list[float] = []
    tests: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for fold in range(10):
        train = [record for record in source_records if fold_id(record) != fold]
        test = [
            record for record in target_records
            if fold_id(record) == fold and record["choice"] in {"A", "B"}
        ]
        scores, fit_diagnostics = fit_group(train, option_ids, "complete_case", l2)
        probabilities.extend(probability_a(scores, record, id_to_index) for record in test)
        tests.extend(test)
        diagnostics.append({
            "fold": fold,
            "train_total": len(train),
            "train_binary": fit_diagnostics["included_records"],
            "test_binary": len(test),
            "fit_converged": fit_diagnostics["converged"],
            "connected_components": fit_diagnostics["connected_components"],
        })
    return evaluate(probabilities, tests), diagnostics


def main() -> int:
    args = parse_args()
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    option_ids = sorted(int(task["outcome_id"]) for task in tasks)
    id_to_index = {option_id: index for index, option_id in enumerate(option_ids)}
    choices, load_metadata = load_choices(args.raw, args.labels)
    records = {
        condition: [
            record for record in choices
            if record["provider"] == "openai" and record["condition"] == condition
        ]
        for condition in CONDITIONS
    }

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "load": load_metadata,
        "fold_definition": "order * 5 + sample; each fold tests one replicate of every task pair",
        "log_score_definition": "mean(log2(p_observed)) + 1 bit, so p=0.5 chance is 0 and perfect prediction is 1",
        "evaluations": {},
    }
    for target_condition in CONDITIONS:
        for source_condition in CONDITIONS:
            metrics, fold_diagnostics = cross_validate(
                records[source_condition], records[target_condition], option_ids, id_to_index, args.l2
            )
            relation = "within_condition_baseline" if source_condition == target_condition else "cross_condition_transfer"
            row = {
                "training_condition": source_condition,
                "test_condition": target_condition,
                "relationship": relation,
                **metrics,
            }
            rows.append(row)
            diagnostics["evaluations"][f"{source_condition} -> {target_condition}"] = fold_diagnostics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "training_condition", "test_condition", "relationship", "n_test_choices",
        "mean_probability_correct", "hard_accuracy", "mean_log_score_bits",
        "log_score_bits_above_chance",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    diagnostics_path = args.output.with_name(args.output.stem + "_diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "results": str(args.output),
        "diagnostics": str(diagnostics_path),
        "metrics": rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
