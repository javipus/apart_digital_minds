#!/usr/bin/env python3
"""Evaluate how stated-condition BT scores predict 4o consequential choices."""

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


DEFAULT_OUTPUT = RQ2_DIR / "results" / "experiment1_stated_bt_transfer.csv"
CONSEQUENTIAL_CONDITIONS = ("consequential_tokens", "consequential_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--l2", type=float, default=1e-6)
    return parser.parse_args()


def probability_a(scores: np.ndarray, record: dict[str, Any], id_to_index: dict[int, int]) -> float:
    difference = scores[id_to_index[record["task_a_id"]]] - scores[id_to_index[record["task_b_id"]]]
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, difference)))))


def evaluate_probabilities(probabilities_a: list[float], records: list[dict[str, Any]]) -> dict[str, float | int]:
    if len(probabilities_a) != len(records):
        raise ValueError("Probability and record counts differ")
    probabilities_correct: list[float] = []
    hard_correct: list[float] = []
    for probability, record in zip(probabilities_a, records):
        if abs(probability - 0.5) < 1e-12:
            hard_value = 0.5
        elif record["choice"] == "A":
            hard_value = float(probability > 0.5)
        else:
            hard_value = float(probability < 0.5)
        if record["choice"] == "A":
            probabilities_correct.append(probability)
            hard_correct.append(hard_value)
        elif record["choice"] == "B":
            probabilities_correct.append(1.0 - probability)
            hard_correct.append(hard_value)
        else:
            raise ValueError("Evaluation records must be binary")
    clipped = np.clip(np.asarray(probabilities_correct), 1e-15, 1.0)
    return {
        "n_choices": len(records),
        "mean_probability_correct": float(np.mean(clipped)),
        "mean_log_score_nats": float(np.mean(np.log(clipped))),
        "log_loss_nats": float(-np.mean(np.log(clipped))),
        "hard_accuracy": float(np.mean(hard_correct)),
    }


def score_model(
    scores: np.ndarray,
    records: list[dict[str, Any]],
    id_to_index: dict[int, int],
) -> dict[str, float | int]:
    probabilities = [probability_a(scores, record, id_to_index) for record in records]
    return evaluate_probabilities(probabilities, records)


def cross_validated_consequential_baseline(
    records: list[dict[str, Any]],
    option_ids: list[int],
    id_to_index: dict[int, int],
    l2: float,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Hold out one order×sample replicate for every pair in each of 10 folds."""
    all_probabilities: list[float] = []
    all_test_records: list[dict[str, Any]] = []
    fold_diagnostics: list[dict[str, Any]] = []
    for fold in range(10):
        train = [record for record in records if record["order"] * 5 + record["sample"] != fold]
        test = [record for record in records if record["order"] * 5 + record["sample"] == fold]
        scores, diagnostics = fit_group(train, option_ids, "complete_case", l2)
        binary_test = [record for record in test if record["choice"] in {"A", "B"}]
        all_probabilities.extend(probability_a(scores, record, id_to_index) for record in binary_test)
        all_test_records.extend(binary_test)
        fold_diagnostics.append({
            "fold": fold,
            "train_binary_choices": diagnostics["included_records"],
            "test_binary_choices": len(binary_test),
            "fit_converged": diagnostics["converged"],
            "connected_components": diagnostics["connected_components"],
        })
    return evaluate_probabilities(all_probabilities, all_test_records), fold_diagnostics


def cross_validated_position_baseline(
    records: list[dict[str, Any]],
) -> dict[str, float | int]:
    probabilities: list[float] = []
    test_records: list[dict[str, Any]] = []
    for fold in range(10):
        train = [
            record for record in records
            if record["choice"] in {"A", "B"}
            and record["order"] * 5 + record["sample"] != fold
        ]
        test = [
            record for record in records
            if record["choice"] in {"A", "B"}
            and record["order"] * 5 + record["sample"] == fold
        ]
        probability_a_value = sum(record["choice"] == "A" for record in train) / len(train)
        probabilities.extend([probability_a_value] * len(test))
        test_records.extend(test)
    return evaluate_probabilities(probabilities, test_records)


def main() -> int:
    args = parse_args()
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    option_ids = sorted(int(task["outcome_id"]) for task in tasks)
    id_to_index = {option_id: index for index, option_id in enumerate(option_ids)}
    choices, load_metadata = load_choices(args.raw, args.labels)
    openai = [record for record in choices if record["provider"] == "openai"]
    stated = [record for record in openai if record["condition"] == "stated"]
    stated_scores, stated_diagnostics = fit_group(stated, option_ids, "complete_case", args.l2)

    output_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "load": load_metadata,
        "stated_fit": stated_diagnostics,
        "conditions": {},
        "metric_definitions": {
            "mean_probability_correct": "Mean probability assigned to the observed A/B choice.",
            "mean_log_score_nats": "Mean natural log probability assigned to the observed A/B choice; higher is better and 0 is perfect.",
            "hard_accuracy": "Fraction for which the option with probability above 0.5 was observed.",
        },
    }

    for condition in CONSEQUENTIAL_CONDITIONS:
        condition_records = [record for record in openai if record["condition"] == condition]
        binary = [record for record in condition_records if record["choice"] in {"A", "B"}]
        position_a_rate = sum(record["choice"] == "A" for record in binary) / len(binary)

        transfer_metrics = score_model(stated_scores, binary, id_to_index)
        consequential_scores, consequential_diagnostics = fit_group(
            condition_records, option_ids, "complete_case", args.l2
        )
        in_sample_metrics = score_model(consequential_scores, binary, id_to_index)
        cross_validated_metrics, fold_diagnostics = cross_validated_consequential_baseline(
            condition_records, option_ids, id_to_index, args.l2
        )
        position_in_sample_metrics = evaluate_probabilities([position_a_rate] * len(binary), binary)
        position_cross_validated_metrics = cross_validated_position_baseline(condition_records)
        chance_metrics = evaluate_probabilities([0.5] * len(binary), binary)

        for model_source, metrics in (
            ("stated_bt_transfer", transfer_metrics),
            ("consequential_bt_in_sample", in_sample_metrics),
            ("consequential_bt_10fold_cv", cross_validated_metrics),
            ("position_only_in_sample", position_in_sample_metrics),
            ("position_only_10fold_cv", position_cross_validated_metrics),
            ("chance_p_0.5", chance_metrics),
        ):
            output_rows.append({
                "condition": condition,
                "model_source": model_source,
                **metrics,
            })
        diagnostics["conditions"][condition] = {
            "total_records": len(condition_records),
            "binary_records": len(binary),
            "unclear_records": len(condition_records) - len(binary),
            "displayed_option_a_choice_rate": position_a_rate,
            "consequential_full_fit": consequential_diagnostics,
            "cross_validation_folds": fold_diagnostics,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "condition", "model_source", "n_choices", "mean_probability_correct",
            "mean_log_score_nats", "log_loss_nats", "hard_accuracy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    diagnostics_path = args.output.with_name(args.output.stem + "_diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "results": str(args.output),
        "diagnostics": str(diagnostics_path),
        "metrics": output_rows,
        "position_a_rates": {
            condition: diagnostics["conditions"][condition]["displayed_option_a_choice_rate"]
            for condition in CONSEQUENTIAL_CONDITIONS
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
