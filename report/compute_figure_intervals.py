#!/usr/bin/env python3
"""Compute uncertainty intervals used in the submission figures."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "rq2_revealed_preferences" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from fit_experiment1_bt import fit_bradley_terry, fit_group, load_choices  # noqa: E402


TASKS = ROOT / "rq2_revealed_preferences" / "experiment_1" / "tasks.json"
OUTPUT = ROOT / "report" / "figure_stats.json"
BOOTSTRAP_REPLICATES = 10_000
SEED = 20260817
CONDITIONS = ("stated", "consequential_tokens", "consequential_time")
MODELS = {
    "4o-mini": {
        "raw": ROOT / "rq2_revealed_preferences" / "experiment_1" / "data" / "raw_responses.jsonl",
        "labels": ROOT / "rq2_revealed_preferences" / "experiment_1" / "data" / "consequential_labels.jsonl",
        "empirical": ROOT / "rq2_revealed_preferences" / "results" / "4o-mini" / "experiment1_empirical_pair_probabilities.csv",
    },
    "Luna": {
        "raw": ROOT / "rq2_revealed_preferences" / "experiment_1" / "data" / "luna_raw_responses.jsonl",
        "labels": ROOT / "rq2_revealed_preferences" / "experiment_1" / "data" / "luna_consequential_labels_luna.jsonl",
        "empirical": ROOT / "rq2_revealed_preferences" / "results" / "luna" / "experiment1_empirical_pair_probabilities.csv",
    },
    "Terra": {
        "raw": ROOT / "rq2_revealed_preferences" / "experiment_1" / "data" / "terra_raw_responses.jsonl",
        "labels": ROOT / "rq2_revealed_preferences" / "experiment_1" / "data" / "terra_consequential_labels_luna.jsonl",
        "empirical": ROOT / "rq2_revealed_preferences" / "results" / "terra" / "experiment1_empirical_pair_probabilities.csv",
    },
}


def fold_id(record: dict[str, Any]) -> int:
    return int(record["order"]) * 5 + int(record["sample"])


def probability_a(scores: np.ndarray, record: dict[str, Any], id_to_index: dict[int, int]) -> float:
    difference = float(
        scores[id_to_index[int(record["task_a_id"])]]
        - scores[id_to_index[int(record["task_b_id"])]]
    )
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, difference))))


def pair_bootstrap_interval(
    values_by_pair: dict[tuple[int, int], tuple[float, int]],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    sums = np.asarray([value[0] for value in values_by_pair.values()], dtype=float)
    counts = np.asarray([value[1] for value in values_by_pair.values()], dtype=float)
    point = float(sums.sum() / counts.sum())
    sample_indices = rng.integers(0, len(sums), size=(BOOTSTRAP_REPLICATES, len(sums)))
    samples = sums[sample_indices].sum(axis=1) / counts[sample_indices].sum(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return point, float(lower), float(upper)


def mean_pair_bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    sample_indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    samples = values[sample_indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def accuracy_predictions(
    source: list[dict[str, Any]],
    target: list[dict[str, Any]],
    option_ids: list[int],
    id_to_index: dict[int, int],
) -> dict[tuple[int, int], tuple[float, int]]:
    pair_sums: dict[tuple[int, int], float] = defaultdict(float)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for fold in range(10):
        train = [record for record in source if fold_id(record) != fold]
        test = [
            record
            for record in target
            if fold_id(record) == fold and record["choice"] in {"A", "B"}
        ]
        scores, _ = fit_group(train, option_ids, "complete_case", 1e-6)
        for record in test:
            probability = probability_a(scores, record, id_to_index)
            if probability == 0.5:
                correct = 0.5
            elif record["choice"] == "A":
                correct = float(probability > 0.5)
            else:
                correct = float(probability < 0.5)
            pair = tuple(sorted((int(record["task_a_id"]), int(record["task_b_id"]))))
            pair_sums[pair] += correct
            pair_counts[pair] += 1
    return {pair: (pair_sums[pair], pair_counts[pair]) for pair in pair_sums}


def correlation_bootstrap_intervals(
    records: dict[str, list[dict[str, Any]]],
    option_ids: list[int],
    id_to_index: dict[int, int],
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Refit BT after resampling responses within pair/order/condition strata."""
    strata: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    point_scores: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        counts: dict[tuple[int, int, int], list[int]] = defaultdict(lambda: [0, 0])
        for record in records[condition]:
            if record["choice"] not in {"A", "B"}:
                continue
            key = (
                int(record["task_a_id"]),
                int(record["task_b_id"]),
                int(record["order"]),
            )
            counts[key][0] += int(record["choice"] == "A")
            counts[key][1] += 1
        keys = sorted(counts)
        left = np.asarray([id_to_index[key[0]] for key in keys], dtype=np.int64)
        right = np.asarray([id_to_index[key[1]] for key in keys], dtype=np.int64)
        wins = np.asarray([counts[key][0] for key in keys], dtype=np.float64)
        totals = np.asarray([counts[key][1] for key in keys], dtype=np.float64)
        strata[condition] = left, right, wins, totals
        point_scores[condition] = fit_bradley_terry(
            len(option_ids), left, right, wins, totals, l2=1e-6
        )[0]

    pairs = (
        ("stated", "consequential_tokens"),
        ("stated", "consequential_time"),
        ("consequential_tokens", "consequential_time"),
    )
    samples = {pair: np.empty(BOOTSTRAP_REPLICATES, dtype=float) for pair in pairs}
    for replicate in range(BOOTSTRAP_REPLICATES):
        bootstrap_scores: dict[str, np.ndarray] = {}
        for condition, (left, right, wins, totals) in strata.items():
            bootstrap_wins = rng.binomial(totals.astype(np.int64), wins / totals)
            bootstrap_scores[condition] = fit_bradley_terry(
                len(option_ids), left, right, bootstrap_wins, totals, l2=1e-6
            )[0]
        for pair in pairs:
            samples[pair][replicate] = np.corrcoef(
                bootstrap_scores[pair[0]], bootstrap_scores[pair[1]]
            )[0, 1]

    output: dict[str, dict[str, float]] = {}
    for condition_x, condition_y in pairs:
        values = samples[(condition_x, condition_y)]
        lower, upper = np.quantile(values, [0.025, 0.975])
        output[f"{condition_x}__{condition_y}"] = {
            "point": float(np.corrcoef(point_scores[condition_x], point_scores[condition_y])[0, 1]),
            "lower": float(lower),
            "upper": float(upper),
        }
    return output


def load_strength_intervals(path: Path, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[row["condition"]].append(float(row["laplace_preference_strength"]))
    return {
        condition: dict(zip(("point", "lower", "upper"), mean_pair_bootstrap_interval(np.asarray(items), rng)))
        for condition, items in values.items()
    }


def main() -> int:
    rng = np.random.default_rng(SEED)
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))
    option_ids = sorted(int(task["outcome_id"]) for task in tasks)
    id_to_index = {option_id: index for index, option_id in enumerate(option_ids)}
    output: dict[str, Any] = {
        "definitions": {
            "correlation_interval": "95% stratified response-bootstrap interval with BT refitting",
            "accuracy_interval": "95% nonparametric bootstrap interval resampling the 351 task pairs",
            "strength_interval": "95% nonparametric bootstrap interval resampling the 351 task pairs",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
        },
        "models": {},
    }

    for model, paths in MODELS.items():
        choices, _ = load_choices(paths["raw"], paths["labels"])
        records = {
            condition: [
                record
                for record in choices
                if record["provider"] == "openai" and record["condition"] == condition
            ]
            for condition in CONDITIONS
        }
        accuracies: dict[str, dict[str, float]] = {}
        for source, target in [
            ("stated", "consequential_tokens"),
            ("consequential_tokens", "consequential_tokens"),
            ("stated", "consequential_time"),
            ("consequential_time", "consequential_time"),
        ]:
            predictions = accuracy_predictions(
                records[source], records[target], option_ids, id_to_index
            )
            point, lower, upper = pair_bootstrap_interval(predictions, rng)
            accuracies[f"{source}__{target}"] = {
                "point": point,
                "lower": lower,
                "upper": upper,
                "n_pairs": len(predictions),
                "n_choices": sum(count for _, count in predictions.values()),
            }
        output["models"][model] = {
            "pearson": correlation_bootstrap_intervals(
                records, option_ids, id_to_index, rng
            ),
            "hard_accuracy": accuracies,
            "preference_strength": load_strength_intervals(paths["empirical"], rng),
        }

    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
