#!/usr/bin/env python3
"""Compute Laplace-smoothed empirical pairwise choice probabilities for GPT-4o-mini."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from fit_experiment1_bt import DEFAULT_LABELS, DEFAULT_RAW, RQ2_DIR, load_choices


DEFAULT_OUTPUT = RQ2_DIR / "results" / "experiment1_4o_empirical_pair_probabilities.csv"
CONDITIONS = ("stated", "consequential_tokens", "consequential_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "n_pairs": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def first_item_only_null(
    condition_rows: list[dict[str, object]], displayed_a_rate: float
) -> dict[str, object]:
    """Exact expected distribution when only displayed position affects choice."""
    expected_bin_counts = np.zeros(10, dtype=float)
    expected_strength_bin_counts = np.zeros(10, dtype=float)
    expected_strength_sum = 0.0
    expected_2_to_1 = 0.0
    expected_3_to_1 = 0.0
    for row in condition_rows:
        n = int(row["n_choices"])
        first_order_n = int(row["order_0_n"])
        reverse_order_n = n - first_order_n
        for first_wins in range(first_order_n + 1):
            first_probability = (
                math.comb(first_order_n, first_wins)
                * displayed_a_rate ** first_wins
                * (1 - displayed_a_rate) ** (first_order_n - first_wins)
            )
            for reverse_wins in range(reverse_order_n + 1):
                # In the reversed display order, canonical A is displayed B.
                reverse_probability = (
                    math.comb(reverse_order_n, reverse_wins)
                    * (1 - displayed_a_rate) ** reverse_wins
                    * displayed_a_rate ** (reverse_order_n - reverse_wins)
                )
                mass = first_probability * reverse_probability
                laplace_probability = (first_wins + reverse_wins + 1) / (n + 2)
                strength = max(laplace_probability, 1 - laplace_probability)
                bin_index = min(9, int(laplace_probability * 10))
                strength_bin_index = min(9, int((strength - 0.5) / 0.05 + 1e-10))
                expected_bin_counts[bin_index] += mass
                expected_strength_bin_counts[strength_bin_index] += mass
                expected_strength_sum += mass * strength
                expected_2_to_1 += mass * (strength >= 2 / 3 - 1e-12)
                expected_3_to_1 += mass * (strength >= 3 / 4 - 1e-12)
    n_pairs = len(condition_rows)
    return {
        "expected_probability_histogram_10pp_bins": expected_bin_counts.tolist(),
        "expected_strength_histogram_5pp_bins_from_0.50": expected_strength_bin_counts.tolist(),
        "expected_mean_strength": expected_strength_sum / n_pairs,
        "expected_fraction_at_least_2_to_1": expected_2_to_1 / n_pairs,
        "expected_fraction_at_least_3_to_1": expected_3_to_1 / n_pairs,
    }


def bootstrap_mean_strength_p_value(
    condition_rows: list[dict[str, object]],
    displayed_a_rate: float,
    observed_mean_strength: float,
    replicates: int,
    rng: np.random.Generator,
) -> float:
    """Parametric-bootstrap upper-tail p-value under the position-only null."""
    sample_sizes = np.asarray([int(row["n_choices"]) for row in condition_rows])
    first_order_sizes = np.asarray([int(row["order_0_n"]) for row in condition_rows])
    exceedances = 0
    chunk_size = 2_000
    for start in range(0, replicates, chunk_size):
        chunk = min(chunk_size, replicates - start)
        canonical_wins = rng.binomial(
            first_order_sizes, displayed_a_rate, size=(chunk, len(condition_rows))
        )
        canonical_wins += rng.binomial(
            sample_sizes - first_order_sizes,
            1 - displayed_a_rate,
            size=(chunk, len(condition_rows)),
        )
        probabilities = (canonical_wins + 1) / (sample_sizes + 2)
        simulated_means = np.maximum(probabilities, 1 - probabilities).mean(axis=1)
        exceedances += int(np.sum(simulated_means >= observed_mean_strength - 1e-15))
    return (exceedances + 1) / (replicates + 1)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    choices, load_metadata = load_choices(args.raw, args.labels)

    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for record in choices:
        if record["provider"] != "openai" or record["condition"] not in CONDITIONS:
            continue
        if record["choice"] not in {"A", "B"}:
            continue
        canonical_a, canonical_b = sorted((record["task_a_id"], record["task_b_id"]))
        grouped[(record["condition"], canonical_a, canonical_b)].append(record)

    rows: list[dict[str, object]] = []
    for (condition, canonical_a, canonical_b), records in sorted(grouped.items()):
        n = len(records)
        canonical_a_wins = 0
        displayed_a_wins = 0
        order_zero_n = 0
        order_one_n = 0
        for record in records:
            displayed_a_wins += record["choice"] == "A"
            chose_canonical_a = (
                (record["task_a_id"] == canonical_a and record["choice"] == "A")
                or (record["task_b_id"] == canonical_a and record["choice"] == "B")
            )
            canonical_a_wins += chose_canonical_a
            order_zero_n += record["order"] == 0
            order_one_n += record["order"] == 1

        probability = (canonical_a_wins + 1) / (n + 2)
        displayed_a_probability = (displayed_a_wins + 1) / (n + 2)
        rows.append({
            "condition": condition,
            "canonical_task_a_id": canonical_a,
            "canonical_task_b_id": canonical_b,
            "n_choices": n,
            "order_0_n": order_zero_n,
            "order_1_n": order_one_n,
            "canonical_a_wins": canonical_a_wins,
            "laplace_p_choose_canonical_a": probability,
            "laplace_preference_strength": max(probability, 1 - probability),
            "displayed_a_wins": displayed_a_wins,
            "laplace_p_choose_displayed_a": displayed_a_probability,
        })

    summaries: dict[str, object] = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        probabilities = [float(row["laplace_p_choose_canonical_a"]) for row in condition_rows]
        strengths = [float(row["laplace_preference_strength"]) for row in condition_rows]
        observed_mean_strength = float(np.mean(strengths))
        observed_strength_bins = np.zeros(10, dtype=int)
        for strength in strengths:
            observed_strength_bins[min(9, int((strength - 0.5) / 0.05 + 1e-10))] += 1
        n_choices = [int(row["n_choices"]) for row in condition_rows]
        displayed_a_wins = sum(int(row["displayed_a_wins"]) for row in condition_rows)
        total_choices = sum(n_choices)
        displayed_a_rate = displayed_a_wins / total_choices
        summaries[condition] = {
            "canonical_probability": summarize(probabilities),
            "preference_strength": {
                **summarize(strengths),
                "fraction_at_least_2_to_1": float(np.mean(np.asarray(strengths) >= 2 / 3)),
                "fraction_at_least_3_to_1": float(np.mean(np.asarray(strengths) >= 3 / 4)),
                "histogram_5pp_bins_from_0.50": observed_strength_bins.tolist(),
            },
            "displayed_a_raw_choice_rate": displayed_a_rate,
            "pair_sample_sizes": dict(sorted({str(n): n_choices.count(n) for n in set(n_choices)}.items())),
            "first_item_only_null": first_item_only_null(condition_rows, displayed_a_rate),
            "position_only_parametric_bootstrap": {
                "statistic": "mean Laplace-smoothed folded pairwise probability",
                "replicates": args.bootstrap_replicates,
                "seed": args.seed,
                "upper_tail_p_value": bootstrap_mean_strength_p_value(
                    condition_rows,
                    displayed_a_rate,
                    observed_mean_strength,
                    args.bootstrap_replicates,
                    rng,
                ),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition", "canonical_task_a_id", "canonical_task_b_id", "n_choices",
        "order_0_n", "order_1_n", "canonical_a_wins",
        "laplace_p_choose_canonical_a", "laplace_preference_strength",
        "displayed_a_wins", "laplace_p_choose_displayed_a",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "definition": {
            "canonical_a": "The lower outcome ID, held fixed across the two display orders.",
            "laplace_probability": "(canonical A wins + 1) / (binary choices + 2)",
            "strength": "max(P(canonical A), 1 - P(canonical A))",
        },
        "load": load_metadata,
        "summaries": summaries,
    }, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"output": str(args.output), "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
