#!/usr/bin/env python3
"""Fit Bradley-Terry models to Experiment 1 and compare condition scores."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RQ2_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = RQ2_DIR / "experiment_1"
DEFAULT_RAW = EXPERIMENT_DIR / "data" / "raw_responses.jsonl"
DEFAULT_LABELS = EXPERIMENT_DIR / "data" / "consequential_labels.jsonl"
DEFAULT_TASKS = EXPERIMENT_DIR / "tasks.json"
DEFAULT_OUTPUT_DIR = RQ2_DIR / "results"
CONDITIONS = ("stated", "consequential_tokens", "consequential_time")
PROVIDERS = ("pooled", "openai", "anthropic")
HANDLING_METHODS = ("complete_case", "half_unclear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--l2", type=float, default=1e-6)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def load_choices(raw_path: Path, labels_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load stated parser choices and the last successful consequential label."""
    raw_records = read_jsonl(raw_path)
    raw_by_id = {
        record["request_id"]: record
        for record in raw_records
        if record.get("status") == "ok"
    }

    choices: list[dict[str, Any]] = []
    for record in raw_by_id.values():
        if record.get("condition") != "stated":
            continue
        choice = record.get("parsed_choice")
        if choice not in {"A", "B"}:
            raise ValueError(f"Unparsed stated response: {record['request_id']}")
        choices.append({
            "request_id": record["request_id"],
            "provider": record["provider"],
            "condition": "stated",
            "task_a_id": int(record["task_a_id"]),
            "task_b_id": int(record["task_b_id"]),
            "order": int(record["order"]),
            "sample": int(record["sample"]),
            "choice": choice,
        })

    label_records = read_jsonl(labels_path)
    successful_labels: dict[str, dict[str, Any]] = {}
    failed_source_ids: set[str] = set()
    for record in label_records:
        source_id = record.get("source_request_id")
        if not source_id:
            continue
        if record.get("status") == "ok":
            successful_labels[source_id] = record
        else:
            failed_source_ids.add(source_id)

    for record in successful_labels.values():
        choices.append({
            "request_id": record["source_request_id"],
            "provider": record["source_provider"],
            "condition": record["condition"],
            "task_a_id": int(record["task_a_id"]),
            "task_b_id": int(record["task_b_id"]),
            "order": int(record["order"]),
            "sample": int(record["sample"]),
            "choice": record["choice"],
        })

    expected_consequential = {
        request_id
        for request_id, record in raw_by_id.items()
        if record.get("condition") in {"consequential_tokens", "consequential_time"}
    }
    missing_labels = sorted(expected_consequential - set(successful_labels))
    parser_fallbacks: list[str] = []
    unresolved_missing_labels: list[str] = []
    for request_id in missing_labels:
        record = raw_by_id[request_id]
        choice = record.get("parsed_choice")
        if choice not in {"A", "B"}:
            unresolved_missing_labels.append(request_id)
            continue
        parser_fallbacks.append(request_id)
        choices.append({
            "request_id": request_id,
            "provider": record["provider"],
            "condition": record["condition"],
            "task_a_id": int(record["task_a_id"]),
            "task_b_id": int(record["task_b_id"]),
            "order": int(record["order"]),
            "sample": int(record["sample"]),
            "choice": choice,
        })
    metadata = {
        "raw_successful_requests": len(raw_by_id),
        "stated_choices": sum(choice["condition"] == "stated" for choice in choices),
        "successful_consequential_labels": len(successful_labels),
        "missing_label_count_before_parser_fallback": len(missing_labels),
        "source_parser_fallback_count": len(parser_fallbacks),
        "source_parser_fallback_request_ids": parser_fallbacks,
        "unresolved_missing_label_count": len(unresolved_missing_labels),
        "unresolved_missing_source_request_ids": unresolved_missing_labels,
        "failed_label_source_ids": sorted(failed_source_ids),
        "physical_label_records": len(label_records),
    }
    return choices, metadata


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def fit_bradley_terry(
    n_options: int,
    left: np.ndarray,
    right: np.ndarray,
    wins_left: np.ndarray,
    totals: np.ndarray,
    l2: float,
    max_iterations: int = 200,
    gradient_tolerance: float = 1e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a binomial Bradley-Terry model with damped Newton updates."""
    scores = np.zeros(n_options, dtype=np.float64)

    def objective(candidate: np.ndarray) -> float:
        differences = candidate[left] - candidate[right]
        losses = totals - wins_left
        nll = np.sum(
            wins_left * np.logaddexp(0.0, -differences)
            + losses * np.logaddexp(0.0, differences)
        )
        return float(nll + 0.5 * l2 * np.dot(candidate[:-1], candidate[:-1]))

    converged = False
    previous_objective = objective(scores)
    gradient_max = math.inf
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        differences = scores[left] - scores[right]
        probabilities = sigmoid(differences)
        residuals = totals * probabilities - wins_left
        curvature = totals * probabilities * (1.0 - probabilities)

        gradient = np.zeros(n_options, dtype=np.float64)
        np.add.at(gradient, left, residuals)
        np.add.at(gradient, right, -residuals)
        gradient[:-1] += l2 * scores[:-1]

        hessian = np.zeros((n_options - 1, n_options - 1), dtype=np.float64)
        left_free = left < n_options - 1
        right_free = right < n_options - 1
        np.add.at(hessian, (left[left_free], left[left_free]), curvature[left_free])
        np.add.at(hessian, (right[right_free], right[right_free]), curvature[right_free])
        both_free = left_free & right_free
        np.add.at(hessian, (left[both_free], right[both_free]), -curvature[both_free])
        np.add.at(hessian, (right[both_free], left[both_free]), -curvature[both_free])
        hessian.flat[::n_options] += l2

        gradient_max = float(np.max(np.abs(gradient[:-1])))
        step = np.linalg.solve(hessian, gradient[:-1])
        accepted = False
        step_scale = 1.0
        for _ in range(30):
            candidate = scores.copy()
            candidate[:-1] -= step_scale * step
            candidate_objective = objective(candidate)
            if candidate_objective < previous_objective:
                scores = candidate
                previous_objective = candidate_objective
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            if gradient_max < gradient_tolerance:
                converged = True
            break
        if gradient_max < gradient_tolerance or float(np.max(np.abs(step_scale * step))) < 1e-10:
            converged = True
            break

    scores -= scores.mean()
    return scores, {
        "converged": converged,
        "iterations": iteration,
        "negative_log_likelihood": previous_objective,
        "max_absolute_gradient": gradient_max,
        "l2_penalty": l2,
    }


def connected_components(n_options: int, edges: set[tuple[int, int]]) -> int:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(range(n_options))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbors = adjacency[node] & unseen
            unseen -= neighbors
            stack.extend(neighbors)
    return components


def fit_group(
    records: list[dict[str, Any]],
    option_ids: list[int],
    handling: str,
    l2: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    id_to_index = {option_id: index for index, option_id in enumerate(option_ids)}
    left: list[int] = []
    right: list[int] = []
    wins_left: list[float] = []
    totals: list[float] = []
    included_edges: set[tuple[int, int]] = set()
    unclear = 0

    for record in records:
        choice = record["choice"]
        if choice == "UNCLEAR":
            unclear += 1
            if handling == "complete_case":
                continue
            win = 0.5
        elif choice == "A":
            win = 1.0
        elif choice == "B":
            win = 0.0
        else:
            raise ValueError(f"Unexpected choice: {choice}")
        left_index = id_to_index[record["task_a_id"]]
        right_index = id_to_index[record["task_b_id"]]
        left.append(left_index)
        right.append(right_index)
        wins_left.append(win)
        totals.append(1.0)
        included_edges.add(tuple(sorted((left_index, right_index))))

    arrays = (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(wins_left, dtype=np.float64),
        np.asarray(totals, dtype=np.float64),
    )
    scores, fit_diagnostics = fit_bradley_terry(len(option_ids), *arrays, l2=l2)
    diagnostics = {
        **fit_diagnostics,
        "total_records": len(records),
        "included_records": len(left),
        "unclear_records": unclear,
        "unclear_rate": unclear / len(records) if records else math.nan,
        "observed_pair_count": len(included_edges),
        "possible_pair_count": len(option_ids) * (len(option_ids) - 1) // 2,
        "connected_components": connected_components(len(option_ids), included_edges),
    }
    return scores, diagnostics


def rank_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pearson = float(np.corrcoef(x, y)[0, 1])
    rank_x = rank_descending(x).astype(float)
    rank_y = rank_descending(y).astype(float)
    spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return {"pearson_r": pearson, "r_squared": pearson**2, "spearman_rho": spearman}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    option_ids = sorted(int(task["outcome_id"]) for task in tasks)
    task_text = {int(task["outcome_id"]): task["consequential"] for task in tasks}
    choices, load_metadata = load_choices(args.raw, args.labels)
    provider_scopes = [
        provider_scope
        for provider_scope in PROVIDERS
        if provider_scope == "pooled"
        or any(record["provider"] == provider_scope for record in choices)
    ]

    scores_by_key: dict[tuple[str, str, str], np.ndarray] = {}
    diagnostics: dict[str, Any] = {"load": load_metadata, "fits": {}}
    score_rows: list[dict[str, Any]] = []

    for provider_scope in provider_scopes:
        for handling in HANDLING_METHODS:
            for condition in CONDITIONS:
                group = [
                    record for record in choices
                    if record["condition"] == condition
                    and (provider_scope == "pooled" or record["provider"] == provider_scope)
                ]
                scores, fit_diagnostics = fit_group(group, option_ids, handling, args.l2)
                key = (provider_scope, handling, condition)
                scores_by_key[key] = scores
                diagnostics["fits"][" / ".join(key)] = fit_diagnostics
                z_scores = (scores - scores.mean()) / scores.std(ddof=0)
                ranks = rank_descending(scores)
                for option_id, score, z_score, rank in zip(option_ids, scores, z_scores, ranks):
                    score_rows.append({
                        "provider_scope": provider_scope,
                        "unclear_handling": handling,
                        "condition": condition,
                        "outcome_id": option_id,
                        "task": task_text[option_id],
                        "bt_score": float(score),
                        "z_score": float(z_score),
                        "rank": int(rank),
                    })

    correlation_rows: list[dict[str, Any]] = []
    for provider_scope in provider_scopes:
        for handling in HANDLING_METHODS:
            for condition_x, condition_y in itertools.combinations(CONDITIONS, 2):
                values = correlation(
                    scores_by_key[(provider_scope, handling, condition_x)],
                    scores_by_key[(provider_scope, handling, condition_y)],
                )
                correlation_rows.append({
                    "provider_scope": provider_scope,
                    "unclear_handling": handling,
                    "condition_x": condition_x,
                    "condition_y": condition_y,
                    "n_tasks": len(option_ids),
                    **values,
                })

    score_path = args.output_dir / "experiment1_bt_scores.csv"
    correlation_path = args.output_dir / "experiment1_bt_correlations.csv"
    diagnostics_path = args.output_dir / "experiment1_bt_diagnostics.json"
    write_csv(
        score_path,
        score_rows,
        ["provider_scope", "unclear_handling", "condition", "outcome_id", "task", "bt_score", "z_score", "rank"],
    )
    write_csv(
        correlation_path,
        correlation_rows,
        ["provider_scope", "unclear_handling", "condition_x", "condition_y", "n_tasks", "pearson_r", "r_squared", "spearman_rho"],
    )
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "score_output": str(score_path),
        "correlation_output": str(correlation_path),
        "diagnostics_output": str(diagnostics_path),
        "load": load_metadata,
        "complete_case_correlations": [
            row for row in correlation_rows if row["unclear_handling"] == "complete_case"
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
