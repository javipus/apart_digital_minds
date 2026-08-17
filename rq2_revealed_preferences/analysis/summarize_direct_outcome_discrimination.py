#!/usr/bin/env python3
"""Summarize fitted Thurstone discrimination among directly executable outcomes."""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = ROOT / "rq2_revealed_preferences/outcomes/utility_engineering_outcome_suitability.csv"
ARCHIVE = ROOT / "mazeika_reanalysis/data/external/options_hierarchical.zip"
RESULTS_DIR = ROOT / "rq2_revealed_preferences/results"


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def main() -> None:
    with ANNOTATIONS.open(newline="", encoding="utf-8") as handle:
        direct = [row for row in csv.DictReader(handle) if row["suitability_label"] == "direct"]

    direct_by_id = {int(row["outcome_id"]): row for row in direct}
    direct_ids = sorted(direct_by_id)
    pairs = list(itertools.combinations(direct_ids, 2))
    model_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    with zipfile.ZipFile(ARCHIVE) as archive:
        utility_paths = sorted(
            name for name in archive.namelist()
            if "/results_utilities_" in name and name.endswith(".json")
        )
        for path in utility_paths:
            model = path.split("/")[-2]
            payload = json.loads(archive.read(path))
            utilities = payload["utilities"]
            probabilities: list[float] = []

            for option_i, option_j in pairs:
                u_i = utilities[str(option_i)]
                u_j = utilities[str(option_j)]
                delta = float(u_i["mean"]) - float(u_j["mean"])
                noise_sd = math.sqrt(float(u_i["variance"]) + float(u_j["variance"]) + 1e-5)
                p_i = phi(delta / noise_sd)
                p_preferred = max(p_i, 1.0 - p_i)
                probabilities.append(p_preferred)
                if delta >= 0:
                    higher_id, lower_id = option_i, option_j
                else:
                    higher_id, lower_id = option_j, option_i
                pair_rows.append({
                    "model": model,
                    "higher_utility_outcome_id": higher_id,
                    "lower_utility_outcome_id": lower_id,
                    "higher_utility_outcome": direct_by_id[higher_id]["original_outcome"],
                    "lower_utility_outcome": direct_by_id[lower_id]["original_outcome"],
                    "p_higher_utility_outcome_preferred": p_preferred,
                    "absolute_probability_margin": 2.0 * p_preferred - 1.0,
                })

            model_rows.append({
                "model": model,
                "n_direct_outcomes": len(direct_ids),
                "n_pairs": len(pairs),
                "mean_p_higher_utility_outcome_preferred": statistics.fmean(probabilities),
                "median_p_higher_utility_outcome_preferred": statistics.median(probabilities),
                "p25": quantile(probabilities, 0.25),
                "p75": quantile(probabilities, 0.75),
                "share_pairs_p_at_least_0_60": sum(p >= 0.60 for p in probabilities) / len(probabilities),
                "share_pairs_p_at_least_0_65": sum(p >= 0.65 for p in probabilities) / len(probabilities),
                "share_pairs_p_at_least_0_75": sum(p >= 0.75 for p in probabilities) / len(probabilities),
                "share_pairs_p_at_least_0_90": sum(p >= 0.90 for p in probabilities) / len(probabilities),
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RESULTS_DIR / "direct_outcome_discrimination_by_model.csv"
    pair_path = RESULTS_DIR / "direct_outcome_pairwise_probabilities.csv"

    with model_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)

    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    pooled = [float(row["p_higher_utility_outcome_preferred"]) for row in pair_rows]
    pair_groups: dict[tuple[int, int], list[float]] = {}
    descriptions: dict[int, str] = {
        outcome_id: row["original_outcome"] for outcome_id, row in direct_by_id.items()
    }
    for row in pair_rows:
        ids = tuple(sorted((int(row["higher_utility_outcome_id"]), int(row["lower_utility_outcome_id"]))))
        pair_groups.setdefault(ids, []).append(float(row["p_higher_utility_outcome_preferred"]))

    pair_averages = sorted(
        (
            statistics.fmean(values),
            ids[0],
            ids[1],
            descriptions[ids[0]],
            descriptions[ids[1]],
        )
        for ids, values in pair_groups.items()
    )
    summary = {
        "n_models": len(model_rows),
        "n_direct_outcomes": len(direct_ids),
        "n_pairs_per_model": len(pairs),
        "n_pair_model_probabilities": len(pooled),
        "arbitrary_A_mean_probability": 0.5,
        "pooled_mean_p_higher_utility_outcome_preferred": statistics.fmean(pooled),
        "pooled_median_p_higher_utility_outcome_preferred": statistics.median(pooled),
        "pooled_p25": quantile(pooled, 0.25),
        "pooled_p75": quantile(pooled, 0.75),
        "pooled_share_p_at_least_0_60": sum(p >= 0.60 for p in pooled) / len(pooled),
        "pooled_share_p_at_least_0_65": sum(p >= 0.65 for p in pooled) / len(pooled),
        "pooled_share_p_at_least_0_75": sum(p >= 0.75 for p in pooled) / len(pooled),
        "pooled_share_p_at_least_0_90": sum(p >= 0.90 for p in pooled) / len(pooled),
        "model_mean_min": min(float(row["mean_p_higher_utility_outcome_preferred"]) for row in model_rows),
        "model_mean_max": max(float(row["mean_p_higher_utility_outcome_preferred"]) for row in model_rows),
        "least_discriminating_pairs_across_models": [
            {"mean_probability": mean, "outcome_i_id": i, "outcome_j_id": j,
             "outcome_i": text_i, "outcome_j": text_j}
            for mean, i, j, text_i, text_j in pair_averages[:10]
        ],
        "most_discriminating_pairs_across_models": [
            {"mean_probability": mean, "outcome_i_id": i, "outcome_j_id": j,
             "outcome_i": text_i, "outcome_j": text_j}
            for mean, i, j, text_i, text_j in reversed(pair_averages[-10:])
        ],
    }
    summary_path = RESULTS_DIR / "direct_outcome_discrimination_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {model_path}")
    print(f"Wrote {pair_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
