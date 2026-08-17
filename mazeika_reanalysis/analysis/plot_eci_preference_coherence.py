#!/usr/bin/env python3
"""Plot Mazeika et al. preference coherence against Epoch's ECI.

Preference coherence is the paper's precomputed held-out Thurstonian utility
model accuracy. The values are read from the summary files in the published
``options_hierarchical.zip`` archive. ECI values are read from Epoch AI's
precomputed ``epoch_capabilities_index.csv`` table.

The join is deliberately explicit because the two sources use different model
identifiers and because several API aliases were not pinned to dated versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]

# Explicit, audited crosswalk from the paper's model keys to Epoch's version IDs.
# "exact" includes harmless provider/repository suffix differences such as -hf.
# The other tiers are shown with hollow markers and retained for the first-stab
# main estimate; the exact-only result is reported as a sensitivity check.
MODEL_CROSSWALK = {
    "claude-3-5-sonnet": {
        "epoch_version": "claude-3-5-sonnet-20240620",
        "match_quality": "exact",
        "match_note": "Paper configuration pins the same dated Anthropic model.",
    },
    "gemma-2-27b-it": {
        "epoch_version": "gemma-2-27b-it",
        "match_quality": "exact",
        "match_note": "Exact model version.",
    },
    "gemma-2-9b-it": {
        "epoch_version": "gemma-2-9b-it",
        "match_quality": "exact",
        "match_note": "Exact model version.",
    },
    "gpt-35-turbo": {
        "epoch_version": "gpt-3.5-turbo-0125",
        "match_quality": "inferred_api_snapshot",
        "match_note": (
            "Paper used the unpinned gpt-3.5-turbo alias; Jan 2024 is the "
            "likely snapshot active during the Feb 2025 data release."
        ),
    },
    "gpt-4o": {
        "epoch_version": "gpt-4o-2024-08-06",
        "match_quality": "inferred_api_snapshot",
        "match_note": (
            "Paper used the unpinned gpt-4o alias; Aug 2024 is the likely "
            "snapshot active during the Feb 2025 data release."
        ),
    },
    "gpt-4o-mini": {
        "epoch_version": "gpt-4o-mini-2024-07-18",
        "match_quality": "single_eci_snapshot",
        "match_note": "Paper alias was unpinned, but Epoch has one scored snapshot.",
    },
    "llama-2-13b-instruct": {
        "epoch_version": "Llama-2-13b-chat",
        "match_quality": "exact",
        "match_note": "Same chat/instruct weights; Epoch omits the -hf suffix.",
    },
    "llama-31-405b-instruct-fp8": {
        "epoch_version": "Llama-3.1-405B-Instruct",
        "match_quality": "precision_variant",
        "match_note": "Paper evaluated FP8 weights; Epoch does not distinguish precision.",
    },
    "llama-31-70b-instruct": {
        "epoch_version": "Llama-3.1-70B-Instruct",
        "match_quality": "exact",
        "match_note": "Exact model version.",
    },
    "llama-31-8b-instruct": {
        "epoch_version": "Llama-3.1-8B-Instruct",
        "match_quality": "exact",
        "match_note": "Exact model version.",
    },
    "llama-33-70b-instruct": {
        "epoch_version": "Llama-3.3-70B-Instruct",
        "match_quality": "exact",
        "match_note": "Exact model version.",
    },
    "qwen25-72b-instruct": {
        "epoch_version": "qwen2.5-72b-instruct",
        "match_quality": "exact",
        "match_note": "Exact model version, ignoring capitalization.",
    },
}

SHORT_LABELS = {
    "claude-3-5-sonnet": "Claude 3.5 Sonnet",
    "gemma-2-27b-it": "Gemma 2 27B",
    "gemma-2-9b-it": "Gemma 2 9B",
    "gpt-35-turbo": "GPT-3.5 Turbo",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o mini",
    "llama-2-13b-instruct": "Llama 2 13B",
    "llama-31-405b-instruct-fp8": "Llama 3.1 405B FP8",
    "llama-31-70b-instruct": "Llama 3.1 70B",
    "llama-31-8b-instruct": "Llama 3.1 8B",
    "llama-33-70b-instruct": "Llama 3.3 70B",
    "qwen25-72b-instruct": "Qwen 2.5 72B",
}

# Point offsets are in display points and keep this fixed dataset legible.
LABEL_OFFSETS = {
    "claude-3-5-sonnet": (-112, -6),
    "gemma-2-27b-it": (-98, -16),
    "gemma-2-9b-it": (-83, 10),
    "gpt-35-turbo": (-4, 13),
    "gpt-4o": (-4, -23),
    "gpt-4o-mini": (-12, 18),
    "llama-2-13b-instruct": (7, -2),
    "llama-31-405b-instruct-fp8": (-160, 24),
    "llama-31-70b-instruct": (-104, -20),
    "llama-31-8b-instruct": (7, -18),
    "llama-33-70b-instruct": (-122, -8),
    "qwen25-72b-instruct": (-70, 46),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mazeika-archive",
        type=Path,
        default=PROJECT_DIR / "data/external/options_hierarchical.zip",
    )
    parser.add_argument(
        "--eci-csv",
        type=Path,
        default=PROJECT_DIR / "data/external/epoch/epoch_capabilities_index.csv",
    )
    parser.add_argument(
        "--mmlu-csv",
        type=Path,
        default=PROJECT_DIR / "data/external/mazeika/capability_scores.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "results"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--permutation-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_815)
    return parser.parse_args()


def load_precomputed_coherence(archive: Path) -> dict[str, float]:
    """Return model -> held-out utility accuracy in percentage points."""
    pattern = re.compile(
        r"Holdout Metrics:\s*\n(?:.*\n)*?accuracy:\s*([0-9.]+)"
    )
    values: dict[str, float] = {}
    with zipfile.ZipFile(archive) as zf:
        summary_names = sorted(
            name
            for name in zf.namelist()
            if name.endswith(".txt") and "/summary_" in name
        )
        for name in summary_names:
            model = name.split("/")[1]
            text = zf.read(name).decode("utf-8")
            match = pattern.search(text)
            if not match:
                raise ValueError(f"No holdout accuracy found in {name}")
            values[model] = round(100.0 * float(match.group(1)), 6)
    return values


def load_mmlu(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {
            row["Model Name"].strip(): float(row["MMLU"])
            for row in csv.DictReader(stream)
            if row["MMLU"].strip()
        }


def load_eci_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    by_version: dict[str, dict[str, str]] = {}
    for row in rows:
        version = row["Model version"].strip().casefold()
        if version:
            existing = by_version.get(version)
            if existing is not None:
                # Epoch sometimes repeats the same model/version for multiple
                # serving providers. This is harmless when the ECI is equal.
                if existing["ECI Score"].strip() != row["ECI Score"].strip():
                    raise ValueError(
                        f"Conflicting ECI scores for duplicate version: {version}"
                    )
                continue
            by_version[version] = row
    return rows, by_version


def build_join(
    coherence: dict[str, float], eci_by_version: dict[str, dict[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    joined: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []

    for model in sorted(coherence):
        mapping = MODEL_CROSSWALK.get(model)
        if mapping is None:
            coverage.append(
                {
                    "mazeika_model": model,
                    "preference_coherence_pct": coherence[model],
                    "included": False,
                    "epoch_model_version": "",
                    "eci_score": "",
                    "match_quality": "no_scored_eci_match",
                    "match_note": "No defensible precomputed scored ECI match found.",
                }
            )
            continue

        epoch_version = str(mapping["epoch_version"])
        epoch_row = eci_by_version.get(epoch_version.casefold())
        if epoch_row is None:
            raise ValueError(f"Epoch version missing from CSV: {epoch_version}")
        if not epoch_row["ECI Score"].strip():
            raise ValueError(f"Epoch version has no precomputed score: {epoch_version}")

        row: dict[str, object] = {
            "mazeika_model": model,
            "preference_coherence_pct": coherence[model],
            "included": True,
            "epoch_model_version": epoch_row["Model version"],
            "eci_score": float(epoch_row["ECI Score"]),
            "match_quality": mapping["match_quality"],
            "match_note": mapping["match_note"],
            "epoch_release_date": epoch_row["Release date"],
            "epoch_model_name": epoch_row["Model name"],
        }
        joined.append(row)
        coverage.append(row.copy())

    return joined, coverage


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_statistics(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    predictions = []
    correlations = []
    for _ in range(samples):
        indices = rng.integers(0, len(x), len(x))
        x_sample = x[indices]
        y_sample = y[indices]
        if np.std(x_sample) == 0 or np.std(y_sample) == 0:
            continue
        slope, intercept = np.polyfit(x_sample, y_sample, 1)
        predictions.append(intercept + slope * x_grid)
        correlations.append(pearson_r(x_sample, y_sample))
    lower, upper = np.percentile(np.asarray(predictions), [2.5, 97.5], axis=0)
    r_interval = tuple(
        float(value) for value in np.percentile(correlations, [2.5, 97.5])
    )
    return lower, upper, r_interval


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(pearson_r(x, y))
    exceedances = 0
    for _ in range(samples):
        exceedances += abs(pearson_r(x, rng.permutation(y))) >= observed
    return (exceedances + 1.0) / (samples + 1.0)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    preferred = [
        "mazeika_model",
        "preference_coherence_pct",
        "included",
        "epoch_model_version",
        "eci_score",
        "match_quality",
        "match_note",
        "epoch_release_date",
        "epoch_model_name",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_plot(
    joined: list[dict[str, object]],
    correlation: float,
    correlation_ci: tuple[float, float],
    p_value: float,
    output_path: Path,
    bootstrap_samples: int,
    seed: int,
) -> None:
    x = np.asarray([row["eci_score"] for row in joined], dtype=float)
    y = np.asarray([row["preference_coherence_pct"] for row in joined], dtype=float)
    x_grid = np.linspace(x.min() - 1.0, x.max() + 1.0, 240)
    slope, intercept = np.polyfit(x, y, 1)

    band_rng = np.random.default_rng(seed + 1)
    lower, upper, _ = bootstrap_statistics(
        x, y, x_grid, bootstrap_samples, band_rng
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#26354a",
            "xtick.color": "#526175",
            "ytick.color": "#526175",
        }
    )
    fig, ax = plt.subplots(figsize=(11.6, 7.8), dpi=180)
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")

    ax.fill_between(
        x_grid,
        lower,
        upper,
        color="#4f78c4",
        alpha=0.14,
        linewidth=0,
        label="95% paired-bootstrap band",
    )
    ax.plot(
        x_grid,
        intercept + slope * x_grid,
        color="#315ea8",
        linewidth=2.5,
        label="OLS fit",
        zorder=2,
    )

    exact_rows = [row for row in joined if row["match_quality"] == "exact"]
    caveat_rows = [row for row in joined if row["match_quality"] != "exact"]
    for rows, face, edge, marker, label in [
        (exact_rows, "#315ea8", "white", "o", "Direct version match"),
        (caveat_rows, "#f7f9fc", "#d06a32", "^", "Alias/precision caveat"),
    ]:
        ax.scatter(
            [row["eci_score"] for row in rows],
            [row["preference_coherence_pct"] for row in rows],
            s=92,
            marker=marker,
            c=face,
            edgecolors=edge,
            linewidths=2.0,
            label=label,
            zorder=4,
        )

    for row in joined:
        model = str(row["mazeika_model"])
        dx, dy = LABEL_OFFSETS[model]
        ax.annotate(
            SHORT_LABELS[model],
            (float(row["eci_score"]), float(row["preference_coherence_pct"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.2,
            color="#34465d",
            arrowprops={
                "arrowstyle": "-",
                "color": "#aab4c1",
                "linewidth": 0.75,
                "shrinkA": 3,
                "shrinkB": 5,
            },
            zorder=5,
        )

    ax.text(
        0.025,
        0.965,
        f"Pearson r = {correlation:.2f}\n"
        f"95% bootstrap CI [{correlation_ci[0]:.2f}, {correlation_ci[1]:.2f}]\n"
        f"permutation p = {p_value:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.2,
        color="#26354a",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "white",
            "edgecolor": "#c6cfda",
            "linewidth": 1.0,
            "alpha": 0.95,
        },
    )

    ax.set_title(
        "Preference coherence vs. Epoch Capabilities Index",
        loc="left",
        fontsize=20,
        color="#17243a",
        pad=24,
    )
    ax.text(
        0,
        1.015,
        "Mazeika et al. held-out utility-model accuracy • scored ECI overlap "
        f"(n={len(joined)} of 32 models)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.2,
        color="#66758a",
    )
    ax.set_xlabel("Epoch Capabilities Index (ECI)", fontsize=12.5, labelpad=10)
    ax.set_ylabel("Held-out utility-model accuracy (%)", fontsize=12.5, labelpad=10)
    ax.set_xlim(x.min() - 2.5, x.max() + 2.5)
    ax.set_ylim(min(76.5, y.min() - 2.0), max(96.5, y.max() + 2.0))
    ax.grid(True, color="#dbe2eb", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 3, 1, 0]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#c6cfda",
        framealpha=0.95,
        fontsize=9.4,
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.86)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coherence = load_precomputed_coherence(args.mazeika_archive)
    mmlu = load_mmlu(args.mmlu_csv)
    _, eci_by_version = load_eci_rows(args.eci_csv)
    joined, coverage = build_join(coherence, eci_by_version)

    x = np.asarray([row["eci_score"] for row in joined], dtype=float)
    y = np.asarray([row["preference_coherence_pct"] for row in joined], dtype=float)
    exact = [row for row in joined if row["match_quality"] == "exact"]
    exact_x = np.asarray([row["eci_score"] for row in exact], dtype=float)
    exact_y = np.asarray(
        [row["preference_coherence_pct"] for row in exact], dtype=float
    )
    overlap_mmlu = np.asarray(
        [mmlu[str(row["mazeika_model"])] for row in joined], dtype=float
    )
    paper_models = [
        model
        for model in sorted(coherence)
        if model in mmlu and model != "qwen15-05b-instruct"
    ]
    paper_mmlu = np.asarray([mmlu[model] for model in paper_models], dtype=float)
    paper_coherence = np.asarray(
        [coherence[model] for model in paper_models], dtype=float
    )

    correlation = pearson_r(x, y)
    rng = np.random.default_rng(args.seed)
    _, _, correlation_ci = bootstrap_statistics(
        x,
        y,
        np.linspace(x.min(), x.max(), 10),
        args.bootstrap_samples,
        rng,
    )
    p_value = permutation_p_value(
        x,
        y,
        args.permutation_samples,
        np.random.default_rng(args.seed + 2),
    )

    joined_path = args.output_dir / "eci_preference_coherence.csv"
    coverage_path = args.output_dir / "eci_model_coverage.csv"
    summary_path = args.output_dir / "eci_preference_coherence_summary.json"
    plot_path = args.output_dir / "eci_preference_coherence.png"
    write_csv(joined_path, joined)
    write_csv(coverage_path, coverage)

    summary = {
        "preference_coherence_definition": (
            "Mazeika et al. precomputed held-out Thurstonian utility-model "
            "accuracy, in percentage points"
        ),
        "mazeika_models": len(coherence),
        "included_models": len(joined),
        "direct_version_matches": len(exact),
        "matches_with_alias_or_precision_caveats": len(joined) - len(exact),
        "pearson_r_all_included": correlation,
        "pearson_r_bootstrap_95_ci": list(correlation_ci),
        "pearson_r_permutation_p_two_sided": p_value,
        "pearson_r_exact_matches_only": pearson_r(exact_x, exact_y),
        "n_exact_matches_only": len(exact),
        "pearson_r_mmlu_same_12_model_overlap": pearson_r(overlap_mmlu, y),
        "pearson_r_mmlu_paper_sample": pearson_r(paper_mmlu, paper_coherence),
        "n_mmlu_paper_sample": len(paper_models),
        "mmlu_paper_sample_exclusion": "qwen15-05b-instruct",
        "eci_min": float(x.min()),
        "eci_max": float(x.max()),
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "random_seed": args.seed,
        "source_files": {
            "mazeika": str(args.mazeika_archive),
            "epoch": str(args.eci_csv),
            "mmlu": str(args.mmlu_csv),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    draw_plot(
        joined,
        correlation,
        correlation_ci,
        p_value,
        plot_path,
        args.bootstrap_samples,
        args.seed,
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote {plot_path}")
    print(f"Wrote {joined_path}")
    print(f"Wrote {coverage_path}")


if __name__ == "__main__":
    main()
