#!/usr/bin/env python3
"""Fit Bradley-Terry scores to the emergent-values pairwise data.

The script intentionally excludes active-learning pseudolabels. It fits only
the raw model responses in the published training split, then compares the
resulting scores with the authors' published Thurstonian means.

Dependencies: numpy, Pillow (PIL)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATA_URL = (
    "https://huggingface.co/mmazeika/emergent-values-data/resolve/main/"
    "options_hierarchical.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/external/options_hierarchical.zip"),
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--l2", type=float, default=1e-6)
    return parser.parse_args()


def load_results(archive: Path, model: str) -> dict:
    member = f"options_hierarchical/{model}/results_{model}.json"
    with zipfile.ZipFile(archive) as zf:
        try:
            with zf.open(member) as stream:
                return json.load(stream)
        except KeyError as exc:
            available = sorted(
                name.split("/")[1]
                for name in zf.namelist()
                if name.count("/") == 2 and name.endswith("/")
            )
            raise SystemExit(
                f"Model {model!r} not found. Available models: {available}"
            ) from exc


def extract_observed_edges(results: dict) -> tuple[np.ndarray, ...]:
    graph = results["graph_data"]
    training = {tuple(sorted(edge)) for edge in graph["training_edges"]}

    option_ids = sorted(int(option["id"]) for option in results["options"])
    id_to_index = {option_id: index for index, option_id in enumerate(option_ids)}

    left: list[int] = []
    right: list[int] = []
    wins_left: list[float] = []
    totals: list[float] = []

    for edge in graph["edges"].values():
        aux = edge["aux_data"]
        if aux.get("is_pseudolabel", False):
            continue

        left_id = int(edge["option_A"]["id"])
        right_id = int(edge["option_B"]["id"])
        if tuple(sorted((left_id, right_id))) not in training:
            continue

        count_left = float(aux["count_A"])
        count_right = float(aux["count_B"])
        left.append(id_to_index[left_id])
        right.append(id_to_index[right_id])
        wins_left.append(count_left)
        totals.append(count_left + count_right)

    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(wins_left, dtype=np.float64),
        np.asarray(totals, dtype=np.float64),
        np.asarray(option_ids, dtype=np.int64),
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def fit_bradley_terry(
    n_options: int,
    left: np.ndarray,
    right: np.ndarray,
    wins_left: np.ndarray,
    totals: np.ndarray,
    l2: float,
    max_iterations: int = 100,
) -> tuple[np.ndarray, dict]:
    """Fit a binomial Bradley-Terry model with damped Newton updates.

    The last score is fixed at zero during optimization to remove translation
    non-identifiability. The returned scores are centered afterward.
    """

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
        np.add.at(
            hessian,
            (right[right_free], right[right_free]),
            curvature[right_free],
        )
        both_free = left_free & right_free
        np.add.at(
            hessian,
            (left[both_free], right[both_free]),
            -curvature[both_free],
        )
        np.add.at(
            hessian,
            (right[both_free], left[both_free]),
            -curvature[both_free],
        )
        hessian.flat[:: n_options] += l2

        gradient_max = float(np.max(np.abs(gradient[:-1])))
        step = np.linalg.solve(hessian, gradient[:-1])

        step_scale = 1.0
        accepted = False
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
            break
        if gradient_max < 1e-7 or np.max(np.abs(step_scale * step)) < 1e-10:
            converged = True
            break

    scores -= scores.mean()
    diagnostics = {
        "converged": converged,
        "iterations": iteration,
        "negative_log_likelihood": previous_objective,
        "max_absolute_gradient": gradient_max,
        "l2_penalty": l2,
    }
    return scores, diagnostics


def observed_edge_counts(results: dict) -> dict:
    graph = results["graph_data"]
    holdout = {tuple(sorted(edge)) for edge in graph["holdout_edge_indices"]}
    observed_train = 0
    observed_holdout = 0
    pseudolabeled = 0

    for edge in graph["edges"].values():
        aux = edge["aux_data"]
        pair = tuple(
            sorted((int(edge["option_A"]["id"]), int(edge["option_B"]["id"])))
        )
        if aux.get("is_pseudolabel", False):
            pseudolabeled += 1
        elif pair in holdout:
            observed_holdout += 1
        else:
            observed_train += 1

    return {
        "observed_training_edges": observed_train,
        "observed_holdout_edges": observed_holdout,
        "pseudolabeled_training_edges_excluded": pseudolabeled,
    }


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std(ddof=0)


def linear_comparison(x: np.ndarray, y: np.ndarray) -> dict:
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = intercept + slope * x
    residual_sum_squares = float(np.sum((y - predicted) ** 2))
    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "pearson_r": pearson_r,
        "r_squared": float(r_squared),
        "predicted": predicted,
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / filename
    return ImageFont.truetype(str(path), size=size)


def draw_plot(
    x: np.ndarray,
    y: np.ndarray,
    comparison: dict,
    output_path: Path,
    model: str,
    n_training_edges: int,
) -> None:
    width, height = 1600, 1100
    left_margin, right_margin = 170, 80
    top_margin, bottom_margin = 180, 150
    plot_left, plot_right = left_margin, width - right_margin
    plot_top, plot_bottom = top_margin, height - bottom_margin

    image = Image.new("RGB", (width, height), "#fbfcfe")
    draw = ImageDraw.Draw(image, "RGBA")

    title_font = font(46, bold=True)
    subtitle_font = font(25)
    axis_font = font(27, bold=True)
    tick_font = font(22)
    annotation_font = font(26, bold=True)

    draw.text(
        (left_margin, 45),
        f"{model}: Bradley–Terry vs. published Thurstonian utilities",
        fill="#172033",
        font=title_font,
    )
    draw.text(
        (left_margin, 110),
        f"BT fit on {n_training_edges:,} observed training pairs; pseudolabels excluded",
        fill="#58657a",
        font=subtitle_font,
    )

    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    padding = 0.08 * (upper - lower)
    lower -= padding
    upper += padding

    def px(value: float) -> float:
        return plot_left + (value - lower) / (upper - lower) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - lower) / (upper - lower) * (plot_bottom - plot_top)

    tick_start = math.ceil(lower)
    tick_end = math.floor(upper)
    for tick in range(tick_start, tick_end + 1):
        x_pos, y_pos = px(tick), py(tick)
        draw.line((x_pos, plot_top, x_pos, plot_bottom), fill="#dce2ea", width=2)
        draw.line((plot_left, y_pos, plot_right, y_pos), fill="#dce2ea", width=2)
        tick_text = str(tick)
        box = draw.textbbox((0, 0), tick_text, font=tick_font)
        draw.text(
            (x_pos - (box[2] - box[0]) / 2, plot_bottom + 18),
            tick_text,
            fill="#58657a",
            font=tick_font,
        )
        draw.text(
            (plot_left - 45 - (box[2] - box[0]), y_pos - (box[3] - box[1]) / 2),
            tick_text,
            fill="#58657a",
            font=tick_font,
        )

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#465266", width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#465266", width=3)

    # Identity line and fitted linear relationship.
    draw.line(
        (px(lower), py(lower), px(upper), py(upper)),
        fill="#9aa6b5",
        width=4,
    )
    fitted_lower = comparison["intercept"] + comparison["slope"] * lower
    fitted_upper = comparison["intercept"] + comparison["slope"] * upper
    draw.line(
        (px(lower), py(fitted_lower), px(upper), py(fitted_upper)),
        fill="#d04a3a",
        width=6,
    )

    radius = 7
    for x_value, y_value in zip(x, y):
        x_pos, y_pos = px(float(x_value)), py(float(y_value))
        draw.ellipse(
            (x_pos - radius, y_pos - radius, x_pos + radius, y_pos + radius),
            fill="#2878b8aa",
            outline="#195985cc",
            width=1,
        )

    annotation = (
        f"R² = {comparison['r_squared']:.4f}\n"
        f"Pearson r = {comparison['pearson_r']:.4f}\n"
        f"n = {len(x)} outcomes"
    )
    annotation_box = draw.multiline_textbbox(
        (0, 0), annotation, font=annotation_font, spacing=12
    )
    annotation_width = annotation_box[2] - annotation_box[0]
    annotation_height = annotation_box[3] - annotation_box[1]
    annotation_x = plot_right - annotation_width - 38
    annotation_y = plot_top + 28
    draw.rounded_rectangle(
        (
            annotation_x - 24,
            annotation_y - 20,
            annotation_x + annotation_width + 24,
            annotation_y + annotation_height + 24,
        ),
        radius=16,
        fill="#ffffffee",
        outline="#cfd7e2",
        width=2,
    )
    draw.multiline_text(
        (annotation_x, annotation_y),
        annotation,
        fill="#172033",
        font=annotation_font,
        spacing=12,
    )

    x_label = "Published Thurstonian mean (z-score)"
    x_box = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(
        ((width - (x_box[2] - x_box[0])) / 2, height - 75),
        x_label,
        fill="#172033",
        font=axis_font,
    )

    y_label = "Bradley–Terry score (z-score)"
    label_layer = Image.new("RGBA", (600, 70), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text((0, 0), y_label, fill="#172033", font=axis_font)
    rotated = label_layer.rotate(90, expand=True)
    image.paste(
        rotated,
        (35, int((height - rotated.height) / 2) + 35),
        rotated,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def main() -> None:
    args = parse_args()
    results = load_results(args.archive, args.model)
    left, right, wins_left, totals, option_ids = extract_observed_edges(results)
    counts = observed_edge_counts(results)

    scores, fit_diagnostics = fit_bradley_terry(
        n_options=len(option_ids),
        left=left,
        right=right,
        wins_left=wins_left,
        totals=totals,
        l2=args.l2,
    )

    thurstone_means = np.asarray(
        [float(results["utilities"][str(option_id)]["mean"]) for option_id in option_ids]
    )
    thurstone_variances = np.asarray(
        [
            float(results["utilities"][str(option_id)]["variance"])
            for option_id in option_ids
        ]
    )
    thurstone_z = zscore(thurstone_means)
    bt_z = zscore(scores)
    comparison = linear_comparison(thurstone_z, bt_z)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_bt_vs_thurstone"
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    summary_path = args.output_dir / f"{stem}.json"

    descriptions = {
        int(option["id"]): option["description"] for option in results["options"]
    }
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "option_id",
                "description",
                "thurstone_mean",
                "thurstone_variance",
                "thurstone_z",
                "bt_score",
                "bt_z",
                "bt_predicted_from_thurstone_z",
                "residual",
            ]
        )
        for index, option_id in enumerate(option_ids):
            predicted = float(comparison["predicted"][index])
            writer.writerow(
                [
                    int(option_id),
                    descriptions[int(option_id)],
                    float(thurstone_means[index]),
                    float(thurstone_variances[index]),
                    float(thurstone_z[index]),
                    float(scores[index]),
                    float(bt_z[index]),
                    predicted,
                    float(bt_z[index] - predicted),
                ]
            )

    draw_plot(
        thurstone_z,
        bt_z,
        comparison,
        png_path,
        args.model,
        counts["observed_training_edges"],
    )

    summary = {
        "model": args.model,
        "data_source": DATA_URL,
        "archive": str(args.archive),
        "n_options": len(option_ids),
        "n_individual_training_responses": int(totals.sum()),
        **counts,
        "fit": fit_diagnostics,
        "comparison": {
            key: value
            for key, value in comparison.items()
            if key != "predicted"
        },
        "published_thurstonian_holdout_metrics": results["holdout_metrics"],
        "outputs": {"scores_csv": str(csv_path), "scatter_plot": str(png_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
