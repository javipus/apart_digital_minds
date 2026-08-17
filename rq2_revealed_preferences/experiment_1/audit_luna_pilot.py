#!/usr/bin/env python3
"""Audit the GPT-5.6 Luna end-to-end pilot for truncation and unclear labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / "data" / "luna_raw_responses.jsonl"
DEFAULT_LABELS = HERE / "data" / "luna_consequential_labels_luna.jsonl"


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


def token_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "max": None}
    return {"mean": sum(values) / len(values), "max": max(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()

    physical_raw = read_jsonl(args.raw)
    # Append-only files may contain an earlier failed attempt followed by a
    # successful retry. Audit the latest record for each logical request.
    raw_by_id = {record["request_id"]: record for record in physical_raw}
    raw = list(raw_by_id.values())
    successful = [record for record in raw if record.get("status") == "ok"]
    consequential = [
        record for record in successful
        if record.get("condition") in {"consequential_tokens", "consequential_time"}
    ]
    labels = read_jsonl(args.labels)
    successful_labels = [record for record in labels if record.get("status") == "ok"]
    latest_label = {record["source_request_id"]: record for record in successful_labels}

    completion_tokens = [
        int(record["usage"]["completion_tokens"])
        for record in successful if record.get("usage")
    ]
    reasoning_tokens = [
        int(record["usage"].get("completion_tokens_details", {}).get("reasoning_tokens") or 0)
        for record in successful if record.get("usage")
    ]
    labeled_consequential = [
        latest_label[record["request_id"]]
        for record in consequential if record["request_id"] in latest_label
    ]
    parser_comparable = [
        label for label in labeled_consequential
        if label.get("source_parsed_choice") in {"A", "B"} and label.get("choice") in {"A", "B"}
    ]

    report = {
        "generation": {
            "physical_records": len(physical_raw),
            "logical_requests": len(raw),
            "successful": len(successful),
            "errors": len(raw) - len(successful),
            "conditions": dict(sorted(Counter(record.get("condition") for record in successful).items())),
            "condition_order_cells": dict(sorted(Counter(
                f"{record.get('condition')}/order_{record.get('order')}" for record in successful
            ).items())),
            "stop_reasons": dict(sorted(Counter(record.get("stop_reason") for record in successful).items())),
            "empty_responses": sum(not record.get("response_text", "").strip() for record in successful),
            "source_parser_unparsed": sum(record.get("parsed_choice") not in {"A", "B"} for record in successful),
            "completion_tokens": token_summary(completion_tokens),
            "reasoning_tokens": {"total": sum(reasoning_tokens), **token_summary(reasoning_tokens)},
        },
        "labeling": {
            "consequential_completions": len(consequential),
            "physical_label_records": len(labels),
            "successful_labels": len(successful_labels),
            "label_errors": len(labels) - len(successful_labels),
            "missing_labels": len(consequential) - len(labeled_consequential),
            "choices": dict(sorted(Counter(label.get("choice") for label in labeled_consequential).items())),
            "unclear_rate": (
                sum(label.get("choice") == "UNCLEAR" for label in labeled_consequential)
                / len(labeled_consequential)
                if labeled_consequential else None
            ),
            "parser_label_agreement": (
                sum(label["choice"] == label["source_parsed_choice"] for label in parser_comparable)
                / len(parser_comparable)
                if parser_comparable else None
            ),
            "parser_label_comparable": len(parser_comparable),
        },
    }
    print(json.dumps(report, indent=2))

    failures = []
    if report["generation"]["errors"]:
        failures.append("generation errors")
    if report["generation"]["empty_responses"]:
        failures.append("empty responses")
    if any(reason in {"length", "max_tokens"} for reason in report["generation"]["stop_reasons"]):
        failures.append("length-limited responses")
    if report["labeling"]["label_errors"] or report["labeling"]["missing_labels"]:
        failures.append("incomplete labeling")
    if failures:
        raise SystemExit("Pilot audit failed: " + ", ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
