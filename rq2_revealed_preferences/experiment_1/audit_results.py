#!/usr/bin/env python3
"""Audit raw experiment JSONL against the deterministic request manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_experiment import DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_TASKS, build_manifest, load_json


IDENTITY_FIELDS = (
    "experiment_id",
    "provider",
    "model",
    "condition",
    "pair_low_id",
    "pair_high_id",
    "order",
    "sample",
    "task_a_id",
    "task_b_id",
    "task_a",
    "task_b",
    "prompt",
    "system_prompt",
    "temperature",
    "max_output_tokens",
    "service_tier",
)


def _add_usage(total: Counter[str], usage: dict[str, Any] | None) -> None:
    if not usage:
        return
    for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total[key] += value


def audit(raw_path: Path, config_path: Path, tasks_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    tasks = load_json(tasks_path)
    manifest = build_manifest(config, tasks)
    expected = {row["request_id"]: row for row in manifest}

    line_count = 0
    blank_lines = 0
    invalid_json: list[int] = []
    records_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_request_id_lines: list[int] = []

    with raw_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line_count += 1
            if not line.strip():
                blank_lines += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json.append(line_number)
                continue
            request_id = record.get("request_id")
            if not request_id:
                missing_request_id_lines.append(line_number)
                continue
            records_by_id[request_id].append(record)

    duplicate_ids = {key: len(rows) for key, rows in records_by_id.items() if len(rows) > 1}
    successful: dict[str, dict[str, Any]] = {}
    for key, rows in records_by_id.items():
        successful_rows = [row for row in rows if row.get("status") == "ok"]
        if successful_rows:
            successful[key] = successful_rows[-1]
    success_ids = set(successful)
    expected_ids = set(expected)
    unexpected_ids = sorted(set(records_by_id) - expected_ids)
    missing_success_ids = sorted(expected_ids - success_ids)

    status_counts: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    for rows in records_by_id.values():
        for row in rows:
            status_counts[str(row.get("status", "missing"))] += 1
            if row.get("status") == "error":
                error_types[str(row.get("error_type", "unknown"))] += 1

    cell_counts: Counter[str] = Counter()
    parsed_counts: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    stop_reasons_by_cell: Counter[str] = Counter()
    parsed_choices_by_cell: Counter[str] = Counter()
    resolved_models: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    empty_responses: list[str] = []
    mismatched_records: dict[str, list[str]] = {}
    for request_id, record in successful.items():
        cell_counts[f"{record['provider']} / {record['condition']}"] += 1
        parsed_counts[str(record.get("parsed_choice"))] += 1
        stop_reasons[str(record.get("stop_reason"))] += 1
        stop_reasons_by_cell[
            f"{record['provider']} / {record['condition']} / {record.get('stop_reason')}"
        ] += 1
        parsed_choices_by_cell[
            f"{record['provider']} / {record['condition']} / {record.get('parsed_choice')}"
        ] += 1
        resolved_models[str(record.get("resolved_model"))] += 1
        _add_usage(usage, record.get("usage"))
        if not str(record.get("response_text", "")).strip():
            empty_responses.append(request_id)
        if request_id in expected:
            bad_fields = [field for field in IDENTITY_FIELDS if record.get(field) != expected[request_id].get(field)]
            if bad_fields:
                mismatched_records[request_id] = bad_fields

    expected_cells = Counter(
        f"{row['provider']} / {row['condition']}" for row in manifest
    )
    complete = not any((
        invalid_json,
        missing_request_id_lines,
        duplicate_ids,
        unexpected_ids,
        missing_success_ids,
        empty_responses,
        mismatched_records,
        status_counts.get("error", 0),
    )) and line_count - blank_lines == len(expected)

    truncated_by_cell = {
        key: value
        for key, value in sorted(stop_reasons_by_cell.items())
        if key.endswith(" / length") or key.endswith(" / max_tokens")
    }

    return {
        "complete_and_valid": complete,
        "storage_complete_and_valid": complete,
        "raw_path": str(raw_path),
        "expected_manifest_requests": len(expected),
        "physical_lines": line_count,
        "blank_lines": blank_lines,
        "valid_records": sum(len(rows) for rows in records_by_id.values()),
        "unique_request_ids": len(records_by_id),
        "successful_unique_request_ids": len(success_ids),
        "status_counts": dict(sorted(status_counts.items())),
        "duplicate_request_ids": duplicate_ids,
        "invalid_json_lines": invalid_json,
        "missing_request_id_lines": missing_request_id_lines,
        "unexpected_request_ids": unexpected_ids,
        "missing_success_request_ids": missing_success_ids,
        "empty_successful_responses": empty_responses,
        "manifest_field_mismatches": mismatched_records,
        "successful_counts_by_provider_condition": dict(sorted(cell_counts.items())),
        "expected_counts_by_provider_condition": dict(sorted(expected_cells.items())),
        "parsed_choice_counts": dict(sorted(parsed_counts.items())),
        "parsed_choice_counts_by_provider_condition": dict(sorted(parsed_choices_by_cell.items())),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "stop_reason_counts_by_provider_condition": dict(sorted(stop_reasons_by_cell.items())),
        "truncated_completion_counts_by_provider_condition": truncated_by_cell,
        "resolved_model_counts": dict(sorted(resolved_models.items())),
        "usage_totals": dict(sorted(usage.items())),
        "error_type_counts": dict(sorted(error_types.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--report", type=Path, help="Optionally save the audit as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(args.raw, args.config, args.tasks)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["complete_and_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
