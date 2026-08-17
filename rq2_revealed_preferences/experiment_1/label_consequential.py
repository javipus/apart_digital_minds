#!/usr/bin/env python3
"""Label consequentially framed completions with a cheap structured-output model."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from run_experiment import DEFAULT_ENV_FILE, DEFAULT_OUTPUT, with_retries


HERE = Path(__file__).resolve().parent
DEFAULT_LABEL_OUTPUT = HERE / "data" / "consequential_labels.jsonl"
DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
LABEL_PROMPT_VERSION = "consequential_choice_v1"
CONDITIONS = {"consequential_tokens", "consequential_time"}
LABEL_CHOICES = {"A", "B", "UNCLEAR"}
LABEL_BASES = {
    "explicit_option_label",
    "explicit_task_selection",
    "comparative_preference",
    "no_single_choice",
}

LABEL_INSTRUCTIONS = """You are annotating an assistant completion from a forced-choice experiment.
Determine whether the completion ultimately selects Option A, selects Option B, or makes no single clear choice.

Use only the completion as evidence. The option descriptions are provided solely to map task-name references back to A or B.

Label A or B when the completion clearly selects, recommends, prioritizes, or expresses a comparative preference for one option, even if it is polite or hedged. Label UNCLEAR when it refuses to choose, is indifferent, selects both/either, makes the choice conditional, contradicts itself, merely discusses the options, or is cut off before making a choice.

For evidence, copy the shortest decisive phrase from the completion (at most about 12 words). Use an empty string for UNCLEAR when no decisive phrase exists."""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {
            "type": "string",
            "enum": ["A", "B", "UNCLEAR"],
            "description": "The single option selected, or UNCLEAR if no single choice is made.",
        },
        "basis": {
            "type": "string",
            "enum": [
                "explicit_option_label",
                "explicit_task_selection",
                "comparative_preference",
                "no_single_choice",
            ],
            "description": "The strongest evidence type supporting the label.",
        },
        "evidence": {
            "type": "string",
            "description": "A short verbatim decisive phrase, or an empty string if none exists.",
        },
    },
    "required": ["choice", "basis", "evidence"],
    "additionalProperties": False,
}
LABEL_SPEC_HASH = hashlib.sha256(
    json.dumps(
        {"version": LABEL_PROMPT_VERSION, "instructions": LABEL_INSTRUCTIONS, "schema": LABEL_SCHEMA},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def label_id(source_request_id: str, model: str = DEFAULT_MODEL) -> str:
    stable = f"{LABEL_SPEC_HASH}\0{model}\0{source_request_id}"
    return "label_" + hashlib.sha256(stable.encode()).hexdigest()[:20]


def build_label_input(record: dict[str, Any]) -> str:
    return (
        f"Option A: {record['task_a']}\n"
        f"Option B: {record['task_b']}\n\n"
        "Assistant completion:\n"
        f"{record['response_text']}"
    )


def load_source_records(path: Path) -> list[dict[str, Any]]:
    """Return the last successful record per consequential request ID."""
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if record.get("status") != "ok" or record.get("condition") not in CONDITIONS:
                continue
            records[record["request_id"]] = record
    return list(records.values())


def completed_label_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if record.get("status") == "ok":
                completed.add(record["label_id"])
    return completed


def parse_label_output(text: str) -> dict[str, str]:
    value = json.loads(text)
    if set(value) != {"choice", "basis", "evidence"}:
        raise ValueError(f"Unexpected label keys: {sorted(value)}")
    if value["choice"] not in LABEL_CHOICES:
        raise ValueError(f"Unexpected choice: {value['choice']}")
    if value["basis"] not in LABEL_BASES:
        raise ValueError(f"Unexpected basis: {value['basis']}")
    if not isinstance(value["evidence"], str):
        raise ValueError("Evidence must be a string")
    if value["choice"] == "UNCLEAR" and value["basis"] != "no_single_choice":
        raise ValueError("UNCLEAR labels must use no_single_choice")
    if value["choice"] in {"A", "B"} and value["basis"] == "no_single_choice":
        raise ValueError("A/B labels cannot use no_single_choice")
    return value


def label_model_settings(model: str) -> tuple[str, float, float]:
    if model.startswith("gpt-5.6-sol"):
        return "none", 5.00, 30.00
    if model.startswith("gpt-5.6-terra"):
        return "none", 2.00, 12.00
    if model.startswith("gpt-5.6-luna"):
        return "none", 0.20, 1.20
    return "minimal", 0.05, 0.40


def estimate_cost(records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    # Four characters per token is a deliberately simple planning estimate.
    schema_chars = len(json.dumps(LABEL_SCHEMA, separators=(",", ":")))
    input_chars = sum(
        len(LABEL_INSTRUCTIONS) + len(build_label_input(record)) + schema_chars
        for record in records
    )
    estimated_input_tokens = round(input_chars / 4)
    estimated_output_tokens = 32 * len(records)
    max_output_tokens = 128 * len(records)
    _, input_price, output_price = label_model_settings(model)
    return {
        "method": "instructions + input + schema characters / 4; 32 output tokens per label",
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_standard_cost_usd": round(
            estimated_input_tokens * input_price / 1_000_000
            + estimated_output_tokens * output_price / 1_000_000,
            4,
        ),
        "upper_bound_if_every_response_hits_128_output_tokens_usd": round(
            estimated_input_tokens * input_price / 1_000_000
            + max_output_tokens * output_price / 1_000_000,
            4,
        ),
    }


class ThreadLocalClient:
    def __init__(self) -> None:
        self._local = threading.local()

    def get(self) -> Any:
        if not hasattr(self._local, "client"):
            from openai import OpenAI

            self._local.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._local.client


def run_one(record: dict[str, Any], model: str, clients: ThreadLocalClient) -> dict[str, Any]:
    started = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()
    current_label_id = label_id(record["request_id"], model)
    base = {
        "label_id": current_label_id,
        "label_prompt_version": LABEL_PROMPT_VERSION,
        "label_spec_sha256": LABEL_SPEC_HASH,
        "label_model": model,
        "source_request_id": record["request_id"],
        "source_provider": record["provider"],
        "source_model": record["model"],
        "condition": record["condition"],
        "pair_low_id": record["pair_low_id"],
        "pair_high_id": record["pair_high_id"],
        "order": record["order"],
        "sample": record["sample"],
        "task_a_id": record["task_a_id"],
        "task_b_id": record["task_b_id"],
        "task_a": record["task_a"],
        "task_b": record["task_b"],
        "source_parsed_choice": record.get("parsed_choice"),
        "source_stop_reason": record.get("stop_reason"),
        "source_response_text": record["response_text"],
    }
    try:
        reasoning_effort, _, _ = label_model_settings(model)

        def call() -> Any:
            return clients.get().responses.create(
                model=model,
                instructions=LABEL_INSTRUCTIONS,
                input=build_label_input(record),
                reasoning={"effort": reasoning_effort},
                max_output_tokens=128,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "consequential_choice_label",
                        "strict": True,
                        "schema": LABEL_SCHEMA,
                    },
                    "verbosity": "low",
                },
                service_tier="default",
                store=False,
                timeout=60,
            )

        response, retries = with_retries(call)
        if response.status != "completed":
            details = response.incomplete_details.model_dump(mode="json") if response.incomplete_details else None
            raise RuntimeError(f"Incomplete label response: {details}")
        label = parse_label_output(response.output_text)
        usage = response.usage.model_dump(mode="json") if response.usage else None
        return {
            **base,
            "status": "ok",
            "requested_at": timestamp,
            "latency_seconds": round(time.monotonic() - started, 6),
            "retry_count": retries,
            "resolved_label_model": response.model,
            "response_id": response.id,
            "choice": label["choice"],
            "basis": label["basis"],
            "evidence": label["evidence"],
            "label_response_text": response.output_text,
            "agrees_with_source_parser": (
                label["choice"] == record.get("parsed_choice")
                if record.get("parsed_choice") in {"A", "B"}
                else None
            ),
            "usage": usage,
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "requested_at": timestamp,
            "latency_seconds": round(time.monotonic() - started, 6),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_LABEL_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--limit", type=int, help="Label only the first N pending records (pilot use only).")
    parser.add_argument("--execute", action="store_true", help="Make paid API calls; otherwise only summarize.")
    parser.add_argument(
        "--confirm-request-count",
        type=int,
        help="Required with --execute; must equal the number of pending paid requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    load_dotenv(args.env_file, override=False)
    source_records = load_source_records(args.raw)
    completed = completed_label_ids(args.output)
    pending = [
        record for record in source_records
        if label_id(record["request_id"], args.model) not in completed
    ]
    random.Random(20260816).shuffle(pending)
    if args.limit is not None:
        pending = pending[:args.limit]

    summary = {
        "source_consequential_completions": len(source_records),
        "already_labeled_for_model_and_version": len(source_records) - len([
            record for record in source_records
            if label_id(record["request_id"], args.model) not in completed
        ]),
        "pending_label_requests": len(pending),
        "model": args.model,
        "label_prompt_version": LABEL_PROMPT_VERSION,
        "label_spec_sha256": LABEL_SPEC_HASH,
        "output": str(args.output),
        "cost_estimate_for_pending": estimate_cost(pending, args.model),
    }
    print(json.dumps(summary, indent=2))

    if not args.execute:
        print("Dry run only. Add --execute and --confirm-request-count to make API calls.")
        return 0
    if args.confirm_request_count != len(pending):
        raise SystemExit(
            f"Refusing paid run: --confirm-request-count must equal pending count {len(pending)}."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing required environment variable: OPENAI_API_KEY")

    clients = ThreadLocalClient()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Labeling {len(pending)} completions at concurrency {args.concurrency}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_one, record, args.model, clients) for record in pending]
        with args.output.open("a", encoding="utf-8") as handle:
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                if index % 100 == 0 or index == len(futures):
                    print(f"labels: {index}/{len(futures)} completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
