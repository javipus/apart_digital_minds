#!/usr/bin/env python3
"""Generate and run Experiment 1 with resumable, append-only JSONL output."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_TASKS = HERE / "tasks.json"
DEFAULT_OUTPUT = HERE / "data" / "raw_responses.jsonl"
DEFAULT_ENV_FILE = HERE / ".env"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def render_prompt(condition: dict[str, Any], task_a: dict[str, Any], task_b: dict[str, Any]) -> str:
    version = condition["task_version"]
    values = {
        "task_a": task_a[version],
        "task_b": task_b[version],
        "resource_context": condition.get("resource_context", ""),
    }
    return condition["template"].format(**values)


def request_id(fields: dict[str, Any]) -> str:
    stable = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(stable.encode()).hexdigest()[:16]
    return f"{fields['experiment_id']}_{fields['provider']}_{digest}"


def build_manifest(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    providers: set[str] | None = None,
    conditions: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return every model × condition × pair × order × repetition request."""
    manifest: list[dict[str, Any]] = []
    for model in config["models"]:
        if providers and model["provider"] not in providers:
            continue
        for condition in config["conditions"]:
            if conditions and condition["id"] not in conditions:
                continue
            for left, right in itertools.combinations(tasks, 2):
                for order, (task_a, task_b) in enumerate(((left, right), (right, left))):
                    for sample in range(config["samples_per_cell"]):
                        identity = {
                            "experiment_id": config["experiment_id"],
                            "provider": model["provider"],
                            "model": model["model"],
                            "condition": condition["id"],
                            "pair_low_id": min(left["outcome_id"], right["outcome_id"]),
                            "pair_high_id": max(left["outcome_id"], right["outcome_id"]),
                            "order": order,
                            "sample": sample,
                        }
                        manifest.append({
                            **identity,
                            "request_id": request_id(identity),
                            "task_a_id": task_a["outcome_id"],
                            "task_b_id": task_b["outcome_id"],
                            "task_a": task_a[condition["task_version"]],
                            "task_b": task_b[condition["task_version"]],
                            "prompt": render_prompt(condition, task_a, task_b),
                            "system_prompt": config["system_prompt"],
                            "temperature": model["temperature"],
                            "reasoning_effort": model.get("reasoning_effort"),
                            "max_output_tokens": condition["max_output_tokens"],
                            "service_tier": model["service_tier"],
                            "request_timeout_seconds": model["request_timeout_seconds"],
                        })
    return manifest


def parse_choice(text: str) -> str | None:
    """Conservatively auto-label an A/B response; retain raw text regardless."""
    normalized = text.strip()
    strict = re.fullmatch(r'["\']?\s*([AB])\s*["\']?[.!]?', normalized, flags=re.IGNORECASE)
    if strict:
        return strict.group(1).upper()

    patterns = [
        r"\b(?:option|task)\s+([AB])\b",
        r"\b(?:choose|pick|prefer|select|go with|work on)\s+(?:option\s+|task\s+)?([AB])\b",
        r"^\s*([AB])(?:\s|[.):-])",
    ]
    found: set[str] = set()
    for pattern in patterns:
        found.update(match.upper() for match in re.findall(pattern, normalized, flags=re.IGNORECASE))
    return next(iter(found)) if len(found) == 1 else None


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in {408, 409, 429, 500, 502, 503, 504} or status is None


def with_retries(call: Any, attempts: int = 6) -> tuple[Any, int]:
    for attempt in range(attempts):
        try:
            return call(), attempt
        except Exception as exc:
            if attempt == attempts - 1 or not _retryable(exc):
                raise
            delay = min(30.0, (2**attempt) + random.random())
            time.sleep(delay)
    raise RuntimeError("unreachable")


class ProviderClients:
    def __init__(self) -> None:
        self._local = threading.local()

    def openai(self) -> Any:
        if not hasattr(self._local, "openai_client"):
            from openai import OpenAI

            self._local.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._local.openai_client

    def anthropic(self) -> Any:
        if not hasattr(self._local, "anthropic_client"):
            from anthropic import Anthropic

            self._local.anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._local.anthropic_client


def run_one(item: dict[str, Any], clients: ProviderClients) -> dict[str, Any]:
    started = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        if item["provider"] == "openai":
            def call() -> Any:
                request = {
                    "model": item["model"],
                    "messages": [
                        {"role": "system", "content": item["system_prompt"]},
                        {"role": "user", "content": item["prompt"]},
                    ],
                    "temperature": item["temperature"],
                    "max_completion_tokens": item["max_output_tokens"],
                    "n": 1,
                    "service_tier": item["service_tier"],
                    "store": False,
                    "timeout": item["request_timeout_seconds"],
                }
                if item.get("reasoning_effort") is not None:
                    request["reasoning_effort"] = item["reasoning_effort"]
                return clients.openai().chat.completions.create(
                    **request,
                )

            response, retries = with_retries(call)
            response_text = response.choices[0].message.content or ""
            usage = response.usage.model_dump(mode="json") if response.usage else None
            raw = response.model_dump(mode="json")
            resolved_model = response.model
            provider_fingerprint = response.system_fingerprint
            stop_reason = response.choices[0].finish_reason
        elif item["provider"] == "anthropic":
            def call() -> Any:
                return clients.anthropic().messages.create(
                    model=item["model"],
                    system=item["system_prompt"],
                    messages=[{"role": "user", "content": item["prompt"]}],
                    temperature=item["temperature"],
                    max_tokens=item["max_output_tokens"],
                    service_tier=item["service_tier"],
                    timeout=item["request_timeout_seconds"],
                )

            response, retries = with_retries(call)
            response_text = "".join(block.text for block in response.content if block.type == "text")
            usage = response.usage.model_dump(mode="json")
            raw = response.model_dump(mode="json")
            resolved_model = response.model
            provider_fingerprint = None
            stop_reason = response.stop_reason
        else:
            raise ValueError(f"Unsupported provider: {item['provider']}")

        return {
            **item,
            "status": "ok",
            "requested_at": timestamp,
            "latency_seconds": round(time.monotonic() - started, 6),
            "retry_count": retries,
            "resolved_model": resolved_model,
            "provider_fingerprint": provider_fingerprint,
            "stop_reason": stop_reason,
            "response_text": response_text,
            "parsed_choice": parse_choice(response_text),
            "usage": usage,
            "raw_response": raw,
        }
    except Exception as exc:
        return {
            **item,
            "status": "error",
            "requested_at": timestamp,
            "latency_seconds": round(time.monotonic() - started, 6),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def completed_ids(path: Path) -> set[str]:
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
                completed.add(record["request_id"])
    return completed


def stratified_pilot(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select a round-robin pilot balanced across condition and display order."""
    groups: dict[tuple[str, int], deque[dict[str, Any]]] = defaultdict(deque)
    for item in items:
        groups[(item["condition"], item["order"])].append(item)
    selected: list[dict[str, Any]] = []
    strata = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for stratum in strata:
            if groups[stratum] and len(selected) < limit:
                selected.append(groups[stratum].popleft())
                progressed = True
        if not progressed:
            break
    return selected


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()


def validate_environment(providers: set[str]) -> None:
    missing = []
    if "openai" in providers and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if "anthropic" in providers and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", action="append", choices=["openai", "anthropic"])
    parser.add_argument("--condition", action="append")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--limit", type=int, help="Run only the first N pending requests (pilot use only).")
    parser.add_argument(
        "--stratified-pilot",
        action="store_true",
        help="With --limit, balance the pilot across condition and display order.",
    )
    parser.add_argument("--execute", action="store_true", help="Make paid API calls; otherwise only summarize.")
    parser.add_argument(
        "--confirm-request-count",
        type=int,
        help="Required with --execute; must equal the number of pending paid requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file, override=False)
    config = load_json(args.config)
    tasks = load_json(args.tasks)
    providers = set(args.provider or [model["provider"] for model in config["models"]])
    conditions = set(args.condition) if args.condition else None
    manifest = build_manifest(config, tasks, providers, conditions)

    completed = completed_ids(args.output)
    pending = [item for item in manifest if item["request_id"] not in completed]
    random.Random(config["execution_seed"]).shuffle(pending)
    if args.limit is not None:
        pending = stratified_pilot(pending, args.limit) if args.stratified_pilot else pending[: args.limit]

    summary = {
        "experiment_id": config["experiment_id"],
        "tasks": len(tasks),
        "conditions": len({item["condition"] for item in manifest}),
        "models": len({(item["provider"], item["model"]) for item in manifest}),
        "total_manifest_requests": len(manifest),
        "already_completed": len(completed & {item["request_id"] for item in manifest}),
        "pending_requests": len(pending),
        "pending_by_condition_and_order": {
            f"{condition}/order_{order}": sum(
                item["condition"] == condition and item["order"] == order
                for item in pending
            )
            for condition, order in sorted({(item["condition"], item["order"]) for item in pending})
        },
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))

    if args.manifest_out:
        write_jsonl(args.manifest_out, manifest)
        print(f"Wrote manifest: {args.manifest_out}")

    if not args.execute:
        print("Dry run only. Add --execute and --confirm-request-count to make API calls.")
        return 0

    if args.confirm_request_count != len(pending):
        raise SystemExit(
            f"Refusing paid run: --confirm-request-count must equal pending count {len(pending)}."
        )
    validate_environment(providers)

    model_config = {(model["provider"], model["model"]): model for model in config["models"]}
    clients = ProviderClients()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for provider in sorted(providers):
        provider_items = [item for item in pending if item["provider"] == provider]
        if not provider_items:
            continue
        model = model_config[(provider, provider_items[0]["model"])]
        print(f"Running {len(provider_items)} {provider} requests at concurrency {model['concurrency']}...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=model["concurrency"]) as executor:
            futures = [executor.submit(run_one, item, clients) for item in provider_items]
            with args.output.open("a", encoding="utf-8") as handle:
                for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    record = future.result()
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    if index % 100 == 0 or index == len(futures):
                        print(f"{provider}: {index}/{len(futures)} completed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
