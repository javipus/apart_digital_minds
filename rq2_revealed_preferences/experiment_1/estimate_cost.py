#!/usr/bin/env python3
"""Estimate Experiment 1 token usage and standard API cost."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from run_experiment import DEFAULT_CONFIG, DEFAULT_TASKS, build_manifest, load_json


def approximate_input_tokens(text: str, system_prompt: str) -> int:
    """Transparent heuristic for short English chat prompts without provider tokenizers."""
    words = len(re.findall(r"\b[\w’'-]+\b", f"{system_prompt} {text}"))
    return math.ceil(words * 4 / 3) + 10  # punctuation/tokenization and chat-message overhead


def estimate(config: dict[str, Any], tasks: list[dict[str, Any]], framed_output_tokens: int) -> dict[str, Any]:
    manifest = build_manifest(config, tasks)
    models = {(m["provider"], m["model"]): m for m in config["models"]}
    rows = []
    for key, model in models.items():
        requests = [item for item in manifest if (item["provider"], item["model"]) == key]
        input_tokens = sum(approximate_input_tokens(item["prompt"], item["system_prompt"]) for item in requests)
        output_tokens = sum(
            2 if item["condition"] == "stated" else framed_output_tokens
            for item in requests
        )
        price = model["pricing_usd_per_million"]
        input_cost = input_tokens / 1_000_000 * price["input"]
        output_cost = output_tokens / 1_000_000 * price["output"]
        rows.append({
            "provider": model["provider"],
            "model": model["model"],
            "requests": len(requests),
            "estimated_input_tokens": input_tokens,
            "assumed_output_tokens": output_tokens,
            "estimated_input_cost_usd": round(input_cost, 4),
            "estimated_output_cost_usd": round(output_cost, 4),
            "estimated_total_cost_usd": round(input_cost + output_cost, 4),
        })
    return {
        "method": "English word count * 4/3 + 10 chat-overhead tokens per request",
        "stated_output_tokens_per_request": 2,
        "consequential_output_tokens_per_request": framed_output_tokens,
        "models": rows,
        "requests_total": sum(row["requests"] for row in rows),
        "estimated_total_cost_usd": round(sum(row["estimated_total_cost_usd"] for row in rows), 4),
        "estimated_batch_cost_usd": round(sum(row["estimated_total_cost_usd"] for row in rows) / 2, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--framed-output-tokens", type=int, default=25)
    args = parser.parse_args()
    result = estimate(load_json(args.config), load_json(args.tasks), args.framed_output_tokens)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
