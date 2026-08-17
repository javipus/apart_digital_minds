# Experiment 1 runner

This directory contains the versioned task fixtures, prompt and model
configuration, cost estimator, and resumable API runner for the stated versus
consequentially framed task-choice experiment.

## Validate and inspect

```bash
cd rq2_revealed_preferences/experiment_1
python3 -m unittest test_experiment.py
python3 run_experiment.py
python3 estimate_cost.py
```

The runner is dry-run-only unless `--execute` is supplied. To save the complete
request manifest without making API calls:

```bash
python3 run_experiment.py --manifest-out data/request_manifest.jsonl
```

## Pilot

Copy the credential template and add both API keys:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

The local `.env` file is ignored by version control. The runner loads it from
this directory automatically; an existing shell environment variable takes
precedence. You can select another file with `--env-file path/to/file`.

Then run a small provider-specific pilot. The confirmation count is deliberately
required to prevent accidental paid runs.

```bash
python3 run_experiment.py \
  --provider openai \
  --condition stated \
  --limit 20 \
  --execute \
  --confirm-request-count 20
```

Use `--provider anthropic` for Haiku. Successful request IDs are read from the
append-only output file and skipped on subsequent runs. Errors remain in the log
but are retried when the runner is resumed.

## Full run

With an empty output file, the confirmation count for the complete experiment
is 21,060:

```bash
python3 run_experiment.py --execute --confirm-request-count 21060
```

Run a dry run immediately beforehand. If a pilot has already written successful
records to the output file, use the newly reported pending count rather than
21,060.

Raw records include the rendered prompt, condition and option-order metadata,
provider response, token usage, latency, stop reason, conservative automatic
A/B parse, and the untouched provider response object. Consequential-condition
parses should still be audited or labeled separately.

## Audit and label the consequential completions

Audit the raw file against the deterministic 21,060-request manifest:

```bash
python3 audit_results.py --report data/audit_report.json
```

The labeling pass uses the pinned `gpt-5-nano-2025-08-07` snapshot with minimal
reasoning and Structured Outputs. It labels all consequential completions as
`A`, `B`, or `UNCLEAR`, while preserving the original parser result and stop
reason for quality checks. The raw response file is never modified.

First inspect the dry run and its approximate cost:

```bash
python3 label_consequential.py
```

Then run a small pilot:

```bash
python3 label_consequential.py \
  --limit 20 \
  --execute \
  --confirm-request-count 20
```

After inspecting `data/consequential_labels.jsonl`, resume the full pass using
the pending count printed by a fresh dry run. The label file is append-only;
successful label IDs are skipped, while errors are eligible for retry.
