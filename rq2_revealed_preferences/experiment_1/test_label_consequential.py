#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from label_consequential import (
    DEFAULT_MODEL,
    build_label_input,
    label_id,
    load_source_records,
    parse_label_output,
)


class ConsequentialLabelTest(unittest.TestCase):
    def test_label_id_is_stable_and_model_specific(self) -> None:
        self.assertEqual(label_id("request-1"), label_id("request-1", DEFAULT_MODEL))
        self.assertNotEqual(label_id("request-1"), label_id("request-1", "another-model"))

    def test_build_label_input(self) -> None:
        value = build_label_input({"task_a": "alpha", "task_b": "beta", "response_text": "I pick beta."})
        self.assertIn("Option A: alpha", value)
        self.assertIn("Option B: beta", value)
        self.assertTrue(value.endswith("I pick beta."))

    def test_source_filter_and_resume_semantics(self) -> None:
        rows = [
            {"request_id": "one", "status": "ok", "condition": "stated", "response_text": "A"},
            {"request_id": "two", "status": "error", "condition": "consequential_time"},
            {"request_id": "two", "status": "ok", "condition": "consequential_time", "response_text": "B"},
            {"request_id": "three", "status": "ok", "condition": "consequential_tokens", "response_text": "A"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            loaded = load_source_records(path)
        self.assertEqual({row["request_id"] for row in loaded}, {"two", "three"})

    def test_label_validation(self) -> None:
        valid = '{"choice":"B","basis":"explicit_task_selection","evidence":"I pick beta"}'
        self.assertEqual(parse_label_output(valid)["choice"], "B")
        with self.assertRaises(ValueError):
            parse_label_output('{"choice":"UNCLEAR","basis":"comparative_preference","evidence":""}')


if __name__ == "__main__":
    unittest.main()
