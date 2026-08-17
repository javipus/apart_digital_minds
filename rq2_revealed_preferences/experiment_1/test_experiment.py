#!/usr/bin/env python3

import json
import re
import unittest
from pathlib import Path

from run_experiment import build_manifest, parse_choice, stratified_pilot


HERE = Path(__file__).resolve().parent


class ExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((HERE / "config.json").read_text())
        cls.tasks = json.loads((HERE / "tasks.json").read_text())
        cls.manifest = build_manifest(cls.config, cls.tasks)

    def test_task_intersection(self) -> None:
        expected = {133, *range(135, 148), *range(149, 162)}
        self.assertEqual({task["outcome_id"] for task in self.tasks}, expected)
        self.assertEqual(len(self.tasks), 27)

    def test_time_removed(self) -> None:
        time_pattern = re.compile(r"\b(?:minutes?|hours?|days?|weeks?)\b", re.IGNORECASE)
        for task in self.tasks:
            self.assertIsNone(time_pattern.search(task["stated"]))
            self.assertIsNone(time_pattern.search(task["consequential"]))

    def test_full_factorial_count(self) -> None:
        self.assertEqual(len(self.manifest), 21_060)
        self.assertEqual(len({row["request_id"] for row in self.manifest}), 21_060)

    def test_both_orders(self) -> None:
        rows = [
            row for row in self.manifest
            if row["provider"] == "openai"
            and row["condition"] == "stated"
            and row["pair_low_id"] == 133
            and row["pair_high_id"] == 135
        ]
        self.assertEqual(len(rows), 10)
        self.assertEqual({(row["task_a_id"], row["task_b_id"]) for row in rows}, {(133, 135), (135, 133)})

    def test_choice_parser(self) -> None:
        self.assertEqual(parse_choice("A"), "A")
        self.assertEqual(parse_choice('"B".'), "B")
        self.assertEqual(parse_choice("I'd choose option A."), "A")
        self.assertEqual(parse_choice("I prefer task B because it is interesting."), "B")
        self.assertIsNone(parse_choice("Either one is fine."))
        self.assertIsNone(parse_choice("Option A or option B would work."))

    def test_stratified_pilot(self) -> None:
        openai_rows = [row for row in self.manifest if row["provider"] == "openai"]
        pilot = stratified_pilot(openai_rows, 50)
        counts = {
            (condition, order): sum(
                row["condition"] == condition and row["order"] == order for row in pilot
            )
            for condition in {row["condition"] for row in pilot}
            for order in {0, 1}
        }
        self.assertEqual(len(pilot), 50)
        self.assertEqual(set(counts.values()), {8, 9})


if __name__ == "__main__":
    unittest.main()
