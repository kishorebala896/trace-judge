"""Tests for trace_judge. Run: python -m unittest discover -s tests -v"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace_judge import Trace, evaluate  # noqa: E402


def load_example(name):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "examples", name)) as f:
        return Trace(json.load(f))


def trace(**overrides):
    base = {"task_id": "t", "task": "Do the thing.", "steps": []}
    base.update(overrides)
    return Trace(base)


class TestOutcome(unittest.TestCase):
    def test_good_trace_passes(self):
        rep = evaluate(load_example("good-trace.json"))
        self.assertEqual(rep["verdict"], "PASS")
        self.assertAlmostEqual(rep["overall"], 1.0)

    def test_bad_trace_fails(self):
        rep = evaluate(load_example("bad-trace.json"))
        self.assertEqual(rep["verdict"], "FAIL")

    def test_wrong_answer_is_zero(self):
        t = trace(expected_answer="42",
                  steps=[{"kind": "answer", "text": "The answer is 7."}])
        rep = evaluate(t)
        self.assertEqual(rep["results"][0].score, 0.0)


class TestGrounding(unittest.TestCase):
    def test_invented_number_flagged(self):
        t = trace(
            task="What is the cheapest quote?",
            expected_answer="74.50",
            steps=[
                {"kind": "tool_call", "tool": "quotes.get", "args": {"v": "Beta"}},
                {"kind": "tool_result", "tool": "quotes.get", "ok": True,
                 "result": "Beta quote $74.50"},
                {"kind": "answer",
                 "text": "Cheapest is Beta at $74.50, plus $4.99 shipping."},
            ])
        rep = evaluate(t)
        grounding = rep["results"][1]
        self.assertEqual(grounding.score, 0.5)
        self.assertIn("4.99", grounding.findings[0].text)

    def test_checked_derivation_counts_as_grounded(self):
        t = trace(
            task="Add INV-101 and INV-102.",
            expected_answer="343.80",
            steps=[
                {"kind": "tool_result", "tool": "ledger.lookup", "ok": True,
                 "result": "INV-101 total $142.50"},
                {"kind": "tool_result", "tool": "ledger.lookup", "ok": True,
                 "result": "INV-102 total $201.30"},
                {"kind": "answer", "text": "Total $343.80 "
                 "(142.50 + 201.30 = 343.80)."},
            ])
        self.assertEqual(evaluate(t)["results"][1].score, 1.0)

    def test_wrong_derivation_not_grounded(self):
        t = trace(
            task="Add INV-101 and INV-102.",
            expected_answer="343.80",
            steps=[
                {"kind": "tool_result", "tool": "ledger.lookup", "ok": True,
                 "result": "INV-101 total $142.50"},
                {"kind": "tool_result", "tool": "ledger.lookup", "ok": True,
                 "result": "INV-102 total $201.30"},
                {"kind": "answer", "text": "Total $999.99 "
                 "(142.50 + 201.30 = 999.99)."},
            ])
        self.assertLess(evaluate(t)["results"][1].score, 1.0)


class TestToolUse(unittest.TestCase):
    def test_duplicate_call_flagged(self):
        call = {"kind": "tool_call", "tool": "x.search", "args": {"q": "a"}}
        t = trace(steps=[call, dict(call),
                         {"kind": "answer", "text": "done"}])
        rep = evaluate(t)
        self.assertEqual(rep["results"][2].score, 0.75)
        self.assertTrue(any("repeat" in f.text for f in rep["results"][2].findings))

    def test_ignored_error_flagged(self):
        t = trace(steps=[
            {"kind": "tool_call", "tool": "db.get", "args": {"id": 1}},
            {"kind": "tool_result", "tool": "db.get", "ok": False,
             "result": "connection refused"},
            {"kind": "answer", "text": "The value is 5."},
        ])
        rep = evaluate(t)
        self.assertTrue(any("silent failure" in f.text
                            for f in rep["results"][2].findings))

    def test_retried_error_is_fine(self):
        t = trace(steps=[
            {"kind": "tool_call", "tool": "db.get", "args": {"id": 1}},
            {"kind": "tool_result", "tool": "db.get", "ok": False,
             "result": "connection refused"},
            {"kind": "tool_call", "tool": "db.get", "args": {"id": 1}},
            {"kind": "tool_result", "tool": "db.get", "ok": True,
             "result": "value is 5"},
            {"kind": "answer", "text": "The value is 5."},
        ])
        self.assertEqual(evaluate(t)["results"][2].score, 1.0)

    def test_different_args_not_a_duplicate(self):
        t = trace(steps=[
            {"kind": "tool_call", "tool": "db.get", "args": {"id": 1}},
            {"kind": "tool_call", "tool": "db.get", "args": {"id": 2}},
            {"kind": "answer", "text": "done"},
        ])
        self.assertEqual(evaluate(t)["results"][2].score, 1.0)


class TestEconomy(unittest.TestCase):
    def test_cost_math(self):
        t = trace(par_calls=2, steps=[
            {"kind": "tool_call", "tool": "x", "args": {},
             "tokens_in": 1000000, "tokens_out": 1000000},
            {"kind": "answer", "text": "done"},
        ])
        # 1M in @ $3 + 1M out @ $15 = $18
        self.assertAlmostEqual(t.cost(), 18.0)
        # 1 call vs par 2 -> full marks
        self.assertEqual(evaluate(t)["results"][4].score, 1.0)


if __name__ == "__main__":
    unittest.main()
