import copy
import unittest

from evaluate import evaluate_records, evidence_f1
from validate_submission import answer_errors, trace_errors, validate_records


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.questions, self.gold, self.answers, self.traces = [], [], [], []
        for number, category in enumerate(("conditions", "comparison", "calculation", "insufficient")):
            question_id = f"q{number}"
            abstain = category == "insufficient"
            self.questions.append({"id": question_id, "question": "Test", "options": [{"id": item, "text": item} for item in ("right", "second", "third", "fourth")]})
            expected = {"id": question_id, "category": category, "answer_ids": [] if abstain else ["right"], "abstain": abstain, "evidence_sets": [] if abstain else [["source1", "source2"], ["source3"]], "rationale": "Reason"}
            if abstain:
                expected["missing_information"] = "Orientation"
            self.gold.append(expected)
            self.answers.append({"id": question_id, "answer_ids": [] if abstain else ["right"], "abstain": abstain, "rationale": "Reason", "citations": [] if abstain else ["source3"]})
            self.traces.append({"id": question_id, "llm_calls": 1, "input_tokens": 100, "output_tokens": 25, "elapsed_ms": 10 + number, "retrieved_block_ids": ["source1", "source2", "source3"], "usage_verified": True, "requests": [{"input_tokens": 100, "output_tokens": 25, "max_tokens": 200}]})
        self.corpus = {"source1", "source2", "source3"}

    def evaluate(self, **kwargs):
        return evaluate_records(self.questions, self.gold, self.answers, self.traces, self.corpus, **kwargs)

    def test_perfect_and_cost(self):
        result = self.evaluate(baseline_cost=456)
        self.assertEqual(result["Q"], 1)
        self.assertEqual(result["mean_cost"], 456)
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["validation_errors"], [])

    def test_content_failure_zeroes_quality_but_preserves_id_accuracy(self):
        result = self.evaluate(rationale_failures={"q0": "Contradiction"}, baseline_cost=456)
        self.assertEqual(result["answer_accuracy"], 1)
        self.assertEqual(result["Q"], .75)
        self.assertEqual(result["score"], 75)
        self.assertEqual(result["per_question"][0]["rationale_error"], "Contradiction")

    def test_rationale_review_cannot_be_reused_for_changed_answer(self):
        import csv
        import tempfile
        from pathlib import Path
        from evaluate import answer_hash, read_rationale_audit
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["name", "id", "answer_sha256", "valid", "reason"])
                writer.writeheader()
                writer.writerow({"name": "final", "id": "q0", "answer_sha256": answer_hash(self.answers[0]),
                                 "valid": "false", "reason": "Contradiction"})
            self.assertIn("q0", read_rationale_audit(path, "final", self.answers))
            self.answers[0]["rationale"] = "Changed"
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_rationale_audit(path, "final", self.answers)

    def test_best_gold_evidence_set(self):
        self.assertEqual(evidence_f1(["source1"], ["source1", "source2"]), 2 / 3)
        self.answers[0]["citations"] = ["source1"]
        result = self.evaluate()
        self.assertAlmostEqual(result["per_question"][0]["score"], 0.8 + 0.2 * 2 / 3)

    def test_wrong_option_zeroes_citation_score(self):
        self.answers[0]["answer_ids"] = ["second"]
        result = self.evaluate()
        self.assertEqual(result["Q"], 0.75)
        self.assertEqual(result["per_question"][0]["score"], 0)

    def test_macro_is_not_micro_average(self):
        self.answers[0]["answer_ids"] = ["second"]
        for rows in (self.questions, self.gold, self.answers, self.traces):
            row = copy.deepcopy(rows[1])
            row["id"] = "extra_comparison"
            rows.append(row)
        result = self.evaluate()
        self.assertEqual(result["answer_accuracy"], 0.8)
        self.assertEqual(result["Q"], 0.75)

    def test_duplicate_and_missing_remain_in_denominator(self):
        self.answers.append(copy.deepcopy(self.answers[0]))
        self.answers = [row for row in self.answers if row["id"] != "q1"]
        result = self.evaluate()
        self.assertEqual(result["Q"], 0.5)
        self.assertTrue(result["validation_errors"])

    def test_non_boolean_abstention_is_invalid(self):
        self.answers[3]["abstain"] = 1
        self.assertEqual(self.evaluate()["Q"], 0.75)

    def test_arbitrary_option_ids_and_permutation(self):
        self.questions[0]["options"].reverse()
        self.assertEqual(self.evaluate()["Q"], 1)

    def test_rationale_and_unknown_citation(self):
        self.answers[0]["rationale"] = "x" * 1201
        self.assertIn("rationale must contain 1 to 1200 characters", answer_errors(self.questions[0], self.answers[0], self.corpus))
        self.answers[0]["citations"] = ["invented"]
        self.assertIn("unknown citation id", answer_errors(self.questions[0], self.answers[0], self.corpus))

    def test_abstention_lists_must_be_empty(self):
        self.answers[3]["citations"] = ["source3"]
        self.assertEqual(self.evaluate()["Q"], 0.75)

    def test_limit_disables_global_efficiency(self):
        self.traces[0]["requests"][0]["max_tokens"] = 8093
        result = self.evaluate(baseline_cost=1000)
        self.assertEqual(result["Q"], 0.75)
        self.assertFalse(result["efficiency_eligible"])
        self.assertEqual(result["E"], 0)
        self.assertEqual(result["score"], 56.25)

    def test_aggregate_limits_and_unverified_usage(self):
        trace = copy.deepcopy(self.traces[0])
        trace["llm_calls"] = 5
        trace["input_tokens"] = 16385
        trace["output_tokens"] = 4097
        trace["usage_verified"] = False
        errors = trace_errors(trace)
        for fragment in ("llm_calls exceeds", "input_tokens exceeds", "output_tokens exceeds", "usage is not verified"):
            self.assertTrue(any(fragment in error for error in errors))

    def test_retry_usage_must_be_counted(self):
        trace = self.traces[0]
        trace["requests"].append({"input_tokens": 50, "output_tokens": 10, "max_tokens": 200})
        self.assertTrue(trace_errors(trace))
        trace.update(llm_calls=2, input_tokens=150, output_tokens=35)
        self.assertFalse(trace_errors(trace))

    def test_no_context_quality_is_blank(self):
        for answer in self.answers:
            answer["citations"] = []
        result = self.evaluate(no_context=True)
        self.assertEqual(result["answer_accuracy"], 1)
        self.assertIsNone(result["Q"])
        self.assertFalse(result["validation_errors"])

    def test_unknown_rows_and_nonretrieved_citation(self):
        self.answers.append({**self.answers[0], "id": "unknown"})
        self.traces[0]["retrieved_block_ids"] = []
        errors = validate_records(self.questions, self.answers, self.traces, self.corpus)
        self.assertTrue(any("unexpected answer" in error for error in errors))
        self.assertTrue(any("citation was not retrieved" in error for error in errors))

    def test_missing_category_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "missing evaluation categories"):
            evaluate_records(self.questions[:3], self.gold[:3], self.answers[:3], self.traces[:3])

    def test_missing_request_budget_is_unverified(self):
        del self.traces[0]["requests"]
        result = self.evaluate(baseline_cost=456)
        self.assertFalse(result["efficiency_eligible"])
        self.assertEqual(result["E"], 0)

    def test_malformed_id_is_reported(self):
        self.answers[0]["id"] = []
        errors = validate_records(self.questions, self.answers, self.traces, self.corpus)
        self.assertIn("answer: invalid id", errors)

    def test_abstention_gold_may_cite_evidence_for_missing_information(self):
        self.gold[3]["evidence_sets"] = [["source1"]]
        self.assertEqual(self.evaluate()["Q"], 1)

    def test_technical_abstention_does_not_inflate_insufficient_score(self):
        for error in ("format", "budget"):
            with self.subTest(error=error):
                self.traces[3]["answer_error"] = error
                result = self.evaluate(baseline_cost=456)
                self.assertEqual(result["Q"], 0.75)
                self.assertEqual(result["answer_accuracy"], 0.75)
                self.assertEqual(result["per_question"][3]["score"], 0)
                self.assertTrue(result["efficiency_eligible"])
                self.assertEqual(result["E"], 1)
                self.assertFalse(result["validation_errors"])

    def test_zero_call_budget_is_known_usage_but_zero_quality(self):
        self.traces[3].update(llm_calls=0, input_tokens=0, output_tokens=0,
                              requests=[], answer_error="budget")
        result = self.evaluate(baseline_cost=456)
        self.assertEqual(result["Q"], 0.75)
        self.assertTrue(result["efficiency_eligible"])
        self.assertEqual(result["mean_llm_calls"], 0.75)


if __name__ == "__main__":
    unittest.main()
