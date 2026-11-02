import argparse
import copy
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from email.message import Message
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlsplit

from llm_client import BudgetError, Client
from predict import guard_output_paths, pack_context, parse_answer, rewrite_queries, run
from retrieval import Index, build_payload


class FixedTokenizer:
    def __init__(self, count=100):
        self.tokens = count

    def count(self, messages):
        return self.tokens


class SizeTokenizer:
    def count(self, messages):
        return len(json.dumps(messages, ensure_ascii=False)) // 4


def runtime():
    return {
        "model_id": "Qwen/Qwen3.5-4B", "timeout_seconds": 3,
        "generation": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "seed": 42,
                       "presence_penalty": 1.5, "repetition_penalty": 1.0,
                       "max_tokens": 1100, "chat_template_kwargs": {"enable_thinking": False}},
        "limits": {"llm_calls": 4, "input_tokens": 16384,
                   "output_tokens": 4096, "request_tokens": 8192},
    }


def completion(content, input_tokens=100, output_tokens=20):
    return {
        "id": "mock-response", "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
    }


def answer(option_id="chosen", source_id="lemma:42"):
    return json.dumps({"answer_ids": [option_id], "abstain": False,
                       "rationale": "По определению", "citations": [source_id]}, ensure_ascii=False)


@contextmanager
def endpoint(responses):
    records = []
    queue = list(responses)

    class MockEndpoint:
        def open(self, request, timeout):
            headers = Message()
            for name, value in request.header_items():
                headers[name] = value
            records.append({"path": urlsplit(request.full_url).path, "headers": headers,
                            "body": json.loads(request.data)})
            status, payload = queue.pop(0) if queue else (500, {"error": "Unexpected extra call"})
            raw = json.dumps(payload).encode()
            if status >= 400:
                raise HTTPError(request.full_url, status, "Mock HTTP error", {}, BytesIO(raw))
            return BytesIO(raw)

    environment = {
        "LLM_BASE_URL": "http://127.0.0.1:9876/v1",
        "LLM_MODEL": "endpoint-qwen", "LLM_API_KEY": "test-key",
        "SUBMISSION_ID": "test-submission",
    }
    with patch.dict(os.environ, environment):
        with patch("llm_client.build_opener", return_value=MockEndpoint()):
            yield records


class ClientTests(unittest.TestCase):
    def test_endpoint_headers_generation_and_verified_usage(self):
        requests = []
        messages = [{"role": "user", "content": "Test"}]
        with endpoint([(200, completion(answer()))]) as records:
            result = Client(runtime(), FixedTokenizer()).complete("q-17", messages, requests)
        self.assertEqual(result, answer())
        self.assertEqual(len(records), 1)
        request = records[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["headers"]["X-Question-ID"], "q-17")
        self.assertEqual(request["headers"]["X-Submission-ID"], "test-submission")
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(request["body"]["model"], "endpoint-qwen")
        self.assertEqual(request["body"]["messages"], messages)
        for name, value in runtime()["generation"].items():
            self.assertEqual(request["body"][name], value)
        self.assertEqual(requests[0]["input_tokens"], 100)
        self.assertEqual(requests[0]["output_tokens"], 20)
        self.assertTrue(requests[0]["usage_verified"])

    def test_budget_checks_prevent_any_extra_http_call(self):
        cases = [
            (FixedTokenizer(7061), []),
            (FixedTokenizer(), [{"input_tokens": 1, "output_tokens": 1}] * 4),
            (FixedTokenizer(), [{"input_tokens": 16253, "output_tokens": 1}]),
            (FixedTokenizer(), [{"input_tokens": 1, "output_tokens": 2997}]),
        ]
        for tokenizer, history in cases:
            with self.subTest(history=history, count=tokenizer.tokens):
                requests = copy.deepcopy(history)
                with endpoint([]) as records:
                    with self.assertRaisesRegex(BudgetError, "budget"):
                        Client(runtime(), tokenizer).complete("q", [], requests)
                self.assertEqual(records, [])
                self.assertEqual(requests, history)

    def test_http_error_is_not_retried_or_marked_verified(self):
        requests = []
        with endpoint([(503, {"error": "Temporary failure"})]) as records:
            with self.assertRaises(Exception):
                Client(runtime(), FixedTokenizer()).complete("q", [], requests)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(requests), 1)
        self.assertFalse(requests[0]["usage_verified"])

    def test_missing_usage_and_actual_overbudget_are_rejected(self):
        for payload in ({"choices": [{"message": {"content": answer()}}]}, completion(answer(), 8000)):
            with self.subTest(payload=payload):
                requests = []
                with endpoint([(200, payload)]) as records:
                    with self.assertRaises(ValueError):
                        Client(runtime(), FixedTokenizer()).complete("q", [], requests)
                self.assertEqual(len(records), 1)
                self.assertEqual(len(requests), 1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.settings = {"chunk_chars": 2200, "overlap_chars": 300, "rrf_k": 30,
                         "top_k": 8, "baseline_top_k": 5, "neighbor_limit": 8}
        self.blocks = [{"block_id": "lemma:42", "page": 1, "section": "Голоморфность",
                        "text": "Голоморфная функция комплексно дифференцируема."}]
        self.questions = [{"id": "question-9", "question": "Что такое голоморфность?",
                           "options": [{"id": value, "text": text} for value, text in
                                       (("third-id", "Непрерывность"), ("chosen", "Дифференцируемость"),
                                        ("opt_9", "Ограниченность"), ("Δ", "Постоянство"))]}]
        self.index_path = self.root / "custom-index"
        self.index_path.mkdir()
        (self.index_path / "index.json").write_text(json.dumps(build_payload(self.blocks, self.settings)), encoding="utf-8")
        self.config = {"runtime": "runtime.json", "system": {"mode": "baseline", "context_tokens": 6000},
                       "retrieval": self.settings}
        (self.root / "runtime.json").write_text(json.dumps(runtime()), encoding="utf-8")
        (self.root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        self.questions_path = self.root / "new-questions.jsonl"

    def arguments(self):
        self.questions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.questions), encoding="utf-8")
        return argparse.Namespace(index=str(self.index_path), questions=str(self.questions_path),
                                  output=str(self.root / "results" / "answers.jsonl"),
                                  trace=str(self.root / "results" / "trace.jsonl"),
                                  config=str(self.root / "config.json"), mode=None)

    def test_predict_without_gold_and_with_permuted_arbitrary_ids(self):
        second = copy.deepcopy(self.questions[0])
        second["id"] = "question-10"
        second["options"].reverse()
        self.questions.append(second)
        args = self.arguments()
        raw = json.dumps({"option_verdicts": {option["id"]: option["id"] == "chosen"
                                              for option in self.questions[0]["options"]},
                          "abstain": False, "rationale": "По определению", "citations": ["lemma:42"]})
        with endpoint([(200, completion(raw)), (200, completion(raw))]) as records:
            with patch("predict.ChatTokenizer", return_value=FixedTokenizer()):
                run(args)
        results = [json.loads(line) for line in Path(args.output).read_text().splitlines()]
        traces = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
        self.assertEqual([row["id"] for row in results], [row["id"] for row in self.questions])
        self.assertEqual([row["answer_ids"] for row in results], [["chosen"], ["chosen"]])
        self.assertEqual([record["headers"]["X-Question-ID"] for record in records], ["question-9", "question-10"])
        for result, trace in zip(results, traces):
            self.assertEqual(set(result), {"id", "answer_ids", "abstain", "rationale", "citations"})
            self.assertEqual(trace["llm_calls"], 1)
            self.assertEqual(trace["input_tokens"], sum(request["input_tokens"] for request in trace["requests"]))
            self.assertEqual(trace["output_tokens"], sum(request["output_tokens"] for request in trace["requests"]))
            self.assertTrue(trace["usage_verified"])
        self.assertFalse(list(self.root.rglob("*gold*")))

    def test_format_repair_counts_both_calls_and_keeps_sampling(self):
        args = self.arguments()
        args.mode = "no_rewrite"
        with endpoint([(200, completion("not json", 100, 7)), (200, completion(answer(), 120, 20))]) as records:
            with patch("predict.ChatTokenizer", return_value=FixedTokenizer()):
                run(args)
        trace = json.loads(Path(args.trace).read_text())
        self.assertEqual(len(records), 2)
        self.assertEqual(trace["llm_calls"], 2)
        self.assertEqual(trace["input_tokens"], 220)
        self.assertEqual(trace["output_tokens"], 27)
        for parameter in runtime()["generation"]:
            self.assertEqual(records[0]["body"][parameter], records[1]["body"][parameter])
        self.assertEqual(json.loads(Path(args.output).read_text())["answer_ids"], ["chosen"])

    def test_rewriting_counts_queries_and_answer_calls(self):
        args = self.arguments()
        args.mode = "no_dense"
        queries = ["голоморфность определение", "комплексная производная"]
        responses = [(200, completion(json.dumps({"queries": queries}), 80, 12)),
                     (200, completion(answer(), 130, 25))]
        with endpoint(responses) as records:
            with patch("predict.ChatTokenizer", return_value=FixedTokenizer()):
                run(args)
        trace = json.loads(Path(args.trace).read_text())
        self.assertEqual(trace["search_queries"], queries)
        self.assertEqual(trace["llm_calls"], 2)
        self.assertEqual(trace["input_tokens"], 210)
        self.assertEqual(trace["output_tokens"], 37)
        self.assertEqual([row["headers"]["X-Question-ID"] for row in records], ["question-9", "question-9"])
        for parameter in runtime()["generation"]:
            self.assertEqual(records[0]["body"][parameter], records[1]["body"][parameter])

    def test_output_guards_cover_runtime_index_symlinks_and_hardlinks(self):
        args = self.arguments()
        config = {**self.config, "_root": str(self.root)}
        for protected in (self.root / "runtime.json", self.index_path / "index.json", self.questions_path,
                          self.root / "config.json", self.index_path / "not-created.json"):
            args.output = str(protected)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                guard_output_paths(args, config)
        for kind, protected in (("symlink", self.root / "runtime.json"),
                                ("hardlink", self.index_path / "index.json")):
            alias = self.root / (kind + "-output.json")
            if kind == "symlink":
                alias.symlink_to(protected)
            else:
                os.link(protected, alias)
            args.output = str(alias)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                guard_output_paths(args, config)
        first = self.root / "first-output.json"
        first.write_text("existing", encoding="utf-8")
        second = self.root / "second-output.json"
        os.link(first, second)
        args.output, args.trace = str(first), str(second)
        with self.assertRaisesRegex(ValueError, "must differ"):
            guard_output_paths(args, config)

    def test_invalid_citations_cannot_escape_context(self):
        with self.assertRaisesRegex(ValueError, "Unknown option or citation"):
            parse_answer(answer(source_id="invented"), self.questions[0], {"lemma:42"})
        obj = json.loads(answer())
        obj["abstain"] = True
        with self.assertRaisesRegex(ValueError, "Abstention"):
            parse_answer(json.dumps(obj), self.questions[0], {"lemma:42"})

    def test_boolean_verdicts_use_every_actual_id_and_follow_permutation(self):
        question = copy.deepcopy(self.questions[0])
        question["options"].reverse()
        payload = {"option_verdicts": {option["id"]: option["id"] in {"chosen", "Δ"}
                                      for option in question["options"]},
                   "abstain": False, "rationale": "По определению", "citations": ["lemma:42"]}
        parsed = parse_answer(json.dumps(payload), question, {"lemma:42"})
        self.assertEqual(parsed["answer_ids"], ["Δ", "chosen"])
        self.assertEqual(set(parsed), {"id", "answer_ids", "abstain", "rationale", "citations"})
        for mutation in ("missing", "unknown", "not_boolean"):
            invalid = copy.deepcopy(payload)
            if mutation == "missing":
                del invalid["option_verdicts"]["Δ"]
            elif mutation == "unknown":
                invalid["option_verdicts"]["invented"] = False
            else:
                invalid["option_verdicts"]["chosen"] = 1
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "boolean verdict"):
                parse_answer(json.dumps(invalid), question, {"lemma:42"})

    def test_explicit_verdict_abstention_overrides_selected_verdicts(self):
        payload = {"option_verdicts": {option["id"]: False for option in self.questions[0]["options"]},
                   "abstain": True, "rationale": "Недостаточно данных", "citations": []}
        parsed = parse_answer(json.dumps(payload), self.questions[0], {"lemma:42"})
        self.assertEqual(parsed["answer_ids"], [])
        payload["option_verdicts"]["chosen"] = True
        payload["citations"] = ["lemma:42"]
        parsed = parse_answer(json.dumps(payload), self.questions[0], {"lemma:42"})
        self.assertTrue(parsed["abstain"])
        self.assertEqual(parsed["answer_ids"], [])
        self.assertEqual(parsed["citations"], [])

    def test_empty_verdicts_normalize_to_abstention(self):
        payload = {"option_verdicts": {}, "abstain": True,
                   "rationale": "Недостаточно данных", "citations": []}
        result = parse_answer(json.dumps(payload), self.questions[0], {"lemma:42"})
        self.assertTrue(result["abstain"])
        self.assertEqual(result["answer_ids"], [])
        self.assertEqual(result["citations"], [])
        payload["abstain"] = False
        self.assertTrue(parse_answer(json.dumps(payload), self.questions[0], {"lemma:42"})["abstain"])

    def test_verdict_abstention_accepts_subsets_and_still_checks_rationale(self):
        payload = {"option_verdicts": {"chosen": True}, "abstain": True,
                   "rationale": "Данных для точного значения нет", "citations": ["explanation-source"]}
        parsed = parse_answer(json.dumps(payload), self.questions[0], {"lemma:42"})
        self.assertEqual(parsed["answer_ids"], [])
        self.assertEqual(parsed["citations"], [])
        payload["rationale"] = " "
        with self.assertRaisesRegex(ValueError, "Rationale"):
            parse_answer(json.dumps(payload), self.questions[0], {"lemma:42"})

    def test_missing_abstain_is_inferred_from_complete_verdicts(self):
        payload = {"option_verdicts": {option["id"]: option["id"] == "chosen"
                                      for option in self.questions[0]["options"]},
                   "rationale": "По определению", "citations": ["lemma:42"]}
        parsed = parse_answer(json.dumps(payload), self.questions[0], {"lemma:42"})
        self.assertFalse(parsed["abstain"])
        self.assertEqual(parsed["answer_ids"], ["chosen"])
        payload["option_verdicts"]["chosen"] = False
        parsed = parse_answer(json.dumps(payload), self.questions[0], {"lemma:42"})
        self.assertTrue(parsed["abstain"])
        self.assertEqual(parsed["citations"], [])

    def test_no_true_verdict_becomes_refusal_but_legacy_remains_strict(self):
        payload = {"option_verdicts": {option["id"]: False for option in self.questions[0]["options"]},
                   "abstain": False, "rationale": "Данных недостаточно", "citations": ["lemma:42"]}
        parsed = parse_answer(json.dumps(payload), self.questions[0], {"lemma:42"})
        self.assertTrue(parsed["abstain"])
        self.assertEqual(parsed["answer_ids"], [])
        self.assertEqual(parsed["citations"], [])
        legacy = {"answer_ids": [], "abstain": False, "rationale": "Данных недостаточно", "citations": ["lemma:42"]}
        with self.assertRaisesRegex(ValueError, "Answer requires"):
            parse_answer(json.dumps(legacy), self.questions[0], {"lemma:42"})
        legacy["abstain"] = True
        with self.assertRaisesRegex(ValueError, "Abstention"):
            parse_answer(json.dumps(legacy), self.questions[0], {"lemma:42"})

    def test_missing_information_projects_to_refusal_reason(self):
        payload = {"option_verdicts": {}, "abstain": True, "citations": [],
                   "missing_information": "Не указана ориентация контура"}
        result = parse_answer(json.dumps(payload), self.questions[0], {"lemma:42"})
        self.assertEqual(result["rationale"], "Недостаточно данных: Не указана ориентация контура")
        self.assertEqual(set(result), {"id", "answer_ids", "abstain", "rationale", "citations"})

    def test_catalog_is_trimmed_before_the_planning_request(self):
        blocks = [{"block_id": f"source-{number:03d}", "page": number + 1, "section": "Определения",
                   "text": f"Определение {number}. " + "Аналитическая функция и окрестность точки " * 5}
                  for number in range(400)]
        payload = build_payload(blocks, self.settings)
        (self.index_path / "index.json").write_text(json.dumps(payload), encoding="utf-8")
        index = Index(self.index_path)
        tokenizer = SizeTokenizer()
        plan = {"block_ids": ["source-000"], "queries": ["аналитическая функция"]}
        with endpoint([(200, completion(json.dumps(plan)))]) as records:
            queries, ids, error = rewrite_queries(Client(runtime(), tokenizer), self.questions[0], [], index, tokenizer)
        self.assertIsNone(error)
        self.assertEqual(ids, ["source-000"])
        self.assertEqual(queries, ["аналитическая функция"])
        self.assertEqual(len(records), 1)
        messages = records[0]["body"]["messages"]
        self.assertLessEqual(tokenizer.count(messages), 6200)
        self.assertLess(messages[1]["content"].count("[source-"), len(blocks))
        self.assertIn("[source-000] Определение 0.", messages[1]["content"])

    def test_unknown_catalog_source_is_rejected_without_extra_call(self):
        index = Index(self.index_path)
        tokenizer = FixedTokenizer()
        plan = {"block_ids": ["invented-source"], "queries": ["голоморфность"]}
        requests = []
        with endpoint([(200, completion(json.dumps(plan)))]) as records:
            queries, ids, error = rewrite_queries(Client(runtime(), tokenizer), self.questions[0], requests, index, tokenizer)
        self.assertEqual(queries, [])
        self.assertEqual(ids, [])
        self.assertIn("Unknown planned source ID", error)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0]["usage_verified"])

    def test_zero_call_budget_fallback_continues_to_next_question(self):
        class QuestionTokenizer:
            def count(self, messages):
                return 7000 if "LONG_SENTINEL" in json.dumps(messages) else 100

        second = copy.deepcopy(self.questions[0])
        second["id"] = "question-10"
        self.questions[0]["question"] = "LONG_SENTINEL"
        self.questions.append(second)
        args = self.arguments()
        with endpoint([(200, completion(answer()))]) as records:
            with patch("predict.ChatTokenizer", return_value=QuestionTokenizer()):
                run(args)
        results = [json.loads(line) for line in Path(args.output).read_text().splitlines()]
        traces = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["abstain"])
        self.assertEqual(results[1]["answer_ids"], ["chosen"])
        self.assertEqual(traces[0]["answer_error"], "budget")
        self.assertEqual(traces[0]["llm_calls"], 0)
        self.assertEqual(traces[0]["requests"], [])
        self.assertTrue(traces[0]["usage_verified"])
        self.assertEqual([row["headers"]["X-Question-ID"] for row in records], ["question-10"])

    def test_repair_budget_fallback_preserves_usage_and_next_question(self):
        limited = runtime()
        limited["limits"]["input_tokens"] = 150
        (self.root / "runtime.json").write_text(json.dumps(limited), encoding="utf-8")
        second = copy.deepcopy(self.questions[0])
        second["id"] = "question-10"
        self.questions.append(second)
        args = self.arguments()
        args.mode = "no_rewrite"
        with endpoint([(200, completion("not json", 100, 7)), (200, completion(answer(), 100, 20))]) as records:
            with patch("predict.ChatTokenizer", return_value=FixedTokenizer()):
                run(args)
        results = [json.loads(line) for line in Path(args.output).read_text().splitlines()]
        traces = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
        self.assertTrue(results[0]["abstain"])
        self.assertEqual(results[1]["answer_ids"], ["chosen"])
        self.assertEqual(traces[0]["answer_error"], "budget")
        self.assertEqual(traces[0]["llm_calls"], 1)
        self.assertEqual(traces[0]["input_tokens"], 100)
        self.assertEqual(traces[0]["output_tokens"], 7)
        self.assertTrue(traces[0]["usage_verified"])
        self.assertEqual(len(records), 2)

    def test_failed_format_repair_is_a_technical_abstention(self):
        args = self.arguments()
        args.mode = "no_rewrite"
        with endpoint([(200, completion("invalid", 100, 7)), (200, completion("invalid", 100, 7))]):
            with patch("predict.ChatTokenizer", return_value=FixedTokenizer()):
                run(args)
        result = json.loads(Path(args.output).read_text())
        trace = json.loads(Path(args.trace).read_text())
        self.assertTrue(result["abstain"])
        self.assertEqual(trace["answer_error"], "format")
        self.assertEqual(trace["llm_calls"], 2)
        self.assertTrue(trace["usage_verified"])

    def test_baseline_and_no_context_remain_single_call_on_bad_format(self):
        for mode in ("baseline", "no_context"):
            args = self.arguments()
            args.mode = mode
            with endpoint([(200, completion("invalid", 100, 7))]) as records:
                with patch("predict.ChatTokenizer", return_value=FixedTokenizer()):
                    run(args)
            trace = json.loads(Path(args.trace).read_text())
            self.assertEqual(len(records), 1)
            self.assertEqual(trace["llm_calls"], 1)
            self.assertEqual(trace["answer_error"], "format")

    def test_long_block_retrieves_usable_window(self):
        text = "Начало " * 20000 + " голоморфная функция комплексно дифференцируема "
        payload = build_payload([{**self.blocks[0], "text": text}], self.settings)
        (self.index_path / "index.json").write_text(json.dumps(payload), encoding="utf-8")
        question = {**self.questions[0], "planned_block_ids": ["lemma:42"]}
        candidates = Index(self.index_path).retrieve(question, self.settings, "no_dense")
        config = {**self.config, "_runtime": runtime()}
        blocks, _ = pack_context(self.questions[0], candidates, SizeTokenizer(), config, "final")
        self.assertTrue(blocks, "A relevant indexed window must fit even when its source block is long")
        self.assertTrue(any("дифференцируема" in block["text"] for block in blocks))
        self.assertEqual({block["block_id"] for block in blocks}, {"lemma:42"})

    def test_planned_source_falls_back_to_first_window_without_ranked_hit(self):
        question = {"id": "unmatched", "question": "banana", "planned_block_ids": ["lemma:42"],
                    "options": [{"id": str(number), "text": "unrelated"} for number in range(4)]}
        index = Index(self.index_path)
        self.assertEqual(index.bm25(question["question"]), [])
        candidates = index.retrieve(question, self.settings, "no_dense")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][0]["block_id"], "lemma:42")
        self.assertEqual(candidates[0][0]["text"], self.blocks[0]["text"])


if __name__ == "__main__":
    unittest.main()
