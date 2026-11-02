"""Check answers, source IDs and token budgets without contacting the model."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


ANSWER_FIELDS = {"id", "answer_ids", "abstain", "rationale", "citations"}
TRACE_FIELDS = {
    "id", "llm_calls", "input_tokens", "output_tokens", "elapsed_ms",
    "retrieved_block_ids",
}
LIMITS = {"llm_calls": 4, "input_tokens": 16384, "output_tokens": 4096}


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            records.append(row)
    return records


def is_integer(value):
    return type(value) is int and value >= 0


def string_list(value):
    return isinstance(value, list) and all(
        isinstance(item, str) and bool(item.strip()) for item in value
    )


def unique_string_list(value):
    return string_list(value) and len(value) == len(set(value))


def question_errors(question):
    errors = []
    if not isinstance(question.get("id"), str) or not question["id"].strip():
        errors.append("invalid question id")
    if not isinstance(question.get("question"), str) or not question["question"].strip():
        errors.append("empty question")
    options = question.get("options")
    if not isinstance(options, list) or not 4 <= len(options) <= 6:
        return errors + ["expected 4 to 6 options"]
    option_ids = []
    for option in options:
        if not isinstance(option, dict):
            errors.append("invalid option")
            continue
        if not isinstance(option.get("id"), str) or not option["id"].strip():
            errors.append("invalid option id")
        else:
            option_ids.append(option["id"])
        if not isinstance(option.get("text"), str) or not option["text"].strip():
            errors.append("empty option text")
    if len(option_ids) != len(set(option_ids)):
        errors.append("duplicate option id")
    return errors


def answer_errors(question, answer, corpus_ids=None, no_context=False):
    errors = []
    if set(answer) != ANSWER_FIELDS:
        errors.append("answer fields do not match the schema")
    if answer.get("id") != question.get("id"):
        errors.append("answer id mismatch")
    answer_ids = answer.get("answer_ids")
    citations = answer.get("citations")
    if not unique_string_list(answer_ids):
        errors.append("answer_ids must be unique strings")
    else:
        valid_ids = {item["id"] for item in question.get("options", []) if isinstance(item, dict) and "id" in item}
        if not set(answer_ids) <= valid_ids:
            errors.append("unknown option id")
    if not unique_string_list(citations):
        errors.append("citations must be unique strings")
    else:
        if len(citations) > 4:
            errors.append("more than four citations")
        if corpus_ids is not None and not set(citations) <= set(corpus_ids):
            errors.append("unknown citation id")
    if type(answer.get("abstain")) is not bool:
        errors.append("abstain must be boolean")
    elif answer["abstain"]:
        if answer_ids != [] or citations != []:
            errors.append("abstention must have empty answer_ids and citations")
    else:
        if not answer_ids:
            errors.append("non-abstention must select an option")
        if not no_context and not citations:
            errors.append("non-abstention must cite a source")
    rationale = answer.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1200:
        errors.append("rationale must contain 1 to 1200 characters")
    return errors


def trace_errors(trace, corpus_ids=None):
    errors = []
    if not TRACE_FIELDS <= set(trace):
        errors.append("missing trace fields")
    for field, limit in LIMITS.items():
        if not is_integer(trace.get(field)):
            errors.append(f"invalid {field}")
        elif trace[field] > limit:
            errors.append(f"{field} exceeds {limit}")
    elapsed = trace.get("elapsed_ms")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        errors.append("invalid elapsed_ms")
    retrieved = trace.get("retrieved_block_ids")
    if not unique_string_list(retrieved):
        errors.append("retrieved_block_ids must be unique strings")
    elif corpus_ids is not None and not set(retrieved) <= set(corpus_ids):
        errors.append("unknown retrieved source id")
    if trace.get("llm_calls") == 0 and (
        trace.get("input_tokens") != 0 or trace.get("output_tokens") != 0
    ):
        errors.append("tokens recorded without a model call")
    if type(trace.get("usage_verified")) is not bool:
        errors.append("missing usage_verified boolean")
    elif not trace["usage_verified"]:
        errors.append("model usage is not verified")
    requests = trace.get("requests")
    if requests is None and trace.get("llm_calls") != 0:
        errors.append("missing per-request usage")
    if requests is not None:
        if not isinstance(requests, list) or not all(isinstance(item, dict) for item in requests):
            errors.append("invalid requests list")
        else:
            if len(requests) != trace.get("llm_calls"):
                errors.append("request count does not match llm_calls")
            for number, request in enumerate(requests, 1):
                fields = ("input_tokens", "output_tokens", "max_tokens")
                if not all(is_integer(request.get(field)) for field in fields):
                    errors.append(f"request {number}: missing token usage or maximum")
                    continue
                if request["max_tokens"] < 1:
                    errors.append(f"request {number}: invalid max_tokens")
                if request["input_tokens"] + request["max_tokens"] > 8192:
                    errors.append(f"request {number}: input plus allowed output exceeds 8192")
                if request["output_tokens"] > request["max_tokens"]:
                    errors.append(f"request {number}: output exceeds max_tokens")
                if request.get("usage_verified") is False:
                    errors.append(f"request {number}: model usage is not verified")
            for field in ("input_tokens", "output_tokens"):
                if all(is_integer(item.get(field)) for item in requests):
                    if sum(item[field] for item in requests) != trace.get(field):
                        errors.append(f"request sum does not match {field}")
    return errors


def validate_records(questions, answers, traces, corpus_ids=None, no_context=False):
    errors = []
    valid_rows = {}
    for label, rows in (("question", questions), ("answer", answers), ("trace", traces)):
        valid_rows[label] = []
        for row in rows:
            if not isinstance(row.get("id"), str) or not row["id"].strip():
                errors.append(f"{label}: invalid id")
            else:
                valid_rows[label].append(row)
    question_ids = [row["id"] for row in valid_rows["question"]]
    for label in ("question", "answer", "trace"):
        counts = Counter(row["id"] for row in valid_rows[label])
        for question_id, count in counts.items():
            if count > 1:
                errors.append(f"{question_id}: duplicate {label}")
        if label != "question":
            for question_id in set(question_ids) - set(counts):
                errors.append(f"{question_id}: missing {label}")
            for question_id in set(counts) - set(question_ids):
                errors.append(f"{question_id}: unexpected {label}")
    answer_map = {row["id"]: row for row in valid_rows["answer"]}
    trace_map = {row["id"]: row for row in valid_rows["trace"]}
    for question in valid_rows["question"]:
        question_id = question.get("id")
        errors.extend(f"{question_id}: {error}" for error in question_errors(question))
        if question_id in answer_map:
            errors.extend(
                f"{question_id}: {error}" for error in answer_errors(
                    question, answer_map[question_id], corpus_ids, no_context
                )
            )
        if question_id in trace_map:
            errors.extend(
                f"{question_id}: {error}" for error in trace_errors(trace_map[question_id], corpus_ids)
            )
            answer = answer_map.get(question_id, {})
            if string_list(answer.get("citations")) and string_list(trace_map[question_id].get("retrieved_block_ids")):
                if not set(answer["citations"]) <= set(trace_map[question_id]["retrieved_block_ids"]):
                    errors.append(f"{question_id}: citation was not retrieved")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--allow-no-context", action="store_true")
    args = parser.parse_args()
    try:
        corpus_ids = None if not args.corpus else {row["block_id"] for row in read_jsonl(args.corpus)}
        errors = validate_records(
            read_jsonl(args.questions), read_jsonl(args.answers), read_jsonl(args.trace),
            corpus_ids, args.allow_no_context,
        )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({"valid": not errors, "errors": errors}, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
