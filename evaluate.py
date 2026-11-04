"""Evaluate exact option matches, best evidence F1 and category-macro quality."""

import argparse
import csv
import json
import math
import statistics
import hashlib
from collections import Counter
from pathlib import Path

from validate_submission import (
    answer_errors, question_errors, read_jsonl, trace_errors, unique_string_list, validate_records,
)


CATEGORIES = ("conditions", "comparison", "calculation", "insufficient")
CSV_FIELDS = (
    "name", "answer_accuracy", "Q", "mean_input_tokens", "mean_output_tokens",
    "mean_llm_calls", "mean_cost", "median_latency_ms",
)


def evidence_f1(predicted, expected):
    predicted, expected = set(predicted), set(expected)
    if not predicted and not expected:
        return 1.0
    return 2 * len(predicted & expected) / (len(predicted) + len(expected))


def answer_hash(answer):
    return hashlib.sha256(json.dumps(answer, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_rationale_audit(path, name, answers):
    by_id = {answer["id"]: answer for answer in answers}
    failures = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["name"] != name:
                continue
            identifier = row["id"]
            if identifier not in by_id or answer_hash(by_id[identifier]) != row["answer_sha256"]:
                raise ValueError(f"{name}/{identifier}: rationale audit does not match this answer")
            if row["valid"] != "false" or not row["reason"].strip() or identifier in failures:
                raise ValueError("Invalid or duplicate rationale audit row")
            failures[identifier] = row["reason"]
    return failures


def validate_gold(questions, gold):
    question_map = {question["id"]: question for question in questions}
    counts = Counter(row.get("id") for row in gold)
    if set(counts) != set(question_map) or any(count != 1 for count in counts.values()):
        raise ValueError("gold IDs must match question IDs exactly")
    for row in gold:
        question_id = row["id"]
        if row.get("category") not in CATEGORIES:
            raise ValueError(f"{question_id}: invalid gold category")
        if type(row.get("abstain")) is not bool or not unique_string_list(row.get("answer_ids")):
            raise ValueError(f"{question_id}: invalid gold answer")
        valid_ids = {option["id"] for option in question_map[question_id]["options"]}
        if not set(row["answer_ids"]) <= valid_ids:
            raise ValueError(f"{question_id}: unknown gold option id")
        sets = row.get("evidence_sets")
        if not isinstance(sets, list) or not all(unique_string_list(items) for items in sets):
            raise ValueError(f"{question_id}: invalid evidence_sets")
        if row["abstain"]:
            if row["answer_ids"] or row["category"] != "insufficient":
                raise ValueError(f"{question_id}: inconsistent abstention gold")
            if not isinstance(row.get("missing_information"), str) or not row["missing_information"].strip():
                raise ValueError(f"{question_id}: missing information must be stated")
        elif not row["answer_ids"] or not sets or any(not items for items in sets) or row["category"] == "insufficient":
            raise ValueError(f"{question_id}: inconsistent solvable gold")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError(f"{question_id}: missing gold rationale")


def evaluate_records(questions, gold, answers, traces, corpus_ids=None, no_context=False, baseline_cost=None, rationale_failures=None):
    rationale_failures = rationale_failures or {}
    if not questions:
        raise ValueError("question set is empty")
    if any(question_errors(question) for question in questions):
        raise ValueError("invalid question set")
    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError("duplicate question id")
    for label, rows in (("gold", gold), ("answer", answers), ("trace", traces)):
        if any(not isinstance(row.get("id"), str) or not row["id"].strip() for row in rows):
            raise ValueError(f"invalid {label} id")
    validate_gold(questions, gold)
    validation_errors = validate_records(questions, answers, traces, corpus_ids, no_context)
    answer_counts = Counter(row.get("id") for row in answers)
    trace_counts = Counter(row.get("id") for row in traces)
    answers_by_id = {row.get("id"): row for row in answers}
    traces_by_id = {row.get("id"): row for row in traces}
    gold_by_id = {row["id"]: row for row in gold}
    details = []
    costs, inputs, outputs, calls, latencies = [], [], [], [], []
    category_scores = {category: [] for category in CATEGORIES}
    usage_valid = True
    for question in questions:
        question_id = question["id"]
        expected = gold_by_id[question_id]
        answer = answers_by_id.get(question_id)
        trace = traces_by_id.get(question_id)
        errors = []
        if answer_counts[question_id] != 1:
            errors.append("missing or duplicate answer")
        elif answer is not None:
            errors.extend(answer_errors(question, answer, corpus_ids, no_context))
        current_trace_errors = ["missing or duplicate trace"] if trace_counts[question_id] != 1 else trace_errors(trace, corpus_ids)
        errors.extend(current_trace_errors)
        if current_trace_errors:
            usage_valid = False
        if trace is not None and trace.get("answer_error"):
            errors.append(f"technical answer error: {trace['answer_error']}")
        if answer is not None and trace is not None and unique_string_list(answer.get("citations")) and unique_string_list(trace.get("retrieved_block_ids")):
            if not set(answer["citations"]) <= set(trace["retrieved_block_ids"]):
                errors.append("citation was not retrieved")
        exact = False
        citation_f1 = 0.0
        score = 0.0
        if not errors:
            exact = answer["abstain"] == expected["abstain"] and set(answer["answer_ids"]) == set(expected["answer_ids"])
            if exact:
                if expected["abstain"]:
                    score = 1.0
                elif not no_context:
                    citation_f1 = max(evidence_f1(answer["citations"], items) for items in expected["evidence_sets"])
                    score = 0.8 + 0.2 * citation_f1
        if question_id in rationale_failures:
            score = 0.0
        category_scores[expected["category"]].append(score)
        details.append({"id": question_id, "category": expected["category"], "answer_correct": exact, "evidence_f1": citation_f1, "score": score, "errors": errors,
                        "rationale_error": rationale_failures.get(question_id)})
        if trace is not None and not current_trace_errors:
            inputs.append(trace["input_tokens"])
            outputs.append(trace["output_tokens"])
            calls.append(trace["llm_calls"])
            latencies.append(trace["elapsed_ms"])
            costs.append(trace["input_tokens"] + 4 * trace["output_tokens"] + 256 * trace["llm_calls"])
    macro = {category: statistics.mean(scores) if scores else None for category, scores in category_scores.items()}
    missing_categories = [category for category, score in macro.items() if score is None]
    if missing_categories and not no_context:
        raise ValueError(f"missing evaluation categories: {', '.join(missing_categories)}")
    quality = None if no_context else statistics.mean(macro.values())
    # Unknown rows and uncheckable usage disable the global efficiency term.
    expected_ids = {row["id"] for row in questions}
    if set(answers_by_id) != expected_ids or set(traces_by_id) != expected_ids:
        usage_valid = False
    complete_usage = usage_valid and len(costs) == len(questions)
    mean_cost = statistics.mean(costs) if complete_usage else None
    efficiency = None
    total_score = None
    if baseline_cost is not None:
        if type(baseline_cost) not in (int, float) or not math.isfinite(baseline_cost) or baseline_cost < 0:
            raise ValueError("baseline cost must be nonnegative")
        efficiency = min(1.0, baseline_cost / max(mean_cost, 1)) if complete_usage else 0.0
        if quality is not None:
            total_score = 75 * quality + 25 * quality * efficiency
    return {
        "answer_accuracy": statistics.mean(item["answer_correct"] for item in details),
        "Q": quality,
        "mean_input_tokens": statistics.mean(inputs) if complete_usage else None,
        "mean_output_tokens": statistics.mean(outputs) if complete_usage else None,
        "mean_llm_calls": statistics.mean(calls) if complete_usage else None,
        "mean_cost": mean_cost,
        "median_latency_ms": statistics.median(latencies) if complete_usage else None,
        "category_scores": macro, "efficiency_eligible": complete_usage,
        "E": efficiency, "score": total_score,
        "validation_errors": validation_errors, "per_question": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "gold", "answers", "trace"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--allow-no-context", action="store_true")
    parser.add_argument("--baseline-cost", type=float)
    parser.add_argument("--name", default="final")
    parser.add_argument("--output")
    parser.add_argument("--csv", help="Append one row with the required experiment columns")
    parser.add_argument("--rationale-audit", help="Content review for the saved answers")
    args = parser.parse_args()
    try:
        corpus_ids = None if not args.corpus else {row["block_id"] for row in read_jsonl(args.corpus)}
        answers = read_jsonl(args.answers)
        failures = read_rationale_audit(args.rationale_audit, args.name, answers) if args.rationale_audit else None
        result = evaluate_records(
            read_jsonl(args.questions), read_jsonl(args.gold), answers,
            read_jsonl(args.trace), corpus_ids, args.allow_no_context, args.baseline_cost,
            failures,
        )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    result["name"] = args.name
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    if args.csv:
        path = Path(args.csv)
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
            if not exists:
                writer.writeheader()
            writer.writerow(result)
    print(serialized, end="")
    return int(bool(result["validation_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
