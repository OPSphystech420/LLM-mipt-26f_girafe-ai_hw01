"""Run the same public questions in five fixed configurations."""
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

from evaluate import evaluate_records, read_rationale_audit
from common import read_jsonl, sha256_file, load_config

MODES = ("no_context", "baseline", "full", "no_math", "no_stemming", "final")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", default="dev_questions.jsonl")
    parser.add_argument("--gold", default="dev_gold.jsonl")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--runs", default=".local/runs")
    parser.add_argument("--output", default="experiments.csv")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--rationale-audit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    questions, gold = read_jsonl(args.questions), read_jsonl(args.gold)
    selected_mode = load_config(args.config)["system"]["mode"]
    corpus_ids = {b["block_id"] for b in json.loads((Path(args.index) / "index.json").read_text())["blocks"]}
    collected = []
    for mode in args.modes:
        answers = root / "dev_answers.jsonl" if mode == "final" else runs / f"{mode}_answers.jsonl"
        trace = root / "dev_trace.jsonl" if mode == "final" else runs / f"{mode}_trace.jsonl"
        command = [sys.executable, str(root / "predict.py"), "--index", args.index,
                   "--questions", args.questions, "--output", str(answers), "--trace", str(trace),
                   "--config", args.config, "--mode", selected_mode if mode == "final" else mode]
        environment = os.environ.copy()
        environment["SUBMISSION_ID"] = environment.get("SUBMISSION_ID", "feldman-hw01") + "-" + mode
        subprocess.run(command, check=True, env=environment)
        predictions = read_jsonl(answers)
        failures = read_rationale_audit(args.rationale_audit, mode, predictions) if args.rationale_audit else None
        metrics = evaluate_records(questions, gold, predictions, read_jsonl(trace),
                                   corpus_ids=corpus_ids, no_context=mode == "no_context", rationale_failures=failures)
        metrics["name"] = mode
        (runs / f"{mode}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        collected.append(metrics)
        print(json.dumps({"name": mode, "Q": metrics.get("Q"), "accuracy": metrics.get("answer_accuracy")}), flush=True)
    # The CSV is replaced only after all selected runs complete.
    from evaluate import CSV_FIELDS
    import csv
    with Path(args.output).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for metrics in collected:
            writer.writerow({field: metrics.get(field) for field in CSV_FIELDS})
    manifest = {"modes": args.modes, "questions_sha256": sha256_file(args.questions),
                "gold_sha256": sha256_file(args.gold), "config_sha256": sha256_file(args.config)}
    (runs / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
