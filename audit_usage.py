"""Compare saved requests with the endpoint journal."""
import argparse
import json
from pathlib import Path

from common import read_jsonl, sha256_file
from validate_submission import trace_errors


def audit(traces, journal, runtime, submission_id):
    records = {row["request_id"]: row for row in journal if "request_id" in row}
    seen = set()
    memory, totals = [], {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
    maximum = 0
    for trace in traces:
        errors = trace_errors(trace)
        if errors:
            raise ValueError(f"{trace['id']}: {errors}")
        for request in trace["requests"]:
            request_id = request["request_id"]
            if request_id in seen:
                raise ValueError("A request is counted more than once")
            seen.add(request_id)
            record = records[request_id]
            if record["question_id"] != trace["id"] or record["submission_id"] != submission_id:
                raise ValueError("Question or submission header mismatch")
            if record["status"] != "ok" or record["model_revision"] != runtime["revision"]:
                raise ValueError("Unexpected model or endpoint status")
            if record["request_sha256"] != request["prompt_sha256"] or record["response_sha256"] != request["response_sha256"]:
                raise ValueError("Request or response differs from the endpoint record")
            for local, remote in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
                if request[local] != record["usage"][remote]:
                    raise ValueError("Token usage mismatch")
                totals[local] += request[local]
            for key in ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "repetition_penalty"):
                if record["sampling"].get(key) != runtime["generation"].get(key):
                    raise ValueError(f"Sampling mismatch: {key}")
            for key, value in runtime["quantization"].items():
                if key != "backend" and record["quantization"].get(key) != value:
                    raise ValueError(f"Precision mismatch: {key}")
            if record["thinking"] != runtime["generation"]["chat_template_kwargs"]["enable_thinking"]:
                raise ValueError("Thinking mode mismatch")
            if request["max_tokens"] != runtime["generation"]["max_tokens"]:
                raise ValueError("Allowed output mismatch")
            totals["llm_calls"] += 1
            maximum = max(maximum, request["input_tokens"] + request["max_tokens"])
            memory.append(record["peak_memory_gb"])
    return {"questions": len(traces), **totals, "max_request_tokens": maximum,
            "peak_mlx_memory_gb": max(memory, default=0), "endpoint_match": True}


def main():
    parser = argparse.ArgumentParser()
    for name in ("trace", "endpoint-log", "runtime", "submission-id", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    traces = read_jsonl(args.trace)
    result = audit(traces, read_jsonl(args.endpoint_log),
                   json.loads(Path(args.runtime).read_text()), args.submission_id)
    result.update(trace_sha256=sha256_file(args.trace), runtime_sha256=sha256_file(args.runtime))
    recorded_hashes = sorted({trace["runtime_sha256"] for trace in traces if "runtime_sha256" in trace})
    if recorded_hashes:
        result["trace_runtime_sha256"] = recorded_hashes
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
