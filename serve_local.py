"""Local Qwen Chat Completions endpoint with per-request usage audit."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import time
import uuid


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def visible_content(raw, thinking):
    if not thinking:
        return raw, True
    if "</think>" not in raw:
        return "", False
    return raw.split("</think>", 1)[1].lstrip(), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(".local/qwen-mlx"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--audit", type=Path, default=Path(".local/endpoint.jsonl"))
    parser.add_argument("--ram-limit-gb", type=float, default=16)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    manifest = json.loads((args.model / "source_manifest.json").read_text())
    if (manifest["model_id"] != "Qwen/Qwen3.5-4B" or
            manifest["revision"] != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"):
        raise ValueError("Unexpected local model source")
    if not 0 < args.ram_limit_gb <= 64:
        raise ValueError("Invalid RAM limit")
    memory_limit = int(args.ram_limit_gb * 1e9)
    mx.set_memory_limit(memory_limit)
    model, tokenizer = load(str(args.model.resolve()), lazy=False)
    load_peak_memory_gb = mx.get_peak_memory() / 1e9
    if load_peak_memory_gb > args.ram_limit_gb:
        raise MemoryError("Model load exceeds the RAM limit")
    print(f"Model load peak MLX memory: {load_peak_memory_gb:.6f} GB", flush=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit_stream = args.audit.open("a", encoding="utf-8", buffering=1)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in {"/health", "/v1/models"}:
                self.send_json(200, {
                    "data": [{"id": "Qwen/Qwen3.5-4B"}],
                    "source": manifest,
                    "load_peak_memory_gb": load_peak_memory_gb,
                    "ram_limit_gb": args.ram_limit_gb,
                })
            else:
                self.send_json(404, {"error": "Unknown path"})

        def do_POST(self):
            started = time.perf_counter()
            request_id = "chatcmpl-" + uuid.uuid4().hex
            record = {
                "request_id": request_id,
                "question_id": self.headers.get("X-Question-ID"),
                "submission_id": self.headers.get("X-Submission-ID"),
                "timestamp": time.time(),
                "model_revision": manifest["revision"],
                "quantization": manifest["quantization"],
                "ram_limit_gb": args.ram_limit_gb,
            }
            try:
                if self.path != "/v1/chat/completions":
                    self.send_json(404, {"error": "Unknown path"})
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2_000_000:
                    raise ValueError("Invalid request size")
                raw = self.rfile.read(size).decode("utf-8")
                body = json.loads(raw)
                record["request_sha256"] = digest(raw)
                if body.get("stream", False):
                    raise ValueError("Only non-streaming completions are supported")
                if body.get("model") != "Qwen/Qwen3.5-4B":
                    raise ValueError("This endpoint serves Qwen/Qwen3.5-4B")
                if not record["question_id"] or not record["submission_id"]:
                    raise ValueError("Question and submission headers are required")
                messages = body["messages"]
                template_kwargs = body.get("chat_template_kwargs", {})
                thinking = template_kwargs.get("enable_thinking", True)
                if type(thinking) is not bool:
                    raise ValueError("enable_thinking must be boolean")
                prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **template_kwargs,
                )
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
                if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
                    raise ValueError("Invalid max_tokens")
                if len(prompt_tokens) + max_tokens > 8192:
                    raise ValueError("Request exceeds the context budget")
                record.update({
                    "prompt_sha256": digest(prompt),
                    "max_tokens": max_tokens,
                    "model": body["model"],
                    "thinking": thinking,
                    "penalty_scope": "generated_history",
                    "sampling": {key: body.get(key) for key in (
                        "temperature", "top_p", "top_k", "min_p", "seed",
                        "presence_penalty", "repetition_penalty",
                    )},
                })
                mx.random.seed(int(body.get("seed", 42)))
                sampler = make_sampler(
                    temp=float(body.get("temperature", 0)),
                    top_p=float(body.get("top_p", 1)),
                    top_k=int(body.get("top_k", 0)),
                    min_p=float(body.get("min_p", 0)),
                )
                penalties = make_logits_processors(
                    repetition_penalty=float(body.get("repetition_penalty", 1)),
                    repetition_context_size=max_tokens,
                    presence_penalty=float(body.get("presence_penalty", 0)),
                    presence_context_size=max_tokens,
                )

                def generated_only(tokens, logits):
                    history = tokens[len(prompt_tokens):]
                    for penalty in penalties:
                        logits = penalty(history, logits)
                    return logits

                parts = []
                last = None
                # No prompt cache is supplied or retained between requests.
                for last in stream_generate(
                    model, tokenizer, prompt=prompt_tokens,
                    max_tokens=max_tokens, sampler=sampler,
                    logits_processors=[generated_only] if penalties else [],
                ):
                    parts.append(last.text)
                if last is None:
                    raise RuntimeError("Generation produced no token usage")
                raw_output = "".join(parts)
                content, reasoning_completed = visible_content(raw_output, thinking)
                usage = {
                    "prompt_tokens": int(last.prompt_tokens),
                    "completion_tokens": int(last.generation_tokens),
                    "total_tokens": int(last.prompt_tokens + last.generation_tokens),
                }
                response = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "Qwen/Qwen3.5-4B",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": last.finish_reason,
                    }],
                    "usage": usage,
                }
                record.update({
                    "status": "ok", "usage": usage,
                    "response_sha256": digest(json.dumps(response, ensure_ascii=False)),
                    "content_sha256": digest(content),
                    "raw_output_sha256": digest(raw_output),
                    "reasoning_completed": reasoning_completed,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                    "peak_memory_gb": last.peak_memory,
                    "prompt_tps": last.prompt_tps,
                    "generation_tps": last.generation_tps,
                })
                audit_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                try:
                    self.send_json(200, response)
                except BrokenPipeError:
                    pass
            except (ValueError, KeyError, TypeError) as exc:
                record.update({"status": "rejected", "error": str(exc)})
                audit_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.send_json(400, {"error": str(exc), "id": request_id})
            except Exception as exc:
                record.update({"status": "error", "error": repr(exc)})
                audit_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.send_json(500, {"error": str(exc), "id": request_id})
            finally:
                mx.clear_cache()

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Qwen ready at http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    finally:
        audit_stream.close()
        server.server_close()


if __name__ == "__main__":
    main()
