"""OpenAI-compatible client with an explicit per-question budget."""
from __future__ import annotations

import hashlib
import json
import os
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Endpoint redirects are disabled")


class BudgetError(ValueError):
    pass


class Client:
    def __init__(self, runtime, tokenizer):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.base_url = os.environ.get("LLM_BASE_URL", runtime.get("development_base_url", "http://127.0.0.1:8081/v1")).rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL must be an HTTP endpoint")
        self.model = os.environ.get("LLM_MODEL", runtime["model_id"])
        self.key = os.environ.get("LLM_API_KEY", "local")
        self.submission = os.environ.get("SUBMISSION_ID", "feldman-hw01")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def complete(self, question_id, messages, requests):
        generation = self.runtime["generation"]
        limits = self.runtime["limits"]
        predicted_input = self.tokenizer.count(messages)
        previous_input = sum(r["input_tokens"] for r in requests)
        previous_output = sum(r["output_tokens"] for r in requests)
        if (len(requests) >= limits["llm_calls"] or
            predicted_input + generation["max_tokens"] + 32 > limits["request_tokens"] or
            previous_input + predicted_input + 32 > limits["input_tokens"] or
            previous_output + generation["max_tokens"] > limits["output_tokens"]):
            raise BudgetError("Generation budget exhausted before request")
        body = {"model": self.model, "messages": messages, "stream": False, **generation}
        raw = json.dumps(body, ensure_ascii=False).encode()
        digest = hashlib.sha256(raw).hexdigest()
        entry = {"input_tokens": 0, "output_tokens": 0, "max_tokens": generation["max_tokens"],
                 "estimated_input_tokens": predicted_input, "prompt_sha256": digest,
                 "status": "pending", "usage_verified": False}
        requests.append(entry)
        request = Request(self.base_url + "/chat/completions", data=raw,
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key,
                                   "X-Question-ID": question_id, "X-Submission-ID": self.submission})
        started = time.perf_counter()
        try:
            with self.opener.open(request, timeout=self.runtime["timeout_seconds"]) as response:
                data = response.read()
            payload = json.loads(data)
            usage = payload.get("usage", {})
            for field in ("prompt_tokens", "completion_tokens"):
                if type(usage.get(field)) is not int or usage[field] < 0:
                    raise ValueError("Endpoint omitted verifiable token usage")
            entry.update(input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
                         usage_verified=True, status="ok", request_id=payload.get("id"),
                         response_sha256=hashlib.sha256(data).hexdigest())
            if (entry["input_tokens"] + generation["max_tokens"] > limits["request_tokens"] or
                entry["output_tokens"] > generation["max_tokens"] or
                previous_input + entry["input_tokens"] > limits["input_tokens"] or
                previous_output + entry["output_tokens"] > limits["output_tokens"]):
                entry["status"] = "budget_error"
                raise ValueError("Endpoint usage exceeds configured budget")
            content = payload["choices"][0]["message"].get("content")
            if not isinstance(content, str):
                raise ValueError("Endpoint returned no content")
            entry["finish_reason"] = payload["choices"][0].get("finish_reason")
            return content
        except Exception as error:
            entry["status"] = "error"
            entry["error"] = type(error).__name__
            # No automatic HTTP retry: a timed-out request may already have generated tokens.
            raise
        finally:
            entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)

