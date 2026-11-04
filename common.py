"""Shared file and configuration helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: empty line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    runtime_path = path.parent / config["runtime"]
    config["_runtime"] = json.loads(runtime_path.read_text(encoding="utf-8"))
    config["_root"] = str(path.parent)
    return config


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_questions(rows):
    seen = set()
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Question ids must be unique nonempty strings")
        seen.add(identifier)
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"{identifier}: empty question")
        options = row.get("options")
        if not isinstance(options, list) or not 4 <= len(options) <= 6:
            raise ValueError(f"{identifier}: expected 4-6 options")
        ids = [opt.get("id") for opt in options]
        if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{identifier}: invalid option ids")
        if any(not isinstance(opt.get("text"), str) or not opt["text"].strip() for opt in options):
            raise ValueError(f"{identifier}: empty option")
