"""Exact text-only Qwen chat template token counting."""
import json
from pathlib import Path

from jinja2 import Environment
from tokenizers import Tokenizer
from assets import asset_source, prepare_asset

FILES = ("tokenizer.json", "chat_template.jinja")


def prepare_tokenizer(output, runtime, cache_root=None):
    destination = Path(output)
    return {name: prepare_asset(destination / name,
                                asset_source(runtime["model_id"], runtime["revision"], name),
                                cache_root, timeout=120) for name in FILES}


class ChatTokenizer:
    def __init__(self, path, enable_thinking=False):
        path = Path(path)
        self.enable_thinking = enable_thinking
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        environment = Environment(trim_blocks=True, lstrip_blocks=True)
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False)
        environment.globals["raise_exception"] = self.raise_exception
        self.template = environment.from_string((path / "chat_template.jinja").read_text(encoding="utf-8"))

    @staticmethod
    def raise_exception(message):
        raise ValueError(message)

    def render(self, messages):
        return self.template.render(messages=messages, add_generation_prompt=True, enable_thinking=self.enable_thinking)

    def count(self, messages):
        return len(self.tokenizer.encode(self.render(messages), add_special_tokens=False).ids)

