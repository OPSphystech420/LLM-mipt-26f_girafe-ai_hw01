"""Pinned multilingual E5 embeddings, ONNX CPU only."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from assets import asset_source, prepare_asset

ort.disable_telemetry_events()


def prepare_embedding(output, settings, cache_root=None):
    directory = Path(output) / "embedding"
    directory.mkdir(parents=True, exist_ok=True)
    return {"embedding/" + name: prepare_asset(directory / name,
                                               asset_source(settings["model_id"], settings["revision"], remote),
                                               cache_root)
            for remote, name in (("tokenizer.json", "tokenizer.json"), (settings["file"], "model.onnx"))}


class Encoder:
    def __init__(self, path):
        path = Path(path) / "embedding"
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path / "model.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])

    def encode(self, texts, prefix="query: "):
        vectors = []
        for start in range(0, len(texts), 8):
            batch = self.tokenizer.encode_batch([prefix + text for text in texts[start:start + 8]])
            values = {"input_ids": np.asarray([b.ids for b in batch], dtype=np.int64),
                      "attention_mask": np.asarray([b.attention_mask for b in batch], dtype=np.int64),
                      "token_type_ids": np.asarray([b.type_ids for b in batch], dtype=np.int64)}
            inputs = {node.name: values[node.name] for node in self.session.get_inputs()}
            hidden = self.session.run(None, inputs)[0]
            mask = values["attention_mask"][..., None]
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
            vectors.append(pooled.astype(np.float32))
        return np.concatenate(vectors, axis=0)
