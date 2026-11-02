"""Build an offline CPU search index."""
import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

from common import load_config, read_jsonl, sha256_file
from retrieval import build_payload
from token_budget import prepare_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    blocks = read_jsonl(args.corpus)
    ids = set()
    for block in blocks:
        if not all(k in block for k in ("block_id", "page", "section", "text")):
            raise ValueError("Corpus fields: block_id,page,section,text")
        if not isinstance(block["block_id"], str) or not block["block_id"] or block["block_id"] in ids:
            raise ValueError("Corpus ids must be unique nonempty strings")
        ids.add(block["block_id"])
        if type(block["page"]) is not int or block["page"] < 1:
            raise ValueError("Page must be a positive integer")
        if not isinstance(block["section"], str) or not isinstance(block["text"], str) or not block["text"].strip():
            raise ValueError("Invalid corpus text")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = Path(config["_root"]) / ".cache" / "assets"
    assets = prepare_tokenizer(out, config["_runtime"], cache_root)
    payload = build_payload(blocks, config["retrieval"])
    if not config["retrieval"].get("single_search", False) and "embedding" in config["retrieval"] and config["retrieval"]["embedding"].get("enabled", True):
        from dense import Encoder, prepare_embedding
        import numpy as np
        assets.update(prepare_embedding(out, config["retrieval"]["embedding"], cache_root))
        encoder = Encoder(out)
        vectors = encoder.encode([d["text"] for d in payload["documents"]], prefix="passage: ")
        np.save(out / "vectors.npy", vectors, allow_pickle=False)
    variants = {
        "index.json": payload,
        "index_no_math.json": build_payload(blocks, config["retrieval"], math_normalization=False),
        "index_no_stemming.json": build_payload(blocks, config["retrieval"], stemming=False),
    }
    index_files = {}
    for filename, variant_payload in variants.items():
        target = out / filename
        target.write_text(json.dumps(variant_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        index_files[filename] = target.stat().st_size
    target = out / "index.json"
    stats = {"corpus_sha256": sha256_file(args.corpus), "blocks": len(blocks),
             "chunks": len(payload["documents"]), "build_seconds": time.perf_counter() - started,
             "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024),
             "index_bytes": target.stat().st_size, "index_files": index_files, "index_version": 3,
             "python": platform.python_version(),
             "retrieval": config["retrieval"], "assets": assets}
    stats["output_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    stats["tokenizer"] = {name: sha256_file(out / name) for name in ("tokenizer.json", "chat_template.jinja")}
    (out / "manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats), file=sys.stderr)


if __name__ == "__main__":
    main()
