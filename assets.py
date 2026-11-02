"""Pinned model files with source manifests and a verified local build cache."""
import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from urllib.request import urlopen

from common import sha256_file


def asset_source(model_id, revision, remote_file):
    return {"model_id": model_id, "revision": revision, "file": remote_file}


def metadata_path(path):
    path = Path(path)
    return path.with_name(path.name + ".source.json")


def cache_path(cache_root, source):
    key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return Path(cache_root) / key / "asset"


def verified_metadata(path, source):
    path = Path(path)
    try:
        record = json.loads(metadata_path(path).read_text(encoding="utf-8"))
        if record.get("version") != 1 or record.get("source") != source:
            return None
        if not path.is_file() or record.get("bytes") != path.stat().st_size:
            return None
        if record.get("sha256") != sha256_file(path):
            return None
        return record
    except (OSError, ValueError, TypeError):
        return None


def temporary_path(path):
    path = Path(path)
    return path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")


def write_metadata(path, record):
    target = metadata_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(target)
    try:
        temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def copy_asset(source_path, target, record):
    source_path, target = Path(source_path), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != target.resolve():
        temporary = temporary_path(target)
        try:
            shutil.copyfile(source_path, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    write_metadata(target, record)


def save_to_cache(path, source, record, cache_root):
    if cache_root is not None:
        cached = cache_path(cache_root, source)
        if verified_metadata(cached, source) is None:
            copy_asset(path, cached, record)


def prepare_asset(target, source, cache_root=None, timeout=240):
    target = Path(target)
    record = verified_metadata(target, source)
    if record is not None:
        save_to_cache(target, source, record, cache_root)
        return record
    if cache_root is not None:
        cached = cache_path(cache_root, source)
        record = verified_metadata(cached, source)
        if record is not None:
            copy_asset(cached, target, record)
            return record
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{source['model_id']}/resolve/{source['revision']}/{source['file']}"
    temporary = temporary_path(target)
    try:
        with urlopen(url, timeout=timeout) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        record = {"version": 1, "source": source, "source_url": url,
                  "sha256": sha256_file(temporary), "bytes": temporary.stat().st_size}
        temporary.replace(target)
        write_metadata(target, record)
    finally:
        temporary.unlink(missing_ok=True)
    save_to_cache(target, source, record, cache_root)
    return record


def adopt_built_index(index, config, cache_root):
    """One-time migration of the completed build made with the current pins."""
    index = Path(index)
    manifest_path = index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedding = config["retrieval"]["embedding"]
    if manifest.get("retrieval", {}).get("embedding") != embedding:
        raise ValueError("Built index embedding settings differ from the current pins")
    runtime = config["_runtime"]
    specifications = [
        (name, asset_source(runtime["model_id"], runtime["revision"], name))
        for name in ("tokenizer.json", "chat_template.jinja")
    ] + [
        ("embedding/" + name, asset_source(embedding["model_id"], embedding["revision"], remote))
        for remote, name in (("tokenizer.json", "tokenizer.json"), (embedding["file"], "model.onnx"))
    ]
    records = {}
    for relative, source in specifications:
        path = index / relative
        digest = sha256_file(path)
        expected = manifest.get("tokenizer", {}).get(relative)
        previous = manifest.get("assets", {}).get(relative)
        if expected is not None and expected != digest:
            raise ValueError(f"Built tokenizer hash mismatch: {relative}")
        if previous is not None and (previous.get("source") != source or previous.get("sha256") != digest):
            raise ValueError(f"Built source or asset hash mismatch: {relative}")
        record = {"version": 1, "source": source,
                  "source_url": f"https://huggingface.co/{source['model_id']}/resolve/{source['revision']}/{source['file']}",
                  "sha256": digest, "bytes": path.stat().st_size,
                  "adopted_build_manifest_sha256": sha256_file(manifest_path)}
        write_metadata(path, record)
        save_to_cache(path, source, record, cache_root)
        records[relative] = record
    return records


def main():
    from common import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adopt-built-index", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    records = adopt_built_index(args.adopt_built_index, config, Path(config["_root"]) / ".cache" / "assets")
    print(json.dumps({"cached_assets": len(records), "bytes": sum(record["bytes"] for record in records.values())}))


if __name__ == "__main__":
    main()
