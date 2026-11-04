"""Download the pinned Qwen weights and convert them for Apple Silicon."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen


MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def download_source(cache, workers):
    source = cache / "official" / REVISION
    source.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
    with urlopen(url, timeout=60) as stream:
        metadata = json.load(stream)
    if metadata["sha"] != REVISION:
        raise ValueError("Unexpected model revision")
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if not (name.endswith((".json", ".jinja", ".safetensors", ".txt")) or name == "LICENSE"):
            continue
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        asset_url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        lfs = item.get("lfs")
        expected_size = item.get("size")
        if target.exists() and target.stat().st_size == expected_size:
            continue
        if not lfs:
            with urlopen(asset_url, timeout=120) as stream:
                target.write_bytes(stream.read())
            continue
        expected_hash = lfs["sha256"]
        expected_size = lfs["size"]
        part_dir = source / (name + ".parts")
        part_dir.mkdir(exist_ok=True)
        chunk_size = 64 * 1024 * 1024
        ranges = [(n, start, min(start + chunk_size, expected_size) - 1)
                  for n, start in enumerate(range(0, expected_size, chunk_size))]

        def download_range(spec):
            number, start, end = spec
            part = part_dir / str(number)
            size = end - start + 1
            if part.exists() and part.stat().st_size == size:
                return
            for attempt in range(4):
                try:
                    offset = part.stat().st_size if part.exists() else 0
                    if offset == size:
                        return
                    if offset > size:
                        part.unlink()
                        offset = 0
                    request_start = start + offset
                    request = Request(asset_url, headers={"Range": f"bytes={request_start}-{end}"})
                    with urlopen(request, timeout=180) as stream:
                        if stream.status != 206:
                            raise ValueError("Range download was not honored")
                        expected = f"bytes {request_start}-{end}/{expected_size}"
                        if stream.headers.get("Content-Range") != expected:
                            raise ValueError("Unexpected Content-Range")
                        with part.open("ab") as output:
                            shutil.copyfileobj(stream, output, length=1024 * 1024)
                    if part.stat().st_size != size:
                        raise ValueError("Incomplete model chunk")
                    return
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)

        print(f"Downloading {name}: {expected_size / 1e9:.2f} GB", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for done, _ in enumerate(executor.map(download_range, ranges), 1):
                print(f"  {done}/{len(ranges)} chunks", flush=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        sha = hashlib.sha256()
        with temporary.open("wb") as output:
            for number, _, _ in ranges:
                with (part_dir / str(number)).open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        sha.update(chunk)
                        output.write(chunk)
        if sha.hexdigest() != expected_hash:
            raise ValueError(f"Downloaded file is corrupted: {name}")
        temporary.replace(target)
        shutil.rmtree(part_dir)
    return str(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--cache", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--download-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(".local/qwen-mlx" if args.bits == 4 else f".local/qwen-mlx-{args.bits}")
    cache = args.cache.resolve()
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from mlx_lm.convert import convert

    quantization = ({"mode": "none", "bits": 16, "dtype": "bfloat16"}
                    if args.bits == 16 else
                    {"mode": "affine", "bits": args.bits, "group_size": 64})
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "quantization": quantization,
        "mlx_lm_version": "0.32.2",
    }
    if args.output.exists():
        saved = args.output / "source_manifest.json"
        if saved.exists() and json.loads(saved.read_text()) == manifest:
            print(args.output.resolve())
            return
        raise ValueError("Output already exists without the expected manifest")
    if not 1 <= args.download_workers <= 32:
        raise ValueError("download-workers must be between 1 and 32")
    source = download_source(cache, args.download_workers)
    convert(
        source,
        mlx_path=str(args.output),
        quantize=args.bits != 16,
        q_group_size=64,
        q_bits=args.bits if args.bits != 16 else None,
        q_mode="affine",
        dtype="bfloat16",
        trust_remote_code=False,
    )
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
