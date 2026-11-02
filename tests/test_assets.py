import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from assets import asset_source, cache_path, metadata_path, prepare_asset, verified_metadata


class AssetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / "index" / "model.onnx"
        self.cache = self.root / "cache"
        self.source = asset_source("test/model", "fixed-revision", "onnx/model.onnx")

    def download(self, content=b"original", source=None):
        with patch("assets.urlopen", return_value=BytesIO(content)) as fetch:
            result = prepare_asset(self.target, source or self.source, self.cache)
        return result, fetch.call_count

    def test_clean_index_restores_verified_assets_without_network(self):
        record, calls = self.download()
        self.assertEqual(calls, 1)
        self.target.unlink()
        metadata_path(self.target).unlink()
        with patch("assets.urlopen", side_effect=AssertionError("unexpected network")):
            restored = prepare_asset(self.target, self.source, self.cache)
        self.assertEqual(restored, record)
        self.assertEqual(self.target.read_bytes(), b"original")

    def test_changed_revision_never_reuses_previous_file(self):
        self.download()
        other = asset_source("test/model", "another-revision", "onnx/model.onnx")
        result, calls = self.download(b"new-revision", other)
        self.assertEqual(calls, 1)
        self.assertEqual(result["source"], other)
        self.assertEqual(self.target.read_bytes(), b"new-revision")

    def test_corrupt_output_is_repaired_from_verified_cache(self):
        self.download()
        self.target.write_bytes(b"damaged")
        with patch("assets.urlopen", side_effect=AssertionError("unexpected network")):
            prepare_asset(self.target, self.source, self.cache)
        self.assertEqual(self.target.read_bytes(), b"original")

    def test_corrupt_cache_is_not_trusted(self):
        self.download()
        self.target.unlink()
        cache_path(self.cache, self.source).write_bytes(b"corrupted")
        _, calls = self.download(b"restored")
        self.assertEqual(calls, 1)
        self.assertEqual(self.target.read_bytes(), b"restored")

    def test_missing_source_manifest_does_not_certify_old_file(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"unknown origin")
        _, calls = self.download(b"verified")
        self.assertEqual(calls, 1)
        self.assertEqual(self.target.read_bytes(), b"verified")

    def test_changed_file_pin_and_sha_are_checked(self):
        self.download()
        other = asset_source("test/model", "fixed-revision", "onnx/other.onnx")
        self.assertIsNone(verified_metadata(self.target, other))
        sidecar = metadata_path(self.target)
        record = json.loads(sidecar.read_text())
        record["sha256"] = "wrong hash"
        sidecar.write_text(json.dumps(record))
        self.assertIsNone(verified_metadata(self.target, self.source))


if __name__ == "__main__":
    unittest.main()
