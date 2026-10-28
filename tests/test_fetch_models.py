"""Pinned-model download contracts using tiny in-memory mocked responses."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import fetch_models


class InterruptedResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.read_count = 0

    def read(self, size=-1):
        self.read_count += 1
        if self.read_count > 1:
            raise URLError("connection interrupted")
        return super().read(min(size, 3) if size >= 0 else 3)


class FetchModelsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.payload = b"tiny synthetic model for integrity tests\n"
        self.asset = fetch_models.ModelAsset(
            "models/tiny-model.bin", len(self.payload),
            hashlib.sha256(self.payload).hexdigest(),
        )
        self.destination = self.root / self.asset.relative_path

    def write_existing(self, payload):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.write_bytes(payload)

    def assert_no_parts(self):
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_manifest_has_exact_four_project_relative_model_destinations(self):
        self.assertEqual({asset.relative_path for asset in fetch_models.ASSETS}, {
            "models/best.pt", "models/best_transfer_model.pth",
            "models/yolox_nano.onnx", "models/pose_landmarker_lite.task",
        })
        for asset in fetch_models.ASSETS:
            self.assertEqual(asset.url,
                "https://github.com/Anone92-yuntech/integrated-analysis/releases/download/v1.0.0/"
                + Path(asset.relative_path).name)

    def test_manifest_validation_rejects_paths_outside_flat_models_directory(self):
        for path in ("../escape", "/models/example", "best.pt", "models/../escape",
                     "models/nested/example", "C:/models/example", "models\\example"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                fetch_models.ModelAsset(path, 1, "a" * 64)
        for size, digest in ((0, "a" * 64), (True, "a" * 64), (1.2, "a" * 64),
                             (1, "x" * 64), (1, "a" * 63)):
            with self.assertRaises(ValueError):
                fetch_models.ModelAsset("models/example", size, digest)

    def test_valid_existing_model_never_uses_network_or_changes_file(self):
        self.write_existing(self.payload)
        stat = self.destination.stat()
        with patch("fetch_models.urllib.request.urlopen") as network:
            self.assertEqual(fetch_models.ensure_model(self.asset, root=self.root), "verified")
        network.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(self.destination.stat().st_mtime_ns, stat.st_mtime_ns)
        self.assert_no_parts()

    def test_wrong_existing_size_and_hash_are_preserved_without_network(self):
        for wrong in (b"wrong size", b"x" * len(self.payload)):
            with self.subTest(wrong=wrong):
                self.write_existing(wrong)
                with patch("fetch_models.urllib.request.urlopen") as network:
                    with self.assertRaises(fetch_models.ModelIntegrityError):
                        fetch_models.ensure_model(self.asset, root=self.root)
                network.assert_not_called()
                self.assertEqual(self.destination.read_bytes(), wrong)
                self.assert_no_parts()

    def test_existing_directory_is_preserved(self):
        self.destination.mkdir(parents=True)
        marker = self.destination / "keep.txt"
        marker.write_bytes(b"preserve")
        with patch("fetch_models.urllib.request.urlopen") as network:
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(self.asset, root=self.root)
        network.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"preserve")
        self.assert_no_parts()

    def test_success_uses_public_release_and_writes_only_verified_complete_file(self):
        def response(request, timeout):
            self.assertFalse(self.destination.exists())
            self.assertEqual(request.full_url, fetch_models.RELEASE_BASE + "/tiny-model.bin")
            self.assertEqual(timeout, 60)
            self.assertFalse(request.has_header("Authorization"))
            return io.BytesIO(self.payload)

        with patch("fetch_models.urllib.request.urlopen", side_effect=response) as network:
            self.assertEqual(fetch_models.ensure_model(self.asset, root=self.root), "downloaded")
        network.assert_called_once()
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assert_no_parts()

    def test_chunked_response_is_hashed_and_joined_in_order(self):
        with patch("fetch_models.CHUNK_SIZE", 3), patch(
            "fetch_models.urllib.request.urlopen", return_value=io.BytesIO(self.payload)
        ):
            self.assertEqual(fetch_models.ensure_model(self.asset, root=self.root), "downloaded")
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assert_no_parts()

    def test_wrong_hash_never_publishes_a_model(self):
        with patch("fetch_models.urllib.request.urlopen",
                   return_value=io.BytesIO(b"x" * len(self.payload))):
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(self.asset, root=self.root)
        self.assertFalse(self.destination.exists())
        self.assert_no_parts()

    def test_short_response_is_rejected_even_if_its_hash_matches_manifest(self):
        short = self.payload[:-1]
        asset = fetch_models.ModelAsset(self.asset.relative_path, len(self.payload),
                                       hashlib.sha256(short).hexdigest())
        with patch("fetch_models.urllib.request.urlopen", return_value=io.BytesIO(short)):
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(asset, root=self.root)
        self.assertFalse(self.destination.exists())
        self.assert_no_parts()

    def test_oversized_response_stops_without_publishing(self):
        with patch("fetch_models.urllib.request.urlopen",
                   return_value=io.BytesIO(self.payload + b"unexpected")):
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(self.asset, root=self.root)
        self.assertFalse(self.destination.exists())
        self.assert_no_parts()

    def test_interruption_preserves_other_process_temporary_file(self):
        self.destination.parent.mkdir()
        unrelated = self.destination.parent / "another-download.part"
        unrelated.write_bytes(b"keep")
        with patch("fetch_models.urllib.request.urlopen",
                   return_value=InterruptedResponse(self.payload)):
            with self.assertRaises(URLError):
                fetch_models.ensure_model(self.asset, root=self.root)
        self.assertFalse(self.destination.exists())
        self.assertEqual(unrelated.read_bytes(), b"keep")
        self.assertEqual(list(self.root.rglob("*.part")), [unrelated])

    def test_network_failure_cleans_its_temporary_file(self):
        with patch("fetch_models.urllib.request.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(URLError):
                fetch_models.ensure_model(self.asset, root=self.root)
        self.assertFalse(self.destination.exists())
        self.assert_no_parts()

    def test_verify_only_missing_model_does_not_create_directory_or_use_network(self):
        with patch("fetch_models.urllib.request.urlopen") as network:
            with self.assertRaises(FileNotFoundError):
                fetch_models.ensure_model(self.asset, root=self.root, verify_only=True)
        network.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_verify_only_hash_checks_existing_file_without_modification(self):
        self.write_existing(b"x" * len(self.payload))
        with patch("fetch_models.urllib.request.urlopen") as network:
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(self.asset, root=self.root, verify_only=True)
        network.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"x" * len(self.payload))
        self.assert_no_parts()

    def test_concurrent_corrupt_destination_is_not_overwritten(self):
        wrong = b"x" * len(self.payload)

        def response(*_args, **_kwargs):
            self.destination.write_bytes(wrong)
            return io.BytesIO(self.payload)

        with patch("fetch_models.urllib.request.urlopen", side_effect=response):
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(self.asset, root=self.root)
        self.assertEqual(self.destination.read_bytes(), wrong)
        self.assert_no_parts()

    def test_concurrent_correct_destination_is_verified_without_replacement(self):
        def response(*_args, **_kwargs):
            self.destination.write_bytes(self.payload)
            return io.BytesIO(self.payload)

        with patch("fetch_models.urllib.request.urlopen", side_effect=response):
            self.assertEqual(fetch_models.ensure_model(self.asset, root=self.root), "verified")
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assert_no_parts()

    def test_destination_created_at_atomic_publication_is_never_overwritten(self):
        real_link = fetch_models.os.link
        wrong = b"racing writer owns this"

        def competing_link(source, destination):
            Path(destination).write_bytes(wrong)
            return real_link(source, destination)

        with patch("fetch_models.urllib.request.urlopen", return_value=io.BytesIO(self.payload)), \
                patch("fetch_models.os.link", side_effect=competing_link):
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(self.asset, root=self.root)
        self.assertEqual(self.destination.read_bytes(), wrong)
        self.assert_no_parts()

    def test_filesystem_publication_error_does_not_leave_partial_final_file(self):
        with patch("fetch_models.urllib.request.urlopen", return_value=io.BytesIO(self.payload)), \
                patch("fetch_models.os.link", side_effect=OSError("hard links unavailable")):
            with self.assertRaises(OSError):
                fetch_models.ensure_model(self.asset, root=self.root)
        self.assertFalse(self.destination.exists())
        self.assert_no_parts()

    def test_cli_verify_only_checks_every_manifest_asset(self):
        with patch.object(fetch_models, "ensure_model", return_value="verified") as ensure, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(fetch_models.main(["--verify-only"]), 0)
        self.assertEqual([call.args[0] for call in ensure.call_args_list], list(fetch_models.ASSETS))
        self.assertTrue(all(call.kwargs["verify_only"] for call in ensure.call_args_list))

    def test_cli_continues_after_failure_and_returns_nonzero(self):
        with patch.object(fetch_models, "ensure_model", side_effect=[
            FileNotFoundError("missing"), "verified", "verified", "verified",
        ]) as ensure, redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(fetch_models.main(["--verify-only"]), 1)
        self.assertEqual(ensure.call_count, 4)
        self.assertEqual(output.getvalue().count("verified:"), 3)
        self.assertIn("missing", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
