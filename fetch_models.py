"""Download the four pinned public Release models, or verify local copies.

Uses only Python's standard library and public URLs without authentication.
Existing files are checked, never repaired or overwritten automatically. A new
download becomes visible at its final path only after size and SHA256 match.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parent
RELEASE_BASE = "https://github.com/Anone92-yuntech/integrated-analysis/releases/download/v1.0.0"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ModelAsset:
    relative_path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "models"
                or ".." in path.parts or "\\" in self.relative_path):
            raise ValueError("model path must be models/<filename> using forward slashes")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("model size must be a positive integer")
        if (not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in self.sha256)):
            raise ValueError("model SHA256 must contain exactly 64 hexadecimal characters")

    @property
    def url(self) -> str:
        # Release assets have flat names; every local destination is models/.
        filename = PurePosixPath(self.relative_path).name
        return f"{RELEASE_BASE}/{urllib.parse.quote(filename, safe='')}"


ASSETS = (
    ModelAsset(
        "models/best.pt", 40516645,
        "f5fce7480e8fb4aebd800eeaae24299979cd7aa6df49c8085b0a67ee6ecf906f",
    ),
    ModelAsset(
        "models/best_transfer_model.pth", 79069865,
        "fccef63f6464c678852224f7cbf24c9c6fb9b9e682f0fd15f774ec7df4b245b3",
    ),
    ModelAsset(
        "models/yolox_nano.onnx", 3659407,
        "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    ),
    ModelAsset(
        "models/pose_landmarker_lite.task", 5777746,
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
)


class ModelIntegrityError(RuntimeError):
    """An existing file or downloaded response differs from the pinned model."""


def _check_integrity(asset: ModelAsset, size: int, digest: str, path: Path) -> None:
    if size != asset.size:
        raise ModelIntegrityError(
            f"{path}: size mismatch (expected {asset.size} bytes, got {size}); "
            "existing files are preserved"
        )
    if digest.lower() != asset.sha256.lower():
        raise ModelIntegrityError(
            f"{path}: SHA256 mismatch (expected {asset.sha256}, got {digest}); "
            "existing files are preserved"
        )


def _exists(path: Path) -> bool:
    # Broken symlinks also occupy a path and must never be replaced.
    return path.exists() or path.is_symlink()


def _verify_existing(path: Path, asset: ModelAsset) -> None:
    if path.is_symlink() or not path.is_file():
        raise ModelIntegrityError(f"{path}: existing path is not a regular file; left unchanged")
    size = path.stat().st_size
    if size != asset.size:
        _check_integrity(asset, size, "", path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    _check_integrity(asset, size, digest.hexdigest(), path)


def ensure_model(asset: ModelAsset, *, root: Path = ROOT,
                 verify_only: bool = False) -> str:
    """Return verified/downloaded without overwriting any existing destination.

    Download to a unique same-directory temporary file, verify and close it,
    then atomically link the complete file to its final name. Unlike replace(),
    link() cannot overwrite a file another process creates during download.
    The temporary link is removed on every exit. A filesystem without hard-link
    support fails without publishing a partial model or replacing user files.
    """
    path = Path(root) / asset.relative_path
    if _exists(path):
        _verify_existing(path, asset)
        return "verified"
    if verify_only:
        raise FileNotFoundError(f"{path}: model is missing (--verify-only does not download)")

    request = urllib.request.Request(
        asset.url, headers={"User-Agent": "integrated-analysis-model-fetcher/1.0"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=path.name + ".", suffix=".part",
            dir=path.parent, delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(CHUNK_SIZE):
                    size += len(chunk)
                    if size > asset.size:
                        raise ModelIntegrityError(
                            f"{path}: download exceeds expected size ({asset.size} bytes)"
                        )
                    destination.write(chunk)
                    digest.update(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        _check_integrity(asset, size, digest.hexdigest(), path)

        try:
            os.link(temporary_path, path)
        except FileExistsError:
            # A concurrent writer wins ownership of its file, even if corrupt.
            _verify_existing(path, asset)
            return "verified"
        return "downloaded"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="check all four models without downloading or changing files",
    )
    arguments = parser.parse_args(argv)
    failed = False
    for asset in ASSETS:
        try:
            status = ensure_model(asset, root=ROOT, verify_only=arguments.verify_only)
        except (OSError, ModelIntegrityError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            failed = True
        else:
            print(f"{status}: {asset.relative_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
