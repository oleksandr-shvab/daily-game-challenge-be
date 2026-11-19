"""The SHA-256 of a file, the content-hash key of a clue file (SPEC-04 AC-41). Standard library
only, so the builder and promote share it without loading a content library (R-4)."""

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
