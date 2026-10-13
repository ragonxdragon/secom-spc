"""Download the UCI SECOM dataset into data/ and verify its SHA-256.

Usage: uv run python scripts/fetch_data.py

The archive is untrusted input. It is checked against a pinned hash before
anything is extracted, and only the two expected member names are read.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
ZIP_SHA256 = "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e"
MEMBERS = ("secom.data", "secom_labels.data")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    if all((DATA_DIR / m).exists() for m in MEMBERS):
        print("data already present, nothing to do")
        return 0
    archive = DATA_DIR / "secom.zip"
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, archive)  # noqa: S310 - fixed https URL
    digest = sha256(archive)
    if digest != ZIP_SHA256:
        archive.unlink()
        print(f"SHA-256 mismatch: got {digest}, expected {ZIP_SHA256}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for member in MEMBERS:
            if member not in names:
                print(f"missing archive member {member}", file=sys.stderr)
                return 1
            (DATA_DIR / member).write_bytes(zf.read(member))
    print("ok:", ", ".join(MEMBERS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
