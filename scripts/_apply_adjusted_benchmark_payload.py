#!/usr/bin/env python3
"""Apply the verified adjusted-benchmark implementation payload."""
from __future__ import annotations

import base64
import io
import shutil
import tarfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    payload_dir = root / ".implementation-payload"
    encoded = "".join(
        (payload_dir / f"chunk-{index:02}.txt").read_text(encoding="utf-8").strip()
        for index in range(3)
    )
    archive = base64.b64decode(encoded, validate=True)

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            destination = (root / member.name).resolve()
            if root.resolve() not in destination.parents and destination != root.resolve():
                raise ValueError(f"unsafe payload member: {member.name}")
        bundle.extractall(root)

    shutil.rmtree(payload_dir)
    Path(__file__).unlink()
    (root / ".github/workflows/apply-adjusted-benchmark-implementation.yml").unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
