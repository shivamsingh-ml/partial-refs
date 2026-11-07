"""Run IDs and provenance records (HANDOFF §9).

run_id = YYYYMMDD-HHMM-<short git sha>[-dirty]-<tag>; every run writes provenance.json next to its
outputs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": sha, "git_dirty": bool(status) if status is not None else None}


def make_run_id(tag: str, now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now()
    g = git_state()
    short = (g["git_sha"] or "nogit")[:7] + ("-dirty" if g["git_dirty"] else "")
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in tag)
    return f"{now:%Y%m%d-%H%M}-{short}-{safe_tag}"


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _versions(packages: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def write_provenance(out_dir: str | os.PathLike, extra: dict[str, Any] | None = None,
                     packages: list[str] | None = None) -> Path:
    """Write provenance.json into ``out_dir`` and return its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        **git_state(),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "node": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "packages": _versions(packages or ["numpy", "scipy", "pandas", "pyarrow", "nibabel"]),
    }
    if extra:
        record.update(extra)
    path = out / "provenance.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    return path
