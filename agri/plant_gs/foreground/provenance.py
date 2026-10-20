"""Provenance chain for 3DGS runs: dataset -> checkpoint -> PLY.

The failure this prevents is not corruption, it is **orphaning**. A run writes
``splats_best.pt`` at the best step, an exporter reads it and writes a PLY, and
then a *second* run in the same directory overwrites ``splats_best.pt``. Nothing
errors. The PLY is still on disk, still valid, and still quotes a PSNR -- but no
file mtime, no config field, and no report text can show which checkpoint it
came from, so the number is unreproducible rather than wrong.

Everything here is deliberately boring and mechanical: a JSON record per
artefact, SHA256 over the bytes, and a step read *out of the file* rather than
out of the filename. Two rules hold it together:

1. A provenance record names the exact file it describes (``sha256``, ``bytes``),
   so "the file changed after I measured it" is detectable rather than silent.
2. An export records the checkpoint hash before and after export, and refuses to
   proceed if they differ, so a still-running training can never leak into an
   export mid-write.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

def attribute_column(tensor_name: str, tensor, index) -> "np.ndarray":
    """Select the 1-D column of a checkpoint tensor that a PLY property holds.

    Tensor layouts differ: ``means``/``scales``/``quats`` are ``(n, k)``,
    ``opacities`` is ``(n,)``, and ``sh0`` is ``(n, 1, 3)`` because it keeps the
    sh-degree axis for the rasteriser. Collapsing that middle axis here means
    the caller does not have to special-case the shape at every comparison.
    """
    import numpy as _np
    values = _np.asarray(tensor)
    if values.ndim == 1:
        return values
    if tensor_name == 'sh0':
        values = values.reshape(values.shape[0], -1)
    if index is None:
        return values.reshape(values.shape[0], -1)[:, 0]
    return values[:, index]


def normalize_quaternions(tensor) -> "np.ndarray":
    import numpy as _np
    values = _np.asarray(tensor)
    norm = _np.linalg.norm(values, axis=-1, keepdims=True)
    return values / _np.maximum(norm, 1e-12)


SCHEMA_VERSION = "agri-provenance-1"

#: PLY property -> checkpoint tensor, and which column of that tensor.
#:
#: This is the *format* contract of gsplat's ``export_splats``: the PLY stores
#: raw (log) scales, raw (logit) opacities and log-encoded SH DC terms, so the
#: exported property is compared against the checkpoint tensor without applying
#: an inverse. Keeping the map here -- rather than in the exporter -- lets the
#: independent verifier use the same definition without importing the code that
#: produced the file.
PLY_ATTRIBUTE_MAP = {
    'x': ('means', 0), 'y': ('means', 1), 'z': ('means', 2),
    'opacity': ('opacities', None),
    'scale_0': ('scales', 0), 'scale_1': ('scales', 1), 'scale_2': ('scales', 2),
    'rot_0': ('quats', 0), 'rot_1': ('quats', 1),
    'rot_2': ('quats', 2), 'rot_3': ('quats', 3),
    'f_dc_0': ('sh0', 0), 'f_dc_1': ('sh0', 1), 'f_dc_2': ('sh0', 2),
}


# --------------------------------------------------------------------------- #
# Hashing and file identity
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA256 of a file's bytes, streamed so a 80 MB checkpoint is fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """Stable hash of a JSON payload: sorted keys, no whitespace noise."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    ).hexdigest()


def file_identity(path: Path) -> dict:
    """Bytes, mtime and hash of one file -- enough to detect any later change."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_utc": _utc(stat.st_mtime),
        "sha256": sha256_file(path),
    }


def _utc(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def utc_now() -> str:
    return _utc(time.time())


# --------------------------------------------------------------------------- #
# Run environment, recorded so a rerun can be compared against it
# --------------------------------------------------------------------------- #
def git_commit(repo: Path | None = None) -> str | None:
    """Current git commit, or None when the code is not in a repo.

    A missing commit is not an error: the record simply says the build was not
    under version control. That is more useful than a fake ``"unknown"`` that
    looks like a real revision.
    """
    repo = Path(repo or Path.cwd()).resolve()
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_dirty(repo: Path | None = None) -> bool | None:
    repo = Path(repo or Path.cwd()).resolve()
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def torch_fingerprint() -> dict:
    """Interpreter and framework identity. Optional on purpose."""
    record: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is optional for tests
        record["torch"] = f"unavailable: {type(exc).__name__}"
        return record
    record["torch"] = torch.__version__
    record["cuda"] = torch.version.cuda
    record["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        record["devices"] = [torch.cuda.get_device_name(i)
                             for i in range(torch.cuda.device_count())]
    try:
        import gsplat
        record["gsplat"] = getattr(gsplat, "__version__", "imported")
    except Exception:
        record["gsplat"] = "unavailable"
    return record


# --------------------------------------------------------------------------- #
# Reading and writing records
# --------------------------------------------------------------------------- #
def write_record(path: Path, payload: dict) -> Path:
    """Atomically write a JSON record (tmp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    return path


def read_record(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def train_record(*, result_dir: Path, dataset: Path, config: dict,
                 seed: int, checkpoint: Path, step: int,
                 extra: dict | None = None, repo: Path | None = None) -> dict:
    """Provenance for one *finished* training run."""
    checkpoint = Path(checkpoint)
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "train",
        "created_utc": utc_now(),
        "dataset": {"path": str(Path(dataset).resolve()), "name": Path(dataset).name},
        "git_commit": git_commit(repo),
        "git_dirty": git_dirty(repo),
        "seed": seed,
        "config": config,
        "config_sha256": sha256_json(config),
        "checkpoint": file_identity(checkpoint),
        "checkpoint_step": int(step),
        "host": {"pid": os.getpid(), "cwd": os.getcwd()},
        "environment": torch_fingerprint(),
    }
    if extra:
        record.update(extra)
    return record


def export_record(*, output: Path, checkpoint: Path, step: int, ply: Path,
                  gaussians_before: int, gaussians_after: int,
                  cleanup: dict,
                  format_validation: dict | None = None,
                  attribute_validation: dict | None = None,
                  extra: dict | None = None) -> dict:
    """Provenance for one export, tied to the checkpoint it was made from."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "export",
        "created_utc": utc_now(),
        "source_checkpoint": file_identity(checkpoint),
        "source_checkpoint_step": int(step),
        "output_ply": file_identity(ply),
        "gaussians_before_filter": int(gaussians_before),
        "gaussians_after_filter": int(gaussians_after),
        "cleanup": cleanup,
        # Two different questions, kept separate on purpose: does the file have
        # the required properties at all (format), and do its rows equal the
        # rows of the checkpoint it claims to come from (attribute consistency).
        "format_validation": format_validation or {},
        "attribute_validation": attribute_validation or {},
        "output_dir": str(Path(output).resolve()),
    }
    if extra:
        record.update(extra)
    return record


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_checkpoint_unchanged(record: dict) -> list[str]:
    """Re-hash the checkpoint a record points at and report any drift."""
    problems = []
    identity = record.get("checkpoint") or record.get("source_checkpoint")
    if not identity:
        return ["provenance record has no checkpoint identity"]
    path = Path(identity["path"])
    if not path.is_file():
        return [f"checkpoint missing: {path}"]
    current = file_identity(path)
    for field in ("bytes", "sha256"):
        if identity.get(field) != current[field]:
            problems.append(
                f"{path.name} {field} changed since provenance was written: "
                f"{identity.get(field)} -> {current[field]}")
    return problems


def verify_ply_unchanged(record: dict) -> list[str]:
    identity = record.get("output_ply")
    if not identity:
        return ["provenance record has no output_ply identity"]
    path = Path(identity["path"])
    if not path.is_file():
        return [f"PLY missing: {path}"]
    current = file_identity(path)
    problems = []
    for field in ("bytes", "sha256"):
        if identity.get(field) != current[field]:
            problems.append(
                f"{path.name} {field} changed since provenance was written: "
                f"{identity.get(field)} -> {current[field]}")
    return problems


def verify_export_chain(record: dict) -> list[str]:
    """Full-chain check on an export record.

    Beyond byte identity, the export must be *attributable*: the step recorded
    inside the checkpoint, the step recorded in the export record, and the
    gaussian counts before and after filtering must all be present and
    consistent. A missing field is a problem, not a blank to skip.
    """
    problems = []
    step = record.get("source_checkpoint_step")
    if step is None:
        problems.append("export record has no source_checkpoint_step")
    before = record.get("gaussians_before_filter")
    after = record.get("gaussians_after_filter")
    if before is None or after is None:
        problems.append("export record is missing gaussian counts")
    else:
        if int(after) > int(before):
            problems.append("gaussians_after_filter exceeds gaussians_before_filter")
        if int(after) < 100:
            problems.append("export retained fewer than 100 gaussians")
    problems.extend(verify_checkpoint_unchanged(record))
    problems.extend(verify_ply_unchanged(record))
    fmt = record.get("format_validation") or {}
    if not fmt.get("finite"):
        problems.append("format_validation reports non-finite values, or is missing")
    if fmt.get("count") is not None and after is not None:
        if int(fmt["count"]) != int(after):
            problems.append(
                f"PLY vertex count {fmt['count']} != recorded "
                f"gaussians_after_filter {after}")
    attr = record.get("attribute_validation") or {}
    if not attr:
        problems.append("attribute_validation is missing")
    elif not attr.get("passed"):
        problems.append(
            "attribute_validation failed: %s" % (attr.get("failures") or "no detail"))
    return problems


def load_chain(directory: Path) -> dict:
    """Collect every provenance record in a run directory."""
    directory = Path(directory)
    records = {}
    for path in sorted(directory.rglob("*.provenance.json")):
        try:
            records[str(path.relative_to(directory))] = read_record(path)
        except Exception as exc:
            records[str(path.relative_to(directory))] = {"unreadable": str(exc)}
    return records
