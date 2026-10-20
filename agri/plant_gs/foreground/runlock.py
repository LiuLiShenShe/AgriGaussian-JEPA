"""Exclusive run-directory lock, and the state machine that guards writes.

Stage 4B lost a Cotton/1 reconstruction because two training processes were
pointed at one ``results/train_30k`` directory at the same time. Nothing
crashed. The directory ended up with:

* ``history.jsonl`` holding two interleaved eval streams,
* checkpoint files from two different save cadences,
* and a ``splats_best.pt`` whose content depended on which process wrote last.

The exported PLY then quoted a PSNR that no single training run had produced,
and no file on disk could say which checkpoint it came from.

This module makes that collision **impossible to enter silently**:

``RunLock``
    An advisory ``flock`` on ``<result>/.train.lock``. The second process to
    ask for a directory that is already in use fails immediately instead of
    waiting and then interleaving. ``flock`` is released by the kernel when the
    holder dies, so a crashed run does not wedge the directory forever.

``describe_result``
    Classifies a result directory as empty / complete / partial / foreign
    *while the lock is held*, so the decision to write is made under the same
    exclusive guarantee as the write itself.

The lock file lives *inside* the result directory, so the lock is per
directory. Two paths that name the same directory -- relative vs absolute, or
through a symlink -- resolve to the same inode and therefore contend for the
same lock. :func:`resolve_result_path` does that resolution explicitly rather
than trusting the caller's spelling.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import time
from pathlib import Path

LOCK_NAME = ".train.lock"

#: Marker written into a result directory to record which run owns it.
RUN_MARKER = ".run_state.json"


class LockUnavailable(RuntimeError):
    """Raised when another live process already holds the directory lock."""


class ResultStateError(RuntimeError):
    """Raised when the target directory is not safe to write into."""


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def resolve_result_path(path: Path | str) -> Path:
    """Canonical identity of a result directory.

    ``Path.resolve()`` collapses symlinks, so ``results/train`` and a symlink
    pointing at it -- and any relative spelling of either -- all land on the
    same path. The lock file is created inside that directory, so two spellings
    of one directory contend for one lock.

    The path is *not* created here: a caller asking about a directory that does
    not exist yet should get an answer about that fact, not a side effect.
    """
    return Path(os.path.realpath(os.path.abspath(os.fspath(path))))


# --------------------------------------------------------------------------- #
# The lock
# --------------------------------------------------------------------------- #
class RunLock:
    """Exclusive, non-blocking advisory lock over one result directory.

    Usage::

        with RunLock(result_dir) as lock:
            ...  # exclusive section

    The lock is taken on first entry and released on exit, including on
    exceptions. If the lock cannot be taken, :class:`LockUnavailable` is
    raised immediately -- the point is to fail fast, not to queue.
    """

    def __init__(self, result_dir: Path | str, *, create: bool = True):
        self.requested = Path(result_dir)
        self.path = resolve_result_path(result_dir)
        self._handle = None
        self._create = create

    # -- lifecycle --------------------------------------------------------- #
    def acquire(self) -> "RunLock":
        if self._handle is not None:
            raise RuntimeError("lock already held by this object")
        if self._create:
            self.path.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_dir():
            raise ResultStateError(f"result directory does not exist: {self.path}")

        lock_file = self.path / LOCK_NAME
        handle = open(lock_file, "a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = self._read_holder(lock_file)
            handle.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise LockUnavailable(
                    f"another process is already using {self.path}"
                    + (f" (held by pid {holder})" if holder else "")
                    + "; refusing to write into the same result directory"
                ) from exc
            raise
        # Record the holder so a failure message can name it. The lock itself
        # is the kernel's, not this file's: truncating the file does not
        # release the lock, so this is safe to rewrite.
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({
                "pid": os.getpid(),
                "acquired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cwd": os.getcwd(),
                "result_dir": str(self.path),
            }))
            handle.flush()
            os.fsync(handle.fileno())
        except OSError:
            pass
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._handle.close()
            self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    @staticmethod
    def _read_holder(lock_file: Path) -> str | None:
        try:
            payload = json.loads(lock_file.read_text() or "{}")
        except Exception:
            return None
        pid = payload.get("pid")
        return str(pid) if pid else None

    # -- context manager ---------------------------------------------------- #
    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


# --------------------------------------------------------------------------- #
# Result directory state
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def describe_result(result_dir: Path | str) -> dict:
    """Classify a result directory without writing anything.

    Returns a dict with ``state`` in:

    ``empty``
        No run has ever touched this directory. Safe to start.
    ``complete``
        A finished run: ``splats.pt`` and ``status.json`` with
        ``status == "complete"``. Must not be silently retrained or overwritten.
    ``partial``
        A run started but did not finish (interrupted, crashed, or killed).
        Refuse by default; only a genuine resume may continue it.
    ``foreign``
        The directory holds artefacts from a *different* configuration than the
        one now requested -- a different step count, seed, or dataset. Writing
        here would mix two experiments.

    The caller is expected to hold :class:`RunLock` while acting on this, so
    the classification and the subsequent write are one atomic decision.
    """
    path = resolve_result_path(result_dir)
    info = {
        "path": str(path),
        "state": "empty",
        "status": None,
        "config": None,
        "config_sha256": None,
        "steps": None,
        "best_step": None,
        "checkpoints": [],
        "has_history": False,
        "has_final": False,
        "problems": [],
    }
    if not path.is_dir():
        return info

    status = _read_json(path / "status.json")
    config = _read_json(path / "config.json")
    info["status"] = status or None
    info["config"] = config or None
    info["config_sha256"] = config.get("config_sha256") if config else None
    info["steps"] = status.get("steps") if status else None
    info["best_step"] = status.get("best_step") if status else None
    info["has_history"] = (path / "history.jsonl").is_file()
    info["has_final"] = (path / "splats.pt").is_file()
    info["checkpoints"] = sorted(
        p.name for p in path.glob("splats_*.pt") if p.is_file())

    artefacts = [name for name in ("status.json", "config.json", "history.jsonl",
                                   "splats.pt", "splats_best.pt")
                 if (path / name).exists()]
    info["artefacts"] = artefacts

    if not artefacts:
        info["state"] = "empty"
        return info

    complete = bool(status.get("status") == "complete") and info["has_final"]
    if complete:
        info["state"] = "complete"
        return info

    # Something is there but the run did not finish. Distinguish "same
    # experiment, interrupted" from "different experiment entirely".
    info["state"] = "partial"
    info["problems"].append(
        f"result directory {path} holds an unfinished run "
        f"(artefacts: {', '.join(artefacts)})")
    return info


def compare_config(existing: dict | None, requested: dict) -> list[str]:
    """Keys on which two run configs disagree.

    ``result`` and ``data`` are compared by resolved path so the same run
    spelled two ways is not reported as a mismatch, while a genuinely
    different dataset or step count is.
    """
    if not existing:
        return []
    problems = []
    for key, value in requested.items():
        if key in ("result", "data"):
            old = existing.get(key)
            if old is None:
                continue
            try:
                same = resolve_result_path(old) == resolve_result_path(value)
            except Exception:
                same = old == value
            if not same:
                problems.append(f"{key}: existing={old!r} requested={value!r}")
            continue
        if key not in existing:
            continue
        if existing[key] != value:
            problems.append(f"{key}: existing={existing[key]!r} requested={value!r}")
    return problems


def assert_writable(result_dir: Path | str, requested_config: dict,
                    *, resume: bool = False) -> dict:
    """Decide whether a run may start in ``result_dir``.

    Call this **while holding the lock**. It raises :class:`ResultStateError`
    with an actionable message rather than returning a flag, because every
    caller so far treated "not writable" as fatal anyway.

    Rules:

    * ``empty`` -> start from scratch.
    * ``complete`` -> refuse. A finished run is not silently retrained.
    * ``partial`` -> refuse unless ``resume`` is requested; even then the
      caller must find a genuinely resumable checkpoint.
    * config mismatch -> always refuse. That directory belongs to a different
      experiment.
    """
    info = describe_result(result_dir)
    problems = []
    if info["state"] == "complete" and not resume:
        problems.append(
            f"{info['path']} already holds a COMPLETE run "
            f"(steps={info['steps']}, best_step={info['best_step']}); "
            "refusing to silently retrain or overwrite it. Use a new result "
            "directory, or pass --resume only to continue an interrupted run.")
    if info["state"] == "partial":
        mismatch = compare_config(info["config"], requested_config)
        if mismatch:
            problems.append(
                f"{info['path']} holds artefacts from a DIFFERENT configuration: "
                + "; ".join(mismatch)
                + ". Refusing to mix two experiments in one directory; use a "
                  "new result directory.")
        elif not resume:
            problems.append(
                f"{info['path']} holds an INCOMPLETE run; refusing to start a "
                "second run over it. Pass --resume to continue that run (which "
                "requires a resumable checkpoint), or use a new result directory.")
    if problems:
        raise ResultStateError("\n".join(problems))
    return info


def write_run_marker(result_dir: Path | str, payload: dict) -> Path:
    """Record which run owns a directory, and with which config."""
    path = resolve_result_path(result_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / RUN_MARKER
    body = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)
    return target
