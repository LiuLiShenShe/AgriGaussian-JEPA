#!/usr/bin/env python
"""Locate the project's two interpreters, without conda.

Why this exists
---------------
Every reproduction command in this repo names the interpreter it needs, because
there are two of them (Python 3.13 / torch 2.9.1 for 3DGS, Python 3.9 /
torch 2.0.1 for Gaussian-JEPA). Those commands used to hard-code the paths of a
conda install. That made them unrunnable anywhere else, and it made the conda
environment look like a requirement rather than an implementation detail.

``env_resolve.py`` resolves the paths the same way the shell activation script
does, in this order:

1. ``AGRI_PY_GS`` / ``AGRI_PY_JEPA`` if already set in the environment;
2. ``$AGRI_ENVS_ROOT/agri-gs/bin/python`` (and ``agri-jepa``), which is where
   ``env/bootstrap.sh`` puts them, relative to the repository root;
3. a ``--root`` override, for a checkout that was bootstrapped elsewhere.

It prints shell assignments, so the two ways to use it are:

    eval "$(python -m agri.plant_gs.paa.env_resolve)"      # inside this repo
    python -m agri.plant_gs.paa.env_resolve --check        # verify both work

A missing interpreter is reported with the bootstrap command, not silently
replaced by whatever ``python`` is on PATH -- silently falling back is how a
3DGS command ends up running under the JEPA interpreter (no gsplat) or under a
system python with no torch at all.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = os.environ.get("AGRI_ENVS_ROOT") or str(REPO / "envs")

INTERPRETERS = {
    "AGRI_PY_GS": ("agri-gs", "gsplat", "3DGS reconstruction, PLY validation, semantics"),
    "AGRI_PY_JEPA": ("agri-jepa", "pointnet2_ops", "Gaussian-JEPA forward/backward, CUDA extensions"),
}


def resolve(root: Path) -> dict:
    """Return {env var: path or None}, preferring an explicit setting."""
    out = {}
    for var, (dirname, _probe, _why) in INTERPRETERS.items():
        explicit = os.environ.get(var)
        if explicit and Path(explicit).is_file():
            out[var] = str(Path(explicit).resolve())
            continue
        candidate = root / dirname / "bin" / "python"
        out[var] = str(candidate.resolve()) if candidate.is_file() else None
    return out


def check(interpreter: str, module: str) -> tuple:
    """Import ``module`` with ``interpreter``; return (ok, detail)."""
    try:
        proc = subprocess.run([interpreter, "-c", f"import {module}"],
                              capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return False, tail[-1] if tail else "import failed"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT),
                    help="directory containing agri-gs/ and agri-jepa/")
    ap.add_argument("--check", action="store_true",
                    help="verify each interpreter imports its key module")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    resolved = resolve(args.root)
    ok = True
    for var, (_dirname, probe, why) in INTERPRETERS.items():
        path = resolved[var]
        if path is None:
            ok = False
            if not args.quiet:
                print(f"# {var}: NOT FOUND under {args.root} -- run: "
                      f"bash env/bootstrap.sh gs   # or jepa", file=sys.stderr)
            continue
        if args.check:
            good, detail = check(path, probe)
            if not good:
                ok = False
            if not args.quiet:
                print(f"# {var}={path}  probe {probe}: {detail}  ({why})",
                      file=sys.stderr)
        else:
            if not args.quiet:
                print(f"# {var}={path}  ({why})", file=sys.stderr)
    if args.check:
        return 0 if ok else 1
    for var, path in resolved.items():
        if path:
            print(f"export {var}={shlex.quote(path)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
