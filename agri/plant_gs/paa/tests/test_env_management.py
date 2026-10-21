"""Tests for the conda-free environment setup.

The migration is only real if it is protected against silent regression. The
three ways it could regress are all tested here:

* someone re-adds a conda manifest or a `conda activate` to an executable file;
* the lock files drift away from the pinned CUDA builds;
* the interpreter resolver falls back to whatever `python` is on PATH instead
  of reporting that the project env is missing.

The tests read files only. They never create an environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]  # .../Gaussian-JEPA
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agri.plant_gs.paa import env_resolve  # noqa: E402


# --------------------------------------------------------------------------- #
# No conda in the repository's executable surface
# --------------------------------------------------------------------------- #
def test_no_conda_manifest_exists():
    assert not (REPO / "environment.yml").exists()
    assert not (REPO / "environment.yaml").exists()
    assert not (REPO / "conda-lock.yml").exists()


def test_no_executable_file_invokes_conda():
    """Comments may mention conda; code may not call it."""
    import re
    # A real invocation is followed by a name-like argument. Prose such as
    # "conda install. That made them..." has a sentence break there instead.
    call = re.compile(
        r"\bconda\s+(activate|create|run|install|env)\s+[A-Za-z0-9_./-]+")
    offenders = []
    for pattern in ("*.sh", "*.py", "*.yaml", "*.yml"):
        for path in REPO.rglob(pattern):
            if any(part in {".git", "envs", "data", "results", "reports"}
                   for part in path.parts):
                continue
            if path.name == "test_env_management.py":
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                # A comment documenting the migration is fine; an executed or
                # copy-pasteable command is not.
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue
                if call.search(line):
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "conda invocation found:\n" + "\n".join(offenders)


def test_readme_documents_the_uv_install():
    text = (REPO / "README.md").read_text()
    assert "env/bootstrap.sh" in text
    assert "conda env create" not in text
    guide = (REPO / "env" / "README.md").read_text()
    assert "uv" in guide and "bootstrap.sh" in guide


# --------------------------------------------------------------------------- #
# Lock files keep the CUDA variants the build depends on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lock,expected", [
    ("requirements-gs.lock.txt", "torch==2.9.1+cu128"),
    ("requirements-jepa.lock.txt", "torch==2.0.1+cu118"),
])
def test_lockfile_pins_the_cuda_variant(lock, expected):
    text = (REPO / "env" / lock).read_text()
    assert expected in text
    for line in text.splitlines():
        if line.startswith("torch=="):
            assert line.strip() == expected


def test_jepa_lock_keeps_setuptools_below_81():
    """PyTorch 2.0.1's cpp_extension imports pkg_resources."""
    text = (REPO / "env" / "requirements-jepa.lock.txt").read_text()
    version = next(line.split("==")[1] for line in text.splitlines()
                   if line.startswith("setuptools=="))
    assert int(version.split(".")[0]) < 81


def test_bootstrap_never_borrows_a_path_interpreter():
    """--python-preference only-managed is what keeps conda out of the venv."""
    text = (REPO / "env" / "bootstrap.sh").read_text()
    assert text.count("--python-preference only-managed") >= 2


# --------------------------------------------------------------------------- #
# Interpreter resolution
# --------------------------------------------------------------------------- #
def test_resolver_prefers_explicit_environment_variable(tmp_path, monkeypatch):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("AGRI_PY_JEPA", str(fake))
    resolved = env_resolve.resolve(tmp_path)
    assert resolved["AGRI_PY_JEPA"] == str(fake.resolve())


def test_resolver_reports_missing_instead_of_falling_back(tmp_path, monkeypatch):
    monkeypatch.delenv("AGRI_PY_GS", raising=False)
    monkeypatch.delenv("AGRI_PY_JEPA", raising=False)
    resolved = env_resolve.resolve(tmp_path)
    assert resolved["AGRI_PY_GS"] is None
    assert resolved["AGRI_PY_JEPA"] is None


def test_resolver_finds_envs_under_root(tmp_path, monkeypatch):
    monkeypatch.delenv("AGRI_PY_GS", raising=False)
    monkeypatch.delenv("AGRI_PY_JEPA", raising=False)
    for name in ("agri-gs", "agri-jepa"):
        target = tmp_path / name / "bin" / "python"
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/sh\n")
    resolved = env_resolve.resolve(tmp_path)
    assert resolved["AGRI_PY_GS"].endswith("agri-gs/bin/python")
    assert resolved["AGRI_PY_JEPA"].endswith("agri-jepa/bin/python")


def test_resolver_cli_prints_exports(tmp_path, monkeypatch):
    monkeypatch.delenv("AGRI_PY_GS", raising=False)
    monkeypatch.delenv("AGRI_PY_JEPA", raising=False)
    target = tmp_path / "agri-gs" / "bin" / "python"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\n")
    proc = subprocess.run(
        [sys.executable, "-m", "agri.plant_gs.paa.env_resolve",
         "--root", str(tmp_path), "--quiet"],
        cwd=REPO, capture_output=True, text=True,
        env={k: v for k, v in __import__("os").environ.items()
             if not k.startswith("AGRI_PY_")} | {"PYTHONPATH": str(REPO)})
    assert "export AGRI_PY_GS=" in proc.stdout
    # JEPA is absent under this root, so its line must not be emitted.
    assert "AGRI_PY_JEPA=" not in proc.stdout
    assert proc.returncode == 1
