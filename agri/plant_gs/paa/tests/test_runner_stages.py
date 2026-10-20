"""Two-stage runner contract: separate runs, no silent overwrite, real resumption.

The runner is a shell script, so these tests source its helper library
(``stages.sh``) and drive it with fake stage directories. That exercises the
actual decision code -- stage classification, gate evaluation, manifest
writing -- without a GPU or a PAA sample.
"""
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
RUNNER = REPO / 'agri' / 'plant_gs' / 'paa' / 'run_paa_pilot.sh'
STAGES = REPO / 'agri' / 'plant_gs' / 'paa' / 'stages.sh'


def bash(script: str, *args, cwd=None):
    return subprocess.run(
        ['bash', '-c', script, '_', *map(str, args)],
        capture_output=True, text=True,
        env={**os.environ, 'PY_GS': 'python', 'REPO': str(REPO)},
        cwd=str(cwd or REPO), timeout=120)


def source_stages(body: str):
    return f'set -euo pipefail\nsource "{STAGES}"\nPY_GS=python\nREPO="{REPO}"\n{body}'


def test_runner_declares_two_separate_stage_directories():
    text = RUNNER.read_text()
    assert 'results/train_7k' in text or 'results/$name' in text
    assert 'train_7k' in text and 'train_30k' in text
    # The formal stage must be its own run_stage invocation, not a resume of 7k.
    calls = [line.strip() for line in text.splitlines() if line.strip().startswith('run_stage ')]
    assert any('train_7k' in c for c in calls)
    assert any('train_30k' in c for c in calls)


def test_runner_never_resumes_the_7k_model_into_the_30k_stage():
    text = RUNNER.read_text()
    assert 'splats_resume' not in text.split('run_stage train_30k')[0].split('run_stage train_7k')[-1]
    # The 30k stage reads its own directory only.
    block = text[text.index('run_stage train_30k'):]
    assert 'train_7k' not in block.split('\n')[0]


def test_runner_sources_the_shared_stage_library():
    text = RUNNER.read_text()
    assert 'stages.sh' in text and 'source ' in text


def test_runner_has_no_inline_copy_of_stage_state():
    """One implementation, so the runner and the tests cannot drift."""
    assert 'stage_state() {' not in RUNNER.read_text()
    assert 'stage_state() {' in STAGES.read_text()


def test_runner_exports_from_the_explicit_30k_best_checkpoint():
    text = RUNNER.read_text()
    assert 'BEST="$WS/results/train_30k/splats_best.pt"' in text
    export_block = text[text.index('export from the 30k best'):]
    assert '--checkpoint "$BEST"' in export_block


def test_runner_verifies_provenance_before_registering():
    text = RUNNER.read_text()
    assert text.index('verify_provenance.py') < text.index('registry.py')


def test_runner_fails_when_the_best_checkpoint_is_missing():
    text = RUNNER.read_text()
    assert '[ -f "$BEST" ]' in text


# --------------------------------------------------------------------------- #
# stage_state
# --------------------------------------------------------------------------- #
def _stage(path: Path, *, status=None, splats=True, config=True, history=False):
    path.mkdir(parents=True, exist_ok=True)
    if config:
        (path / 'config.json').write_text('{"steps": 30000}')
    if history:
        (path / 'history.jsonl').write_text('{"step": 0}\n')
    if splats:
        (path / 'splats.pt').write_bytes(b'ckpt')
    if status is not None:
        (path / 'status.json').write_text(json.dumps({'status': status, 'steps': 30000}))


def test_stage_state_reports_empty_for_a_missing_or_clean_directory(tmp_path):
    result = bash(source_stages('stage_state "$1"'), tmp_path / 'nope')
    assert result.stdout.strip() == 'empty'
    (tmp_path / 'clean').mkdir()
    result = bash(source_stages('stage_state "$1"'), tmp_path / 'clean')
    assert result.stdout.strip() == 'empty'


def test_stage_state_requires_both_status_complete_and_terminal_checkpoint(tmp_path):
    complete = tmp_path / 'a'
    _stage(complete, status='complete')
    assert bash(source_stages('stage_state "$1"'), complete).stdout.strip() == 'complete'

    no_checkpoint = tmp_path / 'b'
    _stage(no_checkpoint, status='complete', splats=False)
    assert bash(source_stages('stage_state "$1"'), no_checkpoint).stdout.strip() == 'partial'

    no_status = tmp_path / 'c'
    _stage(no_status)  # splats.pt but no status.json
    assert bash(source_stages('stage_state "$1"'), no_status).stdout.strip() == 'partial'


def test_stage_state_reports_partial_for_an_interrupted_run(tmp_path):
    run = tmp_path / 'run'
    _stage(run, status='training', splats=False, history=True)
    assert bash(source_stages('stage_state "$1"'), run).stdout.strip() == 'partial'

    # Exactly the Stage 4B Cotton/1 shape: history + checkpoints, no terminal file.
    cotton = tmp_path / 'cotton'
    cotton.mkdir()
    (cotton / 'config.json').write_text('{}')
    (cotton / 'history.jsonl').write_text('{"step": 0}\n')
    (cotton / 'splats_002000.pt').write_bytes(b'x')
    assert bash(source_stages('stage_state "$1"'), cotton).stdout.strip() == 'partial'


def test_a_completed_stage_is_never_re_run(tmp_path):
    """run_stage returns early on `complete`; prove it by watching for a write."""
    run = tmp_path / 'results' / 'train_7k'
    _stage(run, status='complete')
    before = (run / 'splats.pt').stat().st_mtime_ns
    body = (
        'WS="%s"\nRESUME=0\nPY_GS=python\nFG=/nonexistent\nREPO="%s"\nSEED=0\n'
        'run_stage train_7k 7000 1000 2000\necho RETURNED\n' % (tmp_path, REPO))
    result = bash(source_stages(body))
    assert result.returncode == 0, result.stderr
    assert 'already complete; skipping' in result.stdout
    assert 'RETURNED' in result.stdout
    assert (run / 'splats.pt').stat().st_mtime_ns == before


def test_an_incomplete_stage_is_refused_without_resume(tmp_path):
    run = tmp_path / 'results' / 'train_7k'
    _stage(run, status='training', splats=False, history=True)
    before = (run / 'history.jsonl').read_bytes()
    body = (
        'WS="%s"\nRESUME=0\nPY_GS=python\nFG=/nonexistent\nREPO="%s"\nSEED=0\n'
        'run_stage train_7k 7000 1000 2000\n' % (tmp_path, REPO))
    result = bash(source_stages(body))
    assert result.returncode != 0
    assert 'INCOMPLETE run' in result.stderr
    assert (run / 'history.jsonl').read_bytes() == before


# --------------------------------------------------------------------------- #
# gate_check
# --------------------------------------------------------------------------- #
def _write_validation(path, psnr=23.9, iou=0.97, bg=0.0078):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'foreground_psnr': psnr, 'silhouette_iou': iou,
                                'background_alpha': bg}))


def test_gate_passes_on_a_healthy_7k_run(tmp_path):
    _write_validation(tmp_path / 'v.json')
    result = bash(source_stages('gate_check "$1" 18 0.85 0.05'), tmp_path / 'v.json')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'GATE_PASSED' in result.stdout


@pytest.mark.parametrize('field,value,reason', [
    ('psnr', 10.0, 'foreground_psnr'),
    ('iou', 0.50, 'silhouette_iou'),
    ('bg', 0.40, 'background_alpha'),
])
def test_gate_fails_on_each_metric(tmp_path, field, value, reason):
    kwargs = {field: value}
    _write_validation(tmp_path / 'v.json', **kwargs)
    result = bash(source_stages('gate_check "$1" 18 0.85 0.05'), tmp_path / 'v.json')
    assert result.returncode == 1
    assert 'GATE_FAILED' in result.stdout
    assert reason in result.stdout


def test_gate_distinguishes_unreadable_from_failed(tmp_path):
    missing = bash(source_stages('gate_check "$1" 18 0.85 0.05'), tmp_path / 'nope.json')
    assert missing.returncode == 2
    assert 'GATE_ERROR' in missing.stderr

    broken = tmp_path / 'broken.json'
    broken.write_text('{"foreground_psnr": 1}')
    result = bash(source_stages('gate_check "$1" 18 0.85 0.05'), broken)
    assert result.returncode == 2
    assert 'missing' in result.stdout


# --------------------------------------------------------------------------- #
# stage_manifest
# --------------------------------------------------------------------------- #
def test_stage_manifest_records_code_and_command(tmp_path):
    stage = tmp_path / 'train_7k'
    stage.mkdir()
    result = bash(source_stages('stage_manifest "$1" train_7k 7000 "python train.py --steps 7000"'),
                  stage)
    assert result.returncode == 0, result.stderr
    payload = json.loads((stage / 'stage_manifest.json').read_text())
    assert payload['stage'] == 'train_7k'
    assert payload['steps'] == 7000
    assert payload['command'] == 'python train.py --steps 7000'
    assert payload['finished_utc'].endswith('Z')
    assert payload['git_commit']


def test_stage_manifest_is_json_even_after_a_retry(tmp_path):
    stage = tmp_path / 'train_30k'
    stage.mkdir()
    for _ in range(2):
        bash(source_stages('stage_manifest "$1" train_30k 30000 "cmd"'), stage)
    payload = json.loads((stage / 'stage_manifest.json').read_text())
    assert payload['stage'] == 'train_30k'