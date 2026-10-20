"""Trainer entry-point contract: what it refuses, and what it refuses to claim.

The training loop needs a GPU and a prepared dataset, so these tests drive the
*decision* layer instead -- argument handling, resume validation, and the
ordering that puts the lock before the state inspection. That is where the
Stage 4B accident lived.
"""
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import train  # noqa: E402
from runlock import LOCK_NAME, RunLock, resolve_result_path  # noqa: E402


def test_trainer_module_imports_os_regression():
    """The Stage 4B lock called ``os.getpid()`` with no ``import os``.

    It raised ``NameError`` on every invocation, so the "fix" never locked
    anything. Guard the import itself, not just the behaviour.
    """
    assert hasattr(train, 'os')
    assert train.os.getpid() > 0


def test_lock_is_taken_before_the_result_directory_is_inspected():
    """Order is the fix: inspect-then-lock is the race that lost Cotton/1."""
    source = inspect.getsource(train.main)
    lock_at = source.index('RunLock(')
    inspect_markers = [source.index('assert_writable('),
                       source.index("'config.json'"), source.index('status.json')]
    assert lock_at < min(inspect_markers), (
        'main() inspects the result directory before acquiring the lock')


def test_lock_is_released_through_a_finally_block():
    source = inspect.getsource(train.main)
    assert 'finally:' in source and 'lock.release()' in source
    assert source.index('finally:') > source.index('RunLock(')


def run_trainer(*args, env=None):
    import os
    full = dict(os.environ)
    full['PYTHONPATH'] = str(CODE)
    if env:
        full.update(env)
    return subprocess.run(
        [sys.executable, '-u', str(CODE / 'train.py'), *map(str, args)],
        capture_output=True, text=True, env=full, timeout=300)


def test_append_history_is_rejected_as_a_resume_mechanism(tmp_path):
    proc = run_trainer('--data', tmp_path / 'dataset',
                       '--result', tmp_path / 'train_30k', '--append-history')
    assert proc.returncode != 0
    assert '--append-history is not a resume mechanism' in proc.stderr
    assert '--resume' in proc.stderr


def test_trainer_refuses_a_completed_directory(tmp_path):
    result = tmp_path / 'train_30k'
    result.mkdir()
    (result / 'splats.pt').write_bytes(b'ckpt')
    (result / 'config.json').write_text(json.dumps({'steps': 30000}))
    (result / 'status.json').write_text(json.dumps({'status': 'complete', 'steps': 30000}))
    proc = run_trainer('--data', tmp_path / 'dataset', '--result', result,
                       '--steps', 30000)
    assert proc.returncode != 0
    assert 'COMPLETE run' in proc.stderr
    # A refusal may leave the lock file and a run marker behind, but it must
    # not have added any run artefact of its own.
    assert sorted(p.name for p in result.iterdir()) == [
        '.train.lock', 'config.json', 'splats.pt', 'status.json']


def test_trainer_refuses_a_partial_directory_without_resume(tmp_path):
    result = tmp_path / 'train_30k'
    result.mkdir()
    (result / 'config.json').write_text(json.dumps({'steps': 30000}))
    (result / 'history.jsonl').write_text('{"step": 0}\n')
    before = (result / 'history.jsonl').read_bytes()
    proc = run_trainer('--data', tmp_path / 'dataset', '--result', result,
                       '--steps', 30000)
    assert proc.returncode != 0
    assert 'INCOMPLETE run' in proc.stderr
    assert (result / 'history.jsonl').read_bytes() == before


def test_trainer_refuses_resume_without_a_resumable_checkpoint(tmp_path):
    result = tmp_path / 'train_30k'
    result.mkdir()
    (result / 'config.json').write_text(json.dumps({'steps': 30000}))
    (result / 'history.jsonl').write_text('{"step": 0}\n')
    proc = run_trainer('--data', tmp_path / 'dataset', '--result', result,
                       '--steps', 30000, '--resume')
    assert proc.returncode != 0
    assert 'REFUSING TO RESUME' in proc.stderr
    assert 'splats_resume.pt is absent' in proc.stderr


def test_refusal_happens_before_the_dataset_is_read(tmp_path):
    """A refusal must not depend on, or pay for, loading a dataset."""
    result = tmp_path / 'train_30k'
    result.mkdir()
    (result / 'splats.pt').write_bytes(b'ckpt')
    (result / 'config.json').write_text(json.dumps({'steps': 30000}))
    (result / 'status.json').write_text(json.dumps({'status': 'complete', 'steps': 30000}))
    proc = run_trainer('--data', tmp_path / 'does-not-exist',
                       '--result', result, '--steps', 30000)
    assert 'COMPLETE run' in proc.stderr
    assert 'No such file' not in proc.stderr


# --------------------------------------------------------------------------- #
# load_resume_state: the honest-resume gate
# --------------------------------------------------------------------------- #
def _config_sha(**overrides):
    config = {'steps': 30000, 'seed': 42, 'save_every': 5000}
    config.update(overrides)
    import provenance
    return provenance.sha256_json(config), config


def _write_resume(result: Path, *, resumable=True, data=None, sha=None,
                  omit=()):
    payload = {'step': 12000, 'resumable': resumable, 'run_id': 'r1',
               'config_sha256': sha, 'data': data, 'best_score': 1.0,
               'best_step': 11000, 'optimizers': {'means': {'state': {}}},
               'strategy_state': {'scene_scale': 1.0},
               'rng': {'numpy_rng': {}}}
    for key in omit:
        payload.pop(key, None)
    torch.save(payload, result / train.RESUME_CHECKPOINT)


def test_resume_state_loads_when_everything_matches(tmp_path):
    result = tmp_path / 'train_30k'
    result.mkdir()
    sha, _ = _config_sha()
    dataset = tmp_path / 'dataset'
    _write_resume(result, data=str(dataset.resolve()), sha=sha)
    payload = train.load_resume_state(result, sha, dataset)
    assert payload['step'] == 12000


def test_resume_refuses_a_configuration_mismatch(tmp_path):
    result = tmp_path / 'train_30k'
    result.mkdir()
    sha, _ = _config_sha()
    other_sha, _ = _config_sha(steps=7000)
    dataset = tmp_path / 'dataset'
    _write_resume(result, data=str(dataset.resolve()), sha=other_sha)
    with pytest.raises(SystemExit, match='different configuration'):
        train.load_resume_state(result, sha, dataset)


def test_resume_refuses_a_dataset_mismatch(tmp_path):
    result = tmp_path / 'train_30k'
    result.mkdir()
    sha, _ = _config_sha()
    _write_resume(result, data=str((tmp_path / 'other').resolve()), sha=sha)
    with pytest.raises(SystemExit, match='checkpoint dataset'):
        train.load_resume_state(result, sha, tmp_path / 'dataset')


def test_resume_refuses_a_model_only_checkpoint(tmp_path):
    result = tmp_path / 'train_30k'
    result.mkdir()
    sha, _ = _config_sha()
    _write_resume(result, resumable=False, data=str(tmp_path.resolve()), sha=sha)
    with pytest.raises(SystemExit, match='not a resumable checkpoint'):
        train.load_resume_state(result, sha, tmp_path)


@pytest.mark.parametrize('missing', ['optimizers', 'strategy_state', 'rng'])
def test_resume_refuses_when_a_required_state_block_is_missing(tmp_path, missing):
    result = tmp_path / 'train_30k'
    result.mkdir()
    sha, _ = _config_sha()
    _write_resume(result, data=str(tmp_path.resolve()), sha=sha, omit=(missing,))
    with pytest.raises(SystemExit, match=missing):
        train.load_resume_state(result, sha, tmp_path)


# --------------------------------------------------------------------------- #
# Checkpoint writing
# --------------------------------------------------------------------------- #
def test_checkpoint_writes_are_atomic(tmp_path):
    """A reader must never see a partial file, even under a killed writer."""
    calls = []
    real_replace = train.os.replace

    def spy(src, dst):
        calls.append((Path(src).name, Path(dst).name))
        return real_replace(src, dst)

    train.os.replace = spy
    try:
        train.atomic_torch_save({'a': 1}, tmp_path / 'splats_resume.pt')
    finally:
        train.os.replace = real_replace
    assert calls and calls[0][1] == 'splats_resume.pt'
    assert not list(tmp_path.glob('*.tmp'))
    assert torch.load(tmp_path / 'splats_resume.pt', weights_only=False) == {'a': 1}


def test_a_failed_checkpoint_write_leaves_no_temp_file(tmp_path):
    class Boom:
        def __reduce__(self):
            raise RuntimeError('cannot serialise')

    with pytest.raises(Exception):
        train.atomic_torch_save({'x': Boom()}, tmp_path / 'splats_resume.pt')
    assert not list(tmp_path.glob('*.tmp'))
    assert not (tmp_path / 'splats_resume.pt').exists()


def test_periodic_and_final_checkpoints_are_resumable_but_best_is_not(tmp_path):
    """Only the files a resume accepts may claim to be resumable."""
    class FakeData:
        root = tmp_path / 'dataset'
        normalization = {'center': [0, 0, 0], 'radius': 1.0, 'scale': 1.0}

    splats = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(torch.zeros(4, 3)),
        'scales': torch.nn.Parameter(torch.zeros(4, 3)),
        'quats': torch.nn.Parameter(torch.zeros(4, 4)),
        'opacities': torch.nn.Parameter(torch.zeros(4)),
        'sh0': torch.nn.Parameter(torch.zeros(4, 1, 3)),
        'shN': torch.nn.Parameter(torch.zeros(4, 15, 3)),
    })
    config = {'steps': 30000}
    plain = train.checkpoint(splats, FakeData(), 3, 11000, config)
    full = train.checkpoint(splats, FakeData(), 3, 11000, config,
                            resume_state={'optimizers': {}, 'strategy_state': {},
                                          'rng': {}})
    assert plain['resumable'] is False
    assert full['resumable'] is True
    assert full['config_sha256'] == plain['config_sha256']
    assert full['step'] == 11000


def test_strategy_state_round_trips_through_the_cpu_snapshot():
    state = {'grad2d': torch.ones(3), 'count': torch.zeros(3), 'scene_scale': 1.0,
             'radii': None}
    snapshot = train._strategy_state_cpu(state)
    assert snapshot['scene_scale'] == 1.0
    assert snapshot['radii'] is None
    restored = train._strategy_state_to(snapshot, 'cpu')
    assert torch.equal(restored['grad2d'], state['grad2d'])