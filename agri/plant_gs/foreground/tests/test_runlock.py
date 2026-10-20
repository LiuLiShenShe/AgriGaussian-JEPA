"""Concurrency contract for training result directories.

These tests describe the Stage 4B accident directly: two trainers pointed at
one ``results/train_30k`` directory. Every test here uses real processes --
``flock`` is kernel state, so an in-process mock would prove nothing.
"""
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from runlock import (  # noqa: E402
    LOCK_NAME, LockUnavailable, ResultStateError, RunLock,
    assert_writable, compare_config, describe_result, resolve_result_path,
)


def spawn(script: str, *args, env=None):
    """Run a snippet with the foreground package importable."""
    full_env = dict(os.environ)
    full_env['PYTHONPATH'] = str(CODE)
    if env:
        full_env.update(env)
    return subprocess.Popen(
        [sys.executable, '-u', '-c', textwrap.dedent(script), *map(str, args)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=full_env)


def wait_for_line(process, needle, timeout=30):
    """Block until the child prints ``needle``, so tests never sleep blindly."""
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        seen.append(line)
        if needle in line:
            return seen
    out, err = process.communicate(timeout=5)
    raise AssertionError(
        f'child never printed {needle!r}; stdout={seen} stderr={err}')


HOLD_LOCK = """
import sys, time
sys.path.insert(0, {code!r})
from runlock import RunLock
lock = RunLock(sys.argv[1]).acquire()
print('LOCKED', flush=True)
time.sleep(float(sys.argv[2]))
lock.release()
print('RELEASED', flush=True)
"""


def test_second_process_is_refused_while_first_holds_the_directory(tmp_path):
    result = tmp_path / 'train_30k'
    first = spawn(HOLD_LOCK.format(code=str(CODE)), result, 30)
    try:
        wait_for_line(first, 'LOCKED')

        second = spawn("""
import sys
sys.path.insert(0, {code!r})
from runlock import RunLock, LockUnavailable
try:
    RunLock(sys.argv[1]).acquire()
except LockUnavailable as exc:
    print('REFUSED', exc)
else:
    print('ACQUIRED')
""".format(code=str(CODE)), result)
        out, err = second.communicate(timeout=30)
        assert second.returncode == 0, err
        assert 'REFUSED' in out, out
        assert 'refusing to write into the same result directory' in out
    finally:
        first.kill()
        first.communicate(timeout=10)


def test_two_simultaneous_launches_produce_exactly_one_winner(tmp_path):
    result = tmp_path / 'train_30k'
    # No coordination: both start together and race for the same directory.
    racer = """
import sys
sys.path.insert(0, {code!r})
from runlock import RunLock, LockUnavailable
try:
    lock = RunLock(sys.argv[1]).acquire()
except LockUnavailable:
    print('REFUSED', flush=True)
else:
    print('WON', flush=True)
    import time; time.sleep(1.5)
    lock.release()
""".format(code=str(CODE))
    procs = [spawn(racer, result) for _ in range(2)]
    outputs = []
    for proc in procs:
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err
        outputs.append(out.strip())
    assert sorted(outputs) == ['REFUSED', 'WON'], outputs


def test_different_result_directories_train_concurrently(tmp_path):
    """Serialising all training would be a regression, not a fix."""
    left, right = tmp_path / 'a' / 'train_30k', tmp_path / 'b' / 'train_30k'
    holder = spawn(HOLD_LOCK.format(code=str(CODE)), left, 1.0)
    try:
        wait_for_line(holder, 'LOCKED')
        other = spawn(HOLD_LOCK.format(code=str(CODE)), right, 0)
        out, err = other.communicate(timeout=30)
        assert other.returncode == 0, err
        assert 'LOCKED' in out and 'RELEASED' in out, out
    finally:
        holder.communicate(timeout=30)


def test_lock_is_released_when_the_holder_exits_abnormally(tmp_path):
    """A killed trainer must not wedge its directory forever."""
    result = tmp_path / 'train_30k'
    holder = spawn(HOLD_LOCK.format(code=str(CODE)), result, 60)
    wait_for_line(holder, 'LOCKED')
    holder.kill()
    holder.communicate(timeout=10)

    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with RunLock(result):
                break
        except LockUnavailable:
            time.sleep(0.1)
    else:
        pytest.fail('lock was not released after the holder was killed')


def test_lock_is_released_after_an_exception(tmp_path):
    result = tmp_path / 'train_30k'
    with pytest.raises(RuntimeError):
        with RunLock(result):
            raise RuntimeError('boom')
    with RunLock(result):  # must not raise
        pass
    assert not RunLock(result).held


def test_lock_released_on_system_exit(tmp_path):
    """``main`` raises ``SystemExit`` for refusals; that must still unlock."""
    result = tmp_path / 'train_30k'
    with pytest.raises(SystemExit):
        with RunLock(result):
            raise SystemExit(2)
    with RunLock(result):
        pass


def test_symlink_relative_and_absolute_paths_contend_for_one_lock(tmp_path):
    real = tmp_path / 'real' / 'train_30k'
    real.mkdir(parents=True)
    link = tmp_path / 'link'
    link.symlink_to(real)
    nested = tmp_path / 'real' / '.' / 'train_30k'

    assert resolve_result_path(link) == resolve_result_path(real)
    assert resolve_result_path(nested) == resolve_result_path(real)

    with RunLock(real):
        for alias in (link, nested, str(real), str(real) + '/'):
            with pytest.raises(LockUnavailable):
                RunLock(alias).acquire()
        # And the other way round: the lock file lives in the real directory.
        assert (real / LOCK_NAME).is_file()


def test_relative_path_alias_is_locked_too(tmp_path, monkeypatch):
    (tmp_path / 'results' / 'train_30k').mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    with RunLock(Path('results/train_30k')):
        with pytest.raises(LockUnavailable):
            RunLock(tmp_path / 'results' / 'train_30k').acquire()
        with pytest.raises(LockUnavailable):
            RunLock(Path('results/./train_30k')).acquire()


# --------------------------------------------------------------------------- #
# The write/no-write decision, which happens while the lock is held
# --------------------------------------------------------------------------- #
def _write_complete(result: Path, steps=30000, config=None):
    result.mkdir(parents=True, exist_ok=True)
    (result / 'splats.pt').write_bytes(b'checkpoint')
    (result / 'history.jsonl').write_text(
        json.dumps({'step': 0, 'loss': 1.0}) + '\n')
    (result / 'config.json').write_text(json.dumps(config or {'steps': steps}))
    (result / 'status.json').write_text(
        json.dumps({'status': 'complete', 'steps': steps, 'best_step': 24000}))


def _write_partial(result: Path, config=None):
    result.mkdir(parents=True, exist_ok=True)
    (result / 'history.jsonl').write_text(
        json.dumps({'step': 0, 'loss': 1.0}) + '\n')
    (result / 'config.json').write_text(json.dumps(config or {'steps': 30000}))
    (result / 'status.json').write_text(
        json.dumps({'status': 'training', 'steps': 12000, 'best_step': None}))


def test_empty_directory_is_writable(tmp_path):
    info = describe_result(tmp_path / 'train_30k')
    assert info['state'] == 'empty'
    assert_writable(tmp_path / 'train_30k', {})


def test_complete_directory_is_never_silently_retrained(tmp_path):
    result = tmp_path / 'train_30k'
    _write_complete(result)
    assert describe_result(result)['state'] == 'complete'
    with pytest.raises(ResultStateError, match='already holds a COMPLETE run'):
        assert_writable(result, {'steps': 30000})


def test_directory_with_history_is_not_silently_appended(tmp_path):
    """The exact Stage 4B shape: a history on disk, no completion marker."""
    result = tmp_path / 'train_30k'
    _write_partial(result)
    assert describe_result(result)['has_history'] is True
    with pytest.raises(ResultStateError, match='INCOMPLETE run'):
        assert_writable(result, {'steps': 30000})


def test_partial_directory_with_a_different_config_is_refused(tmp_path):
    result = tmp_path / 'train_30k'
    _write_partial(result, config={'steps': 7000, 'seed': 0})
    with pytest.raises(ResultStateError, match='DIFFERENT configuration'):
        assert_writable(result, {'steps': 30000, 'seed': 0})


def test_config_comparison_resolves_paths_but_not_semantics():
    same = compare_config(
        {'data': '/a/b/dataset', 'result': '/a/b/train', 'steps': 30000},
        {'data': '/a/b/./dataset', 'result': '/a/b/train', 'steps': 30000})
    assert same == []
    different = compare_config(
        {'data': '/a/b/dataset', 'steps': 30000, 'seed': 42},
        {'data': '/a/b/dataset', 'steps': 7000, 'seed': 42})
    assert any('steps' in problem for problem in different)
    assert not any('seed' in problem for problem in different)


def test_refusal_does_not_touch_the_directory(tmp_path):
    """A refused start must leave every byte where it was."""
    result = tmp_path / 'train_30k'
    _write_complete(result)
    before = {p.name: p.read_bytes() for p in result.iterdir() if p.is_file()}
    with pytest.raises(ResultStateError):
        assert_writable(result, {'steps': 30000})
    after = {p.name: p.read_bytes() for p in result.iterdir() if p.is_file()}
    assert before == after


TRAINER_LIKE = """
import sys, time
sys.path.insert(0, {code!r})
from pathlib import Path
from runlock import RunLock, LockUnavailable, ResultStateError, assert_writable
from common import write_json

result = Path(sys.argv[1]); hold = float(sys.argv[2])
try:
    lock = RunLock(result).acquire()
except LockUnavailable as exc:
    print('REFUSED-LOCK', flush=True); sys.exit(3)
try:
    try:
        assert_writable(result, {{'steps': 30000}})
    except ResultStateError as exc:
        print('REFUSED-STATE', flush=True); sys.exit(4)
    print('STARTED', flush=True)
    time.sleep(hold)
    write_json(result / 'history.jsonl.part', {{'step': 0}})
    with (result / 'history.jsonl').open('a') as fh:
        fh.write('{{"step": 0}}\\n')
    print('WROTE', flush=True)
finally:
    lock.release()
"""


def test_losing_process_writes_nothing(tmp_path):
    result = tmp_path / 'train_30k'
    first = spawn(TRAINER_LIKE.format(code=str(CODE)), result, 2.0)
    wait_for_line(first, 'STARTED')

    second = spawn(TRAINER_LIKE.format(code=str(CODE)), result, 0)
    out, err = second.communicate(timeout=30)
    assert second.returncode == 3, (out, err)
    assert 'REFUSED-LOCK' in out

    first.communicate(timeout=30)
    lines = (result / 'history.jsonl').read_text().splitlines()
    assert lines == ['{"step": 0}'], lines


def test_second_run_on_a_completed_directory_is_refused_by_state(tmp_path):
    result = tmp_path / 'train_30k'
    _write_complete(result)
    proc = spawn(TRAINER_LIKE.format(code=str(CODE)), result, 0)
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 4, (out, err)
    assert 'REFUSED-STATE' in out