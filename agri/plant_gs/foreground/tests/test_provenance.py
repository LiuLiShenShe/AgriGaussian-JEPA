"""Provenance chain: checkpoint -> PLY, and the checks that make it evidence."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import provenance  # noqa: E402
from export import verify_attribute_consistency  # noqa: E402
from train import checkpoint, export_ply  # noqa: E402


class FakeData:
    """Minimal stand-in for ``Captures`` so no dataset is needed."""

    def __init__(self, root):
        self.root = Path(root)

    @property
    def normalization(self):
        return {'center': [0.0, 0.0, 0.0], 'radius': 1.0, 'scale': 1.0}


def make_splats(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.ParameterDict({
        'means': torch.nn.Parameter(torch.randn(n, 3, generator=g)),
        'scales': torch.nn.Parameter(torch.randn(n, 3, generator=g) * 0.1),
        'quats': torch.nn.Parameter(torch.randn(n, 4, generator=g)),
        'opacities': torch.nn.Parameter(torch.randn(n, generator=g)),
        'sh0': torch.nn.Parameter(torch.randn(n, 1, 3, generator=g)),
        'shN': torch.nn.Parameter(torch.randn(n, 15, 3, generator=g)),
    })


# --------------------------------------------------------------------------- #
# Hashing and records
# --------------------------------------------------------------------------- #
def test_hashes_are_stable_and_order_independent_for_configs():
    a = provenance.sha256_json({'steps': 30000, 'seed': 42})
    b = provenance.sha256_json({'seed': 42, 'steps': 30000})
    assert a == b
    assert a != provenance.sha256_json({'steps': 7000, 'seed': 42})


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib
    target = tmp_path / 'blob.bin'
    target.write_bytes(b'x' * 10_000)
    assert provenance.sha256_file(target) == \
        hashlib.sha256(b'x' * 10_000).hexdigest()


def test_file_identity_records_bytes_mtime_and_hash(tmp_path):
    target = tmp_path / 'ckpt.pt'
    target.write_bytes(b'payload')
    identity = provenance.file_identity(target)
    assert identity['bytes'] == 7
    assert len(identity['sha256']) == 64
    assert identity['mtime_utc'].endswith('Z')


def test_write_record_is_atomic_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / 'nested' / 'record.json'
    provenance.write_record(target, {'a': 1})
    provenance.write_record(target, {'a': 2})
    assert json.loads(target.read_text()) == {'a': 2}
    assert not list(target.parent.glob('*.tmp'))


def test_git_commit_is_read_from_a_real_repository(tmp_path):
    import subprocess
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.email', 't@example.com'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.name', 't'], cwd=repo, check=True)
    (repo / 'f.txt').write_text('hello')
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-qm', 'init'], cwd=repo, check=True)
    commit = provenance.git_commit(repo)
    assert commit and len(commit) == 40
    assert provenance.git_dirty(repo) is False
    (repo / 'f.txt').write_text('changed')
    assert provenance.git_dirty(repo) is True


def test_git_commit_is_none_outside_a_repository(tmp_path):
    assert provenance.git_commit(tmp_path) is None


# --------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------- #
TOTAL_ROWS = 200
KEEP_ROWS = 150


def _export_fixture(tmp_path, keep_rows=KEEP_ROWS):
    """Build a checkpoint, a filtered PLY and a matching export record."""
    config = {'steps': 30000, 'seed': 42}
    data = FakeData(tmp_path / 'dataset')
    splats = make_splats(n=TOTAL_ROWS)
    payload = checkpoint(splats, data, 3, 24000, config)
    ckpt = tmp_path / 'splats_best.pt'
    torch.save(payload, ckpt)

    keep = np.zeros(TOTAL_ROWS, dtype=bool)
    keep[:keep_rows] = True
    cleaned = {k: v[torch.from_numpy(keep)].contiguous() for k, v in splats.items()}
    out = tmp_path / 'final'
    out.mkdir()
    export_ply(cleaned, out / 'point_cloud.ply')
    np.savez_compressed(out / 'cleanup_scores.npz', keep=keep)

    import hashlib
    retained = np.flatnonzero(keep).astype(np.int64)
    consistent = verify_attribute_consistency(out / 'point_cloud.ply', cleaned)
    record = provenance.export_record(
        output=out, checkpoint=ckpt, step=24000, ply=out / 'point_cloud.ply',
        gaussians_before=TOTAL_ROWS, gaussians_after=int(keep.sum()),
        cleanup={'retained_index_sha256': hashlib.sha256(retained.tobytes()).hexdigest(),
                 'retained_index_count': len(retained)},
        format_validation={'count': int(keep.sum()), 'finite': True,
                           'properties': len(PlyData.read(str(out / 'point_cloud.ply'))
                                             ['vertex'].data.dtype.names)},
        attribute_validation=consistent,
        extra={'checkpoint_sha256_before': provenance.sha256_file(ckpt),
               'checkpoint_sha256_after': provenance.sha256_file(ckpt),
               'checkpoint_unchanged_during_export': True})
    provenance.write_record(out / 'export.provenance.json', record)
    return out, ckpt, keep, cleaned


def test_clean_chain_passes(tmp_path):
    out, _, _, _ = _export_fixture(tmp_path)
    from verify_provenance import verify
    result = verify(out)
    assert result['status'] == 'pass', result['problems']
    assert result['details']['gaussians_before_filter'] == TOTAL_ROWS
    assert result['details']['gaussians_after_filter'] == KEEP_ROWS
    assert result['details']['checkpoint_step'] == 24000


def test_exporter_detects_a_checkpoint_that_changed_after_export(tmp_path):
    out, ckpt, _, _ = _export_fixture(tmp_path)
    torch.save({'splats': {}, 'step': 29999}, ckpt)  # a second run overwrote it
    from verify_provenance import verify
    result = verify(out)
    assert result['status'] == 'fail'
    assert any('changed since provenance' in problem for problem in result['problems'])


def test_exporter_detects_a_ply_that_changed_after_provenance(tmp_path):
    out, _, _, _ = _export_fixture(tmp_path)
    with (out / 'point_cloud.ply').open('ab') as handle:
        handle.write(b'# tampered\n')
    from verify_provenance import verify
    result = verify(out)
    assert result['status'] == 'fail'
    assert any('point_cloud.ply' in problem for problem in result['problems'])


def test_attribute_consistency_catches_a_ply_from_a_different_checkpoint(tmp_path):
    out, _, _, cleaned = _export_fixture(tmp_path)
    other = {k: (v + 1.0) for k, v in cleaned.items()}
    report = verify_attribute_consistency(out / 'point_cloud.ply', other)
    assert report['passed'] is False
    assert any('x' in failure for failure in report['failures'])


def test_chain_flags_a_missing_record_field(tmp_path):
    out, _, _, _ = _export_fixture(tmp_path)
    record = provenance.read_record(out / 'export.provenance.json')
    record.pop('gaussians_after_filter')
    problems = provenance.verify_export_chain(record)
    assert any('gaussian counts' in problem for problem in problems)


def test_chain_flags_retaining_more_than_it_started_with(tmp_path):
    out, _, _, _ = _export_fixture(tmp_path)
    record = provenance.read_record(out / 'export.provenance.json')
    record['gaussians_after_filter'] = 999
    problems = provenance.verify_export_chain(record)
    assert any('exceeds' in problem or 'vertex count' in problem for problem in problems)


def test_chain_rejects_an_attribute_consistency_failure(tmp_path):
    out, _, _, _ = _export_fixture(tmp_path)
    record = provenance.read_record(out / 'export.provenance.json')
    record['attribute_validation'] = {'passed': False, 'failures': ['x: max |delta| = 1e+00']}
    problems = provenance.verify_export_chain(record)
    assert any('attribute_validation failed' in problem for problem in problems)


def test_chain_reports_missing_checkpoint_rather_than_passing(tmp_path):
    out, ckpt, _, _ = _export_fixture(tmp_path)
    ckpt.unlink()
    problems = provenance.verify_export_chain(
        provenance.read_record(out / 'export.provenance.json'))
    assert any('missing' in problem for problem in problems)


def test_export_record_is_tied_to_the_step_inside_the_checkpoint(tmp_path):
    config = {'steps': 30000}
    data = FakeData(tmp_path / 'dataset')
    splats = make_splats(n=8)
    ckpt = tmp_path / 'splats_best.pt'
    torch.save(checkpoint(splats, data, 3, 12000, config), ckpt)
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert payload['step'] == 12000
    assert payload['config_sha256'] == provenance.sha256_json(config)
    # A model-only checkpoint is explicitly *not* a resume point.
    assert payload['resumable'] is False


def test_retained_index_digest_is_order_sensitive(tmp_path):
    import hashlib
    a = np.array([0, 1, 2], dtype=np.int64)
    b = np.array([0, 2, 1], dtype=np.int64)
    assert hashlib.sha256(a.tobytes()).hexdigest() != hashlib.sha256(b.tobytes()).hexdigest()
    assert a.sum() == b.sum()  # a checksum of values would have missed this


def test_cleanup_scores_keep_mask_round_trips(tmp_path):
    keep = np.zeros(10, dtype=bool)
    keep[[1, 4, 9]] = True
    path = tmp_path / 'cleanup_scores.npz'
    np.savez_compressed(path, keep=keep, mask_ratio=np.arange(10.0))
    loaded = np.load(path)
    assert np.array_equal(loaded['keep'], keep)
    assert np.flatnonzero(loaded['keep']).tolist() == [1, 4, 9]