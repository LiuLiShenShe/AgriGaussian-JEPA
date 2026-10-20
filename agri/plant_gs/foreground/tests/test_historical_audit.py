"""Retrospective audit of exports that predate hash-based provenance.

The point of the tool is to answer a question mtimes cannot: does the PLY still
agree, attribute for attribute, with the checkpoint it claims to come from?
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import provenance  # noqa: E402
from audit_historical_export import audit  # noqa: E402
from train import export_ply  # noqa: E402


def _splats(n=200, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.ParameterDict({
        'means': torch.nn.Parameter(torch.randn(n, 3, generator=g)),
        'scales': torch.nn.Parameter(torch.randn(n, 3, generator=g) * 0.1),
        'quats': torch.nn.Parameter(torch.randn(n, 4, generator=g)),
        'opacities': torch.nn.Parameter(torch.randn(n, generator=g)),
        'sh0': torch.nn.Parameter(torch.randn(n, 1, 3, generator=g)),
        'shN': torch.nn.Parameter(torch.randn(n, 15, 3, generator=g)),
    })


def _workspace(tmp_path, *, step=24000, report_step=None, keep_rows=150,
               overwrite_checkpoint=None, omit=()):
    ws = tmp_path / 'ws'
    (ws / 'final').mkdir(parents=True)
    splats = _splats()
    ckpt = ws / 'splats_best.pt'
    torch.save({'splats': {k: v.detach() for k, v in splats.items()},
                'step': step, 'sh_degree': 3,
                'normalization': {'center': [0, 0, 0], 'radius': 1.0, 'scale': 1.0},
                'config': {}}, ckpt)

    keep = np.zeros(200, dtype=bool)
    keep[:keep_rows] = True
    cleaned = {k: v[torch.from_numpy(keep)].contiguous() for k, v in splats.items()}
    export_ply(cleaned, ws / 'final' / 'point_cloud.ply')
    np.savez_compressed(ws / 'final' / 'cleanup_scores.npz', keep=keep)

    (ws / 'final' / 'report.json').write_text(json.dumps({
        'source_checkpoint': str(ckpt.resolve()),
        'step': report_step if report_step is not None else step,
        'raw_gaussians': 200, 'retained_gaussians': keep_rows}))

    if overwrite_checkpoint is not None:
        torch.save({'splats': {k: v.detach() for k, v in overwrite_checkpoint.items()},
                    'step': step, 'config': {}}, ckpt)

    if 'report' in omit:
        (ws / 'final' / 'report.json').unlink()
    if 'scores' in omit:
        (ws / 'final' / 'cleanup_scores.npz').unlink()
    if 'ply' in omit:
        (ws / 'final' / 'point_cloud.ply').unlink()
    if 'checkpoint' in omit:
        ckpt.unlink()
    return ws


def test_a_consistent_export_audits_as_pass(tmp_path):
    ws = _workspace(tmp_path)
    result = audit(ws)
    assert result['status'] == 'pass', result['problems']
    assert result['evidence']['checkpoint_step'] == 24000
    assert result['evidence']['report_step'] == 24000
    assert result['evidence']['ply_rows'] == 150
    assert result['evidence']['kept_rows'] == 150
    assert result['evidence']['source_checkpoint_sha256']
    assert result['evidence']['ply_sha256']


def test_a_replaced_checkpoint_audits_as_fail(tmp_path):
    """The Cotton/1 shape: the checkpoint was overwritten after the export."""
    ws = _workspace(tmp_path, overwrite_checkpoint=_splats(seed=99))
    result = audit(ws)
    assert result['status'] == 'fail'
    assert any('max |delta|' in problem for problem in result['problems'])


def test_a_mismatched_step_audits_as_fail(tmp_path):
    ws = _workspace(tmp_path, step=29999, report_step=24000)
    result = audit(ws)
    assert result['status'] == 'fail'
    assert any('checkpoint stores step 29999' in problem for problem in result['problems'])


def test_missing_artifacts_are_insufficient_not_inconsistent(tmp_path):
    for omit, needle in (('report', 'missing'), ('scores', 'keep-mask'),
                          ('ply', 'missing'), ('checkpoint', 'missing')):
        ws = _workspace(tmp_path / omit, omit=(omit,))
        result = audit(ws)
        assert result['status'] == 'insufficient', (omit, result)
        assert any(needle in problem for problem in result['problems'])


def test_the_audit_never_writes_to_the_workspace(tmp_path):
    ws = _workspace(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in ws.rglob('*') if p.is_file()}
    audit(ws)
    after = {p: p.stat().st_mtime_ns for p in ws.rglob('*') if p.is_file()}
    assert before == after


def test_the_audit_records_hashes_it_computed_itself(tmp_path):
    """No recorded hash existed at the time; these are derived now."""
    ws = _workspace(tmp_path)
    result = audit(ws)
    ckpt = Path(result['evidence']['source_checkpoint'])
    ply = Path(result['evidence']['ply'])
    assert result['evidence']['source_checkpoint_sha256'] == provenance.sha256_file(ckpt)
    assert result['evidence']['ply_sha256'] == provenance.sha256_file(ply)