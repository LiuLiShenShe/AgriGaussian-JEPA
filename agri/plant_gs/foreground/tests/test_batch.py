"""Batch orchestration contracts; GPU reconstruction is verified separately."""
import importlib.util
from pathlib import Path
import sys

import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location('foreground_batch', CODE / 'batch.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_discovers_rgb_plants_without_inventing_masks_or_including_tool_dirs(tmp_path):
    rgb, sam, out = [tmp_path / name for name in ['rgb', 'sam', 'out']]
    for name in ['PlantA', 'PlantB', 'ffmpeg_bin']:
        (rgb / name).mkdir(parents=True)
    (rgb / 'PlantA/0000.jpg').write_bytes(b'image')
    (rgb / 'PlantB/0000.jpg').write_bytes(b'image')
    (sam / 'PlantA').mkdir(parents=True)
    (sam / 'PlantA/mask_0000.png').write_bytes(b'mask')
    (sam / 'orphan').mkdir()
    jobs = batch.discover_jobs(rgb, sam, out)
    assert [j['name'] for j in jobs] == ['PlantA', 'PlantB']
    assert [j['status'] for j in jobs] == ['pending', 'missing_sam']
    assert jobs[0]['rgb_count'] == jobs[0]['mask_count'] == 1
    assert jobs[1]['mask_count'] == 0
    assert jobs[0]['workspace'] == str((out / 'PlantA').resolve())


def test_pairing_gate_audits_small_corrupt_exclusions_without_relaxing_pixel_checks():
    report = {'pairs': [{'sam_id': str(i), 'rgb': f'{i}.jpg',
                         'full_exact_fraction': 1.0} for i in range(38)],
              'paired_mask_count': 38, 'unmatched_masks': [
                  {'sam_id': '38', 'error': 'cannot identify image file mask.png'},
                  {'sam_id': '39', 'error': 'expected unique exact RGB match, got 0; best MAE=30'}]}
    audit = batch.audit_pairing(report, 40, .10)
    assert audit['excluded'] == 2 and audit['paired'] == 38
    assert audit['pixel_threshold'] == .999
    report['pairs'][0]['full_exact_fraction'] = .998
    with pytest.raises(ValueError, match='pixel'):
        batch.audit_pairing(report, 40, .10)
    report['pairs'][0]['full_exact_fraction'] = 1.0
    report['unmatched_masks'][0]['error'] = 'expected unique exact RGB match, got 2; best MAE=0'
    with pytest.raises(ValueError, match='Unreviewed'):
        batch.audit_pairing(report, 40, .10)
    report['unmatched_masks'][0]['error'] = 'cannot identify image file mask.png'
    with pytest.raises(ValueError, match='budget'):
        batch.audit_pairing(report, 40, .01)
    report['pairs'][1]['rgb'] = report['pairs'][0]['rgb']
    with pytest.raises(ValueError, match='one-to-one'):
        batch.audit_pairing(report, 40, .10)


def test_source_digest_checks_all_contents_names_and_preserves_sources(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    for root in [left, right]:
        root.mkdir()
        (root / '0000.jpg').write_bytes(b'first image')
        (root / '0001.jpg').write_bytes(b'second image')
    first = batch.source_digest(left, ('*.jpg',))
    assert first == batch.source_digest(right, ('*.jpg',))
    (right / '0001.jpg').write_bytes(b'changed image')
    assert first != batch.source_digest(right, ('*.jpg',))
    assert (left / '0001.jpg').read_bytes() == b'second image'
    (right / '0001.jpg').write_bytes(b'second image')
    (right / '0001.jpg').rename(right / 'renamed.jpg')
    assert first != batch.source_digest(right, ('*.jpg',))


def test_asset_is_not_complete_until_export_and_compatibility_are_both_valid(tmp_path):
    import json
    final = tmp_path / 'final'
    final.mkdir()
    assert batch.asset_info(tmp_path) is None
    (final / 'point_cloud.ply').write_bytes(b'ply\nexample')
    report = {'status': 'complete', 'retained_gaussians': 123,
              'format_validation': {'count': 123, 'finite': True},
              'clean_validation': {'foreground_psnr': 25., 'silhouette_iou': .97, 'background_alpha': .01}}
    (final / 'report.json').write_text(json.dumps(report))
    assert batch.asset_info(tmp_path) is None
    (final / 'gsjepa_compatibility.json').write_text(json.dumps({'gsjepa_compatible': True}))
    asset = batch.asset_info(tmp_path)
    assert asset['gaussians'] == 123 and asset['silhouette_iou'] == .97
    assert asset['ply'] == str((final / 'point_cloud.ply').resolve())
    assert asset['quality_warnings'] == []
    report['clean_validation']['background_alpha'] = .12
    (final / 'report.json').write_text(json.dumps(report))
    assert 'background_alpha' in batch.asset_info(tmp_path)['quality_warnings']
    (final / 'gsjepa_compatibility.json').write_text(json.dumps({'gsjepa_compatible': False}))
    assert batch.asset_info(tmp_path) is None


def test_prepared_job_resumes_without_gpu_and_rejects_changed_raw_inputs(tmp_path):
    import json
    from types import SimpleNamespace
    rgb, sam, out = [tmp_path / n for n in ['rgb', 'sam', 'out']]
    for root in [rgb, sam]:
        (root / 'PlantA').mkdir(parents=True)
    for i in range(30):
        (rgb / 'PlantA' / f'{i}.jpg').write_bytes(b'raw rgb')
        (sam / 'PlantA' / f'mask_{i}.png').write_bytes(b'raw mask')
    jobs = batch.discover_jobs(rgb, sam, out)
    job = jobs[0]
    ws = Path(job['workspace'])
    ws.mkdir(parents=True)
    report = {'source_rgb': job['rgb'], 'source_sam': job['sam'],
              'pairing_accepted': True, 'paired_mask_count': 30,
              'pairs': [{'sam_id': str(i), 'rgb': f'{i}.jpg', 'full_exact_fraction': 1.0}
                        for i in range(30)], 'unmatched_masks': [], 'corrupt_rgb': []}
    (ws / 'preparation.json').write_text(json.dumps(report))
    runner = batch.Runner(SimpleNamespace(output_root=out, until='prepare',
                                         max_excluded_fraction=.1), jobs)
    runner.run_job(job, 0)
    assert job['status'] == 'prepared'
    assert json.loads((out / 'manifest.json').read_text())['counts'] == {'prepared': 1}
    assert json.loads((ws / 'pairing_audit.json').read_text())['paired'] == 30
    assert (rgb / 'PlantA/0.jpg').read_bytes() == b'raw rgb'
    (rgb / 'PlantA/0.jpg').write_bytes(b'external edit')
    runner.run_job(job, 0)
    assert job['status'] == 'failed'
    assert 'Source files changed' in job['error']


def test_pairing_gate_rejects_nan_verification_values():
    report = {'pairs': [{'sam_id': str(i), 'rgb': f'{i}.jpg', 'full_exact_fraction': 1.0}
                        for i in range(30)], 'paired_mask_count': 30, 'unmatched_masks': []}
    report['pairs'][0]['full_exact_fraction'] = float('nan')
    with pytest.raises(ValueError, match='pixel'):
        batch.audit_pairing(report, 30)


def test_training_directory_reuses_completed_retry_but_never_overwrites_partial_run(tmp_path):
    import json
    base = tmp_path / 'results/train_30k'
    assert batch.training_directory(tmp_path, 30000) == (base, False)
    base.mkdir(parents=True)
    (base / 'config.json').write_text('{}')
    retry, complete = batch.training_directory(tmp_path, 30000)
    assert not complete and retry != base and not retry.exists()
    retry.mkdir()
    (retry / 'status.json').write_text(json.dumps({'status': 'complete', 'steps': 30000}))
    (retry / 'splats_best.pt').write_bytes(b'checkpoint')
    assert batch.training_directory(tmp_path, 30000, str(retry)) == (retry, True)
    with pytest.raises(ValueError, match='outside'):
        batch.training_directory(tmp_path, 30000, str(tmp_path / 'outside'))


def test_explicit_exclusion_review_is_snapshot_bound_and_never_relaxes_pairs():
    import copy
    report = {'source_rgb': '/rgb/plant', 'source_sam': '/sam/plant',
              'pairs': [{'sam_id': str(i), 'rgb': f'{i}.jpg', 'full_exact_fraction': 1.0}
                        for i in range(30)], 'paired_mask_count': 30,
              'unmatched_masks': [{'sam_id': str(i), 'error': 'expected unique exact RGB match, got 2;'}
                                  for i in range(30, 35)], 'corrupt_rgb': []}
    with pytest.raises(ValueError):
        batch.audit_pairing(report, 35)
    review = {'approved': True, 'reason': 'Reviewed duplicate source RGB files; exclude ambiguous masks, do not pair them.',
              'pairing_signature': batch.pairing_signature(report),
              'max_excluded_count': 5, 'reviewed_exclusions': copy.deepcopy(report['unmatched_masks'])}
    result = batch.audit_pairing(report, 35, review=review)
    assert result['manual_review'] and result['excluded'] == 5
    # Timing / acceptance bookkeeping can change during prepare --resume, but inputs and pairs cannot.
    report['pairing_accepted'] = True
    report['elapsed_seconds'] = 1
    assert batch.audit_pairing(report, 35, review=review)['manual_review']
    altered = copy.deepcopy(report)
    altered['unmatched_masks'][0]['error'] = 'new unseen exclusion'
    with pytest.raises(ValueError, match='review'):
        batch.audit_pairing(altered, 35, review=review)
    altered = copy.deepcopy(report)
    altered['pairs'][0]['full_exact_fraction'] = .8
    review['pairing_signature'] = batch.pairing_signature(altered)
    with pytest.raises(ValueError, match='pixel'):
        batch.audit_pairing(altered, 35, review=review)


def test_runner_resumes_only_reviewed_exclusion_count(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    rgb, sam, out = [tmp_path / n for n in ['rgb', 'sam', 'out']]
    for root in [rgb, sam]:
        (root / 'PlantA').mkdir(parents=True)
    for i in range(35):
        (rgb / 'PlantA' / f'{i}.jpg').write_bytes(b'raw rgb')
        (sam / 'PlantA' / f'mask_{i}.png').write_bytes(b'raw mask')
    jobs = batch.discover_jobs(rgb, sam, out)
    job = jobs[0]
    ws = Path(job['workspace'])
    ws.mkdir(parents=True)
    report = {'source_rgb': job['rgb'], 'source_sam': job['sam'],
              'pairing_accepted': False, 'paired_mask_count': 30,
              'pairs': [{'sam_id': str(i), 'rgb': f'{i}.jpg', 'full_exact_fraction': 1.0}
                        for i in range(30)], 'corrupt_rgb': [],
              'unmatched_masks': [{'sam_id': str(i), 'error': 'missing reviewed crop'}
                                  for i in range(30, 35)]}
    batch._write(ws / 'preparation.json', report)
    batch._write(ws / 'pairing_exclusion_review.json',
                 {'approved': True, 'reason': 'Checked missing crops, use remaining verified frames.',
                  'max_excluded_count': 5, 'reviewed_exclusions': report['unmatched_masks'],
                  'pairing_signature': batch.pairing_signature(report)})
    runner = batch.Runner(SimpleNamespace(output_root=out, until='prepare',
                                         max_excluded_fraction=.1), jobs)
    calls = []

    def prepare_only(job, gpu, stage, command):
        assert stage == 'prepare' and '--resume' in command
        assert command[command.index('--max-unmatched') + 1] == 5
        calls.append(command)
        report.update(pairing_accepted=True, elapsed_seconds=1.)
        batch._write(ws / 'preparation.json', report)

    monkeypatch.setattr(runner, 'command', prepare_only)
    runner.run_job(job, 0)
    assert job['status'] == 'prepared' and len(calls) == 1
    assert json.loads((ws / 'pairing_audit.json').read_text())['manual_review'] is True
