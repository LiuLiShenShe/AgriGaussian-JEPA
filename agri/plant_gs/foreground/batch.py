#!/usr/bin/env python
"""Read-only-source, resumable orchestration of the foreground reconstruction tools."""
from pathlib import Path


def discover_jobs(rgb_root, sam_root, output_root):
    rgb_root, sam_root, output_root = map(lambda p: Path(p).resolve(),
                                        (rgb_root, sam_root, output_root))
    jobs = []
    for rgb in sorted(rgb_root.iterdir()):
        if not rgb.is_dir():
            continue
        rgb_count = len(list(rgb.glob('*.jpg')))
        if not rgb_count:
            continue
        sam = sam_root / rgb.name
        mask_count = len(list(sam.glob('mask_*.png')))
        jobs.append({'name': rgb.name, 'rgb': str(rgb), 'sam': str(sam),
                     'workspace': str(output_root / rgb.name),
                     'rgb_count': rgb_count, 'mask_count': mask_count,
                     'status': 'pending' if mask_count else 'missing_sam'})
    return jobs


def pairing_signature(report):
    """Bind a manual exclusion review to immutable source paths, verified pairs and failures."""
    import hashlib
    import json
    snapshot = {k: report.get(k) for k in ['source_rgb', 'source_sam', 'pairs',
                                          'paired_mask_count', 'unmatched_masks', 'corrupt_rgb']}
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def audit_pairing(report, mask_count, max_fraction=.10, review=None):
    """Fail closed: never relax exact pixels or accept ambiguous associations."""
    pairs = report['pairs']
    excluded = report['unmatched_masks']
    if report['paired_mask_count'] != len(pairs) or len(pairs) + len(excluded) != mask_count:
        raise ValueError('Pair accounting does not match source mask count')
    if len(pairs) < 30:
        raise ValueError('Fewer than 30 verified masked views')
    if any(not (.999 <= p.get('full_exact_fraction', 0) <= 1.) for p in pairs):
        raise ValueError('Full-resolution pixel verification failed')
    if len({p['rgb'] for p in pairs}) != len(pairs) or len({p['sam_id'] for p in pairs}) != len(pairs):
        raise ValueError('Pairing must be one-to-one')
    manual = review is not None
    budget = int(mask_count * max_fraction)
    if manual:
        if not (review.get('approved') is True and isinstance(review.get('reason'), str)
                and review['reason'].strip()
                and review.get('pairing_signature') == pairing_signature(report)
                and review.get('reviewed_exclusions') == excluded
                and type(review.get('max_excluded_count')) is int
                and review['max_excluded_count'] == len(excluded)):
            raise ValueError('Manual exclusion review is invalid or does not match current pairing snapshot')
        budget = review['max_excluded_count']
    if len(excluded) > budget:
        raise ValueError('Excluded SAM inputs exceed explicit budget')
    allowed = ('cannot identify image file', 'image file is truncated',
               'broken data stream', 'expected unique exact RGB match, got 0;')
    for row in excluded:
        if not manual and not any(s in row['error'] for s in allowed):
            raise ValueError('Unreviewed/ambiguous exclusion: ' + row['error'])
    return {'paired': len(pairs), 'excluded': len(excluded), 'mask_count': mask_count,
            'max_excluded_fraction': max_fraction, 'max_excluded_count': budget,
            'excluded_fraction': len(excluded) / mask_count, 'pixel_threshold': .999,
            'one_to_one': True, 'exclusions': excluded, 'manual_review': manual,
            'review': review}


def source_digest(root, patterns):
    """Stable filename+SHA256 manifest, including corrupt files, without modifying input."""
    import hashlib
    root = Path(root)
    files = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
    digest = hashlib.sha256()
    for path in files:
        file_digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                file_digest.update(chunk)
        digest.update(path.name.encode('utf8') + b'\0' + file_digest.digest())
    return {'count': len(files), 'sha256': digest.hexdigest()}


def asset_info(workspace):
    """Require both reconstruction and 14D Gaussian compatibility, not just a PLY filename."""
    import json
    final = Path(workspace) / 'final'
    try:
        ply = final / 'point_cloud.ply'
        report = json.loads((final / 'report.json').read_text())
        compatible = json.loads((final / 'gsjepa_compatibility.json').read_text())
        if not (ply.stat().st_size and report['status'] == 'complete'
                and report['format_validation']['finite']
                and report['retained_gaussians'] >= 100
                and compatible.get('gsjepa_compatible') is True):
            return None
        metrics = report['clean_validation']
        warnings = []
        if metrics['silhouette_iou'] < .90:
            warnings.append('silhouette_iou')
        if metrics['background_alpha'] > .03:
            warnings.append('background_alpha')
        return {'ply': str(ply.resolve()), 'preview': str((final / 'preview.jpg').resolve()),
                'gaussians': report['retained_gaussians'],
                **{k: metrics[k] for k in ['foreground_psnr', 'silhouette_iou', 'background_alpha']},
                'quality_warnings': warnings, 'gsjepa_compatible': True}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _json(path):
    import json
    return json.loads(Path(path).read_text())


def _write(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def training_directory(workspace, steps, preferred=None):
    """Reuse completed training (including retries), never overwrite partial optimizer state."""
    from datetime import datetime, timezone
    root = Path(workspace).resolve() / 'results'
    result = root / f'train_{steps // 1000}k'
    if preferred:
        selected = Path(preferred).resolve()
        if selected.parent != root:
            raise ValueError('Training directory is outside this workspace results')
        result = selected
    status = result / 'status.json'
    complete = status.exists() and _json(status).get('status') == 'complete'
    if complete:
        if _json(status).get('steps') != steps:
            raise ValueError('Completed training has a different step count')
        if not (result / 'splats_best.pt').exists():
            raise ValueError('Completed training is missing its best checkpoint')
        return result, True
    if result.exists():
        result = root / (f'train_{steps // 1000}k_retry_' +
                         datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    return result, False


class Runner:
    def __init__(self, args, jobs):
        import threading
        self.args, self.jobs = args, jobs
        self.lock = threading.Lock()
        self.code = Path(__file__).resolve().parent

    def update(self, job, **changes):
        import csv
        from collections import Counter
        with self.lock:
            job.update(changes, updated_at=_now())
            _write(Path(job['workspace']) / 'job.json', job)
            summary = {'updated_at': _now(), 'controller_pid': __import__('os').getpid(),
                       'target': 'all verified SAM foreground including plant/pot/marker',
                       'counts': dict(Counter(j['status'] for j in self.jobs)), 'jobs': self.jobs}
            _write(self.args.output_root / 'manifest.json', summary)
            path = self.args.output_root / 'manifest.csv'
            temp = path.with_suffix('.csv.tmp')
            fields = ['name', 'status', 'stage', 'gpu', 'paired_masks', 'corrupt_rgb',
                      'gaussians', 'foreground_psnr', 'silhouette_iou', 'background_alpha',
                      'ply', 'preview', 'quality_warnings', 'error', 'workspace']
            with temp.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.jobs)
            temp.replace(path)
        print(_now(), job['name'], job['status'], job.get('stage', ''), flush=True)

    def command(self, job, gpu, stage, command):
        import os
        import shlex
        import subprocess
        import sys
        ws = Path(job['workspace'])
        log = ws / 'logs' / (stage + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        self.update(job, status='running', stage=stage, gpu=gpu, log=str(log), error=None)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OMP_NUM_THREADS='8', QT_QPA_PLATFORM='offscreen')
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        with log.open('a') as stream:
            stream.write('\n' + _now() + ' ' + shlex.join(map(str, command)) + '\n')
            stream.flush()
            subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                           env=env, check=True, cwd=self.code)

    def fingerprints(self, job):
        return {'rgb': source_digest(job['rgb'], ('*.jpg',)),
                'sam': source_digest(job['sam'], ('*.png', '*.json'))}

    def reuse(self, job, source):
        source = Path(source).resolve()
        info = asset_info(source)
        if info is None:
            raise ValueError('Reuse source is not a validated asset: ' + str(source))
        previous = _json(source / 'preparation.json')
        current = self.fingerprints(job)
        old = {'rgb': source_digest(previous['source_rgb'], ('*.jpg',)),
               'sam': source_digest(previous['source_sam'], ('*.png', '*.json'))}
        if current != old:
            raise ValueError('Cannot reuse an asset from different source contents')
        ws = Path(job['workspace'])
        ws.mkdir(parents=True, exist_ok=True)
        final = ws / 'final'
        if not final.exists():
            final.symlink_to(source / 'final', target_is_directory=True)
        elif final.resolve() != (source / 'final').resolve():
            raise ValueError('Refusing to overwrite a different final directory')
        _write(ws / 'source_fingerprints.json', current)
        _write(ws / 'reuse_provenance.json', {'source_workspace': str(source),
               'source_rgb': previous['source_rgb'], 'source_sam': previous['source_sam'],
               'all_source_files_identical': True, 'fingerprints': current, 'verified_at': _now()})
        self.update(job, status='reused', stage='complete', reused_from=str(source),
                    paired_masks=previous['paired_mask_count'],
                    corrupt_rgb=len(previous['corrupt_rgb']), **info)

    def run_job(self, job, gpu):
        import fcntl
        import sys
        ws = Path(job['workspace'])
        ws.mkdir(parents=True, exist_ok=True)
        with (ws / '.reconstruction.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                old_job = ws / 'job.json'
                if old_job.exists():
                    previous = _json(old_job)
                    if any(previous[k] != job[k] for k in ['rgb', 'sam', 'workspace']):
                        raise ValueError('Workspace source directories changed')
                fingerprints = self.fingerprints(job)
                fingerprint_path = ws / 'source_fingerprints.json'
                if fingerprint_path.exists() and _json(fingerprint_path) != fingerprints:
                    raise ValueError('Source files changed since this workspace was started')
                _write(fingerprint_path, fingerprints)
                info = asset_info(ws)
                if info is not None:
                    self.update(job, status='complete', stage='complete', **info)
                    return
                preparation = ws / 'preparation.json'
                review_path = ws / 'pairing_exclusion_review.json'
                review = _json(review_path) if review_path.exists() else None
                max_unmatched = max(0, int(job['mask_count'] * self.args.max_excluded_fraction))
                if review is not None:
                    # Explicit reviews are allowed only for already inspected, unchanged reports.
                    reviewed = audit_pairing(_json(preparation), job['mask_count'],
                                             self.args.max_excluded_fraction, review=review)
                    max_unmatched = reviewed['excluded']
                if not preparation.exists() or not _json(preparation).get('pairing_accepted'):
                    cmd = [sys.executable, self.code / 'prepare.py', '--rgb', job['rgb'],
                           '--sam', job['sam'], '--workspace', ws, '--max-unmatched',
                           max_unmatched]
                    if preparation.exists():
                        cmd += ['--resume']
                    self.command(job, gpu, 'prepare', cmd)
                report = _json(preparation)
                if report['source_rgb'] != job['rgb'] or report['source_sam'] != job['sam']:
                    raise ValueError('Prepared RGB/SAM directories differ from batch inputs')
                audit = audit_pairing(report, job['mask_count'], self.args.max_excluded_fraction,
                                      review=review)
                _write(ws / 'pairing_audit.json', audit)
                self.update(job, status='running', stage='paired', gpu=gpu,
                            paired_masks=report['paired_mask_count'],
                            excluded_masks=len(report['unmatched_masks']),
                            corrupt_rgb=len(report['corrupt_rgb']))
                if self.args.until == 'prepare':
                    self.update(job, status='prepared', stage='paired')
                    return
                self.command(job, gpu, 'sfm', ['bash', self.code / 'run_sfm.sh', ws])
                if not (ws / 'dataset/meta.json').exists():
                    self.command(job, gpu, 'dataset', [sys.executable, self.code / 'build_dataset.py',
                                                     '--workspace', ws])
                meta = _json(ws / 'dataset/meta.json')
                self.update(job, status='running', stage='dataset',
                            registered_rgb=meta['registered_rgb'],
                            masked_registered_views=meta['masked_registered_views'],
                            initial_foreground_points=meta['foreground_points'])
                if self.args.until == 'dataset':
                    self.update(job, status='dataset_ready', stage='dataset')
                    return
                # No optimizer resume is supported; never append a second training run to an old one.
                result, complete = training_directory(ws, self.args.steps, job.get('result'))
                if not complete:
                    self.update(job, result=str(result))
                    self.command(job, gpu, 'train', [sys.executable, self.code / 'train.py',
                                 '--data', ws / 'dataset', '--result', result, '--steps', self.args.steps])
                self.command(job, gpu, 'export', [sys.executable, self.code / 'export.py',
                             '--checkpoint', result / 'splats_best.pt', '--output', ws / 'final'])
                self.command(job, gpu, 'compatibility', [sys.executable,
                             self.code.parent / 'verify_gsjepa_compatible.py', ws / 'final/point_cloud.ply',
                             '--json', ws / 'final/gsjepa_compatibility.json'])
                if self.fingerprints(job) != fingerprints:
                    raise ValueError('Input file content changed while reconstruction was running')
                info = asset_info(ws)
                if info is None:
                    raise ValueError('Export did not produce a validated Gaussian asset')
                self.update(job, status='complete', stage='complete', result=str(result), **info)
            except Exception as exc:
                self.update(job, status='failed', error=f'{type(exc).__name__}: {exc}')


def main():
    import argparse
    import fcntl
    import queue
    from concurrent.futures import ThreadPoolExecutor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-root', type=Path, required=True)
    parser.add_argument('--sam-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[1])
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--max-excluded-fraction', type=float, default=.10)
    parser.add_argument('--until', choices=['prepare', 'dataset', 'complete'], default='complete')
    parser.add_argument('--exclude', nargs='*', default=[])
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--reuse', action='append', default=[], metavar='NAME=WORKSPACE')
    args = parser.parse_args()
    args.rgb_root, args.sam_root, args.output_root = [p.resolve() for p in
                                                    (args.rgb_root, args.sam_root, args.output_root)]
    if not 0 <= args.max_excluded_fraction <= .10:
        parser.error('The unattended exclusion budget must be at most 10%; inspect larger losses manually')
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus or min(args.gpus) < 0:
        parser.error('GPU IDs must be unique non-negative integers')
    if args.steps < 1000:
        parser.error('Use the individual train.py for short diagnostic runs')
    for source in [args.rgb_root, args.sam_root]:
        if source == args.output_root or source in args.output_root.parents:
            parser.error('Output root must not be inside either source directory')
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.batch.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A batch controller is already running for this output root')
        jobs = discover_jobs(args.rgb_root, args.sam_root, args.output_root)
        if args.only:
            missing = set(args.only) - {j['name'] for j in jobs}
            if missing:
                parser.error('Unknown RGB groups: ' + ', '.join(sorted(missing)))
        runner = Runner(args, jobs)
        _write(args.output_root / 'batch_config.json', {k: str(v) if isinstance(v, Path) else v
                                                      for k, v in vars(args).items()})
        reuse = dict(item.split('=', 1) for item in args.reuse)
        todo = queue.Queue()
        for job in jobs:
            previous = Path(job['workspace']) / 'job.json'
            if previous.exists():
                saved = _json(previous)
                if any(saved.get(k) != job[k] for k in ['rgb', 'sam', 'workspace']):
                    raise SystemExit('An existing job has different source directories')
                job.update({k: v for k, v in saved.items() if k not in ['rgb_count', 'mask_count']})
            if job['status'] == 'missing_sam' and job['mask_count']:
                job['status'] = 'pending'
            if not job['mask_count']:
                runner.update(job, status='missing_sam', stage='inventory', error='No corresponding SAM masks')
            elif job['name'] in args.exclude or (args.only and job['name'] not in args.only):
                runner.update(job)
            elif job['name'] in reuse:
                runner.reuse(job, reuse[job['name']])
            else:
                runner.update(job, status='queued', error=None)
                todo.put(job)

        def worker(gpu):
            while True:
                try:
                    job = todo.get_nowait()
                except queue.Empty:
                    return
                runner.run_job(job, gpu)
                todo.task_done()
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            list(executor.map(worker, args.gpus))
        print('BATCH_FINISHED', _now(), flush=True)
        if any(j['status'] == 'failed' for j in jobs):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
