#!/usr/bin/env python
"""Recover source RGB for every SAM mask and create an isolated SfM workspace."""
import argparse
import csv
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import find_rgb_match, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rgb', type=Path, required=True)
    p.add_argument('--sam', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--sample-stride', type=int, default=16)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-unmatched', type=int, default=3, help='Explicit audited exclusion budget; never force a pairing.')
    args = p.parse_args()
    ws = args.workspace.resolve()
    previous = None
    if (ws / 'preparation.json').exists():
        if not args.resume:
            raise SystemExit('Workspace already prepared; pass --resume for audited retry.')
        import json
        previous = json.loads((ws / 'preparation.json').read_text())
        if previous['source_rgb'] != str(args.rgb.resolve()) or previous['source_sam'] != str(args.sam.resolve()):
            raise SystemExit('Cannot resume with different input directories.')
        if not (ws / 'preparation_attempt1.json').exists():
            (ws / 'preparation_attempt1.json').write_text((ws / 'preparation.json').read_text())
    for sub in ['images', 'masks', 'logs', 'colmap/sparse']:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    stride = args.sample_stride
    samples, names, broken, sources = [], [], [], []
    t0 = time.time()
    for source in sorted(args.rgb.glob('*.jpg')):
        try:
            with Image.open(source) as im:
                rgb = np.asarray(im.convert('RGB'))
                size = im.size
            samples.append(rgb[::stride, ::stride].copy())
            names.append(source.name)
            sources.append(source)
            target_size = tuple(round(x * args.scale) for x in size)
            if not (ws / 'images' / source.name).exists():
                Image.fromarray(rgb).resize(target_size, Image.Resampling.LANCZOS).save(
                    ws / 'images' / source.name, quality=95, subsampling=0)
        except Exception as exc:
            broken.append({'rgb': source.name, 'error': str(exc), 'bytes': source.stat().st_size})
            print('EXCLUDED', source.name, str(exc), flush=True)
    rgb_samples = np.stack(samples)
    pairs = list(previous['pairs']) if previous else []
    used = {row['rgb'] for row in pairs}
    paired_ids = {row['sam_id'] for row in pairs}
    unmatched = []
    for mask_path in sorted(args.sam.glob('mask_*.png')):
        sam_id = mask_path.stem.removeprefix('mask_')
        crop_path = args.sam / f'crop_{sam_id}.png'
        if sam_id in paired_ids:
            continue
        try:
            with Image.open(mask_path) as im:
                mask = np.asarray(im.convert('L')) > 127
            with Image.open(crop_path) as im:
                crop = np.asarray(im.convert('RGB'))
            if crop.shape[:2] != mask.shape or mask.shape != (size[1], size[0]):
                raise ValueError('RGB/crop/mask dimensions differ')
            match = find_rgb_match(crop[::stride, ::stride], mask[::stride, ::stride], rgb_samples, names, max_mae=0.2)
            if match['rgb'] in used:
                raise ValueError('Two masks map to the same source RGB')
            with Image.open(args.rgb / match['rgb']) as im:
                original = np.asarray(im.convert('RGB'))
            exact_full = float((original[mask] == crop[mask]).all(axis=1).mean())
            if exact_full < 0.999:
                raise ValueError(f'Full-resolution exact verification failed: {exact_full}')
            match.update({'sam_id': sam_id, 'mask': mask_path.name, 'crop': crop_path.name,
                          'full_exact_fraction': exact_full, 'foreground_fraction': float(mask.mean()),
                          'rgb_sha256': hashlib.sha256((args.rgb / match['rgb']).read_bytes()).hexdigest(),
                          'mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest()})
            used.add(match['rgb'])
            resized = cv2.resize(mask.astype(np.uint8) * 255, target_size, interpolation=cv2.INTER_NEAREST)
            Image.fromarray(resized).save(ws / 'masks' / (Path(match['rgb']).stem + '.png'))
            pairs.append(match)
            if len(pairs) % 25 == 0 or sam_id == '0207':
                print('PAIRED', len(pairs), mask_path.name, '->', match['rgb'],
                      'exact', exact_full, 'elapsed', round(time.time() - t0, 1), flush=True)
        except Exception as exc:
            unmatched.append({'sam_id': sam_id, 'error': str(exc)})
            print('PAIRING_FAILED', sam_id, str(exc), flush=True)
    with (ws / 'pairing.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]) if pairs else ['sam_id', 'rgb'])
        writer.writeheader()
        writer.writerows(pairs)
    report = {'source_rgb': str(args.rgb.resolve()), 'source_sam': str(args.sam.resolve()),
              'target': 'all SAM foreground (plant/pot/blue block)',
              'source_rgb_count': len(list(args.rgb.glob('*.jpg'))), 'usable_rgb_count': len(names),
              'paired_mask_count': len(pairs), 'unmatched_masks': unmatched,
              'corrupt_rgb': broken, 'rgb_without_mask': sorted(set(names) - used),
              'working_size': list(target_size), 'scale': args.scale,
              'sample_stride': stride, 'pairs': pairs,
              'elapsed_seconds': round(time.time() - t0 + (previous['elapsed_seconds'] if previous else 0), 2),
              'max_unmatched': args.max_unmatched,
              'pairing_accepted': len(unmatched) <= args.max_unmatched and len(pairs) >= 20}
    write_json(ws / 'preparation.json', report)
    print('PREPARATION', {k: report[k] for k in ['usable_rgb_count', 'paired_mask_count', 'working_size', 'elapsed_seconds']}, flush=True)
    if not report['pairing_accepted']:
        raise SystemExit('Too many uncertain associations; SfM/training not started.')
    if unmatched:
        print('AUDITED_EXCLUSIONS', unmatched, flush=True)


if __name__ == '__main__':
    main()
