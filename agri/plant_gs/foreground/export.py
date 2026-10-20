#!/usr/bin/env python
"""Clean true 3DGS parameters; export standard splats and an ordinary center cloud.

The exporter is the point in the pipeline where an *unverifiable* number can
enter a report. It reads ``splats_best.pt``, filters, writes a PLY, and prints a
PSNR. If training is still writing that checkpoint, or if a second run has
already replaced it, the PLY is real but its provenance is a guess.

So export refuses to start unless all of the following hold:

* the owning training run is finished (``status.json`` is ``complete`` and the
  terminal ``splats.pt`` exists),
* no other process holds the result-directory lock,
* the checkpoint's SHA256 is identical before and after the export.

It then writes a provenance record tying its input bytes, its output bytes, the
step inside the checkpoint, and the gaussian counts to each other -- see
``provenance.py``.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from common import write_json, silhouette_support
from train import Captures, export_ply, evaluate, C0
import provenance
import runlock

#: PLY attribute -> checkpoint tensor, verified element-for-element after export.
#: Defined in provenance.py so the independent verifier uses the same map.
ATTRIBUTE_MAP = provenance.PLY_ATTRIBUTE_MAP


def point_cloud(splats, path):
    xyz = splats['means'].cpu().numpy()
    rgb = ((splats['sh0'][:,0].cpu().numpy()*C0+.5).clip(0,1)*255).astype(np.uint8)
    vertex = np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for j,k in enumerate(['x','y','z']): vertex[k]=xyz[:,j]
    for j,k in enumerate(['red','green','blue']): vertex[k]=rgb[:,j]
    _atomic_ply(PlyData([PlyElement.describe(vertex,'vertex')],text=False), path)


def _atomic_ply(ply, path):
    """Write a PLY via a temp file so a partial PLY is never visible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    ply.write(str(tmp))
    with open(tmp, 'rb') as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def validate_ply(path):
    data=PlyData.read(str(path))['vertex'].data
    required=['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    missing=[n for n in required if n not in data.dtype.names]
    finite=all(np.isfinite(data[n]).all() for n in data.dtype.names)
    if missing or not finite or len(data)==0:
        raise RuntimeError(f'Invalid export: missing={missing}, finite={finite}, count={len(data)}')
    xyz=np.column_stack([data[k] for k in ['x','y','z']])
    return {'count':len(data),'properties':len(data.dtype.names),'finite':finite,
            'bbox_min':xyz.min(axis=0).tolist(),'bbox_max':xyz.max(axis=0).tolist()}


def verify_attribute_consistency(ply_path, splats):
    """Confirm PLY rows are exactly ``splats`` rows, in order.

    The cleanup step drops gaussians, so the PLY has *fewer* rows than the
    checkpoint. Comparing the two arrays element-wise would be wrong. What is
    checkable -- and what this does -- is that every PLY row equals the
    corresponding checkpoint row **after** the same keep-mask and in the same
    order. That is the mapping the exporter used, so a mismatch means the PLY
    did not come from this checkpoint (or came from it through different code).

    Returns a dict of per-attribute max absolute error and a pass flag.
    """
    data = PlyData.read(str(ply_path))['vertex'].data
    count = len(data)
    errors = {}
    failures = []
    for ply_name, (tensor_name, index) in ATTRIBUTE_MAP.items():
        if tensor_name not in splats:
            continue
        tensor = splats[tensor_name].detach().cpu().numpy()
        if tensor_name == 'quats':
            tensor = provenance.normalize_quaternions(tensor)
        column = provenance.attribute_column(tensor_name, tensor, index)
        column = np.asarray(column).reshape(count, -1)
        if column.shape[1] != 1:
            column = column[:, :1]
        observed = np.asarray(data[ply_name]).reshape(-1, 1)
        delta = float(np.abs(column - observed).max())
        errors[ply_name] = delta
        tolerance = 2e-6 if observed.dtype.itemsize == 4 else 1e-9
        if delta > tolerance:
            failures.append(f'{ply_name}: max |delta| = {delta:.3e}')
    return {
        'passed': not failures,
        'rows_compared': int(count),
        'max_abs_error': errors,
        'failures': failures,
        'note': 'PLY rows are compared against the cleaned checkpoint in the '
                'same order the keep-mask produces; the arrays are not the same '
                'length as the unfiltered checkpoint by design.',
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mask-ratio',type=float,default=.8)
    p.add_argument('--min-opacity',type=float,default=.01)
    p.add_argument('--mask-dilation',type=int,default=3)
    p.add_argument('--allow-incomplete',action='store_true',
                   help='export even though the training run is not marked '
                        'complete (records the fact in provenance; for debugging '
                        'an interrupted run, never for a reported metric)')
    args=p.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    result_dir = checkpoint_path.parent

    # --- never export a directory that is still being written ------------- #
    try:
        lock = runlock.RunLock(result_dir, create=False).acquire()
    except runlock.LockUnavailable as exc:
        raise SystemExit(
            'REFUSING TO EXPORT: %s. Exporting now would read a checkpoint that '
            'a live training process is still updating.' % exc)
    except runlock.ResultStateError as exc:
        raise SystemExit('REFUSING TO EXPORT: %s' % exc)

    try:
        status_file = result_dir / 'status.json'
        status = json.loads(status_file.read_text()) if status_file.is_file() else {}
        finished = status.get('status') == 'complete' and (result_dir/'splats.pt').is_file()
        if not finished and not args.allow_incomplete:
            raise SystemExit(
                'REFUSING TO EXPORT: %s does not look like a finished training '
                'run (status=%r, splats.pt=%s). An export from an unfinished run '
                'produces a PLY whose reported metrics belong to no complete '
                'experiment. Use --allow-incomplete only for debugging.'
                % (result_dir, status.get('status'), (result_dir/'splats.pt').is_file()))

        started_utc = provenance.utc_now()
        t0 = time.time()
        pre_identity = provenance.file_identity(checkpoint_path)

        args.output.mkdir(parents=True,exist_ok=True)
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        splats=checkpoint['splats']
        frames=json.loads((Path(checkpoint['data'])/'frames.json').read_text())
        masks=[]
        for frame in frames:
            with Image.open(frame['mask']) as im:
                mask=np.asarray(im.convert('L'))>127
            if args.mask_dilation:
                k=2*args.mask_dilation+1
                mask=cv2.dilate(mask.astype(np.uint8),np.ones((k,k),np.uint8))>0
            masks.append(mask)
        normalized=splats['means'].numpy()
        norm=checkpoint['normalization']
        world=normalized/norm['scale']+np.asarray(norm['center'])
        positives,viewed=silhouette_support(world,frames,masks)
        consistency=positives/np.maximum(viewed,1)
        finite=np.ones(len(world),bool)
        for value in splats.values(): finite &= torch.isfinite(value).flatten(1).all(-1).numpy() if value.ndim>1 else torch.isfinite(value).numpy()
        opacity=splats['opacities'].sigmoid().numpy()
        scales=splats['scales'].exp().numpy()
        # The sparse foreground defines an object-centered normalized volume.
        # A generous bound removes remote floaters without dropping tips or block.
        within=np.linalg.norm(normalized,axis=1)<2.5
        keep=finite & (opacity>=args.min_opacity) & (consistency>=args.mask_ratio) & (viewed>=10) & within
        keep &= np.isfinite(scales).all(axis=1) & (scales.max(axis=1)<.3)
        print('CLEANUP',len(world),'->',int(keep.sum()),'REMOVED',int((~keep).sum()),flush=True)
        if keep.sum()<100:
            raise RuntimeError('Cleanup retained too few Gaussians; do not export an empty object')
        cleaned={k:v[torch.from_numpy(keep)].contiguous() for k,v in splats.items()}
        torch.save({**checkpoint,'splats':cleaned,'source_checkpoint':str(checkpoint_path),
                    'cleanup':{'mask_ratio':args.mask_ratio,'dilation':args.mask_dilation}},
                   args.output/'splats_foreground.pt.tmp')
        os.replace(args.output/'splats_foreground.pt.tmp', args.output/'splats_foreground.pt')
        export_ply(cleaned,args.output/'point_cloud.ply')
        point_cloud(cleaned,args.output/'gaussian_centers.ply')
        valid=validate_ply(args.output/'point_cloud.ply')
        attributes=verify_attribute_consistency(args.output/'point_cloud.ply', cleaned)
        if not attributes['passed']:
            raise RuntimeError(
                'exported PLY does not match the cleaned checkpoint: %s'
                % attributes['failures'])

        data=Captures(checkpoint['data'],downscale=checkpoint['config'].get('downscale',1))
        gpu={k:v.to('cuda') for k,v in splats.items()}
        raw_metrics=evaluate(gpu,data,checkpoint['sh_degree'])
        del gpu
        torch.cuda.empty_cache()
        gpu={k:v.to('cuda') for k,v in cleaned.items()}
        cleaned_metrics=evaluate(gpu,data,checkpoint['sh_degree'],out=args.output,step=checkpoint['step'])
        np.savez_compressed(args.output/'cleanup_scores.npz',mask_ratio=consistency,observed_views=viewed,
                            foreground_views=positives,opacity=opacity,keep=keep)

        # The retained-row index vector is the link between "gaussians before"
        # and "gaussians after". Hashing it lets a later reader re-derive the
        # mapping without the intermediate arrays.
        retained_indices = np.flatnonzero(keep).astype(np.int64)
        retained_sha = hashlib.sha256(retained_indices.tobytes()).hexdigest()

        # Re-hash the checkpoint: if training appended to it during the export,
        # the PLY is a mix and must not be reported.
        post_identity = provenance.file_identity(checkpoint_path)
        if pre_identity['sha256'] != post_identity['sha256'] or \
           pre_identity['bytes'] != post_identity['bytes']:
            raise RuntimeError(
                'the checkpoint changed while the export ran (%s -> %s); the '
                'exported PLY has no single source. Refusing to write a '
                'provenance record for it.'
                % (pre_identity['sha256'][:12], post_identity['sha256'][:12]))

        finished_utc = provenance.utc_now()
        report={'status':'complete','source_checkpoint':str(checkpoint_path),'step':checkpoint['step'],
                'raw_gaussians':len(world),'retained_gaussians':int(keep.sum()),'removed_gaussians':int((~keep).sum()),
                'cleanup_mask_ratio':args.mask_ratio,'mask_dilation_pixels':args.mask_dilation,
                'min_opacity':args.min_opacity,'kept_mask_ratio_min':float(consistency[keep].min()),
                'kept_mask_ratio_median':float(np.median(consistency[keep])),
                'normalization':norm,'format_validation':valid,
                'attribute_consistency':attributes,
                'raw_validation':{k:v for k,v in raw_metrics.items() if k!='per_view'},
                'clean_validation':{k:v for k,v in cleaned_metrics.items() if k!='per_view'},
                'target':'all paired SAM foreground, including plant/pot/blue block',
                'point_cloud':str((args.output/'point_cloud.ply').resolve()),
                'gaussian_centers':str((args.output/'gaussian_centers.ply').resolve()),
                'preview':str((args.output/'preview.jpg').resolve()),
                'training_complete': bool(finished),
                'limitations':'2D segmentation/camera errors and Gaussian finite tails prevent an absolute zero-background guarantee; measured unmasked rendered alpha is reported.'}
        write_json(args.output/'report.json',report)

        output_dir = Path(args.output).resolve()
        provenance.write_record(output_dir/'export.provenance.json', provenance.export_record(
            output=output_dir,
            checkpoint=checkpoint_path,
            step=checkpoint['step'],
            ply=output_dir/'point_cloud.ply',
            gaussians_before=len(world),
            gaussians_after=int(keep.sum()),
            cleanup={'mask_ratio': args.mask_ratio,
                     'mask_dilation_pixels': args.mask_dilation,
                     'min_opacity': args.min_opacity,
                     'retained_index_sha256': retained_sha,
                     'retained_index_count': int(len(retained_indices))},
            format_validation=valid,
            attribute_validation=attributes,
            extra={'checkpoint_sha256_before': pre_identity['sha256'],
                   'checkpoint_sha256_after': post_identity['sha256'],
                   'checkpoint_unchanged_during_export': True,
                   'export_started_utc': started_utc,
                   'export_finished_utc': finished_utc,
                   'export_seconds': round(time.time()-t0, 2),
                   'training_complete': bool(finished),
                   'raw_validation': {k: v for k, v in raw_metrics.items() if k != 'per_view'},
                   'clean_validation': {k: v for k, v in cleaned_metrics.items() if k != 'per_view'}}))
        print(json.dumps({k: v for k, v in report.items() if k != 'format_validation'},
                         indent=2),flush=True)
    finally:
        lock.release()


if __name__=='__main__':
    main()
