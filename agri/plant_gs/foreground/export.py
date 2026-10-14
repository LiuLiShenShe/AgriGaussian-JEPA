#!/usr/bin/env python
"""Clean true 3DGS parameters; export standard splats and an ordinary center cloud."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from common import write_json, silhouette_support
from train import Captures, export_ply, evaluate, C0


def point_cloud(splats, path):
    xyz = splats['means'].cpu().numpy()
    rgb = ((splats['sh0'][:,0].cpu().numpy()*C0+.5).clip(0,1)*255).astype(np.uint8)
    vertex = np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for j,k in enumerate(['x','y','z']): vertex[k]=xyz[:,j]
    for j,k in enumerate(['red','green','blue']): vertex[k]=rgb[:,j]
    PlyData([PlyElement.describe(vertex,'vertex')],text=False).write(str(path))


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


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mask-ratio',type=float,default=.8)
    p.add_argument('--min-opacity',type=float,default=.01)
    p.add_argument('--mask-dilation',type=int,default=3)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
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
    torch.save({**checkpoint,'splats':cleaned,'source_checkpoint':str(args.checkpoint.resolve()),
                'cleanup':{'mask_ratio':args.mask_ratio,'dilation':args.mask_dilation}},args.output/'splats_foreground.pt')
    export_ply(cleaned,args.output/'point_cloud.ply')
    point_cloud(cleaned,args.output/'gaussian_centers.ply')
    valid=validate_ply(args.output/'point_cloud.ply')
    data=Captures(checkpoint['data'],downscale=checkpoint['config'].get('downscale',1))
    gpu={k:v.to('cuda') for k,v in splats.items()}
    raw_metrics=evaluate(gpu,data,checkpoint['sh_degree'])
    del gpu
    torch.cuda.empty_cache()
    gpu={k:v.to('cuda') for k,v in cleaned.items()}
    cleaned_metrics=evaluate(gpu,data,checkpoint['sh_degree'],out=args.output,step=checkpoint['step'])
    np.savez_compressed(args.output/'cleanup_scores.npz',mask_ratio=consistency,observed_views=viewed,
                        foreground_views=positives,opacity=opacity,keep=keep)
    report={'status':'complete','source_checkpoint':str(args.checkpoint.resolve()),'step':checkpoint['step'],
            'raw_gaussians':len(world),'retained_gaussians':int(keep.sum()),'removed_gaussians':int((~keep).sum()),
            'cleanup_mask_ratio':args.mask_ratio,'mask_dilation_pixels':args.mask_dilation,
            'min_opacity':args.min_opacity,'kept_mask_ratio_min':float(consistency[keep].min()),
            'kept_mask_ratio_median':float(np.median(consistency[keep])),
            'normalization':norm,'format_validation':valid,
            'raw_validation':{k:v for k,v in raw_metrics.items() if k!='per_view'},
            'clean_validation':{k:v for k,v in cleaned_metrics.items() if k!='per_view'},
            'target':'all paired SAM foreground, including plant/pot/blue block',
            'point_cloud':str((args.output/'point_cloud.ply').resolve()),
            'gaussian_centers':str((args.output/'gaussian_centers.ply').resolve()),
            'preview':str((args.output/'preview.jpg').resolve()),
            'limitations':'2D segmentation/camera errors and Gaussian finite tails prevent an absolute zero-background guarantee; measured unmasked rendered alpha is reported.'}
    write_json(args.output/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
