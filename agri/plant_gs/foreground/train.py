#!/usr/bin/env python
"""Object-only gsplat fit: RGB foreground + alpha silhouette, no background RGB."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization, export_splats
from gsplat.strategy import DefaultStrategy
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from common import write_json, evaluation_due
from losses import foreground_losses, masked_ssim

C0 = .28209479177387814


class Captures:
    def __init__(self, root, device='cuda', test_every=8, downscale=1):
        self.root = Path(root)
        self.frames = json.loads((self.root / 'frames.json').read_text())
        points = np.load(self.root / 'points3D_plant.npz')
        xyz = points['xyz'].astype(np.float32)
        self.center = np.median(xyz, axis=0)
        self.radius = float(np.percentile(np.linalg.norm(xyz-self.center, axis=1), 98))
        self.scale = 1 / max(self.radius, 1e-6)
        self.points = (xyz-self.center) * self.scale
        self.rgbs = points['rgb'].astype(np.float32)/255
        self.device = device
        rgbs, masks, Ks, poses = [], [], [], []
        for frame in self.frames:
            with Image.open(frame['image']) as im:
                size = (frame['width']//downscale, frame['height']//downscale)
                rgbs.append(np.array(im.convert('RGB').resize(size, Image.Resampling.LANCZOS)))
            with Image.open(frame['mask']) as im:
                masks.append(np.array(im.convert('L').resize(size, Image.Resampling.NEAREST)) > 127)
            K = np.asarray(frame['K'], np.float32).copy()
            K[0] *= size[0]/frame['width']
            K[1] *= size[1]/frame['height']
            Ks.append(K)
            pose = np.asarray(frame['camtoworld'], np.float32).copy()
            pose[:3, 3] = (pose[:3, 3]-self.center)*self.scale
            poses.append(pose)
        self.width, self.height = size
        self.images = torch.from_numpy(np.stack(rgbs)).to(device)
        self.masks = torch.from_numpy(np.stack(masks)).to(device)
        self.Ks = torch.from_numpy(np.stack(Ks)).to(device)
        self.w2cs = torch.linalg.inv(torch.from_numpy(np.stack(poses)).to(device))
        self.test_idx = list(range(0, len(self.frames), test_every))
        self.train_idx = [i for i in range(len(self.frames)) if i not in self.test_idx]
        print('DATA',len(self.frames),'TRAIN',len(self.train_idx),'VAL',len(self.test_idx),
              'SIZE',size,'INIT_POINTS',len(xyz),'RADIUS',self.radius,flush=True)

    def batch(self, i):
        return self.images[i:i+1].float()/255, self.masks[i:i+1,...,None].float(), self.Ks[i:i+1], self.w2cs[i:i+1]

    @property
    def normalization(self):
        return {'center': self.center.tolist(), 'radius': self.radius, 'scale': self.scale}


def init_splats(data, degree=3):
    points = data.points
    distances, _ = cKDTree(points).query(points, k=4, workers=4)
    spacing = np.sqrt((distances[:,1:]**2).mean(axis=-1)).clip(1e-5)
    spacing = np.minimum(spacing, np.median(spacing)*3)
    n = len(points)
    sh = torch.zeros((n,(degree+1)**2,3),device=data.device)
    sh[:,0] = (torch.from_numpy(data.rgbs).to(data.device)-.5)/C0
    splats = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(torch.from_numpy(points).to(data.device)),
        'scales': torch.nn.Parameter(torch.from_numpy(np.log(spacing).astype(np.float32)[:,None].repeat(3,1)).to(data.device)),
        'quats': torch.nn.Parameter(F.normalize(torch.randn(n,4,device=data.device),dim=-1)),
        'opacities': torch.nn.Parameter(torch.full((n,),math.log(.1/.9),device=data.device)),
        'sh0': torch.nn.Parameter(sh[:,:1].contiguous()),
        'shN': torch.nn.Parameter(sh[:,1:].contiguous()),
    })
    rates = {'means':1.6e-4,'scales':5e-3,'quats':1e-3,'opacities':5e-2,'sh0':2.5e-3,'shN':2.5e-3/20}
    optimizers = {k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in splats.items()}
    return splats, optimizers


def render(splats, K, w2c, width, height, degree, absgrad=False):
    return rasterization(means=splats['means'],quats=splats['quats'],
        scales=splats['scales'].exp(),opacities=splats['opacities'].sigmoid(),
        colors=torch.cat([splats['sh0'],splats['shN']],dim=1),
        viewmats=w2c,Ks=K,width=width,height=height,sh_degree=degree,
        near_plane=.01,packed=False,absgrad=absgrad,backgrounds=None)


@torch.no_grad()
def export_ply(splats, path):
    export_splats(means=splats['means'],scales=splats['scales'],
        quats=F.normalize(splats['quats'],dim=-1),opacities=splats['opacities'],
        sh0=splats['sh0'],shN=splats['shN'],format='ply',save_to=str(path))


@torch.no_grad()
def evaluate(splats, data, degree, out=None, step=0):
    results = []
    panels = []
    preview_indices = set(data.test_idx[j] for j in np.linspace(0,len(data.test_idx)-1,6).astype(int))
    for i in data.test_idx:
        rgb, mask, K, w2c = data.batch(i)
        color, alpha, _ = render(splats,K,w2c,data.width,data.height,degree)
        mse = (((color-rgb)**2)*mask).sum()/(3*mask.sum()).clamp_min(1)
        fg_psnr = float(-10*torch.log10(mse.clamp_min(1e-12)))
        predicted, target = alpha>.5, mask>.5
        intersection = (predicted & target).sum()
        union = (predicted | target).sum()
        iou = float(intersection/union.clamp_min(1))
        bg = 1-mask
        bg_alpha = float((alpha*bg).sum()/bg.sum().clamp_min(1))
        spill_fraction = float((predicted & ~target).sum()/(~target).sum().clamp_min(1))
        result = {'name':data.frames[i]['name'],'foreground_psnr':fg_psnr,
                  'silhouette_iou':iou,'background_alpha':bg_alpha,'background_opaque_fraction':spill_fraction}
        results.append(result)
        if out and i in preview_indices:
            def pil(x):
                return Image.fromarray((x[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)).resize(
                    (data.width//2,data.height//2),Image.Resampling.LANCZOS)
            bg_color = torch.tensor([.12,.20,.28],device=data.device)[None,None,None,:]
            rendered_bg = color + (1-alpha)*bg_color
            rgba = torch.cat([color,alpha],dim=-1)
            # Unpremultiply for a proper transparent PNG (zero-alpha RGB kept 0).
            rgba[...,:3] = color/alpha.clamp_min(1e-6)
            preview_dir = Path(out)/'renders'
            preview_dir.mkdir(parents=True,exist_ok=True)
            Image.fromarray((rgba[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)).save(preview_dir/f'{Path(result["name"]).stem}_transparent.png')
            columns=[pil(rgb*mask),pil(rendered_bg),pil(alpha.repeat(1,1,1,3))]
            panel=Image.new('RGB',(columns[0].width*3,columns[0].height+22),'white')
            draw=ImageDraw.Draw(panel)
            for col, (label, im) in enumerate(zip(['masked RGB','render (UNMASKED)','render alpha'],columns)):
                panel.paste(im,(col*im.width,22))
                draw.text((col*im.width+5,5),f'{result["name"]} {label}',fill='black')
            panel.save(preview_dir/f'step{step:06d}_{Path(result["name"]).stem}.jpg',quality=90)
            panels.append(panel)
    metrics = {k:float(np.mean([r[k] for r in results])) for k in ['foreground_psnr','silhouette_iou','background_alpha','background_opaque_fraction']}
    metrics['per_view']=results
    if out:
        contact = Image.new('RGB',(panels[0].width,sum(p.height for p in panels)))
        y=0
        for panel in panels:
            contact.paste(panel,(0,y)); y+=panel.height
        contact.save(Path(out)/'preview.jpg',quality=90)
        write_json(Path(out)/'validation.json',metrics)
    return metrics


def checkpoint(splats, data, degree, step, config):
    return {'splats':{k:v.detach().cpu() for k,v in splats.items()},'step':step,'sh_degree':degree,
            'normalization':data.normalization,'data':str(data.root.resolve()),'config':config}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--steps',type=int,default=30000)
    p.add_argument('--eval-every',type=int,default=3000)
    p.add_argument('--save-every',type=int,default=5000)
    p.add_argument('--downscale',type=int,default=1)
    p.add_argument('--sh-degree',type=int,default=3)
    p.add_argument('--foreground-alpha-weight',type=float,default=.1)
    p.add_argument('--background-alpha-weight',type=float,default=.75)
    p.add_argument('--max-gaussians',type=int,default=300000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if (args.result/'splats.pt').exists():
        raise SystemExit('Result already completed; use a new result directory')
    args.result.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    write_json(args.result/'config.json',config)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=True
    data=Captures(args.data,downscale=args.downscale)
    splats,opts=init_splats(data,args.sh_degree)
    strategy=DefaultStrategy(prune_opa=.005,grow_grad2d=.0002,
                             grow_scale3d=.01,prune_scale3d=.15,
                             refine_start_iter=300,refine_stop_iter=min(15000,int(args.steps*.8)),
                             reset_every=3000,refine_every=100,
                             pause_refine_after_reset=len(data.train_idx),verbose=False)
    strategy.check_sanity(splats,opts)
    state=strategy.initialize_state(scene_scale=1.)
    best_score=-float('inf'); best_step=None
    start=time.time()
    history=args.result/'history.jsonl'
    rng=np.random.default_rng(args.seed)
    for step in range(args.steps):
        idx=int(rng.choice(data.train_idx))
        rgb,mask,K,w2c=data.batch(idx)
        degree=min(step//1000,args.sh_degree)
        color,alpha,info=render(splats,K,w2c,data.width,data.height,degree,strategy.absgrad)
        strategy.step_pre_backward(splats,opts,state,step,info)
        components=foreground_losses(color,alpha,rgb,mask)
        ssim=masked_ssim(color,rgb,mask)
        loss=.8*components['foreground_rgb']+.2*(1-ssim)
        loss+=args.foreground_alpha_weight*components['foreground_alpha']
        loss+=args.background_alpha_weight*components['background_alpha']
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at {step}')
        loss.backward()
        opts['means'].param_groups[0]['lr']=1.6e-4*(.01**(step/max(1,args.steps-1)))
        for opt in opts.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        if len(splats['means'])>=args.max_gaussians:
            strategy.grow_grad2d=1e6
        else:
            strategy.grow_grad2d=.0002
        strategy.step_post_backward(splats,opts,state,step,info,packed=False)
        if step%100==0 or step==args.steps-1:
            record={'step':step,'loss':float(loss.detach()),'ssim':float(ssim.detach()),
                    'gaussians':len(splats['means']),'elapsed_seconds':round(time.time()-start,2),
                    **{k:float(v.detach()) for k,v in components.items()}}
            with history.open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            write_json(args.result/'status.json',{'status':'training',**record,'total_steps':args.steps})
        if evaluation_due(step, args.eval_every, args.steps, strategy.reset_every,
                          strategy.refine_stop_iter, len(data.train_idx)):
            with torch.no_grad():
                metrics=evaluate(splats,data,degree)
            score=metrics['foreground_psnr']+5*metrics['silhouette_iou']-20*metrics['background_alpha']
            compact={k:v for k,v in metrics.items() if k!='per_view'}
            print('EVAL',step,json.dumps(compact),'score',score,flush=True)
            with history.open('a') as f: f.write(json.dumps({'step':step,'eval':compact,'score':score})+'\n')
            if score>best_score:
                best_score=score;best_step=step
                torch.save(checkpoint(splats,data,args.sh_degree,step,config),args.result/'splats_best.pt')
                export_ply(splats,args.result/'point_cloud_best_raw.ply')
                with torch.no_grad(): evaluate(splats,data,degree,out=args.result,step=step)
        if step>0 and step%args.save_every==0:
            torch.save(checkpoint(splats,data,args.sh_degree,step,config),args.result/f'splats_{step:06d}.pt')
    torch.save(checkpoint(splats,data,args.sh_degree,args.steps,config),args.result/'splats.pt')
    export_ply(splats,args.result/'point_cloud_final_raw.ply')
    write_json(args.result/'status.json',{'status':'complete','steps':args.steps,'gaussians':len(splats['means']),
                'elapsed_seconds':round(time.time()-start,2),'best_step':best_step,'best_score':best_score})
    print('TRAIN_COMPLETE',flush=True)


if __name__=='__main__':
    main()
