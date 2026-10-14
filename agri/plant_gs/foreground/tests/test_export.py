import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import export_ply


def test_ply_preserves_raw_scale_opacity_and_normalized_quaternions(tmp_path):
    n=2
    splats={'means':torch.tensor([[0.,0.,0.],[1.,2.,3.]]),
            'scales':torch.tensor([[-3.,-2.,-1.],[-4.,-3.,-2.]]),
            'opacities':torch.tensor([-2.,1.]),
            'quats':torch.tensor([[2.,0.,0.,0.],[0.,3.,0.,0.]]),
            'sh0':torch.zeros(n,1,3),'shN':torch.zeros(n,15,3)}
    path=tmp_path/'point_cloud.ply'
    export_ply(splats,path)
    vertex=PlyData.read(path)['vertex'].data
    assert len(vertex.dtype.names)==62 or len(vertex.dtype.names)==59
    np.testing.assert_allclose(vertex['scale_0'],[-3.,-4.])
    np.testing.assert_allclose(vertex['opacity'],[-2.,1.])
    np.testing.assert_allclose(vertex['rot_0'],[1.,0.])
    np.testing.assert_allclose(vertex['rot_1'],[0.,1.])
