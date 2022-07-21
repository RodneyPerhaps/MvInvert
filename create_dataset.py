import argparse
from argparse import Namespace
import PIL
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from scipy import ndimage
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import vstack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from torch_sparse_solve import solve
import torch.nn.functional as F
from core.restyle_encoder import infer
from pytorch3d.structures import Meshes
from core.utils import align_stylegan2
import cv2
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)

from core.resnet_50 import resnet50_use
face_encoder = resnet50_use().to('cuda:0')
face_encoder.load_state_dict(torch.load(r'pretrained/params.pt'))
face_encoder.eval()
for param in face_encoder.parameters():
	param.requires_grad = False

class DFRDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, img_paths):
        'Initialization'
        self.img_paths = img_paths

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_paths)

  def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img = self.img_paths[index]

        X = torch.load('data/' + ID + '.pt')

        return X

def create_dataset(args):
    image_path = args.dir
    save_path = args.outdir
    coeff = face_encoder(img_tensor.permute(0,3,1,2))
    id_tensor, exp_tensor, tex_tensor, rot_tensor, gamma_tensor, trans_tensor = coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], torch.cat((coeff[5], coeff[6]), axis=1)

    id_tensor.requires_grad = True
    exp_tensor.requires_grad = True
    tex_tensor.requires_grad = True
    rot_tensor.requires_grad = True
    gamma_tensor.requires_grad = True
    trans_tensor.requires_grad = True



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, help='image directory')
    parser.add_argument('--out_dir', type=str, default=None, help='output path for rendered image')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for preprocessing')
    args = parser.parse_args()
    create_dataset(args)