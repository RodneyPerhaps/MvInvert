import torch
from torchvision import utils
import os
from os import path as osp
import glob

from tqdm import tqdm
from models.losses import photo_loss, lm_loss, reg_loss, reflectance_loss, gamma_loss
from models.recon_model import ReconModel
from models.resnet_50 import resnet50_use
from auxiliary.load_data import save_obj, run_alignment
from auxiliary.face_mask import get_face_mask 
from scipy.io import loadmat
import numpy as np
import argparse
from configs.paths_config import model_paths
from PIL import Image

TAR_SIZE = 224 # size for rendering window
OUT_SIZE = 512
PADDING_RATIO = 0.3 # enlarge the face detection bbox by a margin
FACE_MODEL_PATH = 'BFM/BFM_model_front.mat'

#params at rigid fitting stage
RF_ITERS = 50# iter number for the first frame
RF_LR = 0.01 #learning rate

#params at non-rigid fitting stage
NRF_ITERS = 50#epoch number
NRF_LR = 0.01 #learning rate
NRF_PHOTO_LOSS_W = 1.6
NRF_LM_LOSS_W = 100
NRF_REG_W = 1e-3
NRF_TEX_LOSS_W = 1

def train(args):
	img_fps = []
	args.img = None
	args.dir = '/data/CelebA/Img/img_celeba'
	#args.dir = '/home/jasonperhaps/Github/mvtc_restyle_6_iters/MoFA-test'
	#args.img = '/home/jasonperhaps/Github/mvtc_restyle_6_iters/MoFA-test/5.jpg'
	if args.img != None:
		img_fps.append(args.img)
	if args.dir != None:
		img_fps = sorted(glob.glob(args.dir+'/*.jpg'))
		img_fps += sorted(glob.glob(args.dir+'/*.png'))

	#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
	print('loading facemodel')
	try:
		facemodel = loadmat(FACE_MODEL_PATH)
	except Exception as e:
		print('failed to load %s' % FACE_MODEL_PATH)
	skinmask = torch.tensor(facemodel['skinmask']).cuda()

	model = ReconModel(facemodel, img_size=TAR_SIZE, out_size=OUT_SIZE)
	model.train()
	model.cuda()
	out_dir = '/data/Dataset/Tex'
	#out_dir = './Mofa-out'
	w_dir = os.path.join(out_dir, 'w')
	p_dir = os.path.join(out_dir, 'p')
	if not osp.exists(out_dir):
		os.makedirs(out_dir)
	if not osp.exists(w_dir):
		os.makedirs(w_dir)
	if not osp.exists(p_dir):
		os.makedirs(p_dir)
	print('loading images')
	face_encoder = resnet50_use().to('cuda:0')
	face_encoder.load_state_dict(torch.load(model_paths['3DMM_encoder']))
	face_encoder.eval()
	for param in face_encoder.parameters():
		param.requires_grad = False
	img_fps = img_fps[9267:]
	for img_path in img_fps:
		print(img_path)

		lms, cropped_img = run_alignment(img_path)
		if lms is None:
			continue
		#skin_mask = get_skinmask(cropped_img[:,:,::-1])

		lms = torch.tensor(lms, dtype=torch.float32).cuda().unsqueeze(0)
		img_tensor = torch.tensor(cropped_img, dtype=torch.float32).cuda().unsqueeze(0)
		#skinmask_tensor = torch.tensor(skin_mask, dtype=torch.float32).cuda().unsqueeze(0)
		face_mask = get_face_mask(img_tensor.permute(0,3,1,2))

		coeff = face_encoder(img_tensor.permute(0,3,1,2))
		id_tensor, exp_tensor, tex_tensor, rot_tensor, gamma_tensor, trans_tensor = coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], torch.cat((coeff[5], coeff[6]), axis=1)
	
		id_tensor.requires_grad = True
		exp_tensor.requires_grad = True
		tex_tensor.requires_grad = True
		rot_tensor.requires_grad = True
		gamma_tensor.requires_grad = True
		trans_tensor.requires_grad = True

		print('start rigid fitting')
		rigid_optimizer = torch.optim.Adam([rot_tensor, trans_tensor], lr=RF_LR)
		for i in tqdm(range(RF_ITERS)):
			rigid_optimizer.zero_grad()
			coeff = torch.cat([id_tensor, exp_tensor,
							tex_tensor, rot_tensor,
							gamma_tensor, trans_tensor], dim=1)
			_, pred_lms, _, _ = model(coeff)
			lm_loss_val = lm_loss(pred_lms, lms, img_size=TAR_SIZE)
			lm_loss_val.backward()
			rigid_optimizer.step()

		print('start non-rigid fitting')
		nonrigid_optimizer = torch.optim.Adam([id_tensor, tex_tensor,
						exp_tensor, rot_tensor,
						gamma_tensor, trans_tensor], lr=NRF_LR)
		for i in tqdm(range(NRF_ITERS)):
			nonrigid_optimizer.zero_grad()
			coeff = torch.cat([id_tensor, exp_tensor,
							tex_tensor, rot_tensor,
							gamma_tensor, trans_tensor], dim=1)
			rendered_img, pred_lms, face_texture, _ = model(coeff)
			mask = rendered_img[:, :, :, 3].detach()
			photo_loss_val = photo_loss(rendered_img[:, :, :, :3], img_tensor, (mask>0) * face_mask)
			lm_loss_val = lm_loss(pred_lms, lms, img_size=TAR_SIZE)
			reg_loss_val = reg_loss(id_tensor, exp_tensor, tex_tensor)
			tex_loss_val = reflectance_loss(face_texture, skinmask)
			loss = photo_loss_val*NRF_PHOTO_LOSS_W + \
							lm_loss_val*NRF_LM_LOSS_W + \
							reg_loss_val*NRF_REG_W + \
							tex_loss_val*NRF_TEX_LOSS_W
			loss.backward()
			nonrigid_optimizer.step()

		with torch.no_grad():
			coeff = torch.cat([id_tensor, exp_tensor,
							tex_tensor, rot_tensor,
							gamma_tensor, trans_tensor], dim=1)
			#uv_img, fitted_mesh_img, imgs, tex_diff_poses, ginv, vis_hat, norm_img = model(coeff, img_tensor.permute(0,3,1,2))
			#uv_img, front_face_img, tex_diff_poses = model(coeff, img_tensor.permute(0,3,1,2))
			#uv_img, front_face_img, tex_diff_poses, vis_img, vis_hat_all_uv_img, norm_img, ginv, imgs_dict = 
			#completed_faces, completed_tex, imgs_dict, tex_diff = model(coeff, img_tensor.permute(0,3,1,2))
			poissoned_img = model(coeff, img_tensor.permute(0,3,1,2), face_mask)

			img_name = os.path.basename(img_path)
			utils.save_image(poissoned_img[:,:,:,:3].permute(0,3,1,2), osp.join(out_dir, img_name), nrow=1, normalize=True, range=(0,255))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, default=None, help='image path')
	parser.add_argument('--dir', type=str, default=None, help='image directory')
	args = parser.parse_args()
	train(args)
