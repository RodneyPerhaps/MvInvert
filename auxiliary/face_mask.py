#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import glob
#from .face_parsing.face_parsing_model import BiSeNet
import sys
#sys.path.append('../auxiliary')
from auxiliary.face_parsing.model import BiSeNet

def save_RGBA_face(im, parsing_anno, img_size, save_path='vis_results/parsing_map_on_im.jpg'):
	vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
	face_mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
	num_of_class = np.max(vis_parsing_anno)
	for pi in range(1, num_of_class + 1):
	    index = np.where(vis_parsing_anno == pi)
	    if pi in [1,2,3,4,5,10,12,13]:
	        face_mask[index[0], index[1]] = 255.0
	im = np.array(im)
	img = im.copy().astype(np.uint8)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	seg_img = face_mask.astype(np.uint8)

	img = cv2.resize(img, (img_size, img_size))
	seg_img = cv2.resize(seg_img, (img_size, img_size))
	seg_img = seg_img[:,:,None]

	BGRA_img = np.concatenate((img, seg_img), axis=2)

	cv2.imwrite(save_path, BGRA_img)

def evaluate(outdir='./res/test_res', srcdir='./data', cp='model_final_diss.pth'):
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	n_classes = 19
	net = BiSeNet(n_classes=n_classes)
	net.cuda()
	save_pth = osp.join('fp_weights', cp)
	net.load_state_dict(torch.load(save_pth))
	net.eval()

	to_tensor = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])
	with torch.no_grad():
		img_list = sorted(glob.glob('{}/*.png'.format(srcdir)))
		img_list += sorted(glob.glob('{}/*.jpg'.format(srcdir)))
		for image_path in img_list:
			img = Image.open(image_path).convert('RGB')
			image = img.resize((512, 512), Image.BILINEAR)
			img = to_tensor(image)
			img = torch.unsqueeze(img, 0)
			img = img.cuda()
			out = net(img)[0]
			parsing = out.squeeze(0).cpu().numpy().argmax(0)
			# print(parsing)
			print(np.unique(parsing))
			save_RGBA_face(image, parsing, 224, save_path=osp.join(outdir, os.path.basename(image_path)))

class FaceMask:
	def __init__(self, img_size=224, ckpt_path = 'auxiliary/fp_weights/face_parsing.pth' ):
		self.img_size = img_size
		self.n_classes = 19
		net = BiSeNet(n_classes=self.n_classes)
		net.cuda()
		net.load_state_dict(torch.load(ckpt_path))
		net.eval()
		self.net = net

		self.resize = transforms.Resize([224, 224])
		self.resize_norm = transforms.Compose([
			transforms.Resize([512, 512]),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])
	def infer(self, img_tensor):
		with torch.no_grad():
			img = self.resize_norm(img_tensor / 255)
			out = self.net(img)[0]
			#parsing = out.squeeze(0).cpu().numpy().argmax(0)
			parsing = out.argmax(1)
			#save_RGBA_face(image, parsing, 224, save_path=osp.join(outdir, os.path.basename(image_path)))
			face_mask = torch.zeros((parsing.shape[0], parsing.shape[1], parsing.shape[2]))
			num_of_class = torch.max(parsing)
			for pi in [1,2,3,4,5,10,12,13]:
				index = torch.where(parsing== pi)
				face_mask[index[0], index[1], index[2]] = 255.0

			seg_img = self.resize(face_mask)

			return seg_img.cuda()



def get_face_mask(input, img_size=224):
	"""
	input: NCHW
	output: N1H'W'
	"""
	facemask = FaceMask(img_size)
	mask = facemask.infer(input) / 255

	return mask

if __name__ == '__main__':
	srcdir='/home/jasonperhaps/Github/mvtc_restyle_6_iters/MoFA-test'
	outdir='/home/jasonperhaps/Github/mvtc_restyle_6_iters/MoFA_RGBA'

	evaluate(outdir, srcdir, cp='face_parsing.pth')

