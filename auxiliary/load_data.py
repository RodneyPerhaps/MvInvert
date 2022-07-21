import torch
from scipy.io import loadmat, savemat
from array import array
import numpy as np
from PIL import Image
from preprocess.mtcnn import MTCNN
from utils.align_faces_parallel import get_landmark_from_image_array
import dlib
from .skin import skinmask
from configs.paths_config import model_paths
predictor = dlib.shape_predictor(model_paths["shape_predictor"])
detector = dlib.get_frontal_face_detector()

def load_lm3d(fsimilarity_Lm3D_all_mat='BFM/similarity_Lm3D_all.mat'):
    # load landmarks for standard face, which is used for image preprocessing
    Lm3D = loadmat(fsimilarity_Lm3D_all_mat)
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]
    return Lm3D

lm3D = load_lm3d()
mtcnn = MTCNN()

def get_skinmask(img):

	#img = np.squeeze(img,0)
	skin_img = skinmask(img)
	return skin_img

def run_alignment(img_path):
	img = Image.open(img_path).convert('RGB')
	_, _, lm = mtcnn.detect(img, landmarks=True)
	if lm is None:
		return None, None
	input_img_org, lm_new, transform_params = Preprocess(img, lm[0], lm3D)
	aligned_img = input_img_org[0,:,:,::-1]
	dets = detector(aligned_img, 1)
	if len(dets) == 0:
		return None, None
	lms = get_landmark_from_image_array(aligned_img, predictor)
	return lms, aligned_img


def load_expbasis():
    # load expression basis
    n_vertex = 53215
    exp_bin = open(r'BFM/Exp_Pca.bin', 'rb')
    exp_dim = array('i')
    exp_dim.fromfile(exp_bin, 1)
    expMU = array('f')
    expPC = array('f')
    expMU.fromfile(exp_bin, 3*n_vertex)
    expPC.fromfile(exp_bin, 3*exp_dim[0]*n_vertex)

    expPC = np.array(expPC)
    expPC = np.reshape(expPC, [exp_dim[0], -1])
    expPC = np.transpose(expPC)

    expEV = np.loadtxt(r'BFM/std_exp.txt')

    return expPC, expEV

# calculating least sqaures problem
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def process_img(img, lm, t, s, target_size=224.):
    w0, h0 = img.size
    w = (w0/s*102).astype(np.int32)
    h = (h0/s*102).astype(np.int32)
    img = img.resize((w, h), resample=Image.BICUBIC)

    left = (w/2 - target_size/2 + float((t[0] - w0/2)*102/s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*102/s)).astype(np.int32)
    below = up + target_size

    img = img.crop((left, up, right, below))
    img = np.array(img)
    img = img[:, :, ::-1]  # RGBtoBGR
    img = np.expand_dims(img, 0)
    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                   t[1] + h0/2], axis=1)/s*102
    lm = lm - \
        np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, lm


def Preprocess(img, lm, lm3D):
    # resize and crop input images before sending to the R-Net
    w0, h0 = img.size

    # change from image plane coordinates to 3D sapce coordinates(X-Y plane)
    lm = np.stack([lm[:, 0], h0 - 1 - lm[:, 1]], axis=1)

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks
    # lm3D -> lm
    t, s = POS(lm.transpose(), lm3D.transpose())

    # processing the image
    img_new, lm_new = process_img(img, lm, t, s)

    lm_new = np.stack([lm_new[:, 0], 223 - lm_new[:, 1]], axis=1)
    trans_params = np.array([w0, h0, 102.0/s, t[0, 0], t[1, 0]])

    return img_new, lm_new, trans_params

def save_obj(path, v, f, c):
    # save 3D face to obj file
    with open(path, 'w') as file:
        for i in range(len(v)):
            file.write('v %f %f %f %f %f %f\n' %
                       (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))

        file.write('\n')

        for i in range(len(f)):
            file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    file.close()