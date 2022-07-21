import torch
import scipy
from PIL import Image
import PIL
import cv2
import numpy as np
import os

class _need_const:
    a0 = np.pi
    a1 = 2 * np.pi / np.sqrt(3.0)
    a2 = 2 * np.pi / np.sqrt(8.0)
    c0 = 1 / np.sqrt(4 * np.pi)
    c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
    c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
    d0 = 0.5 / np.sqrt(3.0)

    illu_consts = [a0, a1, a2, c0, c1, c2, d0]

    origin_size = 300
    target_size = 224
    camera_pos = 10.0

def pad_bbox(bbox, img_wh, padding_ratio=0.2):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    size_bb = int(max(width, height) * (1+padding_ratio))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(img_wh[0] - x1, size_bb)
    size_bb = min(img_wh[1] - y1, size_bb)

    return [x1, y1, x1+size_bb, y1+size_bb]


def mymkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

#import sys
#sys.path.append('/home/jasonperhaps/Github/3dmm_fit_v000/core/restyle_encoder')

def align_stylegan2(lm : np.ndarray, img_tensor : torch.tensor, shape2D : torch.tensor, batch_num, img_size=224):
    shape2D_aligned = []
    return_img_tensor = []
    lm_chin = lm[:, 0: 17]  # left-right
    lm_eyebrow_left = lm[:, 17: 22]  # left-right
    lm_eyebrow_right = lm[:, 22: 27]  # left-right
    lm_nose = lm[:, 27: 31]  # top-down
    lm_nostrils = lm[:, 31: 36]  # top-down
    lm_eye_left = lm[:, 36: 42]  # left-clockwise
    lm_eye_right = lm[:, 42: 48]  # left-clockwise
    lm_mouth_outer = lm[:, 48: 60]  # left-clockwise
    lm_mouth_inner = lm[:, 60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = torch.mean(lm_eye_left, 1)
    eye_right = torch.mean(lm_eye_right, 1)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[:, 0]
    mouth_right = lm_mouth_outer[:, 6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - torch.flip(eye_to_mouth, [1]) * torch.tensor([-1, 1], device='cuda:0')
    #x /= torch.hypot(x[:, 0], x[:, 1])
    x = torch.div(x.T, torch.hypot(x[:, 0], x[:, 1])).T
    x = (x.T * torch.maximum(torch.hypot(eye_to_eye[:, 0], eye_to_eye[:, 1]) * 2.0, torch.hypot(eye_to_mouth[:, 0], eye_to_mouth[:, 1]) * 1.8)).T
    y = torch.flip(x, [1]) * torch.tensor([-1, 1], device='cuda:0')
    c = eye_avg + eye_to_mouth * 0.1
    quad = torch.stack([c - x - y, c - x + y, c + x + y, c + x - y]).permute(1,0,2)
    qsize = torch.hypot(x[:, 0], x[:, 1]) * 2

    # read image
    #img_tensor = img_tensor[:, :, :, :3].cpu().numpy()

    shape2D[:, :, 1] = img_size - 1.0 - shape2D[:, :, 1]

    output_size = img_size 
    transform_size = img_size 
    enable_padding = True

    # Pad.
    border = torch.maximum(torch.round(qsize * 0.1), torch.ones_like(qsize) * 3).unsqueeze(1)
    pad = torch.cat((torch.floor(torch.min(quad[:, :, 0], 1, True).values), torch.floor(torch.min(quad[:, :, 1], 1, True).values), torch.ceil(torch.max(quad[:, :, 0], 1, True).values), torch.ceil(torch.max(quad[:, :, 1], 1, True).values)), dim=1)
    zeros = torch.zeros_like(pad[:, 0].unsqueeze(1))
    pad = torch.cat((torch.maximum(-pad[:, 0].unsqueeze(1) + border, zeros), torch.maximum(-pad[:, 1].unsqueeze(1) + border, zeros), torch.maximum(pad[:, 2].unsqueeze(1) - img_size + border, zeros), torch.maximum(pad[:, 3].unsqueeze(1) - img_size + border, zeros)), dim=1)
    pad = torch.maximum(pad, torch.round(qsize * 0.3).unsqueeze(1)).int()

    quad += pad[:, :2].unsqueeze(1)
    shape2D[:, :, 0] += pad[:, 0].unsqueeze(1)
    shape2D[:, :, 1] += pad[:, 1].unsqueeze(1)
    pad = pad.tolist()
    quad = quad.cpu().numpy()
    qsize = qsize.cpu().numpy()
    return_img_tensor = []
    M = []
    for i in range(batch_num):
        q = quad[i]
        q += 0.5
        pts_src = np.float32([q[0], q[1], q[2]])
        pts_dst = np.float32([[0, 0], [0, transform_size], [transform_size, transform_size]])
        M.append(cv2.getAffineTransform(pts_src, pts_dst))
    M = torch.tensor(M, dtype=torch.float32, requires_grad=False, device='cuda:0').permute(0,2,1)
    shape2D = torch.cat((shape2D, torch.ones(shape2D.shape[1], 1).cuda().repeat(batch_num, 1, 1)), 2)
    shape2D = torch.bmm(shape2D, M)
    shape2D[:, :, 1] = img_size - 1.0 - shape2D[:, :, 1]
    for i in range(batch_num):
        p = pad[i]
        q = quad[i]
        qs = qsize[i]

        img = np.uint8(img_tensor[i][:,:,:3].cpu().numpy())
        img = Image.fromarray(img, 'RGB')

        img = np.pad(np.float32(img), ((p[1], p[3]), (p[0], p[2]), (0, 0)), 'reflect')
        #shape2D = np.pad(shape2D, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / p[0], np.float32(w - 1 - x) / p[2]),
                          1.0 - np.minimum(np.float32(y) / p[1], np.float32(h - 1 - y) / p[3]))
        blur = qs * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (q + 0.5).flatten(), PIL.Image.BILINEAR)

        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        return_img_tensor.append(np.array(img))

        # Save aligned image.
    return torch.tensor(return_img_tensor, dtype=torch.float32).cuda(), shape2D