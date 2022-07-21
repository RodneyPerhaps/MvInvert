from argparse import Namespace
import PIL
from PIL import Image
import torch
import torch.nn as nn
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


class ReconModel(nn.Module):
    def __init__(self, face_model, 
                focal=1015, img_size=224, device='cuda:0'):
        super(ReconModel, self).__init__()
        self.facemodel = face_model

        self.focal = focal
        self.img_size = img_size
        self.batch_size = 1

        self.device = torch.device(device)

        R, T = look_at_view_transform(10, 0, 0)
        self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=50,
                                        fov=2*np.arctan(self.img_size//2/self.focal)*180./np.pi)

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.rasterizer = MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
        )
        self.renderer = self.get_renderer(self.device)

        self.kp_inds = torch.tensor(self.facemodel['keypoints']-1).squeeze().long().to(self.device)
        
        meanshape = nn.Parameter(torch.from_numpy(self.facemodel['meanshape'],).float(), requires_grad=False)
        self.register_parameter('meanshape', meanshape)

        idBase = nn.Parameter(torch.from_numpy(self.facemodel['idBase']).float(), requires_grad=False)
        self.register_parameter('idBase', idBase)

        exBase = nn.Parameter(torch.from_numpy(self.facemodel['exBase']).float(), requires_grad=False)
        self.register_parameter('exBase', exBase)

        meantex = nn.Parameter(torch.from_numpy(self.facemodel['meantex']).float(), requires_grad=False)
        self.register_parameter('meantex', meantex)

        texBase = nn.Parameter(torch.from_numpy(self.facemodel['texBase']).float(), requires_grad=False)
        self.register_parameter('texBase', texBase)

        tcoords = nn.Parameter(torch.from_numpy(self.facemodel['tcoords']).float(), requires_grad=False)
        self.register_parameter('tcoords', tcoords)

        tri = nn.Parameter(torch.from_numpy(self.facemodel['tri']).float(), requires_grad=False)
        self.register_parameter('tri', tri)

        point_buf = nn.Parameter(torch.from_numpy(self.facemodel['point_buf']).float(), requires_grad=False)
        self.register_parameter('point_buf', point_buf)

        self.ganinv, self.opts = infer.create_net()
        self.opts.n_iters_per_batch = 9
        self.opts.resize_outputs = False  # generate outputs at full resolution

    def get_renderer(self, device):
        lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]], ambient_color=[[1, 1, 1]],
                             specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            self.rasterizer,
            shader=SoftPhongShader(
                device=device,
                cameras=self.cameras,
                lights=lights,
                blend_params=blend_params
            )
        )
        return renderer

    def Split_coeff(self, coeff):
        id_coeff = coeff[:, :80]  # identity(shape) coeff of dim 80
        ex_coeff = coeff[:, 80:144]  # expression coeff of dim 64
        tex_coeff = coeff[:, 144:224]  # texture(albedo) coeff of dim 80
        angles = coeff[:, 224:227]  # ruler angles(x,y,z) for rotation of dim 3
        gamma = coeff[:, 227:254]  # lighting coeff for 3 channel SH function of dim 27
        translation = coeff[:, 254:]  # translation coeff of dim 3

        return id_coeff, ex_coeff, tex_coeff, angles, gamma, translation

    def Shape_formation(self, id_coeff, ex_coeff):
        n_b = id_coeff.size(0)

        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
                     torch.einsum('ij,aj->ai', self.exBase, ex_coeff) + self.meanshape

        face_shape = face_shape.view(n_b, -1, 3)
        face_shape = face_shape - self.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)

        return face_shape

    def Texture_formation(self, tex_coeff):
        n_b = tex_coeff.size(0)
        face_texture = torch.einsum('ij,aj->ai', self.texBase, tex_coeff) + self.meantex

        face_texture = face_texture.view(n_b, -1, 3)
        return face_texture

    def Compute_norm(self, face_shape):

        face_id = self.tri.long() - 1
        point_id = self.point_buf.long() - 1 
        shape = face_shape
        v1 = shape[:, face_id[:, 0], :]
        v2 = shape[:, face_id[:, 1], :]
        v3 = shape[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)
        empty = torch.zeros((face_norm.size(0), 1, 3), dtype=face_norm.dtype, device=face_norm.device)
        face_norm = torch.cat((face_norm, empty), 1) 
        v_norm = face_norm[:, point_id, :].sum(2)  
        v_norm = v_norm / v_norm.norm(dim=2).unsqueeze(2)

        return v_norm

    def Projection_block(self, face_shape):
        half_image_width = self.img_size // 2
        batchsize = face_shape.shape[0]
        camera_pos = torch.tensor([0.0,0.0,10.0], device=face_shape.device).reshape(1, 1, 3)
        # tensor.reshape(constant([0.0,0.0,10.0]),[1,1,3])
        p_matrix = np.array([self.focal, 0.0, half_image_width, \
                            0.0, self.focal, half_image_width, \
                            0.0, 0.0, 1.0], dtype=np.float32)

        p_matrix = np.tile(p_matrix.reshape(1, 3, 3), [batchsize, 1, 1])
        reverse_z = np.tile(np.reshape(np.array([1.0,0,0,0,1,0,0,0,-1.0], dtype=np.float32),[1,3,3]),
                            [batchsize,1,1])
        
        p_matrix = torch.tensor(p_matrix, device=face_shape.device)
        reverse_z = torch.tensor(reverse_z, device=face_shape.device)
        face_shape = torch.matmul(face_shape,reverse_z) + camera_pos
        aug_projection = torch.matmul(face_shape,p_matrix.permute((0,2,1)))

        #camera_pos = torch.tensor([0.0,0.0,10.0], device=face_shape.device).reshape(1, 1, 3)
        zvertex = 10 - aug_projection[:,:,2:]
        face_projection = aug_projection[:,:,:2]/ \
                        torch.reshape(aug_projection[:,:,2],[batchsize,-1,1])
        return face_projection, zvertex

    @staticmethod
    def Compute_rotation_matrix(angles):
        n_b = angles.size(0)
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1, 1).view(3, n_b, 3, 3)

        if angles.is_cuda: rotXYZ = rotXYZ.cuda()

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)

    @staticmethod
    def Rigid_transform_block(face_shape, rotation, translation):
        face_shape_r = torch.matmul(face_shape, rotation)
        face_shape_t = face_shape_r + translation.view(-1, 1, 3)

        return face_shape_t

    @staticmethod
    def Illumination_layer(face_texture, norm, gamma):

        n_b, num_vertex, _ = face_texture.size()
        n_v_full = n_b * num_vertex
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8

        gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5/ np.sqrt(3.0)

        Y0 = torch.ones(n_v_full).to(gamma.device).float() * a0 * c0
        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(Y0)
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(n_b, num_vertex, 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        return face_color

    def extract_uv_from_input(self, img_tensor, face_shape_t):
        shape2D, _ = self.Projection_block(face_shape_t)
        shape2D = torch.stack([shape2D[:, :, 0], self.img_size-1.0-shape2D[:, :, 1]], dim=2) / self.img_size
        shape2D = shape2D * 2.0 - 1.0
        sample_grid = shape2D.unsqueeze(2)
        vtex = F.grid_sample(img_tensor, sample_grid, align_corners=True, padding_mode='border')
        vtex = vtex.squeeze(3).permute(0,2,1)
        return vtex
    
    def extract_uv_with_img_and_shape2D(self, img_tensor, shape2D):
        shape2D = torch.stack([shape2D[:, :, 0], self.img_size-1.0-shape2D[:, :, 1]], dim=2) / self.img_size
        shape2D = shape2D * 2.0 - 1.0
        sample_grid = shape2D.unsqueeze(2)
        vtex = F.grid_sample(img_tensor, sample_grid, align_corners=True, padding_mode='border')
        vtex = vtex.squeeze(3).permute(0,2,1)
        texture = TexturesVertex(vtex)
        return texture, vtex

    def render_uv(self, uv_tex, batch_num):
        #TSHAPE_SCALE = 1.133
        tShape = torch.cat([self.tcoords*2-1, torch.ones(self.tcoords.shape[0]).unsqueeze(1).cuda()], dim=1).unsqueeze(0)
        tri = self.tri - 1
        mesh = Meshes(tShape.repeat(batch_num, 1, 1), tri.repeat(batch_num, 1, 1), uv_tex)
        rendered_img = self.renderer(mesh)
        rendered_img = torch.clamp(rendered_img, 0.0, 255.0)
        return rendered_img

    def render_fitted_mesh(self, face_shape_t, uv_tex, batch_num):
        tri = self.tri - 1
        mesh = Meshes(face_shape_t, tri.repeat(batch_num, 1, 1), uv_tex)
        rendered_img = self.renderer(mesh)
        rendered_img = torch.clamp(rendered_img, 0.0, 255.0)
        return rendered_img
    
    def render_mesh_from_different_poses(self, vis_hat, shape_dict, vtex, batch_num, img_tensor):
        imgs_dict = {}
        shape2D_dict = {}
        tri = self.tri.long() - 1

        uv_tex = TexturesVertex(vtex)
        tex_diff_poses = {}
        ginv = {}
        samples_W = {}
        samples_T = {}
        T = vtex
        return_tex_color = T

        for k in ['origin', 'bottom', 'right_bottom', 'left_bottom', 'right', 'left', 'front']:
            v = shape_dict[k]
            shape2D, zvertex = self.Projection_block(v)
            mesh = Meshes(v, tri.repeat(batch_num, 1, 1), uv_tex)
            fbuffer, zbuffer, _, _ = self.rasterizer(mesh)
            zbuffer[zbuffer!=-1] = 10 - zbuffer[zbuffer!=-1]

            img = self.renderer(mesh)
            img = torch.clamp(img, 0.0, 255.0)
            face_lms_t = self.get_lms(v, self.kp_inds)
            lms, _ = self.Projection_block(face_lms_t)
            lms = torch.stack([lms[:, :, 0], self.img_size-lms[:, :, 1]], dim=2)
            #visibility = self.Compute_vertex_vis(shape2D, zvertex, zbuffer, fbuffer, tri)
            img, aligned_shape2D = align_stylegan2(lms, img, shape2D.clone(), batch_num)

            imgs_dict[k] = img
            #shape2D_dict[k] = shape2D

            I = imgs_dict[k]
            G = infer.restyle_infer(I[:, :, :, :3].permute(0,3,1,2) / 255.0, self.ganinv, self.opts)
            G = ((G + 1) / 2)
            G[G<0] = 0
            G[G>1] = 1
            G = G * 255
            ginv[k] = G

            tex, T = self.extract_uv_with_img_and_shape2D(G, aligned_shape2D)
            #samples_T[k] = T
            #tex_diff_poses[k] = self.render_uv(tex, batch_num)
            #return_tex_color = vis_hat[k].repeat(3,1,1).permute(1,2,0) * T + (~vis_hat[k]).repeat(3,1,1).permute(1,2,0) * return_tex_color
            #samples_T[k], samples_W[k] = self.Sample_image(shape_dict['front'], shape2D, aligned_shape2D, visibility, G, fbuffer, 1, return_tex_color)
            samples_T['after'] = T
            samples_W['after'] = vis_hat[k].unsqueeze(2)
            return_tex_color = self.Poisson_blending(shape_dict['front'], tri, samples_T, samples_W, POISSON_LAMBDA=0.1)
            samples_T['pre'] = samples_T['after']
            samples_W['pre'] = samples_W['after']
            uv_tex = TexturesVertex(return_tex_color)
            tex_m = self.render_uv(uv_tex, batch_num)
            tex_diff_poses[k] = tex_m

        #return_tex = TexturesVertex(return_tex_color)
        return_tex_v2 = TexturesVertex(return_tex_color)

        complete_uv = self.render_uv(return_tex_v2, batch_num)
        
        return return_tex_v2, return_tex_color, imgs_dict, tex_diff_poses, ginv

    def compute_vis(self, face_shape, fitted_mesh, fitted_norm):
        batch_num = face_shape.shape[0]

        angles =\
        torch.tensor([[ 0.0, -0.71,  0.0],
                [ 0.0,  0.71, 0.0],
                [ 0.0,  0.0, 0.0],
                [-0.2405, -0.71,  0.0],
                [-0.2405,  0.71,  0.0],
                [-0.2405, 0.0, 0.0]], dtype=torch.float32, device='cuda:0', requires_grad=False).unsqueeze(0)

        names = ['right', 'left', 'front', 'right_bottom', 'left_bottom', 'bottom']

        angles = {names[0]: angles[:, 0, :], names[1]: angles[:, 1, :], names[2]: angles[:, 2, :], names[3]: angles[:, 3, :], names[4]: angles[:, 4, :], names[5]: angles[:, 5, :]}

        translation = torch.zeros((1, 3), dtype=torch.float32, requires_grad=False, device='cuda')
        face_norm = self.Compute_norm(face_shape)

        camera = torch.tensor([0, 0, 10], dtype=torch.float32, requires_grad=False, device='cuda')

        vis_dict = {}
        shape_t_dict = {}
        for k, v in angles.items():
            v = v.repeat(1, batch_num, 1).squeeze(0)
            rotation = self.Compute_rotation_matrix(v)
            face_norm_r = face_norm.bmm(rotation)
            face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)

            shape_t_dict[k] = face_shape_t
            face_cam_v = camera - face_shape_t
            shape_normalized = F.normalize(face_cam_v, p=2, dim=2)
            vis_dict[k] = torch.sum(shape_normalized * face_norm_r, dim=2) # S'/||S'||_2 * norm.T

        vis_img = {}
        for k, v in vis_dict.items():
            vis_tex = v.float().repeat(3,1,1).permute(1,2,0)
            vis_tex = TexturesVertex(vis_tex)
            vis_uv_img = self.render_uv(vis_tex, batch_num)
            vis_img[k] = vis_uv_img

        shape_t_dict['origin'] = fitted_mesh
        face_cam_v = camera - fitted_mesh
        shape_normalized = F.normalize(face_cam_v, p=2, dim=2)
        vis_dict['origin'] = torch.sum(shape_normalized * fitted_norm, dim=2) # S'/||S'||_2 * norm.T
        
        vis_hat = {}
        vis_hat_ori = torch.ones(face_shape.shape[1]).cuda()
        for k, v in vis_dict.items():
            if k == 'origin':
                continue
            vis_hat_ori = torch.logical_and(vis_hat_ori, vis_dict['origin'] * 2.0 > v)

        for k, v in vis_dict.items():
            v_hat = torch.ones(face_shape.shape[1]).cuda()
            for k0, v0 in vis_dict.items():
                if k0 == k:
                    continue
                v_hat = torch.logical_and(v_hat, v > v0)
            vis_hat[k] = v_hat

        vis_hat_img = {}
        for k, v in vis_hat.items():
            vis_hat_tex = v.float().repeat(3,1,1).permute(1,2,0)
            vis_hat_tex = TexturesVertex(vis_hat_tex)
            vis_hat_uv_img = self.render_uv(vis_hat_tex, batch_num)
            vis_hat_img[k] = vis_hat_uv_img

        return vis_hat_ori, vis_hat, shape_t_dict, vis_img, vis_hat_img

    def display_from_diff_pose(self, face_shape, com_tex):
        display_faces = {}
        batch_num = face_shape.shape[0]
        angles =\
        torch.tensor([[ 0.0, -30*np.pi/180,  0.0],
                [ 0.0,  30*np.pi/180, 0.0],
                [ 0.0,  -60*np.pi/180, 0.0],
                [ 0.0,  60*np.pi/180, 0.0],
                [ 0.0,  0.0,  0.0]], dtype=torch.float32, device='cuda:0', requires_grad=False).unsqueeze(0)

        names = ['right_30', 'left_30', 'right_60', 'left_60', 'front']
        angles = {names[0]: angles[:, 0, :], names[1]: angles[:, 1, :], names[2]: angles[:, 2, :], names[3]: angles[:, 3, :], names[4]: angles[:, 4, :]}

        translation = torch.zeros((1, 3), dtype=torch.float32, requires_grad=False, device='cuda')
        for k, v in angles.items():
            rotation = self.Compute_rotation_matrix(v)
            face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)

            tri = self.tri - 1
            mesh = Meshes(face_shape_t, tri.repeat(batch_num, 1, 1), com_tex)
            rendered_img = self.renderer(mesh)
            rendered_img = torch.clamp(rendered_img, 0, 255)
            display_faces[k] = rendered_img
        return display_faces

    def get_lms(self, face_shape, kp_inds):
        lms = face_shape[:, kp_inds, :]
        return lms

    def Sample_image(self, face_shape, shape2D, shape2D_aligned, visibility, img, fbuffer, boundarydistT):
        camera = torch.tensor([0.0,0.0,10.0], device=face_shape.device).reshape(1, 1, 3)
        face_cam = camera - face_shape
        face_cam_norm = F.normalize(face_cam, p=2, dim=2)

        shape2D_aligned = torch.stack([shape2D_aligned[:, :, 0], self.img_size-1.0-shape2D_aligned[:, :, 1]], dim=2) / self.img_size
        shape2D_aligned = shape2D_aligned * 2.0 - 1.0
        sample_grid = shape2D_aligned.unsqueeze(2)
        vtex = F.grid_sample(img, sample_grid, align_corners=True, padding_mode='border')
        vtex = vtex.squeeze(3).permute(0,2,1)


        dist2boundary = []
        for i in range(self.batch_size):
            dist2boundary = ndimage.distance_transform_edt(ndimage.binary_fill_holes((fbuffer[i].cpu().numpy()!=-1)[:,:,0]))
        dist2boundary = torch.tensor(dist2boundary, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)

        shape2D = torch.stack([shape2D[:, :, 0], self.img_size-1.0-shape2D[:, :, 1]], dim=2) / self.img_size
        shape2D = shape2D * 2.0 - 1.0
        sample_grid = shape2D.unsqueeze(2)
        vtex2b = F.grid_sample(dist2boundary, sample_grid, align_corners=True, padding_mode='border')
        vtex2b = vtex2b.squeeze(3).permute(0,2,1)

        vnormals = self.Compute_norm(face_shape)
        weight = torch.sum(vnormals*face_cam_norm, 2).unsqueeze(2)
        weight[weight<0] = 0
        weight = weight * visibility
        weight[vtex2b<boundarydistT] = 0

        #vtex = (weight > 0).repeat(3,1,1).permute(2,1,0) * vtex + (weight == 0).repeat(3,1,1).permute(2,1,0) * face_color

        return vtex, weight

    def Compute_vertex_vis(self, shape2D, zvertex, zbuffer, fbuffer, tri):
        shape2D = torch.stack([shape2D[:, :, 0], self.img_size-1.0-shape2D[:, :, 1]], dim=2) / self.img_size
        shape2D = shape2D * 2.0 - 1.0
        sample_grid = shape2D.unsqueeze(2)
        Vz_buffer = F.grid_sample(zbuffer.permute(0,3,1,2), sample_grid, align_corners=True, padding_mode='border')
        Vz_buffer = Vz_buffer.squeeze(3).permute(0,2,1)
        invalid_index = Vz_buffer == -1
        Vz_buffer[invalid_index] = 0
        zvertex[invalid_index] = 0


        visibility = []
        for i in range(self.batch_size):
            vis = torch.zeros(shape2D.shape[1], 1).cuda().type(torch.BoolTensor)
            f = torch.unique(fbuffer[i][fbuffer[i]!=-1])
            vis[torch.unique(torch.cat((tri[f, 0], tri[f, 1], tri[f, 2]), dim=0))] = True
            diffs = torch.absolute(zvertex[i][vis] - Vz_buffer[i][vis])
            diffs, _ = torch.sort(diffs)
            t = diffs[round(0.65*diffs.shape[0])]
            vis[torch.absolute(zvertex[i] - Vz_buffer[i]) < t] = True
            visibility.append(vis)
        return torch.stack(visibility).cuda()

    def Poisson_blending(self, face_shape, tri, samples_T, samples_W, POISSON_LAMBDA=0.1):
        n = face_shape.shape[1]
        m = tri.shape[0]
        XF = lambda i: face_shape[:, tri[:, i], :]
        Na = torch.cross(XF(1) - XF(0), XF(2) - XF(0))
        Area = Na.norm(dim=2) / 2
        N = F.normalize(Na, p=2, dim=2)
        for i in range(self.batch_size):
            I = []
            J = []
            V = []
            for k in range(0, 3):
                s = (k + 1) % 3
                t = (k + 2) % 3
                wk = cupy.fromDlpack(to_dlpack(torch.cross(XF(t)[i]-XF(s)[i], N[i])))
                I.append(cupy.arange(0,m))
                J.append(cupy.fromDlpack(to_dlpack(tri[:, k])))
                V.append(wk)
            I = cupy.concatenate(I, 0)
            J = cupy.concatenate(J, 0)
            V = cupy.concatenate(V, 0)
            return_color = []
            dA = sparse.spdiags(cupy.fromDlpack(to_dlpack(1/(2*Area[i]))), 0, m, m)
            GradMat = {}
            for j in range(0, 3):
                GradMat[j] = dA*sparse.coo_matrix((V[:, j], (I, J)), shape=(m,n))
            y = cupy.zeros((m*3, 3))
            Tweight = []
            rhs = {}
            for k, v in samples_W.items():
                Tweight.append(torch.min(torch.cat((v[i][tri[:, 0]], v[i][tri[:, 1]], v[i][tri[:, 2]]), dim=1), dim=1).values)
                rhs[k] = vstack((GradMat[0], GradMat[1], GradMat[2])).dot(cupy.fromDlpack(to_dlpack(samples_T[k][i])))
            Tweight = torch.stack(Tweight).cuda().T
            _, idx = torch.max(Tweight, 1)
            count = 0
            for k, v in rhs.items():
                mask = torch.bitwise_and(idx == count, Tweight[:, count] > 0)
                mask = cupy.fromDlpack(to_dlpack(mask.repeat(1,3).squeeze(0)))
                y[mask > 0] = rhs[k][mask>0]
                count += 1

            A = vstack((GradMat[0], GradMat[1], GradMat[2], sparse.coo_matrix((cupy.fromDlpack(to_dlpack(POISSON_LAMBDA*(samples_W['pre'][i]>0)[:,0])), (cupy.arange(0, n), cupy.arange(0, n))), shape=(n,n))))
            y = cupy.vstack((y, POISSON_LAMBDA * cupy.fromDlpack(to_dlpack(samples_T['pre'][i]))))
            A_hat = A.T * A
            A_hat = A_hat.tocoo().get()
            A_hat = torch.sparse.DoubleTensor(torch.LongTensor(np.vstack((A_hat.row, A_hat.col))), torch.DoubleTensor(A_hat.data), torch.Size(A_hat.shape)).unsqueeze(0)
            y_hat = A.T * y
            y_hat = torch.tensor(y_hat.get(), dtype=torch.float64).unsqueeze(0)
            return_color.append(solve(A_hat, y_hat).float().squeeze(0))

        return torch.stack(return_color).cuda()

    def reconstruction(self, coeff):
        # The image size is 224 * 224
        # face reconstruction with coeff and BFM model
        id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = self.Split_coeff(coeff)

        # compute face shape
        face_shape = self.Shape_formation(id_coeff, ex_coeff)
        # compute vertex texture(albedo)
        face_texture = self.Texture_formation(tex_coeff)

        # vertex normal
        face_norm = self.Compute_norm(face_shape)
        # rotation matrix
        rotation = self.Compute_rotation_matrix(angles)
        face_norm_r = face_norm.bmm(rotation)
        face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)

        face_lms_t = self.get_lms(face_shape_t, self.kp_inds)
        lms, _ = self.Projection_block(face_lms_t)
        lms = torch.stack([lms[:, :, 0], self.img_size-lms[:, :, 1]], dim=2)
        # compute vertex color using SH function lighting approximation
        face_color = self.Illumination_layer(face_texture, face_norm_r, gamma)

        return face_shape, face_shape_t, face_color, face_norm_r, lms, face_texture#, landmarks_2d, z_buffer, angles, translation, gamma

    def forward(self, coeff, img_tensor=None):
        self.batch_size = coeff.shape[0]

        face_shape, face_shape_t, face_color, face_norm_r, lms, face_texture = self.reconstruction(coeff)

        if img_tensor is not None:
            vis_hat_ori, vis_hat, shape_dict, vis_img, vis_hat_img = self.compute_vis(face_shape, face_shape_t, face_norm_r)

            face_color = torch.clamp(face_color, 0, 255)
            vtex0 = self.extract_uv_from_input(img_tensor, face_shape_t)
            vtex0 = vis_hat_ori.repeat(3,1,1).permute(1,2,0) * vtex0 + (~vis_hat_ori).repeat(3,1,1).permute(1,2,0) * face_color 

            return_tex, return_tex_color, imgs, tex_diff_poses, ginv = self.render_mesh_from_different_poses(vis_hat, shape_dict, vtex0, self.batch_size, img_tensor)
            completed_tex = self.render_uv(return_tex, self.batch_size)

            display_faces = self.display_from_diff_pose(face_shape, return_tex)

            return imgs, tex_diff_poses, ginv, display_faces, completed_tex, face_shape, self.tri - 1, return_tex_color / 255.0, vis_img, vis_hat_img
        
        face_color = TexturesVertex(face_color)

        tri = self.tri - 1
        mesh = Meshes(face_shape_t, tri.repeat(self.batch_size, 1, 1), face_color)
        rendered_img = self.renderer(mesh)
        rendered_img = torch.clamp(rendered_img, 0, 255)
        return rendered_img, lms, face_texture, mesh, 

