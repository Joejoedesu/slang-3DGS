from plyfile import PlyData, PlyElement
import numpy as np
import torch
from torch import nn
import os
import math
import matplotlib.pyplot as plt
import struct
import threading
import time

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def compute_color_from_sh(plt_scene, cam_loc):
    shs_view = plt_scene._feature.transpose(1, 2).view(-1, 3, (plt_scene.max_sh_degree+1)**2)
    cam_loc_torch = torch.tensor(cam_loc, dtype=torch.float)
    xyz_torch = torch.tensor(plt_scene._xyz, dtype=torch.float)
    dir_pp = (xyz_torch - cam_loc_torch.repeat(plt_scene._feature.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(plt_scene.max_sh_degree, shs_view, dir_pp_normalized)
    print(sh2rgb.shape, sh2rgb[0])
    colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
    return colors


class Plt3DGS:
    def __init__(self, max_sh_degree, path):
        self.max_sh_degree = max_sh_degree

        self._width = 980
        self._height = 545
        self.FoVx = 1.4028140929797814
        self.FoVy = 0.8753571332164317
        self.znear = 0.01
        self.zfar = 100.0

        self.load_ply(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = xyz
        # self._features_dc = features_dc
        # self._features_rest = features_extra
        self._opacity = opacities
        self._scaling = scales
        self._rotation = rots
        # self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float).transpose(1, 2))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float).transpose(1, 2))
        self._feature = torch.cat([self._features_dc, self._features_rest], dim=1)
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # active_sh_degree = self.max_sh_degree
    
    def get_color(self, cam_loc):
        return compute_color_from_sh(self, cam_loc)

def sample_depth():
    width = 980
    height = 545
    FoVx = 1.4028140929797814
    FoVy = 0.8753571332164317
    znear = 0.01
    zfar = 100.0
    # x right y down z forward

    # 104
    # ViewM_T = [[-0.6199, -0.0939, -0.7791,  0.0000],
    #     [ 0.1741,  0.9516, -0.2532,  0.0000],
    #     [ 0.7651, -0.2926, -0.5735,  0.0000],
    #     [-0.4240,  0.9184,  3.0589,  1.0000]]
    # ViewM_T = np.array(ViewM_T)
    # ViewM =ViewM_T.transpose()
    # # print(ViewM)
    # projM = [[ 1.1839,  0.0000,  0.0000,  0.0000],
    #     [ 0.0000,  2.1370,  0.0000,  0.0000],
    #     [ 0.0000,  0.0000,  1.0001,  1.0000],
    #     [ 0.0000,  0.0000, -0.0100,  0.0000]]
    # projM = np.array(projM)

    # full_proj_transform = np.matmul(projM, ViewM)
    # full_proj_transform_copy = [[-0.7338, -0.2007, -0.7791, -0.7791],
    #     [ 0.2061,  2.0336, -0.2533, -0.2532],
    #     [ 0.9058, -0.6253, -0.5736, -0.5735],
    #     [-0.5019,  1.9626,  3.0492,  3.0589]]
    # full_proj_transform_copy = np.array(full_proj_transform_copy)


    # 195
    ViewM_T = [[ 0.5077,  0.0065, -0.8615,  0.0000],
        [ 0.0831,  0.9949,  0.0564,  0.0000],
        [ 0.8575, -0.1002,  0.5046,  0.0000],
        [-1.5442, -0.0462,  2.3823,  1.0000]]
    ViewM_T = np.array(ViewM_T)
    ViewM =ViewM_T.transpose()
    # print(ViewM)
    projM = [[ 1.1839,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  2.1370,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  1.0001,  1.0000],
        [ 0.0000,  0.0000, -0.0100,  0.0000]]
    projM = np.array(projM)

    pos = [2.8366644028087706, 0.03986117836440167, 0.11750136379034062]
    pos = np.array(pos)

    full_proj_transform = np.matmul(projM, ViewM)
    full_proj_transform_copy = [[ 0.6010,  0.0138, -0.8616, -0.8615],
        [ 0.0984,  2.1262,  0.0564,  0.0564],
        [ 1.0152, -0.2142,  0.5046,  0.5046],
        [-1.8281, -0.0987,  2.3726,  2.3823]]
    full_proj_transform_copy = np.array(full_proj_transform_copy)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    ply_path = os.path.join(cur_dir, "models/train/point_cloud/iteration_30000/point_cloud.ply")
    plt3dgs = Plt3DGS(3, ply_path)
    print(plt3dgs._xyz.shape)

    colors = compute_color_from_sh(plt3dgs, pos)
    colors = colors.cpu().detach().numpy()
    print(colors.shape)

    num_points = plt3dgs._xyz.shape[0]
    output = np.zeros((num_points, 3))
    # start timer
    start = time.time()
    # print(plt3dgs._xyz[0])

    # canvas = np.zeros((3000, 3000))
    for idx in range(0, num_points):
        point = plt3dgs._xyz[idx]
        point = np.append(point, 1)
        point1 = np.matmul(ViewM, point)
        point2 = np.matmul(projM, point1)
        # point1 = np.matmul(full_proj_transform, point)
        point3 = point2 / point2[3]
        output[idx] = point3[0:3]
        x = math.floor(point3[0] + 1500)
        y = math.floor(point3[1] + 1500)
        if x < 0 or x >= 3000 or y < 0 or y >= 3000:
            continue
        # canvas[x, y] = 1
        # count += 1

    end = time.time()
    print("Time taken in seconds: ", end - start)
    print("finish")

    g_in_scene = []

    canvas = np.zeros((width, height, 3))
    min_depth = 1.0e100
    max_depth = -1.0e100
    for idx in range(0, num_points):
        if output[idx][0] < -100 or output[idx][0] > 100 or output[idx][1] < -100 or output[idx][1] > 100:
            continue
        g_in_scene.append(idx)
        x = int((output[idx][0]/100 + 1) * 0.5 * width)
        y = int((output[idx][1]/100 + 1) * 0.5 * height)
        # canvas[x, y] = [output[idx][2], output[idx][2], output[idx][2]]
        canvas[x, y] = [colors[idx][0], colors[idx][1], colors[idx][2]]

        # if output[idx][2] < min_depth:
        #     min_depth = output[idx][2]
        # if output[idx][2] > max_depth:
        #     max_depth = output[idx][2]
        # if x+1 < width:
        #     canvas[x+1, y] = [1, 1, 1]
        # if y+1 < height:
        #     canvas[x, y+1] = [1, 1, 1]
        # if x+1 < width and y+1 < height:
        #     canvas[x+1, y+1] = [1, 1, 1]
        # if x-1 >= 0:
        #     canvas[x-1, y] = [1, 1, 1]
        # if y-1 >= 0:
        #     canvas[x, y-1] = [1, 1, 1]
        # if x-1 >= 0 and y-1 >= 0:
        #     canvas[x-1, y-1] = [1, 1, 1]
    
    print(len(g_in_scene))
    # print(min_depth, max_depth)
    # canvas = (canvas - min_depth)/(max_depth - min_depth)
    # canvas = np.sqrt(canvas)
    print(canvas[g_in_scene[0]])
    # print(output[g_in_scene[0]])
    canvas = np.rot90(canvas)
    plt.imshow(canvas)
    plt.show()

    # depth_map = np.zeros((height, width))
    # xy_distribution = np.zeros((200, 200))
    # for idx, point in enumerate(plt3dgs._xyz):
    #     x = math.floor(point[0] + 100)
    #     y = math.floor(point[1] + 100)
    #     xy_distribution[x, y] = 1
    # xy_distribution[0, 0] = 1
    # xy_distribution[1, 0] = 1
    # xy_distribution[2, 0] = 1
    # xy_distribution[3, 0] = 1
    
    # plt.imshow(xy_distribution)
    # plt.show()
        # point = np.append(point, 1)
        # point = np.matmul(ViewM, point)
        # point = np.matmul(projM, point)
        # point = point / point[3]
        # x = int((point[0] + 1) * 1000)
        # y = int((point[1] + 1) * 1000)
        # xy_distribution[x, y] += 1
        # # depth_map[y, x] = point[2]

sample_depth()

# def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
#     """Read and unpack the next bytes from a binary file.
#     :param fid:
#     :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
#     :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
#     :param endian_character: Any of {@, =, <, >, !}
#     :return: Tuple of read and unpacked values.
#     """
#     data = fid.read(num_bytes)
#     return struct.unpack(endian_character + format_char_sequence, data)

# # \models\train
# def read_extrinsics_binary(path_to_model_file):
#     """
#     see: src/base/reconstruction.cc
#         void Reconstruction::ReadImagesBinary(const std::string& path)
#         void Reconstruction::WriteImagesBinary(const std::string& path)
#     """
#     images = {}
#     with open(path_to_model_file, "rb") as fid:
#         num_reg_images = read_next_bytes(fid, 8, "Q")[0]
#         for _ in range(num_reg_images):
#             binary_image_properties = read_next_bytes(
#                 fid, num_bytes=64, format_char_sequence="idddddddi")
#             image_id = binary_image_properties[0]
#             qvec = np.array(binary_image_properties[1:5])
#             tvec = np.array(binary_image_properties[5:8])
#             camera_id = binary_image_properties[8]
#             image_name = ""
#             current_char = read_next_bytes(fid, 1, "c")[0]
#             while current_char != b"\x00":   # look for the ASCII 0 entry
#                 image_name += current_char.decode("utf-8")
#                 current_char = read_next_bytes(fid, 1, "c")[0]
#             num_points2D = read_next_bytes(fid, num_bytes=8,
#                                            format_char_sequence="Q")[0]
#             x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
#                                        format_char_sequence="ddq"*num_points2D)
#             xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
#                                    tuple(map(float, x_y_id_s[1::3]))])
#             point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
#             images[image_id] = Image(
#                 id=image_id, qvec=qvec, tvec=tvec,
#                 camera_id=camera_id, name=image_name,
#                 xys=xys, point3D_ids=point3D_ids)
#     return images

# def fov2focal(fov, pixels):
#     return pixels / (2 * math.tan(fov / 2))

# def focal2fov(focal, pixels):
#     return 2*math.atan(pixels/(2*focal))

# def read_intrinsics_binary(path_to_model_file):
#     """
#     see: src/base/reconstruction.cc
#         void Reconstruction::WriteCamerasBinary(const std::string& path)
#         void Reconstruction::ReadCamerasBinary(const std::string& path)
#     """
#     cameras = {}
#     with open(path_to_model_file, "rb") as fid:
#         num_cameras = read_next_bytes(fid, 8, "Q")[0]
#         for _ in range(num_cameras):
#             camera_properties = read_next_bytes(
#                 fid, num_bytes=24, format_char_sequence="iiQQ")
#             camera_id = camera_properties[0]
#             model_id = camera_properties[1]
#             model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
#             width = camera_properties[2]
#             height = camera_properties[3]
#             num_params = CAMERA_MODEL_IDS[model_id].num_params
#             params = read_next_bytes(fid, num_bytes=8*num_params,
#                                      format_char_sequence="d"*num_params)
#             cameras[camera_id] = Camera(id=camera_id,
#                                         model=model_name,
#                                         width=width,
#                                         height=height,
#                                         params=np.array(params))
#         assert len(cameras) == num_cameras
#     return cameras


# def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
#     cam_infos = []
#     for idx, key in enumerate(cam_extrinsics):

#         extr = cam_extrinsics[key]
#         intr = cam_intrinsics[extr.camera_id]
#         height = intr.height
#         width = intr.width

#         uid = intr.id
#         R = np.transpose(qvec2rotmat(extr.qvec))
#         T = np.array(extr.tvec)

#         if intr.model=="SIMPLE_PINHOLE":
#             focal_length_x = intr.params[0]
#             FovY = focal2fov(focal_length_x, height)
#             FovX = focal2fov(focal_length_x, width)
#         elif intr.model=="PINHOLE":
#             focal_length_x = intr.params[0]
#             focal_length_y = intr.params[1]
#             FovY = focal2fov(focal_length_y, height)
#             FovX = focal2fov(focal_length_x, width)
#         else:
#             assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

#         image_path = os.path.join(images_folder, os.path.basename(extr.name))
#         image_name = os.path.basename(image_path).split(".")[0]
#         image = Image.open(image_path)

#         cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                               image_path=image_path, image_name=image_name, width=width, height=height)
#         cam_infos.append(cam_info)
#     return cam_infos


# def load_params(path):
#     llffhold = 8
#     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
#     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
#     cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#     cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

#     reading_dir = "images" if images == None else images
#     cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
#     cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

#     if eval:
#         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
#         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []
#     print(train_cam_infos[0])


