import slangtorch
import torch
import numpy as np
import time
import os
import math
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from torch.autograd import Function
import torch.nn.functional as F
import threading
import random
from PIL import Image

from loader_3DGS import get_sample_camera, Plt3DGS
from loss_util import l1_loss, ssim, get_expon_lr_func, psnr_error

# https://shader-slang.com/slang/user-guide/a1-02-slangpy.html

image_dir_root = "images"
gaussian_dir = "gaussian"
blender_dir = "blender"

image_gaussian_dir = os.path.join(image_dir_root, gaussian_dir)
image_blender_dir = os.path.join(image_dir_root, blender_dir)


def update_learning_rate(optimizer, lr_func, step):
    lr = lr_func(step)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'means':
            param_group['lr'] = lr
            return lr

def setup_1G_rasterizer():
    rasterizer_1G = slangtorch.loadModule("rasterize_1_G_color.slang")

    class Rasterizer1G(Function):
        @staticmethod
        def forward(ctx, width, height, mean, s, r, viewM, projM, color):
            output = torch.zeros((width, height, 4), dtype=torch.float).cuda()
            kernel_with_args = rasterizer_1G.rasterize(mean=mean, s=s, r=r, viewM=viewM, projM=projM, color=color, output=output)
            kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((width + 15)//16, (height + 15)//16, 1))
            ctx.viewM = viewM
            ctx.projM = projM
            ctx.save_for_backward(mean, s, r, color, output)
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            mean, s, r, color, output = ctx.saved_tensors
            viewM = ctx.viewM
            projM = ctx.projM
            grad_mean = torch.zeros_like(mean)
            grad_s = torch.zeros_like(s)
            grad_r = torch.zeros_like(r)
            grad_color = torch.zeros_like(color)

            width, height = grad_output.shape[0], grad_output.shape[1]
            grad_output = grad_output.contiguous()

            kernel_with_args = rasterizer_1G.rasterize.bwd(
                mean=(mean, grad_mean),
                s=(s, grad_s),
                r=(r, grad_r),
                viewM=viewM,
                projM=projM,
                color=(color, grad_color),
                output=(output, grad_output))
            kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((width + 15)//16, (height + 15)//16, 1))

            return None, None, grad_mean, grad_s, grad_r, None, None, grad_color

    return Rasterizer1G()


def rasterize_1_G_color_slang(mean, s, r, came_params, color):
    mean = mean.cuda()
    s = s.cuda()
    r = r.cuda()
    viewM = came_params.M_view.cuda()
    projM = came_params.M_proj.cuda()
    color = color.cuda()

    rasterizer = setup_1G_rasterizer()
    sample = rasterizer.apply(came_params.width, came_params.height, mean, s, r, viewM, projM, color)
    sample = sample.cpu().numpy()
    # print(sample[0, 0])
    # print(sample[1, 0])
    # print(sample[2, 0])
    sample = np.rot90(sample)
    # plt.imshow(sample)
    # plt.show()
    return sample

def setup_nG_rasterizer():
    rasterizer_nG = slangtorch.loadModule("rasterize_n_G_color.slang")

    class RasterizernG(Function):
        @staticmethod
        def forward(ctx, width, height, mean, s, r, viewM, projM, view_angle, gaussian_range, color):
            output = torch.zeros((width, height, 4), dtype=torch.float).cuda().requires_grad_(True)
            kernel_with_args = rasterizer_nG.rasterize(mean=mean, s=s, r=r, viewM=viewM, projM=projM, color=color, view_angle=view_angle, gaussian_range=gaussian_range, output=output)
            kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((width + 15)//16, (height + 15)//16, 1))
            ctx.viewM = viewM
            ctx.projM = projM
            ctx.view_angle = view_angle
            ctx.gaussian_range = gaussian_range
            ctx.save_for_backward(mean, s, r, color, output)
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            mean, s, r, color, output = ctx.saved_tensors
            viewM = ctx.viewM
            projM = ctx.projM
            view_angle = ctx.view_angle
            gaussian_range = ctx.gaussian_range
            grad_mean = torch.zeros_like(mean)
            grad_s = torch.zeros_like(s)
            grad_r = torch.zeros_like(r)
            grad_color = torch.zeros_like(color)

            width, height = grad_output.shape[0], grad_output.shape[1]
            grad_output = grad_output.contiguous()

            kernel_with_args = rasterizer_nG.rasterize.bwd(
                mean=(mean, grad_mean),
                s=(s, grad_s),
                r=(r, grad_r),
                viewM=viewM,
                projM=projM,
                view_angle=view_angle,
                gaussian_range=gaussian_range,
                color=(color, grad_color),
                output=(output, grad_output))
            kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((width + 15)//16, (height + 15)//16, 1))

            return None, None, grad_mean, grad_s, grad_r, None, None, None, None, grad_color

    return RasterizernG()

def rasterize_n_G_color_slang(means, s, r, came_params, color):
    means_came = torch.zeros_like(means)
    for i in range(means.shape[0]):
        means_came[i] = came_params.M_view @ means[i]
    
    # sort the means by depth
    indices = torch.argsort(means_came[:, 2], descending=True)
    s = s[indices]
    r = r[indices]
    color = color[indices]
    means = means[indices]

    means = means.cuda()
    s = s.cuda()
    r = r.cuda()
    viewM = came_params.M_view.cuda()
    projM = came_params.M_proj.cuda()
    color = color.cuda()
    view_angle_rad = came_params.fov * math.pi / 180
    tanfovx = math.tan(view_angle_rad * 0.5)
    tanfovy = math.tan(view_angle_rad * 0.5)
    focal_y = came_params.height / (2.0 * tanfovy)
    focal_x = came_params.width / (2.0 * tanfovx)
    view_angle = torch.Tensor([tanfovx, tanfovy, focal_x, focal_y]).cuda()
    # print(view_angle)
    rasterizer = setup_nG_rasterizer()
    # print(means.shape)
    # print(s.shape)
    # print(r.shape)
    gaussian_range = means.shape[0]
    assert gaussian_range <= 5 # only support 5 gaussians

    sample = rasterizer.apply(came_params.width, came_params.height, means, s, r, viewM, projM, view_angle, gaussian_range, color)
    sample = sample.cpu().numpy()
    # print(sample[13, 13])
    # print(sample[13, 12])
    # print(sample[1, 3])
    # print(sample[2, 3])
    sample = np.rot90(sample)
    plt.imshow(sample)
    plt.show()
    return sample

def setup_scene_rasterizer():
    rasterizer_scene = slangtorch.loadModule("rasterize_n_G_scene.slang")

    class RasterizerG_scene(Function):
        @staticmethod
        def forward(ctx, width, height, mean, s, r, viewM, projM, view_angle, gaussian_range, color, pre_input):                    
            output = torch.zeros((width, height, 5), dtype=torch.float).cuda()
            kernel_with_args = rasterizer_scene.rasterize(mean=mean, s=s, r=r, viewM=viewM, projM=projM, color=color, view_angle=view_angle, gaussian_range=gaussian_range, pre_input=pre_input, output=output)
            kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((width + 15)//16, (height + 15)//16, 1))
            ctx.viewM = viewM
            ctx.projM = projM
            ctx.view_angle = view_angle
            ctx.gaussian_range = gaussian_range
            ctx.pre_input = pre_input
            ctx.save_for_backward(mean, s, r, color, output)
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            mean, s, r, color, output = ctx.saved_tensors
            viewM = ctx.viewM
            projM = ctx.projM
            view_angle = ctx.view_angle
            gaussian_range = ctx.gaussian_range
            pre_input = ctx.pre_input
            grad_mean = torch.zeros_like(mean)
            grad_s = torch.zeros_like(s)
            grad_r = torch.zeros_like(r)
            grad_color = torch.zeros_like(color)

            width, height = grad_output.shape[0], grad_output.shape[1]
            grad_output = grad_output.contiguous()

            kernel_with_args = rasterizer_scene.rasterize.bwd(
                mean=(mean, grad_mean),
                s=(s, grad_s),
                r=(r, grad_r),
                viewM=viewM,
                projM=projM,
                view_angle=view_angle,
                gaussian_range=gaussian_range,
                color=(color, grad_color),
                pre_input=pre_input,
                output=(output, grad_output))
            kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((width + 15)//16, (height + 15)//16, 1))

            return None, None, grad_mean, grad_s, grad_r, None, None, None, None, grad_color, None

    return RasterizerG_scene()

def rasterize_n_G_scene_slang(means, s, r, came_params, color):
    means_came = torch.zeros_like(means)
    for i in range(means.shape[0]):
        means_came[i] = came_params.M_view @ means[i]
    
    # sort the means by depth
    indices = torch.argsort(means_came[:, 2], descending=True)
    s = s[indices]
    r = r[indices]
    color = color[indices]
    means = means[indices]

    means = means.cuda()
    s = s.cuda()
    r = r.cuda()
    viewM = came_params.M_view.cuda()
    projM = came_params.M_proj.cuda()
    color = color.cuda()
    view_angle_rad = came_params.fov * math.pi / 180
    tanfovx = math.tan(view_angle_rad * 0.5)
    tanfovy = math.tan(view_angle_rad * 0.5)
    focal_y = came_params.height / (2.0 * tanfovy)
    focal_x = came_params.width / (2.0 * tanfovx)
    view_angle = torch.Tensor([tanfovx, tanfovy, focal_x, focal_y]).cuda()
    rasterizer = setup_scene_rasterizer()
    gaussian_range = means.shape[0]
    assert gaussian_range <= 5 # only support 5 gaussians
    
    pre_input = torch.zeros((came_params.width, came_params.height, 5), dtype=torch.float)
    # last channel set to 1
    pre_input[:, :, 4] = 1
    pre_input = pre_input.cuda()

    cur_id = 0
    max_gaussian = 2
    while cur_id < means.shape[0]:
        end_id = min(cur_id + max_gaussian, means.shape[0])
        means_i = means[cur_id:end_id]
        s_i = s[cur_id:end_id]
        r_i = r[cur_id:end_id]
        color_i = color[cur_id:end_id]
        pre_input = rasterizer.apply(came_params.width, came_params.height, means_i, s_i, r_i, viewM, projM, view_angle, end_id-cur_id, color_i, pre_input)
        cur_id = end_id


    pre_input = pre_input.cpu().numpy()

    # get the first 4 channels
    pre_input = pre_input[:, :, :4]
    # print(sample[13, 13])
    # print(sample[13, 12])
    # print(sample[1, 3])
    # print(sample[2, 3])
    sample = np.rot90(pre_input)
    plt.imshow(sample)
    plt.show()
    return sample

class CameraParams:
    def __init__(self, eye, center, up, fov, aspect, near, far, width, height):
        self.eye = eye
        self.center = center
        self.up = up
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.width = width
        self.height = height
        self.compute_view_matrix()
        self.compute_proj_matrix()
        self.compute_view_proj_matrix()

    def compute_view_matrix(self, pos_z = False):
        z = (self.eye - self.center) / torch.norm(self.eye - self.center)  # looking into the -z direction
        x = torch.cross(self.up, z)
        x = x / torch.norm(x)
        y = torch.cross(z, x)
        y = y / torch.norm(y)
        # print("z: ", z)
        # print("x: ", x)
        # print("y: ", y)
        R = torch.Tensor([[x[0], x[1], x[2], 0],
                               [y[0], y[1], y[2], 0],
                               [z[0], z[1], z[2],0],
                               [0, 0, 0, 1]])
        T = torch.Tensor([[1, 0, 0, -self.eye[0]],
                        [0, 1, 0, -self.eye[1]],
                        [0, 0, 1, -self.eye[2]],
                        [0, 0, 0, 1]])
        self.M_view = R @ T
        # print("view matrix: ", self.M_view )
        self.M_view

    def compute_proj_matrix(self):
        fov_rad = self.fov * math.pi / 180
        f = 1 / math.tan(fov_rad / 2)
        # print("f: ", f)
        self.M_proj = torch.Tensor([[f / self.aspect, 0, 0, 0],
                               [0, f, 0, 0],
                               [0, 0, (self.far + self.near) / (self.near - self.far), 2 * self.far * self.near / (self.near - self.far)],
                               [0, 0, -1, 0]])
        # print("proj matrix: ", self.M_proj)
        self.M_proj
    
    def compute_view_proj_matrix(self):
        # assert hasattr(self, 'M_view') and hasattr(self, 'M_proj')
        self.M_view_proj = self.M_proj @ self.M_view
        return self.M_view_proj


def quart_to_rot(q):
    # normalize the quaternion
    q = q / torch.norm(q)
    # calculate the rotation matrix
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    R = torch.Tensor([[1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
                      [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
                      [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]])
    return R

def s_to_scale(s):
    return torch.Tensor([[s[0], 0, 0],
                         [0, s[1], 0],
                         [0, 0, s[2]]])

def compute_cov2D(mean, camparam, Vrk, debug=False):
    viewM = camparam.M_view
    t = viewM @ mean
    if debug:
        print("t: ", t)
    l = torch.norm(t)
    J = torch.Tensor([[1/t[2], 0, -t[0]/t[2]**2],
                      [0, 1/t[2], -t[1]/t[2]**2],
                      [0,0,0]])
    
    # view_angle_rad = camparam.fov * math.pi / 180
    # tanfovx = math.tan(view_angle_rad * 0.5)
    # tanfovy = math.tan(view_angle_rad * 0.5)
    # # focal_y = camparam.height / (2.0 * tanfovy)
    # # focal_x = camparam.width / (2.0 * tanfovx)
    # focal_y = 1.0 / (2.0 * tanfovy)
    # focal_x = 1.0  / (2.0 * tanfovx)
    # limx = 1.3 * tanfovx
    # limy = 1.3 * tanfovy
    # txtz = t[0] / t[2]
    # tytz = t[1] / t[2]
    # print("previous t0 t1 ", t[0], t[1])
    # t[0] = min(limx, max(-limx, txtz)) * t[2]
    # t[1] = min(limy, max(-limy, tytz)) * t[2]
    # print("after t0 t1 ", t[0], t[1])
    # print("focal x, y ", focal_x, focal_y)

    # J_ = torch.Tensor([[focal_x/t[2], 0, -(focal_x*t[0])/t[2]**2],
    #                   [0, focal_y/t[2], -(focal_y * t[1])/t[2]**2],
    #                   [0,0,0]])
    # print("J: ", J)
    # print("J_: ", J_)


    if debug:
        print("J: ", J)
    W = torch.Tensor([[viewM[0][0], viewM[0][1], viewM[0][2]],
                      [viewM[1][0], viewM[1][1], viewM[1][2]],
                      [viewM[2][0], viewM[2][1], viewM[2][2]]])
    if debug:
        print("W: ", W)
    T = J @ W
    cov2D = T @ Vrk @ T.t()
    t_cov2D = torch.Tensor([[cov2D[0][0] + 0.3, cov2D[0][1]],
                            [cov2D[1][0], cov2D[1][1] + 0.3]])
    if debug:
        print("t_cov2D: ", t_cov2D)
    return t_cov2D                


def rast_1_G_color(mean, s, r, came_params, color, debug=False):
    height = came_params.height
    width = came_params.width
    R = quart_to_rot(r)
    S = s_to_scale(s)
    cov3D = R @ S @ S.t() @ R.t()
    if debug:
        print("R: ", R)
        print("S: ", S)
        print("cov3D: ", cov3D)

    # cov2D = compute_cov2D(mean, came_params.M_view, cov3D, debug=debug)
    cov2D = compute_cov2D(mean, came_params, cov3D, debug=debug)
    if debug:
        print("cov2D: ", cov2D)

    det = torch.det(cov2D)
    # det_c = cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[1][0]
    # print("det: ", det)
    # print("det_c: ", det_c)
    # assert det_c == det.item()

    det_inv = 1.0 / det
    conic = [cov2D[1][1] * det_inv, -cov2D[0][1] * det_inv, cov2D[0][0] * det_inv]
    mid = 0.5 * (cov2D[0][0] + cov2D[1][1])
    lambda1 = mid + math.sqrt(max(0.1, mid**2 - det))
    lambda2 = mid - math.sqrt(max(0.1, mid**2 - det))
    radius = math.ceil(3 * math.sqrt(max(lambda1, lambda2)))

    mean_came = came_params.M_view @ mean

    depth = mean_came[2]
    mean_NDC = came_params.M_proj @ mean_came

    for i in range(mean_NDC.shape[0]):
        mean_NDC[i] = mean_NDC[i] / mean_NDC[3]

    # above is preprocess, TODO: sh to rgb
    mean_screen = torch.Tensor([width * (mean_NDC[0] + 1) / 2, height * (mean_NDC[1] + 1) / 2])
    if debug:
        print("det: ", det)
        print("conic: ", conic)
        print("radius: ", radius)
        print("mean_came: ", mean_came)
        print("mean_NDC: ", mean_NDC)
        print("mean_screen: ", mean_screen)
    
    # we just need the conic and the mean_screen

    # rasterize
    output = torch.zeros((width, height, 4), dtype=torch.float)
    
    for x in range(width):
        for y in range(height):
            # debug = False
            # if abs(x - mean_screen[0]) < radius//2 and abs(y - mean_screen[1]) < radius//2:
            #     debug = True
            # if x == 9 and y == 12:
            #     debug = True
            x_ = float(x) - mean_screen[0]
            y_ = float(y) - mean_screen[1]
            if debug:
                print("=========")
                print(x, y, x_, y_)
            power_con = -0.5 * (conic[0] * x_**2 + 2 * conic[1] * x_ * y_ + conic[2] * y_**2)
            if debug:
                print("power_con: ", power_con)
            if power_con > 0:
                continue

            alpha = min(0.99, color[3] * torch.exp(power_con))
            if debug:
                print("alpha: ", alpha)
                print("color: ", color)
                print("exp: ", torch.exp(power_con))
            if alpha < 1.0/255.0:
                continue

            test_T = 1.0 * (1-alpha) # do not really need this for 1 Gaussian
            if debug:
                print("test_T: ", test_T)
            if test_T < 0.0001:
                continue
            # print("||||||||||||||||||")
            # print(x, y, alpha)
            # print(power_con, alpha, test_T)
            # print("||||||||||||||||||")
            color_copy = color.clone()
            color_copy[3] = alpha
            output[x, y] = color_copy

    # output[0, 0] = torch.Tensor([1, 0, 0, 1])
    # output[1, 0] = torch.Tensor([1, 0, 0, 1])
    # torch to np array
    output = output.cpu().numpy()
    output = np.rot90(output)
    # plt.imshow(output)
    # plt.show()
    return output


def world_to_screen(mean, came_params):
    # non batch version
    mean_came = torch.zeros_like(mean)
    for i in range(mean.shape[0]):
        mean_came[i] = came_params.M_view @ mean[i]

    mean_NDC = torch.zeros_like(mean)
    for i in range(mean.shape[0]):
        mean_NDC[i] = came_params.M_proj @ mean_came[i]
    for i in range(mean_NDC.shape[0]):
        mean_NDC[i] = mean_NDC[i] / mean_NDC[i][3]
    mean_screen = torch.zeros_like(mean)
    for i in range(mean_NDC.shape[0]):
        mean_screen[i][0] = came_params.width * (mean_NDC[i][0] + 1) / 2
        mean_screen[i][1] = came_params.height * (mean_NDC[i][1] + 1) / 2
    return mean_screen


def G_2D_projection(mean, s, r, came_params):
    conic = torch.zeros((mean.shape[0], 3))
    for i in range(mean.shape[0]):
        R = quart_to_rot(r[i])
        S = s_to_scale(s[i])
        cov3D = R @ S @ S.t() @ R.t()
        # cov2D = compute_cov2D(mean[i], came_params.M_view, cov3D)
        cov2D = compute_cov2D(mean[i], came_params, cov3D)
        det = torch.det(cov2D)
        det_inv = 1.0 / det
        conic[i][0] = cov2D[1][1] * det_inv
        conic[i][1] = -cov2D[0][1] * det_inv
        conic[i][2] = cov2D[0][0] * det_inv
    return conic


def rast_n_G_color(means, s, r, came_params, color):
    # convert means to view space
    means_came = torch.zeros_like(means)
    for i in range(means.shape[0]):
        means_came[i] = came_params.M_view @ means[i]
    
    # sort the means by depth
    indices = torch.argsort(means_came[:, 2], descending=True)
    means_came = means_came[indices]
    s = s[indices]
    r = r[indices]
    color = color[indices]
    means = means[indices]

    means_screen = world_to_screen(means, came_params)
    conics = G_2D_projection(means, s, r, came_params)

    # rasterize
    output = torch.zeros((came_params.width, came_params.height, 4), dtype=torch.float)
    for i in range(came_params.width):
        for j in range(came_params.height):
            T = 1.0
            done = False
            for k in range(means.shape[0]):
                if done:
                    break
                x_ = i - means_screen[k][0]
                y_ = j - means_screen[k][1]
                power_con = -0.5 * (conics[k][0] * x_**2 + 2 * conics[k][1] * x_ * y_ + conics[k][2] * y_**2)
                if power_con > 0:
                    continue
                alpha = min(0.99, color[k][3] * torch.exp(power_con))
                if alpha < 1.0/255.0:
                    continue
                test_T = T * (1-alpha)
                if test_T < 0.0001:
                    done = True
                    continue
                color_copy = color[k].clone()
                color_copy = color_copy * alpha * T
                output[i, j] = color_copy + output[i, j]
                T = test_T
    output = output.cpu().numpy()
    output = np.rot90(output)
    plt.imshow(output)
    plt.show()
    return output

def compute_conv2D_scene(mean, s, r, focal_x, focal_y, tan_fovx, tan_fovy, viewM, projM, debug=False):
    R = quart_to_rot(r)
    S = s_to_scale(s)
    Vrk = R @ S @ S.t() @ R.t()

    t = viewM @ mean
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]

    J_ = torch.Tensor([[1/t[2], 0, -t[0]/t[2]**2],
                      [0, 1/t[2], -t[1]/t[2]**2],
                      [0,0,0]])
    # J_ = torch.Tensor([[focal_x/t[2], 0, -(focal_x*t[0])/t[2]**2],
    #                   [0, focal_y/t[2], -(focal_y * t[1])/t[2]**2],
    #                   [0,0,0]])
    if debug:
        print("J: ", J_)
    W = torch.Tensor([[viewM[0][0], viewM[0][1], viewM[0][2]],
                      [viewM[1][0], viewM[1][1], viewM[1][2]],
                      [viewM[2][0], viewM[2][1], viewM[2][2]]])
    if debug:
        print("W: ", W)
    T = J_ @ W
    cov2D = T @ Vrk @ T.t()
    t_cov2D = torch.Tensor([[cov2D[0][0] + 0.3, cov2D[0][1]],
                            [cov2D[1][0], cov2D[1][1] + 0.3]])
    if debug:
        print("t_cov2D: ", t_cov2D)
    return t_cov2D      


def rast_3DGS_scene(plt3DGS, cam_para):
    width = plt3DGS._width
    height = plt3DGS._height
    start = time.time()
    means = plt3DGS._xyz
    opacity = plt3DGS._opacity
    colors = plt3DGS._colors
    r = plt3DGS._rotation
    s = plt3DGS._scaling
    means = torch.Tensor(means)
    opacity = torch.Tensor(opacity)
    colors = torch.Tensor(colors)
    r = torch.Tensor(r)
    s = torch.Tensor(s)

    mean_ones = torch.ones((means.shape[0], 1))
    means = torch.cat((means, mean_ones), dim=1)

    viewM = cam_para['viewM']
    projM = cam_para['projM']
    full_projM = cam_para['full_proj_transform']
    viewM = torch.Tensor(viewM)
    projM = torch.Tensor(projM)
    full_projM = torch.Tensor(full_projM)
    means_cam = torch.zeros_like(means)
    with torch.no_grad():
        for i in range(means.shape[0]):
            means_cam[i] = viewM @ means[i]

        indices = torch.argsort(means_cam[:, 2], descending=False)
        means_cam = means_cam[indices]  # at idx 114627, the depth is 0.2

        s = s[indices]
        r = r[indices]
        colors = colors[indices]
        opacity = opacity[indices]
        means = means[indices]

    end = time.time()
    print("time: ", end - start)

    # means = means.detach().cpu().numpy()
    # s = s.detach().cpu().numpy()
    # r = r.detach().cpu().numpy()
    # viewM = viewM.detach().cpu().numpy()
    # projM = projM.detach().cpu().numpy()
    # colors = colors.detach().cpu().numpy()

    num_points = means.shape[0]
    output = torch.zeros((num_points, 3))
    conv2D = torch.zeros((num_points, 2, 2))
    # start timer
    start = time.time()
    
    tanfovx = math.tan(plt3DGS.FoVx * 0.5)
    tanfovy = math.tan(plt3DGS.FoVy * 0.5)
    focal_y = plt3DGS._height / (2.0 * tanfovy)
    focal_x = plt3DGS._width / (2.0 * tanfovx)
    print("focal_x: ", focal_x)
    print("focal_y: ", focal_y)
    print("tanfovx: ", tanfovx)
    print("tanfovy: ", tanfovy)
    print("FoVx: ", plt3DGS.FoVx)
    print("FoVy: ", plt3DGS.FoVy)
    return

    canvas = torch.zeros((width, height, 5))
    # canvas[:, :, 4] = 1
    st = 157000 # start at 150000
    only_points = False
    with torch.no_grad():
        # for idx in range(50000, ):
        for idx in range(st, st+3000):
            debug = False
            if idx % 1000 == 0:
                print("idx: ", idx)
                print(s[idx])
                debug = True
            point = means[idx]
            point1 = viewM @ point
            # if point1[2] < 0.2:
            #     # print("too close")
            #     continue
            point2 = projM @ point1
            # point1 = np.matmul(full_proj_transform, point)
            point3 = point2 / point2[3]
            output[idx] = point3[0:3]
            conv2D[idx] = compute_conv2D_scene(point, s[idx], r[idx], focal_x, focal_y, tanfovx, tanfovy, viewM, projM, debug=debug)
            det = torch.det(conv2D[idx])
            conic = [conv2D[idx][1][1] / det, -conv2D[idx][0][1] / det, conv2D[idx][0][0] / det]
            mid = 0.5 * (conv2D[idx][0][0] + conv2D[idx][1][1])
            lambda1 = mid + math.sqrt(max(0.1, mid**2 - det))
            lambda2 = mid - math.sqrt(max(0.1, mid**2 - det))
            radius = math.ceil(3 * math.sqrt(max(lambda1, lambda2)))
            screen_x = int(((output[idx][0]/100+ 1) * width - 1) * 0.5)
            screen_y = int(((output[idx][1]/100+ 1) * height - 1) * 0.5)

            if only_points:
                if screen_x < 0 or screen_x >= width or screen_y < 0 or screen_y >= height:
                    continue
                canvas[screen_x, screen_y, 0:3] = colors[idx][0:3]
            else:
                if idx % 100 == 0:
                    print("idx: ", idx, radius)
                #     for x in range(max(0, screen_x - radius), min(width, screen_x + radius)):
                #         for y in range(max(0, screen_y - radius), min(height, screen_y + radius)):
                #             x_ = float(x) - screen_x
                #             y_ = float(y) - screen_y
                #             power_con = -0.5 * (conic[0] * x_**2 + 2 * conic[1] * x_ * y_ + conic[2] * y_**2)
                #             if power_con > 0:
                #                 continue
                #             alpha = min(0.99, opacity[idx] * torch.exp(power_con))
                #             if alpha < 1.0/255.0:
                #                 continue
                #             test_T = 1.0 * (1-alpha)
                #             if test_T < 0.0001:
                #                 continue
                #             color_copy = colors[idx].clone()
                #             canvas[x, y, 0:3] = color_copy
                #             canvas[x, y, 4] = alpha
                for x in range(max(0, screen_x - radius), min(width, screen_x + radius)):
                    for y in range(max(0, screen_y - radius), min(height, screen_y + radius)):
                        if canvas[x, y, -1] == 1:
                            continue

                        canvas[x, y, -1] = 1
                        canvas[x, y, 0:3] = colors[idx][0:3]

                # x_ = float(x) - screen_x
                # y_ = float(y) - screen_y
                # power_con = -0.5 * (conic[0] * x_**2 + 2 * conic[1] * x_ * y_ + conic[2] * y_**2)
                # if power_con > 0:
                #     continue
                # alpha = min(0.99, colors[idx][3] * torch.exp(power_con))
                # if alpha < 1.0/255.0:
                #     continue
                # test_T = 1.0 * (1-alpha)
                # if test_T < 0.0001:
                #     continue
                # color_copy = colors[idx].clone()
                # color_copy[3] = alpha
                # canvas[x, y] = color_copy

    end = time.time()
    print("Time taken in seconds: ", end - start)
    print("finish")

    # g_in_scene = []

    # for idx in range(0, num_points):
    #     if output[idx][0] < -100 or output[idx][0] > 100 or output[idx][1] < -100 or output[idx][1] > 100:
    #         continue
    #     g_in_scene.append(idx)
    #     x = int((output[idx][0]/100 + 1) * 0.5 * width)
    #     y = int((output[idx][1]/100 + 1) * 0.5 * height)
    #     # canvas[x, y] = [output[idx][2], output[idx][2], output[idx][2]]
    #     canvas[x, y] = [colors[idx][0], colors[idx][1], colors[idx][2]]
    
    # print(len(g_in_scene))
    # print(min_depth, max_depth)
    # canvas = (canvas - min_depth)/(max_depth - min_depth)
    # canvas = np.sqrt(canvas)
    # print(canvas[g_in_scene[0]])
    # print(output[g_in_scene[0]])
    canvas = canvas.detach().cpu().numpy()
    canvas = canvas[:, :, 0:3]
    canvas = np.rot90(canvas)
    plt.imshow(canvas)
    plt.show()


def rast_3DGS_scene_slang(plt3DGS, cam_para):
    start = time.time()
    means = plt3DGS._xyz
    print(means.shape)
    return
    opacity = plt3DGS._opacity
    colors = plt3DGS._colors
    r = plt3DGS._rotation
    s = plt3DGS._scaling
    means = torch.Tensor(means)
    opacity = torch.Tensor(opacity)
    colors = torch.Tensor(colors)
    r = torch.Tensor(r)
    s = torch.Tensor(s)

    # means = plt3DGS._xyz
    # opacity = plt3DGS._opacity
    # colors = plt3DGS._colors
    # r = plt3DGS._rotation
    # s = plt3DGS._scaling
    means = torch.Tensor(means)
    mean_ones = torch.ones((means.shape[0], 1))
    means = torch.cat((means, mean_ones), dim=1)

    viewM = cam_para['viewM']
    projM = cam_para['projM']
    full_projM = cam_para['full_proj_transform']
    viewM = torch.Tensor(viewM)
    projM = torch.Tensor(projM)
    full_projM = torch.Tensor(full_projM)
    # cam_loc = cam_para['camera_center']
    # cam_loc = np.append(cam_loc, 1)
    # cam_loc = torch.Tensor(cam_loc)

    means_cam = torch.zeros_like(means)
    for i in range(means.shape[0]):
        means_cam[i] = viewM @ means[i]

    indices = torch.argsort(means_cam[:, 2], descending=False)
    means_cam = means_cam[indices]
    s = s[indices]
    r = r[indices]
    colors = colors[indices]
    opacity = opacity[indices]
    means = means[indices]

    end = time.time()
    print("time: ", end - start)

    means = means.cuda()
    s = s.cuda()
    r = r.cuda()
    viewM = viewM.cuda()
    projM = projM.cuda()
    colors = colors.cuda()
    tanfovx = math.tan(plt3DGS.FoVx * 0.5)
    tanfovy = math.tan(plt3DGS.FoVy * 0.5)
    focal_y = plt3DGS._height / (2.0 * tanfovy)
    focal_x = plt3DGS._width / (2.0 * tanfovx)
    view_angle = torch.Tensor([tanfovx, tanfovy, focal_x, focal_y]).cuda()

    rasterizer = setup_scene_rasterizer()
    
    pre_input = torch.zeros((plt3DGS._width, plt3DGS._height, 5), dtype=torch.float)
    # last channel set to 1
    pre_input[:, :, 4] = 1
    pre_input = pre_input.cuda()

    cur_id = 150000
    final_id = cur_id + 10000
    max_gaussian = 1024
    while cur_id < final_id:
        print("cur_id: ", cur_id)
        end_id = min(cur_id + max_gaussian, final_id)
        means_i = means[cur_id:end_id]
        s_i = s[cur_id:end_id]
        r_i = r[cur_id:end_id]
        color_i = colors[cur_id:end_id]
        pre_input = rasterizer.apply(plt3DGS._width, plt3DGS._height, means_i, s_i, r_i, viewM, projM, view_angle, end_id-cur_id, color_i, pre_input)
        cur_id = end_id


    pre_input = pre_input.detach().cpu().numpy()

    # get the first 4 channels
    pre_input = pre_input[:, :, :4]
    # print(sample[13, 13])
    # print(sample[13, 12])
    # print(sample[1, 3])
    # print(sample[2, 3])
    sample = np.rot90(pre_input)
    plt.imshow(sample)
    plt.show()
    return sample


def in_triangle(x, y, x1, y1, x2, y2, x3, y3):
    def tri_sign(x, y, x1, y1, x0, y0):
        return (x-x0)*(y1-y0) - (y-y0)*(x1-x0)
    b1 = tri_sign(x, y, x2, y2, x1, y1) < 0
    b2 = tri_sign(x, y, x3, y3, x2, y2) < 0
    b3 = tri_sign(x, y, x1, y1, x3, y3) < 0
    return b1 and b2 and b3


def batch_rasterize(x, y, x_r, y_r, vertices_screen, color, output):
    for i in range(x_r):
        for j in range(y_r):
            x_ = x + i
            y_ = y + j
            if in_triangle(x_, y_, vertices_screen[0][0], vertices_screen[0][1], vertices_screen[1][0], vertices_screen[1][1], vertices_screen[2][0], vertices_screen[2][1]):
                output[int(x_), int(y_)] = color


def rasterize_tri_3D(vertices, color, camera_params):
    width, height = camera_params.width, camera_params.height

    # a torch object with width x height x rgba channels
    output = torch.zeros((width, height, 4), dtype=torch.float)

    vertices_clip = torch.zeros_like(vertices)
    vertices_NDC = torch.zeros_like(vertices)
    vertices_screen = torch.zeros_like(vertices)

    for i in range(vertices.shape[0]):
        vertices_clip[i] = camera_params.M_view_proj @ vertices[i]
    
    for i in range(vertices.shape[0]):
        vertices_NDC[i] = vertices_clip[i] / vertices_clip[i][3]
    
    for i in range(vertices.shape[0]):    
        vertices_screen[i][0] = width * (vertices_NDC[i][0] + 1) / 2
        vertices_screen[i][1] = height * (vertices_NDC[i][1] + 1) / 2
        vertices_screen[i][2] = 0.5 * vertices_NDC[i][2] + 0.5
        vertices_screen[i][3] = vertices_NDC[i][3]

    # for i in range(vertices.shape[0]):
    #     print(vertices_clip[i])
    # return

    max_t = 16
    x_dim = width // math.sqrt(max_t)
    y_dim = height // math.sqrt(max_t)

    threads = []

    for i in range(int(math.floor(math.sqrt(max_t)))):
        for j in range(int(math.floor(math.sqrt(max_t)))):
            x = x_dim * i
            y = y_dim * j
            thread = threading.Thread(target=batch_rasterize, args=(x, y, int(x_dim), int(y_dim), vertices_screen, color, output))
            threads.append(thread)
            thread.start()
    
    for thread in threads:
        thread.join()

    output[0, 0] = torch.Tensor([1, 0, 0, 1])
    output[1, 0] = torch.Tensor([1, 0, 0, 1])
    # torch to np array
    output = output.cpu().numpy()
    output = np.rot90(output)
    plt.imshow(output)
    plt.show()
    return output


def test_tri_rast():
    cam_para = CameraParams(eye=torch.Tensor([0, 0, 20]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=5, far=500, width=128, height=128)

    # p0 = torch.Tensor([-10, 0, 0, 1])
    # p1 = torch.Tensor([10, 0, 0, 1])
    # p2 = torch.Tensor([0, 10, 0, 1])
    p0 = torch.Tensor([-1, -1, 0, 1])
    p1 = torch.Tensor([1, -1, 0, 1])
    p2 = torch.Tensor([1, 1, 0, 1])

    color = torch.Tensor([0, 1, 0, 1])
    rasterize_tri_3D(torch.stack([p0, p1, p2]), color, cam_para)


def test_rast_1_G_color():
    cam_para = CameraParams(eye=torch.Tensor([0, 0, 50]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=20, far=500, width=30, height=30)
    mean = torch.Tensor([0, 0, 0, 1])
    s = torch.Tensor([100, 100, 100])
    r = torch.Tensor([1, 0, 0, 0])
    color = torch.Tensor([0, 1, 0, 1])
    rast_1_G_color(mean, s, r, cam_para, color, debug=False)


def test_rast_1_G_color_slang():
    cam_para = CameraParams(eye=torch.Tensor([0, 0, 50]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=20, far=500, width=30, height=30)
    mean = torch.Tensor([0, 0, 0, 1])
    s = torch.Tensor([100, 100, 100])
    r = torch.Tensor([1, 0, 0, 0])
    color = torch.Tensor([0, 1, 0, 1])
    rasterize_1_G_color_slang(mean, s, r, cam_para, color)


def test_rast_n_G_color():
    cam_para = CameraParams(eye=torch.Tensor([50, 0, 0]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=10, far=5000, width=30, height=30)
    means = torch.Tensor([[0, 0, 0, 1], [0, 0, -4, 1], [10, 0, -10, 1]])
    s = torch.Tensor([[100, 100, 100], [100, 100, 100], [100, 100, 100]])
    r = torch.Tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    color = torch.Tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    rast_n_G_color(means, s, r, cam_para, color)

def test_rast_n_G_color_slang():
    cam_para = CameraParams(eye=torch.Tensor([50, 0, 0]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=10, far=5000, width=30, height=30)
    means = torch.Tensor([[0, 0, 0, 1], [0, 0, -4, 1], [10, 0, -10, 1]])
    s = torch.Tensor([[100, 100, 100], [100, 100, 100], [100, 100, 100]])
    r = torch.Tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    color = torch.Tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    rasterize_n_G_color_slang(means, s, r, cam_para, color)

def test_rast_n_G_scene_slang_sample():
    cam_para = CameraParams(eye=torch.Tensor([50, 0, 0]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=10, far=5000, width=30, height=30)
    means = torch.Tensor([[0, 0, 0, 1], [0, 0, -4, 1], [10, 0, -10, 1]])
    s = torch.Tensor([[100, 100, 100], [100, 100, 100], [100, 100, 100]])
    r = torch.Tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    color = torch.Tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    rasterize_n_G_scene_slang(means, s, r, cam_para, color)

def test_3DGS_scene_Train():
    sh_degree = 3
    cam_para = get_sample_camera()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    ply_path = os.path.join(cur_dir, "models/train/point_cloud/iteration_30000/point_cloud.ply")
    plt_3DGS = Plt3DGS(sh_degree, ply_path)
    plt_3DGS.get_color(cam_para['camera_center'])
    # rast_3DGS_scene(plt_3DGS, cam_para)
    rast_3DGS_scene_slang(plt_3DGS, cam_para)

def compare_rast_1_G_color():
    cam_para = CameraParams(eye=torch.Tensor([0, 0, 50]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=20, far=500, width=128, height=128)
    mean = torch.Tensor([0, 0, 0, 1])
    s = torch.Tensor([100, 100, 100])
    r = torch.Tensor([1, 0, 0, 0])
    color = torch.Tensor([0, 1, 0, 1])

    mean_1 = mean.clone()
    s_1 = s.clone()
    r_1 = r.clone()
    color_1 = color.clone()

    out_GT = rast_1_G_color(mean, s, r, cam_para, color, debug=False)
    out_slang = rasterize_1_G_color_slang(mean_1, s_1, r_1, cam_para, color_1)
    diff = np.sum((out_GT - out_slang)**2)
    print("diff: ", diff)
    plt.figure
    plt.subplot(1, 2, 1)
    plt.imshow(out_GT)
    plt.subplot(1, 2, 2)
    plt.imshow(out_slang)
    plt.show()


def compare_rast_n_G_color():
    cam_para = CameraParams(eye=torch.Tensor([50, 0, 0]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=10, far=5000, width=30, height=30)
    means = torch.Tensor([[0, 0, 0, 1], [0, 0, -4, 1], [10, 0, -10, 1]])
    s = torch.Tensor([[100, 100, 100], [100, 100, 100], [100, 100, 100]])
    r = torch.Tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    color = torch.Tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])

    means_1 = means.clone()
    s_1 = s.clone()
    r_1 = r.clone()
    color_1 = color.clone()

    out_GT = rast_n_G_color(means, s, r, cam_para, color)
    out_slang = rasterize_n_G_color_slang(means_1, s_1, r_1, cam_para, color_1)
    out_slang_scene = rasterize_n_G_scene_slang(means_1, s_1, r_1, cam_para, color_1)
    diff1 = np.sum((out_GT - out_slang)**2)
    diff2 = np.sum((out_GT - out_slang_scene)**2)
    print("diff: ", diff1, diff2)
    plt.figure
    plt.subplot(1, 3, 1)
    plt.imshow(out_GT)
    plt.subplot(1, 3, 2)
    plt.imshow(out_slang)
    plt.subplot(1, 3, 3)
    plt.imshow(out_slang_scene)
    plt.show()


def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


def get_G_color_op(iteration):
    iterations = 30_000
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01
    lambda_dssim = 0.2
    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002
    return


def optimize_1_G_color():

    rasterizer = setup_1G_rasterizer()

    cam_para = CameraParams(eye=torch.Tensor([0, 0, 50]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=20, far=500, width=32, height=32)
    mean = torch.Tensor([0, 0, 0, 1]).cuda()
    s = torch.Tensor([100, 100, 100]).cuda()
    r = torch.Tensor([1, 0, 0, 0]).cuda()
    color = torch.Tensor([0, 1, 0, 1]).cuda()

    mean_1 = torch.Tensor([0, 10, 0, 1]).cuda()
    mean_1.requires_grad = True
    s_1 = torch.Tensor([100, 100, 100]).cuda()
    s_1.requires_grad = True
    r_1 = torch.Tensor([1, 0, 0, 0]).cuda()
    r_1.requires_grad = True
    color_1 = torch.Tensor([0, 1, 0, 1]).cuda()
    color_1.requires_grad = True

    viewM = cam_para.M_view.cuda()
    projM = cam_para.M_proj.cuda()

    target = rasterizer.apply(cam_para.width, cam_para.height, mean, s, r, viewM, projM, color)
    target_show = target.detach().cpu().numpy()
    target_show = np.rot90(target_show)
    plt.imshow(target_show)
    plt.show()

    learningRate = 5e-3
    numIterations = 4000
    lambda_opt = 0.2
    # optimizer = torch.optim.Adam([mean_1, s_1, r_1, color_1], lr=learningRate)
    optimizer = torch.optim.Adam([
        {'params': mean_1, 'lr': 0.016},
        {'params': s_1, 'lr': 0.005},
        {'params': r_1, 'lr': 0.001},
        {'params': color_1, 'lr': 0.0001}
    ])

    def optimize(i):
        # print("Iteration %d" % i)

        optimizer.zero_grad()

        output = rasterizer.apply(cam_para.width, cam_para.height, mean_1, s_1, r_1, viewM, projM, color_1)
        output.register_hook(set_grad(output))

        Ll1 = l1_loss(output, target)
        loss = (1.0 - lambda_opt) * Ll1 + lambda_opt * (1.0 - ssim(output, target))

        # loss = torch.mean((output - target) ** 2)
        loss.backward()
        optimizer.step()
    
    for i in range(numIterations):
        optimize(i)
        # store image every 10 iterations
        if i % 40 == 0:
            with torch.no_grad():
                output = rasterizer.apply(cam_para.width, cam_para.height, mean_1, s_1, r_1, viewM, projM, color_1)
                output = output.detach().cpu().numpy()
                output = np.rot90(output)
                mean_1_np = mean_1.detach().cpu().numpy()
                plt.imshow(output)
                plt.savefig(os.path.join(image_gaussian_dir, "output_%d.png" % i))
                plt.close()
                print(i, "mean_1: ", mean_1_np)


    output_f = rasterizer.apply(cam_para.width, cam_para.height, mean_1, s_1, r_1, viewM, projM, color_1)
    output_f = output_f.detach().cpu().numpy()
    output_f = np.rot90(output_f)
    plt.imshow(output_f)
    plt.show()


def optimize_n_G_color():
    rasterizer = setup_nG_rasterizer()

    cam_para1 = CameraParams(eye=torch.Tensor([0, 0, 50]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=20, far=500, width=32, height=32)
    cam_para2 = CameraParams(eye=torch.Tensor([50, 0, 0]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=20, far=500, width=32, height=32)
    cam_para3 = CameraParams(eye=torch.Tensor([-50, 0, -50]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 0, 1]), fov=60, aspect=1, near=20, far=500, width=32, height=32)
    cam_para4 = CameraParams(eye=torch.Tensor([50, 0, 50]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 0, 1]), fov=60, aspect=1, near=20, far=500, width=32, height=32)
    cam_para5 = CameraParams(eye=torch.Tensor([50, 0, -10]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 0, 1]), fov=60, aspect=1, near=20, far=500, width=32, height=32)
    cam_para6 = CameraParams(eye=torch.Tensor([10, 0, 50]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 0, 1]), fov=60, aspect=1, near=20, far=500, width=32, height=32)

    cam_paras = [cam_para1, cam_para2, cam_para3, cam_para4, cam_para5, cam_para6]

    # means = torch.Tensor([[0, -10, 0, 1], [0, 6, -4, 1]]).cuda()
    # s = torch.Tensor([[50, 200, 100], [50, 50, 100]]).cuda()
    # r = torch.Tensor([[1, 0, 0, 0], [1, 0, 0, 0]]).cuda()
    # color = torch.Tensor([[0.5, 0, 0, 1], [0, 0.8, 0, 1]]).cuda()

    # means_1 = torch.Tensor([[0, -1, 0, 1], [0, 0, 0, 1]]).cuda()
    # means_1.requires_grad = True
    # s_1 = torch.Tensor([[100, 100, 100], [100, 100, 100]]).cuda()
    # s_1.requires_grad = True
    # r_1 = torch.Tensor([[1, 0, 0, 0], [1, 0, 0, 0]]).cuda()
    # r_1.requires_grad = True
    # color_1 = torch.Tensor([[1, 0, 0, 1], [0, 1, 0, 1]]).cuda()
    # color_1.requires_grad = True

    means = torch.Tensor([[0, -10, 0, 1], [0, 6, -4, 1], [10, 0, -10, 1]]).cuda()
    s = torch.Tensor([[50, 200, 100], [50, 50, 100], [100, 50, 50]]).cuda()
    r = torch.Tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]).cuda()
    color = torch.Tensor([[0.5, 0, 0, 1], [0, 0.8, 0, 1], [0.4, 0, 0.4, 1]]).cuda()

    means_1 = torch.Tensor([[0, -1, 0, 1], [0, 0, 0, 1], [6, 0, -6, 1]]).cuda()
    means_1.requires_grad = True
    s_1 = torch.Tensor([[100, 100, 100], [100, 100, 100], [100, 100, 100]]).cuda()
    s_1.requires_grad = True
    r_1 = torch.Tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]).cuda()
    r_1.requires_grad = True
    color_1 = torch.Tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0.8, 0, 0.8, 1]]).cuda()
    color_1.requires_grad = True

    gaussian_range = means.shape[0]
    targets = []

    for i in range(len(cam_paras)):
        view_angle_rad = cam_paras[i].fov * math.pi / 180
        tanfovx = math.tan(view_angle_rad * 0.5)
        tanfovy = math.tan(view_angle_rad * 0.5)
        focal_y = cam_paras[i].height / (2.0 * tanfovy)
        focal_x = cam_paras[i].width / (2.0 * tanfovx)
        view_angle = torch.Tensor([tanfovx, tanfovy, focal_x, focal_y]).cuda()
        cam_paras[i].M_view = cam_paras[i].M_view.cuda()
        cam_paras[i].M_proj = cam_paras[i].M_proj.cuda()
        target = rasterizer.apply(cam_paras[i].width, cam_paras[i].height, means, s, r, cam_paras[i].M_view, cam_paras[i].M_proj, view_angle, gaussian_range, color)
        target_show = target.detach().cpu().numpy()
        target_show = np.rot90(target_show)
        targets.append(target)
        plt.imshow(target_show)
        # plt.show()
        plt.savefig(os.path.join(image_gaussian_dir, "target_%d.png" % i))
        plt.close()

    
    numIterations = 1000
    optimizer = torch.optim.Adam([
        {'params': means_1, 'lr': 0.016},
        {'params': s_1, 'lr': 1.5},
        {'params': r_1, 'lr': 0.005},
        {'params': color_1, 'lr': 0.001}
    ])

    def optimize(i, means_op, s_op, r_op, color_op):
        sample_cam = random.uniform(0, len(cam_paras))
        if i % 100 == 0:
            sample_cam = 2
        cam_para1 = cam_paras[int(sample_cam)]

        means_came = torch.zeros_like(means_op)
        for j in range(means_op.shape[0]):
            means_came[j] = cam_para1.M_view @ means_op[j]
        
        # sort the means_op by depth
        indices = torch.argsort(means_came[:, 2], descending=True)
        s_op = s_op[indices]
        r_op = r_op[indices]
        color_op = color_op[indices]
        means_op = means_op[indices]

        optimizer.zero_grad()
    
        if i % 100 == 1:
            print("Iteration %d, sample_cam: %d" % (i, int(sample_cam)))

        output = rasterizer.apply(cam_para1.width, cam_para1.height, means_op, s_op, r_op, cam_para1.M_view, cam_para1.M_proj, view_angle, gaussian_range, color_op)
        output.register_hook(set_grad(output))

        cur_target = targets[int(sample_cam)]

        loss = torch.mean((output - cur_target) ** 2)
        loss.backward()
        optimizer.step()

    for i in range(numIterations):
        optimize(i, means_1, s_1, r_1, color_1)
        if i % 100 == 0:
            with torch.no_grad():
                sample_cam = 2
                # target_in_cuda = torch.Tensor(targets[sample_cam]).cuda()
                cam_para1 = cam_paras[int(sample_cam)]
                output = rasterizer.apply(cam_para1.width, cam_para1.height, means_1, s_1, r_1, cam_para1.M_view, cam_para1.M_proj, view_angle, gaussian_range, color_1)
                # ssim_error = ssim(output, target_in_cuda)
                output = output.detach().cpu().numpy()
                output = np.rot90(output)
                # means_1_np = means_1.detach().cpu().numpy()
                plt.imshow(output)
                plt.savefig(os.path.join(image_gaussian_dir, "output_%d.png" % i))
                plt.close()
                # psnr = psnr_error(output, targets[sample_cam].detach().cpu().numpy())
                # print(i, "ssim: ", ssim_error, "psnr: ", psnr)
                # print(i, "mean_1: ", means_1_np)
    
    for i in range(len(cam_paras)):
        cam_para1 = cam_paras[i]
        output_f = rasterizer.apply(cam_para1.width, cam_para1.height, means_1, s_1, r_1, cam_para1.M_view, cam_para1.M_proj, view_angle, gaussian_range, color_1)
        output_f = output_f.detach().cpu().numpy()
        output_f = np.rot90(output_f)
        plt.imshow(output_f)
        # plt.show()
        plt.savefig(os.path.join(image_gaussian_dir, "output_final_%d.png" % i))
        plt.close()
    
    # compare means, s, r, color and means_1, s_1, r_1, color_1 in mse
    means_error = torch.mean((means - means_1) ** 2)
    s_error = torch.mean((s - s_1) ** 2)
    r_error = torch.mean((r - r_1) ** 2)
    color_error = torch.mean((color - color_1) ** 2)
    print("means_error: ", torch.mean(means_error))
    print("s_error: ", torch.mean(s_error))
    print("r_error: ", torch.mean(r_error))
    print("color_error: ", torch.mean(color_error))


def load_png_as_np(image):
    img = Image.open(image)
    img = np.array(img).astype(np.float32)
    img /= 255.0
    return img

def setup_training_set(Train_set):
    # test_set = [("x0_y_10_z0_cube.png", [0,0,-10]), ("x_8_y_6_z0_cube.png", [-8,0,-6]), ("x_9_1_y_4_2_z0_cube.png", [-9.1,0,-4.2]), ("x_10_y0_z0_cube.png", [-10,0,0])]
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    Train_data_dir = os.path.join(cur_dir, "Train")
    Train_data_dir = os.path.join(Train_data_dir, Train_set)
    # read train.txt
    Train_sum_dir = os.path.join(Train_data_dir, "train.txt")
    with open(Train_sum_dir, "r") as f:
        lines = f.readlines()
    test_set = []
    for i in range(1, len(lines)):
        line = lines[i].split()
        image_name = line[0] + ".png"
        cam_para = [float(line[1]), float(line[2]), float(line[3])]
        test_set.append([image_name, cam_para])

    complete_set = []
    for i in range(len(test_set)):
        image_name = test_set[i][0]
        image_dir = os.path.join(Train_data_dir, image_name)
        img1 = load_png_as_np(image_dir).astype(np.float32)
        cam1 = CameraParams(eye=torch.Tensor(test_set[i][1]), center=torch.Tensor([0, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=5, far=500, width=256, height=256)
        set1 = [img1, cam1]
        complete_set.append(set1)
    return complete_set

def optimize_blender():
    task = "Comb"
    rasterizer = setup_nG_rasterizer()
    train_set = setup_training_set(task)
    # train_set = setup_training_set("Cube")

    cam_paras = []
    targets = []
    for i in range(len(train_set)):
        cam_paras.append(train_set[i][1])
        cam_paras[i].M_view = cam_paras[i].M_view.cuda()
        cam_paras[i].M_proj = cam_paras[i].M_proj.cuda()
        target_show = train_set[i][0]
        plt.imshow(target_show)
        # plt.show()
        plt.savefig(os.path.join(image_blender_dir, f"target_{i}.png"))
        plt.close()

        targets.append(torch.Tensor(train_set[i][0]).cuda())

    # pick random one to be the testing set
    # test_index = random.randint(0, len(train_set) - 1)
    test_index = 3
    test_cam = [cam_paras[test_index]]
    test_target = [targets[test_index]]

    guassian_num = 32
    means = []
    r = []
    s = []
    colors = []

    if task == "Cube":
        for i in range(guassian_num):
            x = random.uniform(-1, 1)
            y = random.uniform(-.5, .5)
            z = random.uniform(-0.5, 0.5)
            means.append([x, y, z, 1])
            r.append([1, 0, 0, 0])
            s.append([10, 10, 10])
            colors.append([1, 0.4, 0.4, 1])
    elif task == "Torus":
        for i in range(guassian_num):
            phi = random.uniform(-math.pi, math.pi)
            x = math.cos(phi) * 2
            y = math.sin(phi) * 2
            z = random.uniform(-0.3, 0.3)
            means.append([x, y, z, 1])
            r.append([1, 0, 0, 0])
            s.append([10, 10, 10])
            colors.append([0.4, 1, 0.4, 1])
    elif task == "Comb":
        for_cube = guassian_num // 2
        for i in range(for_cube):
            x = random.uniform(-1, 1)
            y = random.uniform(-.5, .5)
            z = random.uniform(-0.5, 0.5)
            means.append([x, y, z, 1])
            r.append([1, 0, 0, 0])
            s.append([10, 10, 10])
            colors.append([1, 0.4, 0.4, 1])
        for i in range(guassian_num - for_cube):
            phi = random.uniform(-math.pi, math.pi)
            x = math.cos(phi) * 2
            y = math.sin(phi) * 2
            z = random.uniform(-0.3, 0.3)
            means.append([x, y, z, 1])
            r.append([1, 0, 0, 0])
            s.append([10, 10, 10])
            colors.append([0.4, 1, 0.4, 1])
    elif task == "Monkey":
        for i in range(guassian_num):
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            z = random.uniform(-2, 2)
            means.append([x, y, z, 1])
            r.append([1, 0, 0, 0])
            s.append([10, 10, 10])
            colors.append([1, 0.4, 0.4, 1])

    means = torch.Tensor(means).cuda().requires_grad_(True)
    r = torch.Tensor(r).cuda().requires_grad_(True)
    s = torch.Tensor(s).cuda().requires_grad_(True)
    colors = torch.Tensor(colors).cuda().requires_grad_(True)

    cam1 = cam_paras[2]

    view_angle_rad = cam1.fov * math.pi / 180
    tanfovx = math.tan(view_angle_rad * 0.5)
    tanfovy = math.tan(view_angle_rad * 0.5)
    focal_y = cam1.height / (2.0 * tanfovy)
    focal_x = cam1.width / (2.0 * tanfovx)
    view_angle = torch.Tensor([tanfovx, tanfovy, focal_x, focal_y]).cuda()
    # cam1.M_view = cam1.M_view.cuda()
    # cam1.M_proj = cam1.M_proj.cuda()

    target = rasterizer.apply(cam1.width, cam1.height, means, s, r, cam1.M_view, cam1.M_proj, view_angle, guassian_num, colors)
    target_show = target.detach().cpu().numpy()
    # target_show = np.rot90(target_show)
    plt.imshow(target_show)
    plt.show()
    # plt.close()

    numIterations = 2000
    optimizer = torch.optim.Adam([
        {'params': means, 'lr': 0.01, "name": "means"},
        {'params': s, 'lr': 0.04, "name": "s"},
        {'params': r, 'lr': 0.00001, "name": "r"},
        {'params': colors, 'lr': 0.01, "name": "colors"}
    ])

    lr_func = get_expon_lr_func(0.01, 0.0001, 600, 0.5, 2000)

    def optimize(i, means_op, s_op, r_op, colors_op):
        lambda_opt = 0.2
        sample_cam = random.uniform(0, len(cam_paras))
        while sample_cam == test_index:
            sample_cam = random.uniform(0, len(cam_paras))
        cam_para1 = cam_paras[int(sample_cam)]

        means_came = torch.zeros_like(means_op)
        for j in range(means_op.shape[0]):
            means_came[j] = cam_para1.M_view @ means_op[j]
        
        # sort the means_op by depth
        indices = torch.argsort(means_came[:, 2], descending=True)
        s_op = s_op[indices]
        r_op = r_op[indices]
        colors_op = colors_op[indices]
        means_op = means_op[indices]

        update_learning_rate(optimizer, lr_func, i)
        optimizer.zero_grad()
        if i % 100 == 0:
            print("Iteration %d, sample_cam: %d" % (i, int(sample_cam)))

        output = rasterizer.apply(cam_para1.width, cam_para1.height, means_op, s_op, r_op, cam_para1.M_view, cam_para1.M_proj, view_angle, guassian_num, colors_op)
        # print("output: ", output)
        output.register_hook(set_grad(output))

        cur_target = targets[int(sample_cam)]

        Ll1 = l1_loss(output, cur_target)
        loss = (1.0 - lambda_opt) * Ll1 + lambda_opt * (1.0 - ssim(output, cur_target))
        if i % 100 == 0:
            print("Iteration %d, loss: %f" % (i, loss))
            print("optimizer means: ", optimizer.param_groups[0]['lr'])
        loss.backward()
        optimizer.step()

    for i in range(numIterations):
        optimize(i, means, s, r, colors)
        if i % 100 == 0:
            with torch.no_grad():
                sample_cam = test_index
                # if sample_cam == test_index:
                #     sample_cam = 2
                cam_para1 = cam_paras[int(sample_cam)]

                # means_came = torch.zeros_like(means)
                # for j in range(means.shape[0]):
                #     means_came[j] = cam_para1.M_view @ means[j]
                
                # # sort the means by depth
                # indices = torch.argsort(means_came[:, 2], descending=True)
                # s = s[indices]
                # r = r[indices]
                # colors = colors[indices]
                # means = means[indices]

                output = rasterizer.apply(cam_para1.width, cam_para1.height, means, s, r, cam_para1.M_view, cam_para1.M_proj, view_angle, guassian_num, colors)
                output = output.detach().cpu().numpy()
                # output = np.rot90(output)
                means_np = means.detach().cpu().numpy()
                r_np = r.detach().cpu().numpy()
                s_np = s.detach().cpu().numpy()
                colors_np = colors.detach().cpu().numpy()
                plt.imshow(output)
                plt.savefig(os.path.join(image_blender_dir, "output_%d.png" % i))
                plt.close()
                # print(i, "mean: ", means_np)
                # print(i, "r: ", r_np)
                # print(i, "s: ", s_np)
                # print(i, "colors: ", colors_np)

    with torch.no_grad():
        train_psnr = 0
        for i in range(len(cam_paras)):
            cam_para1 = cam_paras[i]
            start_time = time.time()

            means_came = torch.zeros_like(means)
            for j in range(means.shape[0]):
                means_came[j] = cam_para1.M_view @ means[j]
            
            # sort the means by depth
            indices = torch.argsort(means_came[:, 2], descending=True)
            s = s[indices]
            r = r[indices]
            colors = colors[indices]
            means = means[indices]

            end_time1 = time.time()

            output_f = rasterizer.apply(cam_para1.width, cam_para1.height, means, s, r, cam_para1.M_view, cam_para1.M_proj, view_angle, guassian_num, colors)
            output_f = output_f.detach().cpu().numpy()
            end_time = time.time()
            # output_f = np.rot90(output_f)
            plt.imshow(output_f)
            print("sample cam: ", i)
            # plt.show()
            if i == test_index:
                plt.savefig(os.path.join(image_blender_dir, "output_final_%d_t.png" % test_index))

                psnr = psnr_error(output_f, test_target[0].detach().cpu().numpy())
                print("psnr test: ", psnr)
                print("total time: ", (end_time - start_time)*1000)
                print("render time: ", (end_time - end_time1)*1000)

            else:
                plt.savefig(os.path.join(image_blender_dir, "output_final_%d.png" % i))
                psnr = psnr_error(output_f, targets[i].detach().cpu().numpy())
                train_psnr += psnr
                # print("psnr: ", psnr)
            plt.close()
        print("train psnr: ", train_psnr / (len(cam_paras) - 1))

    # plot the 3D distribution of the gaussian
    means_np = means.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    # add color bar for c
    cbar = plt.colorbar(ax.scatter(means_np[:, 0], means_np[:, 2], means_np[:, 1], c=colors_np[:, 3], marker='o'))
    cbar.set_label('Alpha Value')
    plt.show()


def optimize_blender_scene():
    rasterizer = setup_scene_rasterizer()
    train_set = setup_training_set()
    cam_paras = []
    targets = []
    for i in range(len(train_set)):
        cam_paras.append(train_set[i][1])
        cam_paras[i].M_view = cam_paras[i].M_view.cuda()
        cam_paras[i].M_proj = cam_paras[i].M_proj.cuda()
        target_show = train_set[i][0]
        plt.imshow(target_show)
        # plt.show()
        plt.savefig(os.path.join(image_blender_dir, f"target_{i}.png"))
        plt.close()

        targets.append(torch.Tensor(train_set[i][0]).cuda())

    guassian_num = 512
    means = []
    r = []
    s = []
    colors = []

    for i in range(guassian_num):
        x = random.uniform(-1, 1)
        y = random.uniform(-.5, .5)
        z = random.uniform(-0.5, 0.5)
        means.append([x, y, z, 1])
        r.append([1, 0, 0, 0])
        s.append([10, 10, 10])
        colors.append([1, 0.4, 0.4, 1])

    means = torch.Tensor(means).cuda().requires_grad_(True)
    r = torch.Tensor(r).cuda().requires_grad_(True)
    s = torch.Tensor(s).cuda().requires_grad_(True)
    colors = torch.Tensor(colors).cuda().requires_grad_(True)

    cam1 = cam_paras[0]

    view_angle_rad = cam1.fov * math.pi / 180
    tanfovx = math.tan(view_angle_rad * 0.5)
    tanfovy = math.tan(view_angle_rad * 0.5)
    focal_y = cam1.height / (2.0 * tanfovy)
    focal_x = cam1.width / (2.0 * tanfovx)
    view_angle = torch.Tensor([tanfovx, tanfovy, focal_x, focal_y]).cuda()
    # cam1.M_view = cam1.M_view.cuda()
    # cam1.M_proj = cam1.M_proj.cuda()

    pre_input = torch.zeros((cam1.width, cam1.height, 5), dtype=torch.float).cuda()
    pre_input[:, :, 4] = 1
    st = 0
    cap = 16
    while st < guassian_num:
        ed = min(st + cap, guassian_num)
        output = rasterizer.apply(cam1.width, cam1.height, means[st:ed], s[st:ed], r[st:ed], cam1.M_view, cam1.M_proj, view_angle, ed - st, colors[st:ed], pre_input)
        pre_input = output
        st = ed
    target = pre_input
    target_show = target.detach().cpu().numpy()
    target_show = target_show[:, :, 0:4]
    # target_show = np.rot90(target_show)
    plt.imshow(target_show)
    plt.show()

    numIterations = 2000
    optimizer = torch.optim.Adam([
        {'params': means, 'lr': 0.01},
        {'params': s, 'lr': 0.0001},
        {'params': r, 'lr': 0.00001},
        {'params': colors, 'lr': 0.01}
    ])

    lr_func = get_expon_lr_func(0.01, 0.00001, 600, 0.5, 4000)

    def optimize(i, means_op, s_op, r_op, colors_op):
        lambda_opt = 0.2
        sample_cam = random.uniform(0, len(cam_paras))
        cam_para1 = cam_paras[int(sample_cam)]

        means_came = torch.zeros_like(means_op)
        for j in range(means_op.shape[0]):
            means_came[j] = cam_para1.M_view @ means_op[j]
        
        # sort the means_op by depth
        indices = torch.argsort(means_came[:, 2], descending=True)
        s_op = s_op[indices]
        r_op = r_op[indices]
        colors_op = colors_op[indices]
        means_op = means_op[indices]

        update_learning_rate(optimizer, lr_func, i)

        optimizer.zero_grad()
        if i % 100 == 0:
            print("Iteration %d, sample_cam: %d" % (i, int(sample_cam)))

        pre_input = torch.zeros((cam1.width, cam1.height, 5), dtype=torch.float).cuda()
        pre_input[:, :, 4] = 1

        st = 0
        cap = 16
        while st < guassian_num:
            ed = min(st + cap, guassian_num)
            output = rasterizer.apply(cam_para1.width, cam_para1.height, means_op[st:ed], s_op[st:ed], r_op[st:ed], cam_para1.M_view, cam_para1.M_proj, view_angle, ed - st, colors_op[st:ed], pre_input)
            pre_input = output
            st = ed
        # print("output: ", output)
        output.register_hook(set_grad(output))

        cur_target = targets[int(sample_cam)]

        Ll1 = l1_loss(output[:, :, 0:4], cur_target)
        loss = (1.0 - lambda_opt) * Ll1 + lambda_opt * (1.0 - ssim(output[:, :, 0:4], cur_target))
        if i % 100 == 0:
            print("Iteration %d, loss: %f" % (i, loss))
        loss.backward()
        optimizer.step()

    for i in range(numIterations):
        optimize(i, means, s, r, colors)
        if i % 100 == 0:
            with torch.no_grad():
                sample_cam = 0
                cam_para1 = cam_paras[int(sample_cam)]

                # means_came = torch.zeros_like(means)
                # for j in range(means.shape[0]):
                #     means_came[j] = cam_para1.M_view @ means[j]
                
                # # sort the means by depth
                # indices = torch.argsort(means_came[:, 2], descending=True)
                # s = s[indices]
                # r = r[indices]
                # colors = colors[indices]
                # means = means[indices]

                pre_input = torch.zeros((cam1.width, cam1.height, 5), dtype=torch.float).cuda()
                pre_input[:, :, 4] = 1
                st = 0
                cap = 16
                while st < guassian_num:
                    ed = min(st + cap, guassian_num)
                    output = rasterizer.apply(cam_para1.width, cam_para1.height, means[st:ed], s[st:ed], r[st:ed], cam_para1.M_view, cam_para1.M_proj, view_angle, ed - st, colors[st:ed], pre_input)
                    pre_input = output
                    st = ed
                output = output.detach().cpu().numpy()
                # output = np.rot90(output)
                means_np = means.detach().cpu().numpy()
                r_np = r.detach().cpu().numpy()
                s_np = s.detach().cpu().numpy()
                colors_np = colors.detach().cpu().numpy()
                plt.imshow(output[:, :, 0:4])
                plt.savefig(os.path.join(image_blender_dir, "output_%d.png" % i))
                plt.close()
                # print(i, "mean: ", means_np)
                # print(i, "r: ", r_np)
                # print(i, "s: ", s_np)
                # print(i, "colors: ", colors_np)

    with torch.no_grad():
        for i in range(len(cam_paras)):
            cam_para1 = cam_paras[i]

            means_came = torch.zeros_like(means)
            for j in range(means.shape[0]):
                means_came[j] = cam_para1.M_view @ means[j]
            
            # sort the means by depth
            indices = torch.argsort(means_came[:, 2], descending=True)
            s = s[indices]
            r = r[indices]
            colors = colors[indices]
            means = means[indices]

            pre_input = torch.zeros((cam1.width, cam1.height, 5), dtype=torch.float).cuda()
            pre_input[:, :, 4] = 1
            st = 0
            cap = 16
            while st < guassian_num:
                ed = min(st + cap, guassian_num)
                output = rasterizer.apply(cam_para1.width, cam_para1.height, means[st:ed], s[st:ed], r[st:ed], cam_para1.M_view, cam_para1.M_proj, view_angle, ed - st, colors[st:ed], pre_input)
                pre_input = output
                st = ed

            output_f = pre_input
            output_f = output_f.detach().cpu().numpy()
            # output_f = np.rot90(output_f)
            plt.imshow(output_f[:, :, 0:4])
            # plt.show()
            plt.savefig(os.path.join(image_blender_dir, "output_final_%d.png" % i))
            plt.close()
    

# test_tri_rast()
# test_rast_1_G_color()
# test_rast_1_G_color_slang()
# compare_rast_1_G_color()
# optimize_1_G_color()
# test_rast_n_G_color()
# test_rast_n_G_color_slang()
# test_rast_n_G_scene_slang_sample()
# compare_rast_n_G_color()
# optimize_n_G_color()
test_3DGS_scene_Train()
# optimize_blender()
# optimize_blender_scene()  # doesn't work