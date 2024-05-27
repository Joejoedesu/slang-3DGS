import slangtorch
import torch
import numpy as np
import timeit
import os
import math
import matplotlib.pyplot as plt
from torch.autograd import Function
import torch.nn.functional as F
import threading

from loss_util import l1_loss, ssim

# https://shader-slang.com/slang/user-guide/a1-02-slangpy.html

image_dir = "images"

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

def compute_cov2D(mean, viewM, Vrk, debug=False):
    t = viewM @ mean
    if debug:
        print("t: ", t)
    l = torch.norm(t)
    J = torch.Tensor([[1/t[2], 0, -t[0]/t[2]**2],
                      [0, 1/t[2], -t[1]/t[2]**2],
                      [0,0,0]])
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

    cov2D = compute_cov2D(mean, came_params.M_view, cov3D, debug=debug)
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
        cov2D = compute_cov2D(mean[i], came_params.M_view, cov3D)
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

    # for x in range(width):
    #     for y in range(height):
    #         # print(x, y)
    #         if in_triangle(x, y, vertices_screen[0][0], vertices_screen[0][1], vertices_screen[1][0], vertices_screen[1][1], vertices_screen[2][0], vertices_screen[2][1]):
    #             output[x, y] = color

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
    # plt.imshow(output)
    # plt.show()
    return output


def test_tri_rast():
    cam_para = CameraParams(eye=torch.Tensor([20, 0, 100]), center=torch.Tensor([20, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=50, far=500, width=128, height=128)

    p0 = torch.Tensor([-10, 0, 0, 1])
    p1 = torch.Tensor([10, 0, 0, 1])
    p2 = torch.Tensor([0, 10, 0, 1])

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
    diff = np.mean((out_GT - out_slang)**2)
    print("diff: ", diff)
    plt.figure
    plt.subplot(1, 2, 1)
    plt.imshow(out_GT)
    plt.subplot(1, 2, 2)
    plt.imshow(out_slang)
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

        # Ll1 = l1_loss(output, target)
        # loss = (1.0 - lambda_opt) * Ll1 + lambda_opt * (1.0 - ssim(output, target))

        loss = torch.mean((output - target) ** 2)
        loss.backward()
        optimizer.step()
    
    for i in range(numIterations):
        optimize(i)
        # store image every 10 iterations
        if i % 20 == 0:
            with torch.no_grad():
                output = rasterizer.apply(cam_para.width, cam_para.height, mean_1, s_1, r_1, viewM, projM, color_1)
                output = output.detach().cpu().numpy()
                output = np.rot90(output)
                mean_1_np = mean_1.detach().cpu().numpy()
                plt.imshow(output)
                plt.savefig(os.path.join(image_dir, "output_%d.png" % i))
                plt.close()
                print(i, "mean_1: ", mean_1_np)


    output_f = rasterizer.apply(cam_para.width, cam_para.height, mean_1, s_1, r_1, viewM, projM, color_1)
    output_f = output_f.detach().cpu().numpy()
    output_f = np.rot90(output_f)
    plt.imshow(output_f)
    plt.show()


# test_rast_1_G_color()
# test_rast_1_G_color_slang()
# compare_rast_1_G_color()
# optimize_1_G_color()
test_rast_n_G_color()