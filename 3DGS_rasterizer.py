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

    output[0, 0] = torch.Tensor([1, 0, 0, 1])
    output[1, 0] = torch.Tensor([1, 0, 0, 1])
    # torch to np array
    output = output.cpu().numpy()
    output = np.rot90(output)
    plt.imshow(output)
    plt.show()

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
    plt.imshow(output)
    plt.show()


def test_tri_rast():
    cam_para = CameraParams(eye=torch.Tensor([20, 0, 100]), center=torch.Tensor([20, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=50, far=500, width=128, height=128)

    p0 = torch.Tensor([-10, 0, 0, 1])
    p1 = torch.Tensor([10, 0, 0, 1])
    p2 = torch.Tensor([0, 10, 0, 1])

    color = torch.Tensor([0, 1, 0, 1])
    rasterize_tri_3D(torch.stack([p0, p1, p2]), color, cam_para)

def test_rast_1_G_color():
    cam_para = CameraParams(eye=torch.Tensor([20, 0, 50]), center=torch.Tensor([20, 0, 0]), up=torch.Tensor([0, 1, 0]), fov=60, aspect=1, near=20, far=500, width=30, height=30)
    mean = torch.Tensor([0, 0, 0, 1])
    s = torch.Tensor([100, 100, 100])
    r = torch.Tensor([1, 0, 0, 0])
    color = torch.Tensor([0, 1, 0, 1])
    rast_1_G_color(mean, s, r, cam_para, color, debug=False)

test_rast_1_G_color()