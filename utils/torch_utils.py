# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

""" quaternion order: xyzw """
@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

""" quaternion order: xyzw """
@torch.jit.script
def quat_inv(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

""" quaternion order: xyzw """
""" delta quaternion from quat0 to quat1, represented in quat0's frame. """
@torch.jit.script
def delta_quat_body(quat0, quat1):
    return quat_mul(quat_inv(quat0), quat1)

""" delta quaternion from quat0 to quat1, represented in world frame. """
@torch.jit.script
def delta_quat_world(quat0, quat1):
    return quat_mul(quat1, quat_inv(quat0))

def delta_quat(quat0, quat1, frame='world'):
    if frame == 'world':
        return delta_quat_world(quat0, quat1)
    else:
        return delta_quat_body(quat0, quat1)

@torch.jit.script
def quat_angle_diff(quat_0, quat_1):
    quat_diffs = delta_quat_world(quat_0, quat_1)
    axis_norms = torch.clamp(torch.norm(quat_diffs[..., 0:3], 2, dim = -1), -1., 1.)
    return 2 * torch.asin(axis_norms)

@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

"""
Reference: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L530
"""
@torch.jit.script
def quat_to_exponential_coord(quat):
    norms = torch.norm(quat[..., :3], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quat[..., 3:])
    sin_half_angles_over_angles = 0.5 * torch.sinc(half_angles / torch.pi)

    return quat[..., :3] / sin_half_angles_over_angles

@torch.jit.script
def exponential_coord_to_quat(exp_coord):
    angles = torch.norm(exp_coord, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat([
        exp_coord * sin_half_angles_over_angles, torch.cos(angles * 0.5)
    ], dim = -1)

@torch.jit.script
def transform_point(pos, quat, point):
    return quat_rotate(quat, point) + pos

@torch.jit.script
def transform_point_inverse(pos, quat, point):
    return quat_rotate_inverse(quat, point - pos)

@torch.jit.script
def convert_states_w2b(frame_pos, frame_quat, p, quat, omega, nu):
    p_body = transform_point_inverse(frame_pos, frame_quat, p)
    quat_body = quat_mul(quat_inv(frame_quat), quat)
    omega_body = quat_rotate_inverse(frame_quat, omega)
    nu_body = quat_rotate_inverse(frame_quat, nu - torch.cross(frame_pos, omega, dim=-1))
    return p_body, quat_body, omega_body, nu_body

@torch.jit.script
def convert_states_b2w(frame_pos, frame_quat, p, quat, omega, nu):
    p_world = transform_point(frame_pos, frame_quat, p)
    quat_world = quat_mul(frame_quat, quat)
    omega_world = quat_rotate(frame_quat, omega)
    nu_world = torch.cross(frame_pos, omega_world, dim=-1) + quat_rotate(frame_quat, nu)
    return p_world, quat_world, omega_world, nu_world

@torch.jit.script
def convert_angular_states_w2b(frame_quat, quat, omega):
    quat_body = quat_mul(quat_inv(frame_quat), quat)
    omega_body = quat_rotate_inverse(frame_quat, omega)
    return quat_body, omega_body

@torch.jit.script
def convert_angular_states_b2w(frame_quat, quat, omega):
    quat_world = quat_mul(frame_quat, quat)
    omega_world = quat_rotate(frame_quat, omega)
    return quat_world, omega_world

def num_params_torch_model(model):
    return sum(p.numel() for p in model.parameters())

def grad_norm(params):
    grad_norm = 0.
    for p in params:
        if p.grad is not None:
            grad_norm += torch.sum(p.grad ** 2)
    return torch.sqrt(grad_norm)