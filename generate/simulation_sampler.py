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

from abc import abstractmethod
from typing import Union
import warp as wp

import numpy as np

import torch

from envs.neural_environment import NeuralEnvironment

'''
Compute the contact point 1 from contact point 0, contact normal and contact depth.
Computed contact point 1 is in world frame.
'''

@wp.kernel(enable_backward=False)
def compute_contact_points_0_world(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    # outputs
    contact_point0_world: wp.array(dtype=wp.vec3)
):
    contact_id = wp.tid()
    shape = contact_shape0[contact_id]
    body = shape_body[shape]
    contact_point0_world[contact_id] = wp.transform_point(
        body_q[body], 
        contact_point0[contact_id]
    )

# [NOTE]: Assume contact point 1 is in world frame
@wp.kernel(enable_backward=False)
def compute_contact_points_1(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=float),
    # outputs
    contact_point1: wp.array(dtype=wp.vec3)
):
    contact_id = wp.tid()
    shape = contact_shape0[contact_id]
    body = shape_body[shape]
    point0_world = wp.transform_point(body_q[body], contact_point0[contact_id])

    contact_point1[contact_id] = point0_world \
        - contact_depth[contact_id] * contact_normal[contact_id]
    
class Sampler:
    """Abstract class of data sampler."""
    @abstractmethod
    def sample(
        self, 
        batch_size: int, 
        low: Union[float, torch.Tensor],
        high: Union[float, torch.Tensor],
        data: torch.Tensor
    ):
        pass

class SobolSampler(Sampler):
    """Systematic sampling using Sobol sequences."""

    def __init__(
        self,
        seed=None,
        scramble=False,
    ):
        self.scramble = scramble
        self.seed = seed

    def sample(
        self, 
        batch_size: int, 
        low: Union[float, torch.Tensor],
        high: Union[float, torch.Tensor],
        data: torch.Tensor
    ):
        soboleng = torch.quasirandom.SobolEngine(
            data.shape[1], scramble=self.scramble, seed=self.seed
        )
        soboleng.draw(batch_size, dtype=torch.float32, out=data)
        data[...] = data * (high - low) + low


class UniformSampler(Sampler):
    """Random sampling using uniform distribution."""
    def __init__(self):
        pass

    def sample(
        self, 
        batch_size: int, 
        low: Union[float, torch.Tensor],
        high: Union[float, torch.Tensor],
        data: torch.Tensor
    ):
        assert data.shape[0] == batch_size
        data.uniform_()
        data[...] = data * (high - low) + low

class WarpSimDataGenerator:
    """Generic data generator for WarpSim environments."""

    def __init__(
        self,
        env: NeuralEnvironment,
        joint_q_min: Union[float, np.ndarray],
        joint_q_max: Union[float, np.ndarray],
        joint_qd_min: Union[float, np.ndarray],
        joint_qd_max: Union[float, np.ndarray],
        joint_act_scale: Union[float, np.ndarray],
        contact_prob: float = 0.,
        sampler=UniformSampler()
    ):
        self.env = env
        self.num_envs = env.num_envs

        # joint position limits
        if isinstance(joint_q_min, np.ndarray):
            assert len(joint_q_min) == self.env.dof_q_per_env
            q_lower = joint_q_min.copy()
        else:
            q_lower = np.full(self.env.dof_q_per_env, joint_q_min)
    
        if isinstance(joint_q_max, np.ndarray):
            assert len(joint_q_max) == self.env.dof_q_per_env
            q_upper = joint_q_max.copy()
        else:
            q_upper = np.full(self.env.dof_q_per_env, joint_q_max)
        
        # joint velocity limits
        if isinstance(joint_qd_min, np.ndarray):
            assert len(joint_qd_min) == self.env.dof_qd_per_env
            qd_lower = joint_qd_min
        else:
            qd_lower = np.full(self.env.dof_qd_per_env, joint_qd_min)
        if isinstance(joint_qd_max, np.ndarray):
            assert len(joint_qd_max) == self.env.dof_qd_per_env
            qd_upper = joint_qd_max
        else:
            qd_upper = np.full(self.env.dof_qd_per_env, joint_qd_max)

        states_min = np.concatenate([q_lower, qd_lower])
        states_max = np.concatenate([q_upper, qd_upper])

        actions_min = np.full(self.env.action_dim, -1.)
        actions_max = np.full(self.env.action_dim, 1.)
        for i in range(env.action_dim):
            actions_min[i] = env.action_limits[i][0]
            actions_max[i] = env.action_limits[i][1]

        if isinstance(joint_act_scale, np.ndarray):
            assert len(joint_act_scale) == self.env.joint_act_dim
            self.joint_act_scale = torch.tensor(
                joint_act_scale, 
                dtype=torch.float32, 
                device=self.torch_device
            )
        else:
            self.joint_act_scale = torch.full(
                (self.env.joint_act_dim,),
                joint_act_scale, 
                dtype=torch.float32, 
                device=self.torch_device
            )

        self.contact_prob = contact_prob

        self.states_range = torch.tensor(
            states_max - states_min,
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.states_min = torch.tensor(
            states_min, 
            dtype=torch.float32, 
            device=self.torch_device
        )
        self.states_max = torch.tensor(
            states_max,
            dtype=torch.float32,
            device=self.torch_device
        )

        self.actions_range = torch.tensor(
            actions_max - actions_min,
            dtype=torch.float32,
            device=self.torch_device
        )
        self.actions_min = torch.tensor(
            actions_min, 
            dtype=torch.float32, 
            device=self.torch_device
        )
        self.actions_max = torch.tensor(
            actions_max, 
            dtype=torch.float32, 
            device=self.torch_device
        )

        self.sampler = sampler

    @property
    def state_dim(self):
        return self.env.state_dim

    @property
    def joint_act_dim(self):
        return self.env.joint_act_dim
    
    @property
    def action_dim(self):
        return self.env.action_dim
    
    @property
    def torch_device(self):
        return wp.device_to_torch(self.env.device)
