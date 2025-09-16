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

import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import numpy as np

from generate.trajectory_sampler import TrajectorySampler
from utils import torch_utils

import torch

"""
Trajectory-mode dataset generator for Ant env.
Have a specialized initial states sampler for Ant.
"""
class TrajectorySamplerAnt(TrajectorySampler):
    """
    Data generator that samples random trajectories with a fixed ground assumption.
    For fixed ground, we use the contact settings from the env.
    """

    def sample_initial_states(self, num_envs, initial_states):
        states_body = torch.empty_like(initial_states)
        self.sampler.sample(
            batch_size = num_envs, 
            low = self.states_min,
            high = self.states_max,
            data = states_body
        )
        
        # generate a random orientation for base
        angle = states_body[:, 3:4] * np.pi / 2.
        axis = states_body[:, 4:7]
        axis = torch.nn.functional.normalize(axis, p = 2.0, dim = -1)
        default_exp = torch.tensor(
            [-np.pi * 0.5, 0.0, 0.0], 
            device = initial_states.device
        ).unsqueeze(0).expand(num_envs, 3)
        default_quat = torch_utils.exponential_coord_to_quat(default_exp)
        delta_quat = torch_utils.exponential_coord_to_quat(axis * angle)
        states_body[:, 3:7] = torch_utils.quat_mul(default_quat, delta_quat)
        
        # convert omega and nu of base from local frame to world frame
        omega_body = states_body[:, self.env.dof_q_per_env:self.env.dof_q_per_env + 3]
        nu_body = states_body[:, self.env.dof_q_per_env + 3:self.env.dof_q_per_env + 6]
        omega_world = torch_utils.quat_rotate(
            states_body[:, 3:7], omega_body
        )
        nu_world = torch.cross(states_body[:, 0:3], omega_world, dim = -1) \
            + torch_utils.quat_rotate(states_body[:, 3:7], nu_body)
            
        initial_states.copy_(states_body)
        initial_states[
            :, self.env.dof_q_per_env:self.env.dof_q_per_env + 3
        ].copy_(omega_world)
        initial_states[
            :, self.env.dof_q_per_env + 3:self.env.dof_q_per_env + 6
        ].copy_(nu_world)