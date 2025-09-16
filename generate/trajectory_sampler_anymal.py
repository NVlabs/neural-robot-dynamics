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

import warp as wp

from typing import List, Dict
import numpy as np
from tqdm import tqdm

from generate.trajectory_sampler import TrajectorySampler
from utils import torch_utils

import torch

"""
Trajectory-mode dataset generator for ANYmal env.
Have a specialized initial states sampler and a action-mode generator for ANYmal.
"""
class TrajectorySamplerAnymal(TrajectorySampler):
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
    
    def sample_trajectories_action_mode(
        self,
        num_transitions: int, 
        trajectory_length: int,
        passive: bool = False, 
        initial_states_source: str = 'sample',
        render: bool = False,
        export_video: bool = False,
        export_video_path: str = None
    ) -> List[Dict]:
        
        rollout_batches = {
            'gravity_dir': [],
            'root_body_q': [],
            'states': [],
            'contacts': {
                'contact_normals': [],
                'contact_depths': [],
                'contact_points_0': [],
                'contact_points_1': [],
                'contact_thicknesses': []
            },
            'actions': [],
            'joint_acts': [],
            'next_states': []
        }
        
        # allocate buffers
        states = torch.empty(
            trajectory_length, 
            self.num_envs,
            self.state_dim,
            dtype=torch.float32,
            device=self.torch_device
        )
        next_states = torch.empty(
            trajectory_length, 
            self.num_envs,
            self.state_dim,
            dtype=torch.float32,
            device=self.torch_device
        )
        actions = torch.empty(
            trajectory_length,
            self.num_envs,
            self.action_dim,
            dtype=torch.float32,
            device=self.torch_device
        )
        joint_acts = torch.empty(
            trajectory_length, 
            self.num_envs,
            self.joint_act_dim,
            dtype=torch.float32,
            device=self.torch_device
        )
        root_body_q = torch.empty(
            trajectory_length,
            self.num_envs,
            7,
            dtype=torch.float32,
            device=self.torch_device
        )
        gravity_dir = torch.empty(
            trajectory_length, 
            self.num_envs,
            3,
            dtype=torch.float32,
            device=self.torch_device
        )
        gravity_dir[:, :, self.env.model.up_axis] = -1.0
        
        num_contacts_per_env = self.env.abstract_contacts.num_contacts_per_env
        
        contact_normals = torch.empty(
            trajectory_length,
            self.num_envs,
            num_contacts_per_env,
            3,
            dtype=torch.float32,
            device=self.torch_device
        )
        
        contact_depths = torch.empty(
            trajectory_length,
            self.num_envs,
            num_contacts_per_env,
            dtype=torch.float32,
            device=self.torch_device
        )
        contact_points_0 = torch.empty(
            trajectory_length, 
            self.num_envs,
            num_contacts_per_env,
            3,
            dtype=torch.float32,
            device=self.torch_device
        )
        contact_points_1 = torch.empty(
            trajectory_length, 
            self.num_envs,
            num_contacts_per_env,
            3,
            dtype=torch.float32,
            device=self.torch_device
        )
        contact_thicknesses = torch.empty(
            trajectory_length,
            self.num_envs,
            num_contacts_per_env,
            dtype=torch.float32,
            device=self.torch_device
        )
        
        self.env.set_env_mode('ground-truth')
        
        _eval_collisions = self.env.eval_collisions
        self.env.set_eval_collisions(True)

        if export_video:
            self.env.start_video_export(export_video_path)
        
        initial_states = torch.empty(
            self.num_envs,
            self.state_dim,
            dtype=torch.float32,
            device=self.torch_device
        )
        
        rounds = 0
        total_transitions = 0
        progress_bar = tqdm(total = num_transitions)
        while total_transitions < num_transitions:
            rounds += 1

            if initial_states_source == 'env':
                # randomize initial states according to env
                self.env.reset()
            elif initial_states_source == 'sample':
                # randomize initial states according to given ranges
                self.sample_initial_states(self.num_envs, initial_states)
                self.env.reset(initial_states = initial_states)
            else:
                raise NotImplementedError
            
            # generate random joint_acts
            if self.action_dim:
                self.sampler.sample(
                    batch_size = self.num_envs * trajectory_length, 
                    low = self.actions_min,
                    high = self.actions_max, 
                    data = actions.view(-1, self.action_dim)
                )
            
                if passive:
                    actions *= 0.
            else:
                actions = torch.zeros(
                    trajectory_length, 
                    self.num_envs, 
                    self.action_dim, 
                    device=self.torch_device
                )

            for step in range(trajectory_length):
                if render:
                    self.env.render()
                    
                root_body_q[step, :, :].copy_(self.env.root_body_q)
                states[step, :, :].copy_(self.env.states)
                next_states[step, :, :].copy_(
                    self.env.step(
                        actions[step, :, :],
                        env_mode = 'ground-truth'
                    )
                )
                
                # Acquire joint_act data from env
                joint_acts[step, :, :].copy_(self.env.joint_acts)
                
                # Acquire contact data from env
                contact_points_0[step, ...] = \
                    self.env.abstract_contacts.contact_point0.view(
                        self.num_envs,
                        num_contacts_per_env,
                        3,
                    ).clone()
                
                contact_points_1[step, ...] = \
                    self.env.abstract_contacts.contact_point1.view(
                        self.num_envs, 
                        num_contacts_per_env, 
                        3
                    ).clone()

                contact_thicknesses[step, ...] = \
                    self.env.abstract_contacts.contact_thickness.view(
                        self.num_envs, 
                        num_contacts_per_env
                    ).clone()
                
                contact_depths[step, ...] = \
                    self.env.abstract_contacts.contact_depth.view(
                        self.num_envs,
                        num_contacts_per_env
                    ).clone()
                    
                contact_normals[step, ...] = \
                    self.env.abstract_contacts.contact_normal.view(
                        self.num_envs,
                        num_contacts_per_env,
                        3
                    ).clone()

            # remove invalid trajectories (i.e. NaN, inf, or large values)
            invalid_masks = (
                next_states.isnan().any(dim = 0, keepdim = True).any(dim = 2, keepdim = True) |
                next_states.isinf().any(dim = 0, keepdim = True).any(dim = 2, keepdim = True) |
                (next_states > 1e5).any(dim = 0, keepdim = True).any(dim = 2, keepdim = True) |
                (next_states < -1e5).any(dim = 0, keepdim = True).any(dim = 2, keepdim = True)
            )
            valid_masks = (~invalid_masks)[0, :, 0]

            # save to trajectories
            rollout_batches['gravity_dir'].append(
                gravity_dir[:, valid_masks, ...].detach().cpu().clone()
            )
            rollout_batches['root_body_q'].append(
                root_body_q[:, valid_masks, ...].detach().cpu().clone()
            )
            rollout_batches['states'].append(
                states[:, valid_masks, ...].detach().cpu().clone()
            )
            rollout_batches['contacts']['contact_normals'].append(
                contact_normals[:, valid_masks, ...].detach().cpu().clone()
            )
            rollout_batches['contacts']['contact_depths'].append(
                contact_depths[:, valid_masks, ...].detach().cpu().clone()
            )
            rollout_batches['contacts']['contact_points_0'].append(
                contact_points_0[:, valid_masks, ...].detach().cpu().clone()
            )
            rollout_batches['contacts']['contact_points_1'].append(
                contact_points_1[:, valid_masks, ...].detach().cpu().clone()
            )
            rollout_batches['contacts']['contact_thicknesses'].append(
                contact_thicknesses[:, valid_masks, ...].detach().cpu().clone()
            )
            rollout_batches['actions'].append(
                actions[:, valid_masks, ...].detach().cpu().clone()
            )
            rollout_batches['joint_acts'].append(
                joint_acts[:, valid_masks, ...].detach().cpu().clone()
            )
            rollout_batches['next_states'].append(
                next_states[:, valid_masks, ...].detach().cpu().clone()
            )

            total_transitions += valid_masks.sum().item() * trajectory_length
            progress_bar.update(valid_masks.sum().item() * trajectory_length)

            if invalid_masks.sum().item() > 0:
                print(f'Invalid trajectories removed: {invalid_masks.sum().item()}')

        print(f'\n\nTotal number of transitions generated: {total_transitions}')

        # merge rollout batches
        rollouts = {}
        for key in rollout_batches:
            if isinstance(rollout_batches[key], dict):
                rollouts[key] = {}
                for sub_key in rollout_batches[key]:
                    rollouts[key][sub_key] = torch.cat(
                        rollout_batches[key][sub_key], 
                        dim = 1
                    )
            else:
                rollouts[key] = torch.cat(rollout_batches[key], dim = 1)

        print('[DEBUG] sum(next_states) = ', rollouts['next_states'].sum())
        print('[DEBUG] min(next_states) = ', rollouts['next_states'].min())
        print('[DEBUG] max(next_states) = ', rollouts['next_states'].max())
        
        print('[DEBUG] min(joint_act) = ', rollouts['joint_acts'].view(-1, 12).min(dim=0).values)
        print('[DEBUG] max(joint_act) = ', rollouts['joint_acts'].view(-1, 12).max(dim=0).values)

        self.env.set_eval_collisions(_eval_collisions)
        
        if export_video:
            self.env.end_video_export()
            
        return rollouts