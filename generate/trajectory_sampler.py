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

from typing import List, Dict, Union
import numpy as np
from tqdm import tqdm, trange

import warp as wp

from envs.neural_environment import NeuralEnvironment
from generate.simulation_sampler import WarpSimDataGenerator, UniformSampler

import torch

"""
Base trajectory-mode dataset generator.
In this base generator class, we assume the ground is fixed as defined in the env setup, 
and we use the collision detection in the abstract_contact_environment 
to determine the contact information.
"""
class TrajectorySampler(WarpSimDataGenerator):

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
        super().__init__(env, 
                         joint_q_min, joint_q_max, 
                         joint_qd_min, joint_qd_max,
                         joint_act_scale,
                         contact_prob,
                         sampler)

    def sample_initial_states(self, num_envs, initial_states):
        if self.env.model.joint_type.numpy()[0] == wp.sim.JOINT_FREE:
            raise NotImplementedError
        self.sampler.sample(
            batch_size = num_envs, 
            low = self.states_min,
            high = self.states_max, 
            data = initial_states
        )
        
    def sample_trajectories_joint_act_mode(
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
            if self.joint_act_dim:
                self.sampler.sample(
                    batch_size = self.num_envs * trajectory_length, 
                    low = -self.joint_act_scale,
                    high = self.joint_act_scale,
                    data = joint_acts.view(-1, self.joint_act_dim)
                )
            
                if passive:
                    joint_acts *= 0.

            for step in range(trajectory_length):
                if render:
                    self.env.render()
                    
                root_body_q[step, :, :].copy_(self.env.root_body_q)
                states[step, :, :].copy_(self.env.states)
                next_states[step, :, :].copy_(
                    self.env.step_with_joint_act(
                        joint_acts[step, :, :],
                        env_mode = 'ground-truth'
                    )
                )
                
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
                gravity_dir[:, valid_masks, ...].clone()
            )
            rollout_batches['root_body_q'].append(
                root_body_q[:, valid_masks, ...].clone()
            )
            rollout_batches['states'].append(
                states[:, valid_masks, ...].clone()
            )
            rollout_batches['contacts']['contact_normals'].append(
                contact_normals[:, valid_masks, ...].clone()
            )
            rollout_batches['contacts']['contact_depths'].append(
                contact_depths[:, valid_masks, ...].clone()
            )
            rollout_batches['contacts']['contact_points_0'].append(
                contact_points_0[:, valid_masks, ...].clone()
            )
            rollout_batches['contacts']['contact_points_1'].append(
                contact_points_1[:, valid_masks, ...].clone()
            )
            rollout_batches['contacts']['contact_thicknesses'].append(
                contact_thicknesses[:, valid_masks, ...].clone()
            )
            rollout_batches['joint_acts'].append(
                joint_acts[:, valid_masks, ...].clone()
            )
            rollout_batches['next_states'].append(
                next_states[:, valid_masks, ...].clone()
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
                    rollouts[key][sub_key] = torch.cat(rollout_batches[key][sub_key], dim = 1)
            else:
                rollouts[key] = torch.cat(rollout_batches[key], dim = 1)
        
        self.env.set_eval_collisions(_eval_collisions)
        
        if export_video:
            self.env.end_video_export()
            
        return rollouts

    '''
    Sample trajectories with random initial states and random action sequences, 
    using env's collision detection to determine contact configurations.
    
    NOTE[Jie]: assume no early termination happens
    '''
    def sample_trajectories_action_mode(
        self, 
        num_transitions: int, 
        trajectory_length: int, 
        passive: bool = False
    ) -> List[Dict]:

        rollout_batches = {
            'states': [],
            'actions': [],
            'next_states': []
        }

        progress = trange(
            0, 
            num_transitions, 
            self.num_envs * trajectory_length, 
            desc="Sampling state transitions"
        )
        
        # allocate buffers
        states = torch.empty(
            trajectory_length, 
            self.num_envs,
            self.state_dim,
            dtype=torch.float32,
            device=self.torch_device)
        next_states = torch.empty(
            trajectory_length, 
            self.num_envs,
            self.state_dim,
            dtype=torch.float32,
            device=self.torch_device)
        actions = torch.empty(
            trajectory_length, 
            self.num_envs,
            self.action_dim,
            dtype=torch.float32,
            device=self.torch_device)
        
        self.env.set_env_mode('ground-truth')
        
        _eval_collisions = self.env.eval_collisions
        self.env.set_eval_collisions(True)
        
        rounds = 0
        for _ in progress:
            rounds += 1

            # randomly reset initial states by env
            self.env.reset()
            
            if self.action_dim > 0:
                # randomly sample actions
                self.sampler.sample(
                    batch_size = self.num_envs * trajectory_length, 
                    low = self.actions_min,
                    high = self.actions_max, 
                    data = actions.view(-1, self.action_dim)
                )
            else:
                actions = torch.zeros(
                    trajectory_length, 
                    self.num_envs, 
                    self.action_dim, 
                    device=self.torch_device
                )

            if passive:
                actions *= 0.

            for step in range(trajectory_length):
                states[step, :, :].copy_(self.env.states)
                next_states[step, :, :].copy_(
                    self.env.step(
                        actions[step, :, :],
                        env_mode = 'ground-truth'
                    )
                )

            # save to trajectories
            rollout_batches['states'].append(states.clone())
            rollout_batches['actions'].append(actions.clone())
            rollout_batches['next_states'].append(next_states.clone())

        # merge rollout batches
        rollouts = {}
        for key in rollout_batches:
            rollouts[key] = torch.cat(rollout_batches[key], dim = 1)
        
        self.env.set_eval_collisions(_eval_collisions)
        
        return rollouts