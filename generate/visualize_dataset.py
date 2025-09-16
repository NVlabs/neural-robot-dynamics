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

import argparse
import h5py
import numpy as np
import torch

from envs.neural_environment import NeuralEnvironment
from utils import warp_utils

def visualize_dataset(dataset_path, num_envs, num_transitions):
    dataset = h5py.File(dataset_path, 'r', swmr=True, libver='latest')
    env_name = dataset['data'].attrs['env']
    total_transitions = dataset['data'].attrs['total_transitions']
    print('total transitions = ', total_transitions)

    # create neural env
    env = NeuralEnvironment(
        env_name = env_name,
        num_envs = num_envs,
        warp_env_cfg = {},
        neural_integrator_cfg = {},
        neural_model = None,
        default_env_mode = "ground-truth",
        render = True
    )

    # load datase
    states = dataset['data']['states'][()].astype('float32').transpose(1, 0, 2)
    states = states.reshape(-1, states.shape[-1])
    next_states = dataset['data']['next_states'][()].astype('float32').transpose(1, 0, 2)
    next_states = next_states.reshape(-1, next_states.shape[-1])

    # visualize
    # reshape states and next_states to be (num_envs, total_transitions / num_envs, dofs)
    total_transitions = total_transitions // num_envs * num_envs
    states = torch.tensor(
                states[:total_transitions, :].reshape(num_envs, -1, states.shape[-1]), 
                device = warp_utils.device_to_torch(env.device))
    next_states = torch.tensor(
                    next_states[:total_transitions, :].reshape(num_envs, -1, next_states.shape[-1]), 
                    device = warp_utils.device_to_torch(env.device))

    min_vel, max_vel = np.inf, -np.inf
    for step in range(min(num_transitions, states.shape[1])):
        env.reset(initial_states = states[:, step, :])
        env.render()
    print('Min vel = {}, Max vel = {}'.format(min_vel, max_vel))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', 
                        type=str,
                        required=True,
                        help='Path to the dataset.')
    parser.add_argument('--num-transitions',
                        type=int,
                        default=1000,
                        help='Number of transitions to be visualized.')
    parser.add_argument('--num-envs',
                        type=int,
                        default=10,
                        help='The number of parallel environments.')
    
    args = parser.parse_args()

    visualize_dataset(
        dataset_path = args.dataset_path,
        num_envs = args.num_envs,
        num_transitions = args.num_transitions
    )