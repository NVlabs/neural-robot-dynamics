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

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(base_dir)

import argparse 
import torch
from envs.neural_environment import NeuralEnvironment
from utils.env_utils import ENV_CLS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', 
                        type=str, 
                        default='Cartpole',
                        choices=ENV_CLS.keys())
    parser.add_argument('--num-envs',
                        type=int,
                        default=1)
    parser.add_argument('--random-reset',
                        action='store_true')
    parser.add_argument('--export-video',
                        action = 'store_true')
    parser.add_argument('--export-video-path',
                        type = str,
                        default = 'video.gif')
    
    args = parser.parse_args()

    warp_env_cfg = {'seed': 0,
                    'random_reset': args.random_reset}
    
    env = NeuralEnvironment(
        env_name = args.env_name,
        num_envs = args.num_envs,
        warp_env_cfg = warp_env_cfg,
        default_env_mode = "ground-truth",
        render = True,
    )
                            
    for i in range(10):
        states_min = torch.full((env.state_dim,), torch.inf, device = 'cuda:0')
        states_max = torch.full((env.state_dim,), -torch.inf, device = 'cuda:0')
        if args.export_video and i == 0:
            env.start_video_export(args.export_video_path)
        env.reset()
        states = env.states.clone()
        pos_prev, rot_prev, joint_q_prev = (
            states[:, 0:3], states[:, 3:7], states[:, 7:15]
        )
        for j in range(1000):
            actions = torch.zeros(
                (args.num_envs, env.action_dim), 
                device = 'cuda:0'
            )
            # actions = torch.rand((args.num_envs, env.action_dim), device = 'cuda:0') * 2. - 1.   
            states = env.step(actions).clone()
            states_min = torch.minimum(states_min, states.min(dim = 0).values)
            states_max = torch.maximum(states_max, states.max(dim = 0).values)

            for k in range(1):
                env.render()
        print(states_min, states_max)
        if args.export_video and i == 0:
            env.end_video_export()
    
    