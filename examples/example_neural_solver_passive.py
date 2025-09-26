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
import yaml

from envs.neural_environment import NeuralEnvironment
from utils.python_utils import set_random_seed
from utils.torch_utils import num_params_torch_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env-name', 
        type=str, 
        default='Cartpole',
        choices=["Cartpole", "Ant"]
    )
    parser.add_argument(
        '--num-envs', 
        type=int, 
        default=1
    )
    parser.add_argument(
        '--solver-type', 
        type=str, 
        default='neural', 
        choices=['neural', 'ground-truth']
    )
    parser.add_argument(
        '--nerd-model-path',
        type=str,
        default=None
    )
    parser.add_argument(
        '--use-graph-capture', 
        action='store_true'
    )
    parser.add_argument(
        '--num-rollouts', 
        type=int, 
        default=1
    )
    parser.add_argument(
        '--rollout-horizon', 
        type=int, 
        default=500
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1234
    )

    args = parser.parse_args()
    
    if args.nerd_model_path is None:
        if args.env_name == "Cartpole":
            args.nerd_model_path = '../pretrained_models/NeRD_models/Cartpole/model/nn/model.pt'
        elif args.env_name == "Ant":
            args.nerd_model_path = '../pretrained_models/NeRD_models/Ant/model/nn/model.pt'
        else:
            raise ValueError(f"Environment {args.env_name} not supported")
    
    device = 'cuda:0'

    # Construct neural environment
    newton_env_cfg = {
        "seed": args.seed,
        "random_reset": True
    }
    neural_dynamics_model, neural_solver_cfg = None, None
    if args.solver_type == "neural":
        neural_dynamics_model, robot_name = torch.load(
            args.nerd_model_path, 
            map_location='cuda:0', 
            weights_only=False
        )
        print('Number of Model Parameters: ', num_params_torch_model(neural_dynamics_model))
        neural_dynamics_model.to(device)
        neural_dynamics_model.fix_input_names() # Compatibility with the old models
        model_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(args.nerd_model_path)), '../'
        ))
        cfg_path = os.path.join(model_dir, 'cfg.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        neural_solver_cfg = cfg["env"]["neural_solver_cfg"]

    neural_env = NeuralEnvironment(
        env_name = args.env_name,
        num_envs = args.num_envs,
        newton_env_cfg = newton_env_cfg,
        default_env_mode = args.solver_type,
        neural_model = neural_dynamics_model,
        neural_solver_cfg = neural_solver_cfg,
        use_graph_capture = args.use_graph_capture,
        render = True
    )

    set_random_seed(args.seed)

    # Rollout the passive-motion trajectory
    num_rounds = (args.num_rollouts - 1) // args.num_envs + 1
    rollout_states = torch.zeros(
        num_rounds * args.num_envs,
        args.rollout_horizon + 1,
        neural_env.state_dim,
        device = device
    )

    for round_idx in range(num_rounds):
        neural_env.reset()
        neural_env.init_rnn(neural_env.num_envs)
    
        rollout_states[
            round_idx * args.num_envs:(round_idx + 1) * args.num_envs, 
            0, 
            :
        ].copy_(neural_env.states)

        for step in range(args.rollout_horizon):
            rollout_states[
                round_idx * args.num_envs:(round_idx + 1) * args.num_envs,
                step + 1,
                :
            ].copy_(
                neural_env.step(
                    torch.zeros(
                        (neural_env.num_envs, neural_env.action_dim),
                        device = device
                    )
                )
            )

            neural_env.render()
            
        


    