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

"""
This is the script to generate the results for the passive motion experiments in the paper.
"""
import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

import argparse
import torch
import yaml
import numpy as np

from envs.warp_sim_envs import RenderMode

from envs.neural_environment import NeuralEnvironment
from utils.torch_utils import num_params_torch_model
from utils.python_utils import set_random_seed
from utils import torch_utils
from utils.evaluator import NeuralSimEvaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--env-name', 
                        default = 'Pendulum',
                        type = str)
    parser.add_argument('--model-path',
                        default = None,
                        type = str)
    parser.add_argument('--dataset-path', 
                        default = None, 
                        type = str)
    parser.add_argument('--env-mode',
                        default = 'neural',
                        type = str,
                        choices = ['neural', 'ground-truth'])
    parser.add_argument('--num-envs', 
                        default = 1,
                        type = int)
    parser.add_argument('--num-rollouts',
                        default = 100,
                        type = int)
    parser.add_argument('--rollout-horizon',
                        default = 10,
                        type = int)
    parser.add_argument('--seed', 
                        default = 0,
                        type = int)
    parser.add_argument('--render', 
                        action = 'store_true')
    parser.add_argument('--export-video',
                        action = 'store_true')
    parser.add_argument('--export-video-path',
                        type = str,
                        default = 'video.gif')
    parser.add_argument('--export-usd',
                        action='store_true')
    
    args = parser.parse_args()

    device = 'cuda:0'

    set_random_seed(args.seed)

    env_cfg = {
        "env_name": args.env_name,
        "num_envs": args.num_envs,
        "render": args.render,
        "warp_env_cfg": {
            "seed": args.seed
        },
        "default_env_mode": args.env_mode,
    }
    
    if args.export_usd:
        args.render = True
        env_cfg["render"] = True
        env_cfg["warp_env_cfg"]["render_mode"] = RenderMode.USD

    # Load neural model and neural_integrator_cfg if model_path exists
    if args.model_path is not None:
        model, robot_name = torch.load(args.model_path, map_location='cuda:0')
        print('Number of Model Parameters: ', num_params_torch_model(model))
        model.to(device)
        train_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(args.model_path)), '../'
        ))
        cfg_path = os.path.join(train_dir, 'cfg.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f, Loader = yaml.SafeLoader)
        env_cfg["neural_integrator_cfg"] = cfg["env"]["neural_integrator_cfg"]
    else:
        model = None

    neural_env = NeuralEnvironment(
        neural_model = model,
        **env_cfg
    )
    
    if model is not None:
        assert neural_env.robot_name == robot_name, \
            "neural_env.robot_name is not equal to neural_model's robot_name."
        
    evaluator = NeuralSimEvaluator(
        neural_env, 
        args.dataset_path, 
        args.rollout_horizon, 
        device = device
    )

    set_random_seed(args.seed)

    next_states_diff, trajectories, _ = \
        evaluator.evaluate_action_mode(
            num_traj = args.num_rollouts,
            trajectory_source = "sampler",
            eval_mode = "rollout",
            env_mode = args.env_mode,
            passive = True,
            render = args.render,
            export_video = args.export_video,
            export_video_path = args.export_video_path
        )

    print('=========================================')
    if args.env_name == 'Cartpole':
        base_position_idx = [0]
        base_orientation_idx = []
        joint_idx = [1]
    elif args.env_name == 'PendulumWithContact':
        base_position_idx = []
        base_orientation_idx = [0]
        joint_idx = [1]
    elif args.env_name == 'Ant':
        base_position_idx = [0, 1, 2]
        base_orientation_idx = [3, 4, 5, 6]
        joint_idx = [7, 8, 9, 10, 11, 12, 13, 14]
    elif 'Anymal' in args.env_name:
        base_position_idx = [0, 1, 2]
        base_orientation_idx = [3, 4, 5, 6]
        joint_idx = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    elif args.env_name == 'CubeToss':
        base_position_idx = [0, 1, 2]
        base_orientation_idx = [3, 4, 5, 6]
        joint_idx = []
    else:
        raise NotImplementedError
    
    if len(base_position_idx) > 0:
        base_position_error = next_states_diff[..., base_position_idx].norm(dim = -1).mean()
        base_position_error_std = next_states_diff[..., base_position_idx].norm(dim = -1).std()
    else:
        base_position_error = None
    
    if len(base_orientation_idx) > 0:
        if len(base_orientation_idx) == 1:
            base_orientation_error = next_states_diff[..., base_orientation_idx].abs().mean()
            base_orientation_error_std = next_states_diff[..., base_orientation_idx].abs().std()
        else:
            quat_rollout = trajectories['rollout_states'][1:, :, base_orientation_idx]
            quat_target = quat_rollout + next_states_diff[..., base_orientation_idx]
            quat_rollout = quat_rollout.view(-1, 4)
            quat_target = quat_target.view(-1, 4)
            quat_angle_diff = torch_utils.quat_angle_diff(quat_rollout, quat_target)
            base_orientation_error = quat_angle_diff.mean()
            base_orientation_error_std = quat_angle_diff.std()
    else:
        base_orientation_error = None

    if base_position_error is not None:
        print("{:<30} = {:.6f}".format(
            "Base position error mean", base_position_error.cpu().item()
        ))
        print("{:<30} = {:.6f}".format(
            "Base position error std", base_position_error_std.cpu().item()
        ))
    if base_orientation_error is not None:
        print("{:<30} = {:.6f} rad ({:.6f} deg)".format(
            "Base orientaion error mean",
            base_orientation_error.cpu().item(), 
            np.rad2deg(base_orientation_error.cpu().item())
        ))
        print("{:<30} = {:.6f} rad ({:.6f} deg)".format(
            "Base orientation error std",
            base_orientation_error_std.cpu().item(),
            np.rad2deg(base_orientation_error_std.cpu().item())
        ))
    if len(joint_idx) > 0:
        joint_pos_error = next_states_diff[..., joint_idx].abs().mean()
        print('{:<30} = {:.6f} rad ({:.6f} deg)'.format(
            "Joint position error mean",
            joint_pos_error.cpu().item(), 
            np.rad2deg(joint_pos_error.cpu().item())
        ))

    print("{:<30} = {}".format(
        "Joint position Error per dof", 
        next_states_diff[..., joint_idx].abs().mean((0, 1))
    ))
    print('=========================================')
    
    if args.export_usd:
        neural_env.save_usd()
