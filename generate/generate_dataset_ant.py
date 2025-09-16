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
Generate dataset for Ant
"""
import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import os
import argparse
import h5py

from generate.trajectory_sampler_ant import TrajectorySamplerAnt
from utils.python_utils import set_random_seed
from envs.neural_environment import NeuralEnvironment
from utils.commons import (
    JOINT_Q_MIN, JOINT_Q_MAX, 
    JOINT_QD_MIN, JOINT_QD_MAX, 
    JOINT_ACT_SCALE
)

def collect_dataset(
    env_name, 
    num_envs, 
    initial_states_source,
    num_transitions, 
    trajectory_length, 
    dataset_path, 
    passive = False,
    seed = 0, 
    render = False,
    export_video = False,
    export_video_path = None
):
    data_writer = h5py.File(dataset_path, 'w')
    data_grp = data_writer.create_group('data')
    data_grp.attrs['env'] = env_name
    data_grp.attrs['mode'] = "trajectory"
    
    env = NeuralEnvironment(
        env_name = env_name,
        num_envs = num_envs,
        neural_integrator_cfg = {},
        neural_model = None,
        default_env_mode = "ground-truth",
        warp_env_cfg = {
            "seed": seed,
        },
        render = render
    )
    
    robot_name = env.robot_name
    simulation_sampler = TrajectorySamplerAnt(
                            env,
                            joint_q_min = JOINT_Q_MIN[robot_name],
                            joint_q_max = JOINT_Q_MAX[robot_name],
                            joint_qd_min = JOINT_QD_MIN[robot_name],
                            joint_qd_max = JOINT_QD_MAX[robot_name],
                            joint_act_scale = JOINT_ACT_SCALE.get(robot_name, 0.0),
                            contact_prob = 0.
                        )
    
    rollouts = \
        simulation_sampler.sample_trajectories_joint_act_mode(
            num_transitions, 
            trajectory_length, 
            passive,
            initial_states_source=initial_states_source,
            render=render,
            export_video=export_video,
            export_video_path=export_video_path
        )

    data_grp.attrs['total_trajectories'] = rollouts['states'].shape[1]
    data_grp.attrs['total_transitions'] = rollouts['states'].shape[0] * rollouts['states'].shape[1]

    data_grp.create_dataset(
        name = 'gravity_dir',
        data = rollouts['gravity_dir'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'root_body_q',
        data = rollouts['root_body_q'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'states', 
        data = rollouts['states'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'contact_normals',
        data = rollouts['contacts']['contact_normals'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'contact_depths',
        data = rollouts['contacts']['contact_depths'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'contact_points_0',
        data = rollouts['contacts']['contact_points_0'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'contact_points_1',
        data = rollouts['contacts']['contact_points_1'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'contact_thicknesses',
        data = rollouts['contacts']['contact_thicknesses'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'joint_acts', 
        data = rollouts['joint_acts'].detach().cpu().numpy()
    )
    data_grp.create_dataset(
        name = 'next_states', 
        data = rollouts['next_states'].detach().cpu().numpy()
    )
    data_grp.attrs['state_dim'] = rollouts['states'].shape[-1]
    data_grp.attrs['contact_prob'] = 0.
    data_grp.attrs['num_contacts_per_env'] = rollouts['contacts']['contact_depths'].shape[-1]
    data_grp.attrs['joint_act_dim'] = rollouts['joint_acts'].shape[-1]
    data_grp.attrs['next_state_dim'] = rollouts['next_states'].shape[-1]

    data_writer.flush()
    data_writer.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', 
                        type=str, 
                        default='../../data/datasets/',
                        help='Directory to store the generated datasets.')
    parser.add_argument('--env-name', 
                        type=str, 
                        default='Ant', 
                        choices=['Ant'],
                        help='Environment to generate the dataset.' )
    parser.add_argument('--initial-states-source',
                        type=str,
                        default='sample',
                        choices = ['sample', 'env'])
    parser.add_argument('--num-transitions', 
                        type=int,
                        default=1000000,
                        help='The total number of transitions to be collected. ')
    parser.add_argument('--trajectory-length', 
                        type=int,
                        default=100,
                        help='The length of each trajectory. Valid only if mode is trajectory.')
    parser.add_argument('--passive',
                        action='store_true',
                        help="Whether use passive simulation.")
    parser.add_argument('--dataset-name',
                        type=str,
                        default='dataset.hdf5',
                        help='The filename of the newly collected dataset.')
    parser.add_argument('--num-envs',
                        type=int,
                        default=1024,
                        help='The number of parallel environments.')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='The random seed for sampling.')
    parser.add_argument('--render',
                        action='store_true',
                        help='Whether to render the simulation.')
    parser.add_argument('--export-video',
                        action = 'store_true')
    parser.add_argument('--export-video-path',
                        type = str,
                        default = 'video.gif')
    
    args = parser.parse_args()

    set_random_seed(args.seed)

    assert args.env_name == "Ant"
    
    dataset_path = os.path.join(args.dataset_dir, args.env_name, args.dataset_name)
    if os.path.exists(dataset_path):
        answer = input(f'Dataset exists in the specified path {dataset_path}, do you want to clean the old dataset [y/n]')
        if answer != 'y' and answer != 'Y':
            exit()
    
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    collect_dataset(
        env_name=args.env_name, 
        num_envs=args.num_envs,
        initial_states_source=args.initial_states_source,
        num_transitions=args.num_transitions, 
        trajectory_length=args.trajectory_length,
        dataset_path=dataset_path,
        passive=args.passive,
        seed=args.seed,
        render=args.render,
        export_video=args.export_video,
        export_video_path=args.export_video_path
    )

