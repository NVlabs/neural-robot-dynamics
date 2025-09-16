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

import os
import sys

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
)
sys.path.append(base_dir)

import argparse
import yaml
import numpy as np
from rl_games.torch_runner import Runner

from envs.rlgames_env_wrapper import RLGPUAlgoObserver
from utils.python_utils import set_random_seed, print_ok
from eval.eval_rl.run_rl import (
    construct_env, construct_rlg_config, evaluate_policy
)

def init_rlg_runner(policy_path, args):
    # compose rl config
    policy_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(policy_path)), '../')
        )
    rl_cfg_path = os.path.join(policy_dir, 'rl_cfg.yaml')
    
    with open(rl_cfg_path, 'r') as f:
        rl_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    rl_cfg['rl']['config']['player']['games_num'] = args.num_games
    rl_cfg['seed'] = args.seed
    rl_cfg['env']['num_envs'] = args.num_envs
    
    # compose rlg_config
    rlg_config_dict =construct_rlg_config(rl_cfg)
    
    # init runner
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()
    
    return runner
        
parser = argparse.ArgumentParser()

parser.add_argument('--eval-cfg', type=str, required=True)
parser.add_argument('--num-envs', type=int, default=1024)
parser.add_argument('--num-games', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--render', action='store_true')

args = parser.parse_args()

args.render_mode = 'human'
args.export_usd = False

eval_cfg = yaml.load(open(args.eval_cfg, 'r'), Loader=yaml.SafeLoader)

policy_variants = eval_cfg['policies']
sim_variants = eval_cfg['simulators']
env_cfg = eval_cfg['env']

env_cfg['num_envs'] = args.num_envs

for sim_idx, sim_name in enumerate(sim_variants.keys()):
    print_ok(f"==================================================== Evaluating {sim_name} simulation ====================================================")
    # construct the environment
    sim_cfg = sim_variants[sim_name]
    env_cfg['env_mode'] = sim_cfg['env_mode']
    env_cfg['model_path'] = sim_cfg.get('model_path', None)
    env = construct_env(env_cfg, 'cuda:0', args)
    for policy_idx, policy_name in enumerate(policy_variants.keys()):
        agg_results = {'reward': 0, 'steps': 0}
        policy_seeds = os.listdir(policy_variants[policy_name]['policy_folder'])
        policy_subpath = policy_variants[policy_name]['policy_path']
        all_rewards = []
        for policy_seed in policy_seeds:
            policy_path = os.path.join(
                policy_variants[policy_name]['policy_folder'], 
                policy_seed, 
                policy_subpath
            )

            # build rlg runner
            runner = init_rlg_runner(policy_path, args)
            # evaluate
            set_random_seed(args.seed)
            # NOTE: Official rl-games doesn't have returned summary
            evaluate_policy(runner, policy_path)
 