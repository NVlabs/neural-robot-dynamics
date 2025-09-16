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

import yaml

import warp as wp
wp.config.verify_cuda = True

from arguments import get_parser
from utils.python_utils import get_time_stamp, \
    set_random_seed, solve_argv_conflict, handle_cfg_overrides
from algorithms.vanilla_trainer import VanillaTrainer
from algorithms.sequence_model_trainer import SequenceModelTrainer
from envs.neural_environment import NeuralEnvironment

def add_additional_params(parser):
    parser.add_argument(
        '--cfg-overrides', default="", type=str)
    return parser

if __name__ == '__main__':
    args_list = ['--cfg', './cfg/Ant/transformer.yaml',
                 '--logdir', '../../data/trained_models/Ant/test/']

    solve_argv_conflict(args_list)

    parser = get_parser()

    parser = add_additional_params(parser)

    args = parser.parse_args(args_list + sys.argv[1:])

    # load config
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader = yaml.SafeLoader)

    # handle parser overrides
    handle_cfg_overrides(args.cfg_overrides, cfg)

    if not args.no_time_stamp:
        time_stamp = get_time_stamp()
        args.logdir = os.path.join(args.logdir, time_stamp)
        
    # cfg parameter overwrite
    if args.num_envs is not None:
        cfg['env']['num_envs'] = args.num_envs

    cfg['env']['render'] = args.render
    
    if args.seed is None:
        if cfg['algorithm'].get('seed', None) is not None:
            args.seed = cfg['algorithm']['seed']
        else:
            args.seed = 0

    cfg['algorithm']['seed'] = args.seed
    set_random_seed(args.seed)

    args.train = not args.test

    # create cli sub-config in cfg
    vargs = vars(args)
    cfg["cli"] = {}
    for key in vargs.keys():
        cfg["cli"][key] = vargs[key]
    cfg["cli"]['train'] = args.train
    # delete parameters that are already in cfg to avoid ambiguity
    del cfg["cli"]["num_envs"] 
    del cfg["cli"]["seed"]

    """ Create env """
    neural_integrator_name = cfg['env']['neural_integrator_cfg']['name']
    
    neural_env = NeuralEnvironment(**cfg['env'], device = args.device)

    """ Create algorithm """
    algorithm_name = cfg['algorithm'].get('name', 'VanillaTrainer')
    if algorithm_name == 'VanillaTrainer':
        assert neural_integrator_name == 'NeuralIntegrator'
        algo = VanillaTrainer(
            neural_env=neural_env,
            model_checkpoint_path=args.checkpoint,
            cfg=cfg,
            device=args.device
        )
    elif algorithm_name == 'SequenceModelTrainer':
        # some sanity check for the consistency of config file
        if 'transformer' in cfg['network']:
            assert neural_integrator_name == 'TransformerNeuralIntegrator'
            assert (
                cfg['env']['neural_integrator_cfg'].get('num_states_history') ==
                cfg['algorithm']['sample_sequence_length']
            ), (
                "'num_states_history' needs to be the same as " 
                "'sample_sequence_length' in the train config for Transformer."
            )
        elif 'rnn' in cfg['network']:
            assert neural_integrator_name == 'RNNNeuralIntegrator'
            assert (
                cfg['env']['neural_integrator_cfg'].get('reset_seq_length', 1) ==
                cfg['algorithm']['sample_sequence_length']
            ), (
                "'reset_seq_length' needs to be the same as "
                "'sample_sequence_length' in the train config for RNN."
            )
        else:
            raise NotImplementedError
        
        algo = SequenceModelTrainer(
            neural_env=neural_env,
            model_checkpoint_path=args.checkpoint,
            cfg=cfg,
            device=args.device
        )
    else:
        raise NotImplementedError(f'Algorithm {algorithm_name} not recognized')
    
    if args.train:
        algo.train()
    else:
        algo.test()