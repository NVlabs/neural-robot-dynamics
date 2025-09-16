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

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='RL')
    
    parser.add_argument(
        '--cfg', type=str, default='./cfg/Cartpole/transformer.yaml',
        help='specify the config file for the run'
    )

    parser.add_argument(
        '--test', action='store_true',
        help='test a stored policy'
    )

    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help="restore the policy from a checkpoint"
    )

    parser.add_argument(
        '--logdir', type=str, default="../../data/trained_models/Cartpole/transformer/",
        help="directory of logging"
    )

    parser.add_argument(
        '--no-time-stamp', action='store_true',
        help='whether not to add the timestamp in log folder'
    )

    parser.add_argument(
        '--eval-interval', type=int, default=1,
        help='interval between policy evaluation in simulation'
    )
    
    parser.add_argument(
        '--save-interval', type=int, default=50,
        help="interval between two saved checkpoints"
    )

    parser.add_argument(
        '--seed', type=int, default=0,
        help="random seed"
    )

    parser.add_argument(
        '--log-interval', type=int, default=1,
        help="interval between two logging item"
    )

    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help="which device to use"
    )

    parser.add_argument(
        '--render', action='store_true',
        default=False,
        help="whether to render during evaluation"
    )

    parser.add_argument(
        '--num-envs', type=int,
        default=None,
        help="specify the number of envs to overwrite config"
    )

    return parser
