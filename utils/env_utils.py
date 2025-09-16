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

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
)
sys.path.append(base_dir)

from envs.abstract_contact_environment import AbstractContactEnvironment
import envs.warp_sim_envs as warp_sim_envs
from envs.warp_sim_envs import RenderMode

ENV_CLS = {
    "Cartpole": getattr(warp_sim_envs, "CartpoleEnvironment", None),
    "PendulumWithContact": getattr(warp_sim_envs, "PendulumWithContactEnvironment", None),
    "FrankaReach": getattr(warp_sim_envs, "FrankaPandaEnvironment", None),
    "Ant": getattr(warp_sim_envs, "AntEnvironment", None),
    "CubeToss": getattr(warp_sim_envs, "CubeTossingEnvironment", None),
    "Anymal": getattr(warp_sim_envs, "AnymalEnvironment", None),
    "AnymalJointPositionControl": getattr(warp_sim_envs, "AnymalJointPositionControlEnvironment", None),
}

def create_abstract_contact_env(
    env_name,
    num_envs,
    requires_grad=False,
    device="cuda:0",
    render=False,
    **extra_env_args,
) -> AbstractContactEnvironment:
    assert env_name in ENV_CLS, f"No environment named {env_name}."
    env_args = {}
    env_args["num_envs"] = num_envs
    if not render:
        env_args['env_offset'] = (0.0, 0.0, 0.0)
    env_args["requires_grad"] = requires_grad
    env_args["use_graph_capture"] = False
    env_args["device"] = device
    if not render:
        env_args["render_mode"] = RenderMode.NONE
    for key in extra_env_args.keys():
        env_args[key] = extra_env_args[key]
    env = ENV_CLS[env_name](**env_args)
        
    return AbstractContactEnvironment(env)
