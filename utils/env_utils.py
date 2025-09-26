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

from envs.fixed_contact_environment import FixedContactEnvironment
import envs.newton_envs as newton_envs
from envs.newton_envs import RenderMode

ENV_CLS = {
    "Cartpole": getattr(newton_envs, "CartpoleEnvironment", None),
    "Ant": getattr(newton_envs, "AntEnvironment", None)
}

def create_fixed_contact_env(
    env_name,
    num_envs,
    use_graph_capture=False,
    requires_grad=False,
    device="cuda:0",
    render=False,
    **extra_env_args,
) -> FixedContactEnvironment:
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
        
    return FixedContactEnvironment(env, use_graph_capture=use_graph_capture)
