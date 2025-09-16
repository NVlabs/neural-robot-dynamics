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

import warp as wp   
from envs.warp_sim_envs import Environment

import numpy as np


def get_observation_space(
    env: Environment, render_mode: str = None, use_gymnasium=True
):
    if use_gymnasium:
        import gymnasium as gym
    else:
        import gym

    if render_mode == "rgb_array":
        tile_height = env.opengl_tile_render_settings["tile_height"]
        tile_width = env.opengl_tile_render_settings["tile_width"]
        shape = (tile_height, tile_width, 3)
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )
    else:
        return gym.spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(env.observation_dim,),
        )


def get_action_space(env: Environment, use_gymnasium=True):
    if use_gymnasium:
        import gymnasium as gym
    else:
        import gym

    if hasattr(env, 'policy_action_limits'):
        ctrl_limits = np.array(env.policy_action_limits)
    else:
        ctrl_limits = np.array(env.control_limits)

    return gym.spaces.Box(low=ctrl_limits[:, 0], high=ctrl_limits[:, 1])


@wp.kernel
def cost2reward(
    cost: wp.array(dtype=float),
    scale: float,
    bias: float,
    reward: wp.array(dtype=float),
):
    tid = wp.tid()
    reward[tid] = -cost[tid] * scale + bias


@wp.kernel(enable_backward=False)
def eval_timeout(
    max_episode_length: int,
    timeout_buf: wp.array(dtype=bool),
    progress_buf: wp.array(dtype=int),
    reset_buf: wp.array(dtype=bool),
):
    tid = wp.tid()

    timeout_buf[tid] = False
    progress_buf[tid] = progress_buf[tid] + 1

    # reset agents
    if progress_buf[tid] >= max_episode_length:
        reset_buf[tid] = True
        timeout_buf[tid] = True

    if reset_buf[tid]:
        progress_buf[tid] = 0


@wp.kernel(enable_backward=False)
def clear_nonfinite_values(values: wp.array(dtype=float, ndim=2)):
    i, j = wp.tid()
    if not wp.isfinite(values[i, j]):
        values[i, j] = 0.0