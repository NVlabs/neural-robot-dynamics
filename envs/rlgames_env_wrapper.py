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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
)
sys.path.append(base_dir)

from typing import List, Optional, Sequence, Union

import torch
import warp as wp
import numpy as np

from envs.neural_environment import NeuralEnvironment
from envs.warp_sim_envs import RenderMode
from envs.warp_sim_envs.wrapper_utils import (
    cost2reward, get_observation_space, get_action_space
)
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext

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


class RlgamesEnvironment(vecenv.IVecEnv):
    def __init__(
        self,
        env: NeuralEnvironment,
        render_mode: str = None,
        max_episode_length: int = 500,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        control_steps: int = 1,
        image_width: int = 128,
        image_height: int = 128,
    ):
        self.neural_env = env
        self.num_envs = self.neural_env.num_envs

        self.render_mode = render_mode

        self.num_obs = self.neural_env.observation_dim
        self.num_actions = self.neural_env.action_dim
        self.device = self.neural_env.device

        self.max_episode_length = max_episode_length
        
        self.control_steps = control_steps
        
        with wp.ScopedDevice(self.device):
            self.action_limits_wp = wp.array(self.neural_env.action_limits, dtype=wp.float32)
            self.action_buf = wp.empty(
                (self.num_envs, self.num_actions), dtype=wp.float32
            )
            self.rew_buf = wp.empty(self.num_envs, dtype=wp.float32)
            self.done_buf = wp.empty(self.num_envs, dtype=wp.bool)
            self.episode_length_buf = wp.empty(self.num_envs, dtype=wp.int32)
            self.time_step_buf = wp.empty(self.num_envs, dtype=wp.int32)
            self.timeout_buf = wp.empty(self.num_envs, dtype=wp.bool)
            self.cost_buf = wp.empty(self.num_envs, dtype=wp.float32)

        self._step_count = 0
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias

        if self.neural_env.use_graph_capture:
            with wp.ScopedCapture() as capture:
                self.neural_env.update()
            self.graph = capture.graph
        else:
            self.graph = None

        self.extras = {}
        self.obs_dict = {}

        self.render_mode = render_mode
        self.neural_env.render_mode = RenderMode.NONE
        if render_mode == "human" or render_mode == "rgb_array":
            self.neural_env.render_mode = RenderMode.OPENGL
        if render_mode == "rgb_array":
            self.neural_env.use_tiled_rendering = True
            self.neural_env.opengl_render_settings = dict(scaling=3.0, draw_axis=False)
            self.neural_env.opengl_tile_render_settings = dict(
                tile_width=image_width, tile_height=image_height
            )
            self.get_observations = self.get_rgb_observations
            self.obs_buf = wp.empty(
                (self.num_envs, image_height, image_width, 3),
                dtype=wp.uint8,
                device=self.device,
            )
        else:
            self.get_observations = self.get_state_observations
            self.obs_buf = wp.empty(
                (self.num_envs, self.num_obs), dtype=wp.float32, device=self.device
            )
        self.neural_env.setup_renderer()
        self.observation_space = get_observation_space(
            self.neural_env, render_mode, use_gymnasium=False
        )
        self.action_space = get_action_space(self.neural_env, use_gymnasium=False)

    def get_state_observations(self):
        self.neural_env.compute_observations(
            self.obs_buf,
            0,
            0,
        )
        return wp.to_torch(self.obs_buf)

    def get_rgb_observations(self):
        self.neural_env.renderer.get_pixels(
            self.obs_buf, split_up_tiles=True, mode="rgb", use_uint8=True
        )
        return wp.to_torch(self.obs_buf)

    def reset(self):
        self.neural_env.reset()
        self._step_count = 0
        self.done_buf.fill_(0)
        self.episode_length_buf.fill_(0)
        return self.get_observations()

    def render(self):
        self.neural_env.render()

    def clamp_actions(self, actions, action_limits):
        actions = torch.max(torch.min(actions, action_limits[:, 1]), 
                            action_limits[:, 0])

    def step(self, actions):
        """
        Step the environments with the given action

        :param actions: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        self.extras = {}
        
        self.clamp_actions(actions, wp.to_torch(self.action_limits_wp))

        self.action_buf.assign(wp.array(actions))
        
        if self.neural_env.use_graph_capture:
            wp.capture_launch(self.graph)
        else:
            for _ in range(self.control_steps):
                self.neural_env.step(actions)

        # reward is single-step, not cumulative
        self.cost_buf.zero_()
        self.done_buf.zero_()
        self.neural_env.compute_cost_termination(
            self._step_count,
            self.max_episode_length,
            self.cost_buf,
            self.done_buf,
        )

        wp.launch(
            eval_timeout,
            dim=self.num_envs,
            inputs=[
                self.max_episode_length,
            ],
            outputs=[self.timeout_buf, self.episode_length_buf, self.done_buf],
            device=self.device,
        )

        wp.launch(
            cost2reward,
            dim=self.num_envs,
            inputs=[self.cost_buf, self.reward_scale, self.reward_bias],
            outputs=[self.rew_buf],
            device=self.device,
        )
        
        self.render()
        obs = self.get_observations()

        self.neural_env.reset_envs(self.done_buf)

        self._step_count += 1

        rewards = wp.to_torch(self.rew_buf)
        dones = wp.to_torch(self.done_buf)

        # get extras
        self.neural_env.get_extras(self.extras)

        # copy warp data to pytorch
        self.extras["time_outs"] = wp.to_torch(self.timeout_buf).squeeze(-1)
        self.obs_dict["obs"] = obs

        return self.obs_dict, rewards, dones, self.extras

    def close(self):
        """
        Clean up the environment's resources.
        """
        if self.neural_env.renderer:
            self.neural_env.renderer.close()

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        if attr_name == "render_mode":
            return [self.render_mode for _ in range(self.num_envs)]
        return getattr(self.neural_env, attr_name)

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.

        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        setattr(self.neural_env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        return getattr(self.neural_env, method_name)(*method_args, **method_kwargs)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        :param seed: (Optional[int]) The random seed. May be None for completely random seeding.
        :return: (List[Union[None, int]]) Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        pass

    def get_images(self) -> Sequence[np.ndarray]:
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    def env_is_wrapped(self) -> List[bool]:
        """
        Check if environments are wrapped with a given wrapper

        :return: (List[bool]) List of bools for each environment, indicating if wrapped
        """
        raise NotImplementedError

    def get_env_info(self):
        info = {}
        info["action_space"] = self.action_space
        info["observation_space"] = self.observation_space
        return info

    # def get_number_of_agents(self):
    #     return self.num_envs


class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats."""

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(
            self.algo.ppo_device
        )
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if "episode" in infos:
                self.ep_infos.append(infos["episode"])

            if len(infos) > 0 and isinstance(
                infos, dict
            ):  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if (
                        isinstance(v, float)
                        or isinstance(v, int)
                        or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                    ):
                        self.direct_info[k] = v

    def after_clear_stats(self):
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.algo.device))
                    )
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, epoch_num)
            self.ep_infos.clear()

        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"{k}/frame", v, frame)
            self.writer.add_scalar(f"{k}/iter", v, epoch_num)
            self.writer.add_scalar(f"{k}/time", v, total_time)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar("scores/mean", mean_scores, frame)
            self.writer.add_scalar("scores/iter", mean_scores, epoch_num)
            self.writer.add_scalar("scores/time", mean_scores, total_time)


def register_env(
    env: NeuralEnvironment,
    render_mode: str = "human",
    max_episode_length: int = 500,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    control_steps: int = 1,
    image_width: int = 128,
    image_height: int = 128,
    env_name: str = "warp",
):
    env = RlgamesEnvironment(
        env,
        render_mode=render_mode,
        max_episode_length=max_episode_length,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        control_steps=control_steps,
        image_width=image_width,
        image_height=image_height,
    )

    vecenv.register(
        "WARP",
        lambda config_name, num_actors, **kwargs: env,
    )
    env_configurations.register(
        env_name,
        {
            "vecenv_type": "WARP",
            "env_creator": lambda **_: env,
        },
    )