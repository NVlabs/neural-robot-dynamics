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

from typing import Tuple, Union
import torch
import numpy as np

class RunningMeanStd(object):
    def __init__(
        self, 
        epsilon: float = 1e-4, 
        shape: Tuple[int, ...] = (), 
        device = 'cuda:0'
    ):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = torch.zeros(shape, dtype = torch.float32, device = device)
        self.var = torch.ones(shape, dtype = torch.float32, device = device)
        self.count = epsilon

    def to(self, device):
        rms = RunningMeanStd(device = device)
        rms.mean = self.mean.to(device).clone()
        rms.var = self.var.to(device).clone()
        rms.count = self.count
        return rms
    
    @torch.no_grad()
    def update(
        self, 
        arr: torch.tensor, 
        batch_dim = False, 
        time_dim = False
    ) -> None:
        mean_dims = [i for i in range(int(batch_dim) + int(time_dim))]
        batch_mean = torch.mean(arr, dim = mean_dims)
        batch_var = torch.var(arr, dim = mean_dims, unbiased = False)
        batch_count = np.prod(arr.shape[:int(batch_dim) + int(time_dim)])
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, 
        batch_mean: torch.tensor, 
        batch_var: torch.tensor, 
        batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a + m_b + 
            torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, arr:torch.tensor, un_norm = False) -> torch.tensor:
        if not un_norm:
            result = (arr - self.mean) / torch.sqrt(self.var + 1e-5)
        else:
            result = arr * torch.sqrt(self.var + 1e-5) + self.mean
        return result

class RunningMeanStdDict(object):
    def __init__(
        self, 
        epsilon: float = 1e-4, 
        obs_sample: dict = {}, 
        batch_dim = False, 
        time_dim = False, 
        device = 'cuda:0'
    ):
        self.running_mean_stds = {}
        shape_start_idx = int(batch_dim) + int(time_dim)
        for obs_key in obs_sample.keys():
            self.running_mean_stds[obs_key] = RunningMeanStd(
                epsilon = epsilon, 
                shape = obs_sample[obs_key].shape[shape_start_idx:], 
                device = device
            )
    
    @torch.no_grad()
    def update(self, obs: dict, batch_dim = False, time_dim = False) -> None:
        for obs_key in obs.keys():
            self.running_mean_stds[obs_key].update(
                obs[obs_key], 
                batch_dim = batch_dim, 
                time_dim = time_dim
            )
    
    def normalize(self, obs: dict, un_norm = False) -> dict:
        result = {}
        for obs_key in obs.keys():
            if obs_key in self.running_mean_stds:
                result[obs_key] = self.running_mean_stds[obs_key].normalize(
                    obs[obs_key], 
                    un_norm
                )
            else:
                result[obs_key] = obs[obs_key]
        return result
