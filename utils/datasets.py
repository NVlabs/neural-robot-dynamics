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
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

from utils.commons import DATASET_MODES

class BatchTransitionDataset:
    def __init__(
            self,
            batch_size,
            hdf5_dataset_path,
            max_capacity = 100000000,
            device = 'cpu'
        ):
        
        self.batch_size = batch_size
        self.max_capacity = max_capacity
        self.device = device

        self.load_dataset(hdf5_dataset_path)

    def load_dataset(self, hdf5_dataset_path):
        if not os.path.exists(hdf5_dataset_path):
            raise FileNotFoundError(
                f"Dataset file {os.path.abspath(hdf5_dataset_path)} not found."
            )
            
        dataset = h5py.File(hdf5_dataset_path, 'r', swmr=True, libver='latest')

        mode = dataset['data'].attrs['mode']

        assert mode in DATASET_MODES

        # load dataset
        total_transitions = dataset['data'].attrs['total_transitions']
        self.dataset_length = min(self.max_capacity, total_transitions) // self.batch_size
        
        self.dataset = {}
        if mode == 'transition':
            for key in dataset['data'].keys():
                data = dataset['data'][key][()].astype('float32')
                # flatten feature dims if needed (e.g. contact info)
                data = data.reshape(total_transitions, -1)
                self.dataset[key] = torch.tensor(
                    data[:self.dataset_length * self.batch_size, ...].reshape(
                        self.dataset_length,
                        self.batch_size,
                        -1
                    ),
                    device = self.device
                )
        elif mode == 'trajectory':
            # TODO: need to handle trajectory dataset consisted of varying-length trajectories
            self.traj_lengths = None
            for key in dataset['data'].keys():
                if key == 'traj_lengths':
                    self.traj_lengths = dataset['data'][key][()].astype('int32')
                    continue
                data = dataset['data'][key][()].astype('float32') # shape (T, B, dim1, dim2, ...)
                data = np.swapaxes(data, 0, 1) # shape (B, T, dim1, dim2, ...)
                # flatten feature dims if needed (e.g. contact info)
                data = data.reshape(total_transitions, -1)
                self.dataset[key] = torch.tensor(
                    data[:self.dataset_length * self.batch_size, ...].reshape(
                        self.dataset_length,
                        self.batch_size,
                        -1
                    ),
                    device = self.device
                )

    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index):
        assert index < len(self)

        data = {}
        for key in self.dataset.keys():
            data[key] = self.dataset[key][index, ...]

        return data
    
    def shuffle(self):
        for i in range(self.batch_size):
            perm = torch.randperm(self.dataset_length, device = self.device)
            for key in self.dataset.keys():
                self.dataset[key][:, i, :] = self.dataset[key][perm, i, :]

# Return shape (B, 1, dim)
def collate_fn_BatchTransitionDataset(batch):
    output_batch = {}
    for key in batch[0].keys():
        output_batch[key] = batch[0][key].unsqueeze(1)

    return output_batch

class TrajectoryDataset(Dataset):
    def __init__(
        self,
        hdf5_dataset_path,
        sample_sequence_length = 10,
        max_capacity = 100000000
    ):
        self.max_capacity = max_capacity
        self.load_dataset(hdf5_dataset_path)
        self.update_sample_sequence_length(sample_sequence_length)
    
    def load_dataset(self, hdf5_dataset_path):
        dataset = h5py.File(hdf5_dataset_path, 'r', swmr=True, libver='latest')

        mode = dataset['data'].attrs['mode']

        assert mode == 'trajectory', \
            "TrajectoryDataset requires the dataset model to be 'trajectory'."

        # truncate the dataset based on max_capacity
        num_transitions_per_trajectory = dataset['data']['states'].shape[0]
        num_trajectories = min(
            int(np.ceil(self.max_capacity / num_transitions_per_trajectory)), 
            dataset['data']['states'].shape[1]
        )
        
        # load dataset
        self.dataset = {}
        self.traj_lengths = None
        for key in dataset['data'].keys():
            if key == 'traj_lengths':
                self.traj_lengths = dataset['data'][key][:num_trajectories].astype('int32')
                continue
            data = dataset['data'][key][:, :num_trajectories, ...].astype('float32') # shape (T, B, dim1, dim2, ...)
            data = np.swapaxes(data, 0, 1) # shape (B, T, dim1, dim2, ...)
            self.dataset[key] = data.reshape(data.shape[0], data.shape[1], -1) # shape (B, T, dim)
        
        if self.traj_lengths is None:
            self.traj_lengths = np.full(num_trajectories, num_transitions_per_trajectory)

    def update_sample_sequence_length(self, sample_sequence_length):
        self.sample_sequence_length = sample_sequence_length
        self.build_index()

    def build_index(self):
        self.length = 0
        for i in range(len(self.traj_lengths)):
            self.length += max(0, self.traj_lengths[i] - self.sample_sequence_length + 1)
        self.mapping_index2traj = np.zeros((self.length, 2), dtype = int)
        index = 0
        for i in range(len(self.traj_lengths)):
            for j in range(max(0, self.traj_lengths[i] - self.sample_sequence_length + 1)):
                self.mapping_index2traj[index][0] = i
                self.mapping_index2traj[index][1] = j
                index += 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        [traj_idx, traj_step_idx] = self.mapping_index2traj[index]
        trajectory = {}
        for key in self.dataset.keys():
            trajectory[key] = torch.tensor(
                self.dataset[key][
                    traj_idx, 
                    traj_step_idx:traj_step_idx + self.sample_sequence_length
                ]
            )
        return trajectory

    def shuffle(self):
        pass