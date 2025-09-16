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

from algorithms.vanilla_trainer import VanillaTrainer
from utils.datasets import TrajectoryDataset

class SequenceModelTrainer(VanillaTrainer):
    def __init__(
        self,
        neural_env,
        cfg,
        model_checkpoint_path=None,
        device='cuda:0'
    ):
        self.sample_sequence_length = cfg['algorithm'].get('sample_sequence_length', 1)
        super().__init__(neural_env, cfg, model_checkpoint_path, device)
    
    def get_datasets(self, train_dataset_path, valid_datasets_cfg):
        self.train_dataset = TrajectoryDataset(
            sample_sequence_length=self.sample_sequence_length,
            hdf5_dataset_path=train_dataset_path,
            max_capacity=self.dataset_max_capacity)

        valid_dataset_names = valid_datasets_cfg.keys()
        for valid_dataset_name in valid_dataset_names:
            self.valid_datasets[valid_dataset_name] = TrajectoryDataset(
                sample_sequence_length=self.sample_sequence_length,
                hdf5_dataset_path=valid_datasets_cfg[valid_dataset_name])

        self.collate_fn = None