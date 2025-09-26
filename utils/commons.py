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
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import numpy as np

DATASET_MODES = ['transition', 'trajectory']

CONTACT_FREE_DEPTH = 10000.
CONTACT_DEPTH_LOWER_RATIO = -2.
CONTACT_DEPTH_UPPER_RATIO = 4.
MIN_CONTACT_EVENT_THRESHOLD = 0.12

JOINT_Q_MIN = {
    'Cartpole': np.array([-1., -np.pi]),
    'Ant': np.array([
                -10., -10., 1.0,
                -1., -1., -1., -1.,
                np.deg2rad(-40), np.deg2rad(30),
                np.deg2rad(-40), np.deg2rad(-100),
                np.deg2rad(-40), np.deg2rad(-100),
                np.deg2rad(-40), np.deg2rad(30)
            ])
}

JOINT_Q_MAX = {
    'Cartpole': np.array([1., np.pi]),
    'Ant': np.array([
                10., 10., 1.3,
                1., 1., 1., 1.,
                np.deg2rad(40), np.deg2rad(100),
                np.deg2rad(40), np.deg2rad(-30),
                np.deg2rad(40), np.deg2rad(-30),
                np.deg2rad(40), np.deg2rad(100)
            ])
}

JOINT_QD_LIM = {
    'Cartpole': 1.0,
    'Ant': np.array([
                np.pi, np.pi, np.pi,
                1., 1., 1.,
                np.pi * 2., np.pi * 2.,
                np.pi * 2., np.pi * 2.,
                np.pi * 2., np.pi * 2.,
                np.pi * 2., np.pi * 2.
            ])
}

JOINT_F_LIM = {
    'Cartpole': np.array([1500., 0.]),
    'Ant': 20. * np.array([
        0, 0, 0, 0, 0, 0, # dummy joint_f for base floating joint
        1., 
        1.,
        1.,
        1.,
        1.,
        1.,
        1.,
        1.,
    ])
}