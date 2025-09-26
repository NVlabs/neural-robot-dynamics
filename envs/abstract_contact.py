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

import torch

class AbstractContact:
    """
    Class for AbstractContact, a torch view of the rigid contact info in warp model
    
    Each contact is represented by:
    - contact_shape0 (fixed): a shape from robot
    - contact_point0 (fixed): the contact point in the shape0's body's frame
    - contact_shape1 (variable, fixed for now): a shape from the external object
    - contact_point1 (variable): the contact point in the shape1's body frame, or in world frame (for ground)
    - contact_normal (variable, fixed for now): the contact normal in world frame
    """
    def __init__(self, num_contacts_per_env, num_envs, device = 'cuda:0'):
        self.num_contacts_per_env = num_contacts_per_env
        self.num_envs = num_envs
        self.num_total_contacts = num_contacts_per_env * num_envs

        # essential variables
        self.contact_shape0 = torch.empty(
            (self.num_total_contacts,),
            dtype = torch.int32, 
            device = device
        )
        self.contact_shape1 = torch.empty(
            (self.num_total_contacts,), 
            dtype = torch.int32, 
            device = device
        )
        self.contact_point0 = torch.empty(
            (self.num_total_contacts, 3),
            dtype = torch.float32,
            device = device
        )
        self.contact_point1 = torch.empty(
            (self.num_total_contacts, 3),
            dtype = torch.float32,
            device = device
        )
        self.contact_normal = torch.empty(
            (self.num_total_contacts, 3),
            dtype = torch.float32,
            device = device
        )
        self.contact_offset0 = torch.empty(
            (self.num_total_contacts, 3),
            dtype = torch.float32,
            device = device
        )
        self.contact_offset1 = torch.empty(
            (self.num_total_contacts, 3),
            dtype = torch.float32,
            device = device
        )
        self.contact_thickness0 = torch.empty(
            (self.num_total_contacts,),
            dtype = torch.float32,
            device = device
        )
        self.contact_thickness1= torch.empty(
            (self.num_total_contacts,),
            dtype = torch.float32,
            device = device
        )
        
        #  derived variables (not used by warp default integrators)
        self.contact_depth = torch.empty(
            (self.num_total_contacts,),
            dtype = torch.float32,
            device = device
        )
        self.contact_thickness = torch.empty(
            (self.num_total_contacts,),
            dtype = torch.float32,
            device = device
        )
            