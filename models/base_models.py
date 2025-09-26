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

import torch
import torch.nn as nn
from models import model_utils

class MLPBase(nn.Module):
    def __init__(self, in_features, network_cfg, device='cuda:0'):
        super(MLPBase, self).__init__()
        
        self.device = device

        layer_sizes = network_cfg['layer_sizes']
        modules = []
        for i in range(len(layer_sizes)):
            modules.append(nn.Linear(in_features, layer_sizes[i]))
            modules.append(model_utils.get_activation_func(network_cfg['activation']))
            if network_cfg.get('layernorm', False):
                modules.append(torch.nn.LayerNorm(layer_sizes[i]))
            in_features = layer_sizes[i]

        self.body = nn.Sequential(*modules).to(device)
        self.out_features = in_features

    def forward(self, inputs):
        shape = inputs.shape
        inputs_flatten = inputs.view((-1, shape[-1]))
        out = self.body(inputs_flatten).view((*shape[:-1], self.out_features))
        return out

    def to(self, device):
        self.device = device
        self.body.to(device)

class LSTMBase(nn.Module):
    def __init__(self, in_features, network_cfg, device='cuda:0'):
        super().__init__()

        self.hidden_size = network_cfg['hidden_size']
        self.num_layers = network_cfg['num_layers']
        self.device = device
        self.lstm = nn.LSTM(input_size = in_features, 
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            batch_first = True)
        self.lstm.to(self.device)
    
    def forward(self, x):
        output, hidden_states_next = self.lstm(x, self.hidden_states)
        self.hidden_states = hidden_states_next
        return output
    
    def initialize_hidden_states(self, batch_size):
        h = torch.zeros(
            (self.num_layers, batch_size, self.hidden_size), 
            device = self.device
        )
        c = torch.zeros(
            (self.num_layers, batch_size, self.hidden_size), 
            device = self.device
        )
        self.hidden_states = (h, c)
    
    # Hidden_states is in shape (B, 2 * L * H)
    def reset_hidden_states(self, batch_indices=None, hidden_states=None):
        if hidden_states is None:
            if batch_indices is None:
                self.hidden_states[0][:, :, :] = 0.
                self.hidden_states[1][:, :, :] = 0.
            else:
                self.hidden_states[0][:, batch_indices, :] = 0.
                self.hidden_states[1][:, batch_indices, :] = 0.
        else:
            hidden_states_right_order = hidden_states.view(
                self.hidden_states[0].shape[1], 
                2, 
                self.num_layers, 
                self.hidden_size
            ).permute(1, 2, 0, 3)
            if batch_indices is None:
                self.hidden_states[0][:, :, :] = hidden_states_right_order[0]
                self.hidden_states[1][:, :, :] = hidden_states_right_order[1]
            else:
                self.hidden_states[0][:, batch_indices, :] = (
                    hidden_states_right_order[0][:, batch_indices, :]
                )
                self.hidden_states[1][:, batch_indices, :] = (
                    hidden_states_right_order[1][:, batch_indices, :]
                )

    # Get the hidden states in shape B, 2*L*H
    def get_hidden_states(self):
        hidden_states = (
            torch.stack((self.hidden_states[0], self.hidden_states[1]))
            .permute(2, 0, 1, 3)
            .view(
                self.hidden_states[0].shape[1],
                2 * self.num_layers * self.hidden_size
            )
        )
                                
        return hidden_states

    def hidden_states_size(self):
        return 2 * self.num_layers * self.hidden_size
    
    def to(self, device):
        self.device = device
        self.lstm.to(device)

class GRUBase(nn.Module):
    def __init__(self, in_features, network_cfg, device='cuda:0'):
        super().__init__()
        
        self.hidden_size = network_cfg['hidden_size']
        self.num_layers = network_cfg['num_layers']
        self.device = device
        self.gru = nn.GRU(input_size = in_features, 
                          hidden_size = self.hidden_size, 
                          num_layers = self.num_layers,
                          batch_first = True)
        self.gru.to(device)

    def forward(self, x):
        output, hidden_states_next = self.gru(x, self.hidden_states)
        self.hidden_states = hidden_states_next
        return output
    
    def initialize_hidden_states(self, batch_size):
        self.hidden_states = torch.zeros(
            (self.num_layers, batch_size, self.hidden_size), 
            device = self.device
        )

    # Hidden_states is in shape (B, L * H)
    def reset_hidden_states(self, batch_indices=None, hidden_states=None):
        if hidden_states is None:
            if batch_indices is None:
                self.hidden_states[:, :, :] = 0.
            else:
                self.hidden_states[:, batch_indices, :] = 0.
        else:
            hidden_states_right_order = hidden_states.view(
                self.hidden_states.shape[1], 
                self.num_layers, 
                self.hidden_size
            ).permute(1, 0, 2)
            if batch_indices is None:
                self.hidden_states[:, :, :] = hidden_states_right_order
            else:
                self.hidden_states[:, batch_indices, :] = (
                    hidden_states_right_order[:, batch_indices, :]
                )

    # Get the hidden states in shape B, L*H
    def get_hidden_states(self):
        hidden_states = (
            self.hidden_states
            .permute(1, 0, 2)
            .view(
                self.hidden_states.shape[1], 
                self.num_layers * self.hidden_size
            )
        )
        return hidden_states

    def hidden_states_size(self):
        return self.num_layers * self.hidden_size
    
    def to(self, device):
        self.device = device
        self.gru.to(device)
