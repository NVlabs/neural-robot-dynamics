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
from models.base_models import MLPBase, LSTMBase, GRUBase
from models.model_transformer import GPT, GPTConfig

class MLPDeterministic(nn.Module):
    def __init__(
        self,
        input_shape,
        output_dim,
        network_cfg,
        device='cuda:0'
    ):
        super().__init__()
        
        self.device = device

        self.feature_net = MLPBase(
            input_shape[0], 
            network_cfg['mlp'], 
            device=device
        )

        self.output_net = nn.Linear(
            self.feature_net.out_features, 
            output_dim, 
            device=device
        )
    
    def forward(self, inputs, deterministic = False):
        features = self.feature_net(inputs)
        output = self.output_net(features)
        return output
    
    def to(self, device):
        self.device = device
        self.feature_net.to(device)
        self.output_net.to(device)


class ModelMixedInput(nn.Module):
    def __init__(
        self,
        input_sample,
        output_dim,
        input_cfg,
        network_cfg,
        device = 'cuda:0'
    ):
        
        super().__init__()

        self.device = device
        self.model = None
        
        self.input_rms = None
        self.normalize_input = network_cfg.get('normalize_input', False)
        self.output_rms = None
        self.normalize_output = network_cfg.get('normalize_output', False)

        self.encoders, self.feature_dim = self.construct_input_encoders(
            input_cfg, 
            network_cfg['encoder'], 
            input_sample, 
            device = device
        )
        
        if "rnn" in network_cfg:
            self.is_rnn = True
            if network_cfg['rnn']['net'] == 'lstm':
                self.rnn = LSTMBase(
                    self.feature_dim, 
                    network_cfg['rnn'], 
                    device=self.device
                )
            elif network_cfg['rnn']['net'] == 'gru':
                self.rnn = GRUBase(
                    self.feature_dim, 
                    network_cfg['rnn'], 
                    device=self.device
                )
            else:
                raise NotImplementedError
            self.feature_dim = self.rnn.hidden_size
        else:
            self.is_rnn = False
            self.rnn = None

        if "transformer" in network_cfg:
            model_args = dict(
                n_layer=network_cfg['transformer']['n_layer'],
                n_head=network_cfg['transformer']['n_head'],
                n_embd=network_cfg['transformer']['n_embd'],
                block_size=network_cfg['transformer']['block_size'],
                bias=network_cfg['transformer']['bias'],
                vocab_size=self.feature_dim,
                dropout=network_cfg['transformer']['dropout'],
            )
            gptconf = GPTConfig(**model_args)

            self.transformer_model = GPT(gptconf)
            self.transformer_model.to(self.device)

            self.is_transformer = True
            self.feature_dim = self.transformer_model.config.n_embd
        else:
            self.is_transformer = False
            self.transformer_model = None

        if self.model is None:
            self.model = MLPDeterministic(
                (self.feature_dim, ), 
                output_dim, 
                network_cfg['model'], 
                device = device
            )
        
        self.output_tanh = network_cfg.get('output_tanh', False)
    
    # For compatibility with the old models trained from warp.sim
    def fix_input_names(self): 
        if 'joint_acts' in self.encoders:
            self.encoders['joint_f'] = self.encodders.pop('joint_acts')
        if 'joint_acts' in self.input_rms:
            self.input_rms['joint_f'] = self.input_rms.pop('joint_acts')
        if 'joint_acts' in self.low_dim_input_names:
            self.low_dim_input_names[
                self.low_dim_input_names.index('joint_acts')
            ] = 'joint_f'

    def construct_input_encoders(
        self,
        input_cfg,
        encoder_cfg,
        input_sample,
        device = 'cuda:0'
    ):
        encoders = nn.ModuleDict()
        
        '''
        low-dim inputs
        '''
        if len(input_cfg.get('low_dim', [])) > 0:
            low_dim_size = 0
            self.low_dim_input_names = input_cfg.get('low_dim')
            for low_dim_input_name in self.low_dim_input_names:
                assert len(input_sample[low_dim_input_name].shape) in [2, 3] # (B, *) or (B, T, *)
                low_dim_size += input_sample[low_dim_input_name].shape[-1]
            
            assert 'low_dim' in encoder_cfg
            low_dim_encoder = MLPBase(
                low_dim_size, 
                encoder_cfg['low_dim'], 
                device = device
            )
            encoders['low_dim'] = low_dim_encoder
            
        '''
        rgb inputs
        '''
        rgb_input_names = input_cfg.get('rgb', [])
        for rgb_input_name in rgb_input_names:
            assert rgb_input_name in input_sample
            assert len(input_sample[rgb_input_name].shape) in [4, 5] # (B, C, H, W) or (B, T, C, H, W)
            assert 'rgb' in encoder_cfg
            if rgb_input_name in encoder_cfg['rgb']:
                config = encoder_cfg['rgb'][rgb_input_name]
            elif 'default' in encoder_cfg['rgb']:
                config = encoder_cfg['rgb']['default']
            else:
                raise ValueError(
                    f"No '{rgb_input_name}' nor 'default' in encoder_cfg['rgb']"
                )

            rgb_encoder = CNNBase(
                input_sample[rgb_input_name].shape[1:], 
                config, 
                device = device
            )
            encoders[rgb_input_name] = rgb_encoder
        
        feature_dim = 0
        for input_name in encoders:
            feature_dim += encoders[input_name].out_features

        return encoders, feature_dim

    def set_input_rms(self, data_rms):
        self.input_rms = {}
        for input_name in self.encoders:
            if input_name == 'low_dim':
                for low_dim_input_name in self.low_dim_input_names:
                    if low_dim_input_name in data_rms:
                        self.input_rms[low_dim_input_name] = data_rms[low_dim_input_name]
            else:
                self.input_rms[input_name] = data_rms[input_name]
    
    def set_output_rms(self, output_rms):
        self.output_rms = output_rms

    def extract_input_features(self, input_dict): # input can be in shape (B, input_dim) or (T, B, input_dim)
        features = []
        for input_name in self.encoders:
            if input_name == 'low_dim':
                low_dim_input_list = []
                for low_dim_input_name in self.low_dim_input_names:
                    low_dim_input_list.append(input_dict[low_dim_input_name])
                cur_input = torch.cat(low_dim_input_list, dim = -1)
            else:
                cur_input = input_dict[input_name]
            features.append(self.encoders[input_name](cur_input)) # each feature is ((T), B, feature_dim_i)
        features = torch.cat(features, dim = -1)
        return features

    def evaluate(self, input_dict, deterministic = False): # Single-step forward, input in shape (B, T, input_dim), T = 1 for non-transformer models
        if self.normalize_input:
            for obs_key in self.input_rms.keys():
                # input_obs_key = obs_key
                # if obs_key == 'joint_acts': # Compatibility with the old models
                #     input_obs_key = 'joint_f'
                # input_dict[obs_key] = self.input_rms[obs_key].normalize(input_dict[input_obs_key])
                input_dict[obs_key] = self.input_rms[obs_key].normalize(input_dict[obs_key])

        features = self.extract_input_features(input_dict)
        if self.is_rnn:
            features = self.rnn(features)

        if self.is_transformer:
            features = self.transformer_model(features)
                    
        output = self.model(features, deterministic = deterministic)

        if self.output_tanh:
            output = torch.tanh(output)

        if self.normalize_output:
            # output = self.output_rms.normalize(output, un_norm = False)
            output = self.output_rms.normalize(output, un_norm = True)

        return output[:, -1:, :]

    def forward(self, input_dict, deterministic = False, inject_noise = False): # Multi-step sequence forward, input in shape (B, T, input_dim)
        if self.normalize_input:
            for obs_key in self.input_rms.keys():
                input_dict[obs_key] = self.input_rms[obs_key].normalize(input_dict[obs_key])

        if inject_noise:
            for obs_key in input_dict.keys():
                input_dict[obs_key] = (
                    input_dict[obs_key] + 
                    torch.randn_like(input_dict[obs_key]) * 0.01
                )

        features = self.extract_input_features(input_dict) # (B, T, feature_dim)

        if self.is_rnn:
            features = self.rnn(features) # (B, T, rnn_hidden_dim)

        if self.is_transformer:
            features = self.transformer_model(features) # (B, T, transform_embed_dim)

        B, T, feature_dim = features.shape
        features_flatten = features.contiguous().view(-1, feature_dim)
        output_flatten = self.model(features_flatten, deterministic = deterministic)
        output = output_flatten.view(B, T, -1)

        if self.output_tanh:
            output = torch.tanh(output)

        if self.normalize_output:
            output = self.output_rms.normalize(output, un_norm = True)
            
        return output

    def get_rnn_hidden_states(self):
        assert self.rnn is not None
        return self.rnn.get_hidden_states()
    
    def rnn_hidden_states_size(self):
        if self.is_rnn:
            return self.rnn.hidden_states_size()
        else:
            return None
        
    def to(self, device):
        for (_, encoder) in self.encoders.items():
            encoder.to(device)
        if self.rnn is not None:
            self.rnn.to(device)
        self.model.to(device)

    def init_rnn(self, batch_size):
        if self.is_rnn:
            return self.rnn.initialize_hidden_states(batch_size)
    
    def reset_rnn_hidden_states(self, batch_indices=None, hidden_states=None):
        if self.is_rnn:
            self.rnn.reset_hidden_states(batch_indices, hidden_states)
        
    def reset(self, batch_size):
        self.init_rnn(batch_size)