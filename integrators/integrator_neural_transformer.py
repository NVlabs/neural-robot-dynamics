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

from collections import deque
import torch

from integrators.integrator_neural_stateful import StatefulNeuralIntegrator

class TransformerNeuralIntegrator(StatefulNeuralIntegrator):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        
    def reset_states_history(self):
        self.states_history = deque(maxlen=self.num_states_history)
        
    def get_neural_model_inputs(self):
        if len(self.states_history) == 0: # for dummy call
            contacts = self.get_abstract_contacts(self.model)
            processed_model_inputs = {
                "root_body_q": torch.zeros_like(self.root_body_q).unsqueeze(1),
                "states": torch.zeros_like(self.states).unsqueeze(1),
                "states_embedding": torch.zeros_like(self.states_embedding).unsqueeze(1),
                "joint_acts": torch.zeros_like(self.joint_acts).unsqueeze(1),
                "gravity_dir": torch.zeros_like(self.gravity_dir).unsqueeze(1),
                **contacts
            }
        else:
            # assemble the model inputs in world frame
            model_inputs = torch.utils.data.default_collate(self.states_history)
            for k in model_inputs:
                model_inputs[k] = model_inputs[k].permute(1, 0, 2)
            
            processed_model_inputs = self.process_neural_model_inputs(model_inputs)
                    
        return processed_model_inputs
        


    
