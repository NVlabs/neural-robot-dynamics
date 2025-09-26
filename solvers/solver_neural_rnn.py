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

from solvers.solver_neural import NeuralSolver

class RNNNeuralSolver(NeuralSolver):
    def __init__(
        self,
        reset_seq_length = 1,
        **kwargs
    ):
        self.reset_seq_length = reset_seq_length
        self._step_count = 0
        super().__init__(**kwargs)
    
    def reset(self):
        self._step_count = 0
        self.neural_model.init_rnn(self.num_envs)
    
    def before_model_forward(self):
        if self._step_count % self.reset_seq_length == 0:
            self.neural_model.reset_rnn_hidden_states()
            self._step_count = 0
        self._step_count += 1
        

        


    
