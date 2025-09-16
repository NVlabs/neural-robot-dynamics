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

import warp as wp
import numpy as np

from envs.warp_sim_envs import AnymalEnvironment

@wp.kernel
def apply_joint_position_pd_control(
    actions: wp.array(dtype=wp.float32, ndim=1),
    action_scale: wp.float32,
    baseline_joint_q: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_torque_limit: wp.array(dtype=wp.float32),
    Kp: wp.float32,
    Kd: wp.float32,
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_axis_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_axis_start: wp.array(dtype=wp.int32),
    # outputs
    target_joint_q: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32)
):
    joint_id = wp.tid()
    ai = joint_axis_start[joint_id]
    qi = joint_q_start[joint_id]
    qdi = joint_qd_start[joint_id]
    dim = joint_axis_dim[joint_id, 0] + joint_axis_dim[joint_id, 1]
    for j in range(dim):
        qj = qi + j
        qdj = qdi + j
        aj = ai + j
        q = joint_q[qj]
        qd = joint_qd[qdj]
        
        tq = actions[aj] * action_scale + baseline_joint_q[qj]
        
        target_joint_q[aj] = tq
        
        tq = Kp * (tq - q) - Kd * qd
        
        joint_act[aj] = tq
        
class AnymalJointPositionControlEnvironment(AnymalEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.default_joint_q = self.model.joint_q
        self.joint_torque_limit = self.control_gains_wp
        self.action_scale = 0.5
        self.Kp = 85.
        self.Kd = 2.
        
        self.target_joint_q = wp.empty(
            (self.num_envs * self.control_dim), 
            dtype=wp.float32,
            device=self.device
        )
    
    def assign_control(
        self,
        actions: wp.array,
        control: wp.sim.Control,
        state: wp.sim.State
    ):
        if self.task == "dataset":
            # randomize Kp and Kd in dataset generation
            self.Kp = np.random.uniform(low=30., high=200.)
            self.Kd = np.random.uniform(low=0.0, high=1.0)

        wp.launch(
            kernel=apply_joint_position_pd_control,
            dim=self.model.joint_count,
            inputs=[
                wp.from_torch(wp.to_torch(actions).reshape(-1)),
                self.action_scale,
                self.default_joint_q,
                state.joint_q,
                state.joint_qd,
                self.joint_torque_limit,
                self.Kp,
                self.Kd,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_axis_dim,
                self.model.joint_axis_start,
            ],
            outputs=[
                self.target_joint_q,
                control.joint_act
            ],
            device=self.model.device
        )
