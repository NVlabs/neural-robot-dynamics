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

@wp.kernel(enable_backward=False)
def _assign_states(
    states: wp.array(dtype=float, ndim=2),
    q_count: int,
    qd_count: int,
    # output
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float)
):
    tid = wp.tid()
    for i in range(q_count):
        joint_q[tid * q_count + i] = states[tid, i]
    for i in range(qd_count):
        joint_qd[tid * qd_count + i] = states[tid, i + q_count]

@wp.kernel(enable_backward=False)
def _acquire_states(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    q_count: int,
    qd_count: int,
    # output
    states: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    for i in range(q_count):
        states[tid, i] = joint_q[tid * q_count + i]
    for i in range(qd_count):
        states[tid, q_count + i] = joint_qd[tid * qd_count + i]

@wp.kernel(enable_backward=False)
def _assign_joint_acts_from_actions(
    actions: wp.array(dtype=float, ndim=2),
    control_count: int,
    joint_act_dim: int,
    control_dofs: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    control_limits: wp.array(dtype=float, ndim=2),
    # output
    joint_act: wp.array(dtype=float)
):
    tid = wp.tid()
    for i in range(control_count):
        ci = control_dofs[i]
        lo, hi = control_limits[i, 0], control_limits[i, 1]
        joint_act[tid * joint_act_dim + ci] = wp.clamp(
            actions[tid, i], lo, hi
        ) * control_gains[i]

def assign_states_from_torch(warp_env, torch_states):
    assert torch_states.shape[0] <= warp_env.num_envs
    wp.launch(
        _assign_states,
        dim = torch_states.shape[0],
        inputs = [
            wp.from_torch(torch_states),
            warp_env.dof_q_per_env,
            warp_env.dof_qd_per_env
        ],
        outputs = [
            warp_env.state.joint_q,
            warp_env.state.joint_qd
        ],
        device = warp_env.device
    )

def acquire_states_to_torch(warp_env, torch_states):
    assert torch_states.shape[0] == warp_env.num_envs
    wp.launch(
        _acquire_states,
        dim = torch_states.shape[0],
        inputs = [
            warp_env.state.joint_q,
            warp_env.state.joint_qd,
            warp_env.dof_q_per_env,
            warp_env.dof_qd_per_env
        ],
        outputs = [wp.from_torch(torch_states)],
        device = warp_env.device
    )

def assign_joint_acts_from_actions_torch(warp_env, actions):
    assert actions.shape[0] == warp_env.num_envs
    wp.launch(
        _assign_joint_acts_from_actions,
        dim = actions.shape[0],
        inputs = [
            wp.from_torch(actions),
            warp_env.control_dim,
            warp_env.joint_act_dim,
            warp_env.controllable_dofs_wp,
            warp_env.control_gains_wp,
            warp_env.control_limits_wp
        ],
        outputs = [
            warp_env.joint_act
        ],
        device = warp_env.device
    )

''' apply generalized coordinates to maximal coordinates '''
def eval_fk(model, state):
    wp.sim.eval_fk(
        model,
        state.joint_q,
        state.joint_qd,
        None,
        state
    )
    
''' convert new maximal coordinates back to generalized coordinates '''
def eval_ik(model, state):
    wp.sim.eval_ik(
        model,
        state,
        state.joint_q,
        state.joint_qd
    )
    
def device_to_torch(warp_device):
    return wp.device_to_torch(warp_device)