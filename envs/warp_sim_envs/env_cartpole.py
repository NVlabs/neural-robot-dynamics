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
import math
import warp as wp
import warp.sim

from envs.warp_sim_envs import Environment, IntegratorType

@wp.func
def fmod(n: float, M: float):
    return ((n % M) + M) % M

@wp.func
def angle_normalize(x: float):
    return (fmod(x + wp.pi, 2.0 * wp.pi)) - wp.pi

@wp.kernel
def single_cartpole_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    x = joint_q[env_id * 2 + 0]
    th = joint_q[env_id * 2 + 1]
    xdot = joint_qd[env_id * 2 + 0]
    thdot = joint_qd[env_id * 2 + 1]
    u = joint_act[env_id * 2]

    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
    angle = angle_normalize(th)
    c = angle**2.0 + 0.05 * x**2.0 + 0.1 * thdot**2.0 + 0.1 * xdot**2.0

    wp.atomic_add(cost, env_id, c)

    if terminated:
        terminated[env_id] = abs(x) > 4.0 or abs(thdot) > 10.0 or abs(xdot) > 10.0


@wp.kernel
def double_cartpole_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    th1 = joint_q[env_id * 3 + 1]
    thdot1 = joint_qd[env_id * 3 + 1]
    th2 = joint_q[env_id * 3 + 2]
    thdot2 = joint_qd[env_id * 3 + 2]
    u = joint_act[env_id * 3]

    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
    angle1 = angle_normalize(th1)
    angle2 = angle_normalize(th2)
    c = (
        angle1**2.0
        + 0.1 * thdot1**2.0
        + angle2**2.0
        + 0.1 * thdot2**2.0
        + (u * 1e-4) ** 2.0
    )

    wp.atomic_add(cost, env_id, c)

    if terminated:
        terminated[env_id] = abs(angle1) > 0.3 or abs(angle2) > 0.3


@wp.kernel(enable_backward=False)
def reset_init_state_single_cartpole(
    reset: wp.array(dtype=wp.bool),
    seed: int,
    random_reset: bool,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    default_joint_q_init: wp.array(dtype=wp.float32),
    default_joint_qd_init: wp.array(dtype=wp.float32),
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()

    if reset:
        if not reset[env_id]:
            return

    if random_reset:
        random_state = wp.rand_init(seed, env_id)
        joint_q[env_id * dof_q_per_env] = wp.randf(random_state, -1.0, 1.0)
        joint_q[env_id * dof_q_per_env + 1] = wp.randf(random_state, -wp.pi, wp.pi)
        joint_qd[env_id * dof_qd_per_env] = wp.randf(random_state, -1.0, 1.0)
        joint_qd[env_id * dof_qd_per_env + 1] = wp.randf(random_state, -1.0, 1.0)
    else:
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[i]
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[i]


@wp.kernel
def compute_observations_single_cartpole(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    obs: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    obs[tid, 0] = joint_q[tid * dof_q_per_env + 0]
    obs[tid, 1] = angle_normalize(joint_q[tid * dof_q_per_env + 1])
    for i in range(dof_qd_per_env):
        obs[tid, i + dof_q_per_env] = joint_qd[tid * dof_qd_per_env + i]


class CartpoleEnvironment(Environment):
    robot_name = 'Cartpole'
    sim_name = "env_cartpole"
    env_offset = (2.0, 0.0, 2.0)
    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    single_cartpole = True

    sim_substeps_euler = 16
    sim_substeps_featherstone = 5
    sim_substeps_xpbd = 5

    num_rigid_contacts_per_env = 0

    activate_ground_plane = False

    integrator_type = IntegratorType.FEATHERSTONE

    show_joints = True

    controllable_dofs = [0]
    control_gains = [1500.0]
    control_limits = [(-1.0, 1.0)]

    def __init__(self, seed = 42, random_reset = True, **kwargs):
        self.seed = seed
        self.random_reset = random_reset
        super().__init__(**kwargs)

    def create_articulation(self, builder):
        if self.single_cartpole:
            path = "cartpole_single.urdf"
        else:
            path = "cartpole.urdf"
            self.opengl_render_settings["camera_pos"] = (40.0, 1.0, 0.0)
            self.opengl_render_settings["camera_front"] = (-1.0, 0.0, 0.0)
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets", path),
            builder,
            xform=wp.transform(
                (0.0, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5),
            ),
            floating=False,
            armature=0.01,
            stiffness=0.0,
            damping=0.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        # joint initial positions
        if self.single_cartpole:
            builder.joint_q[-2:] = [0.0, 0.1]
        else:
            builder.joint_q[-3:] = [0.0, 0.1, 0.0]

    def compute_cost_termination(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        if self.integrator_type != IntegratorType.FEATHERSTONE:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        wp.launch(
            single_cartpole_cost if self.single_cartpole else double_cartpole_cost,
            dim=self.num_envs,
            inputs=[state.joint_q, state.joint_qd, control.joint_act],
            outputs=[cost, terminated],
            device=self.device,
        )

    def compute_observations(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        if self.single_cartpole:
            if not self.uses_generalized_coordinates:
                # evaluate generalized coordinates
                wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)
            wp.launch(
                compute_observations_single_cartpole,
                dim=self.num_envs,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                ],
                outputs=[observations],
                device=self.device,
            )
        else:
            super().compute_observations(
                state, control, observations, step, horizon_length
            )

    def reset_envs(self, env_ids: wp.array = None):
        if self.single_cartpole and self.uses_generalized_coordinates:
            wp.launch(
                reset_init_state_single_cartpole,
                dim=self.num_envs,
                inputs=[
                    env_ids,
                    self.seed,
                    self.random_reset,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                    self.model.joint_q,
                    self.model.joint_qd,
                ],
                outputs=[self.state.joint_q, self.state.joint_qd],
                device=self.device,
            )
            self.seed += self.num_envs
        else:
            super().reset_envs(env_ids)
        
        # update maximal coordinates after reset
        if env_ids is None or env_ids.numpy().any():
            if self.uses_generalized_coordinates:
                wp.sim.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, None, self.state)
