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

import numpy as np
from scipy.spatial.transform import Rotation

import warp as wp

from envs.warp_sim_envs import Environment, IntegratorType
from envs.warp_sim_envs.utils import update_ground_plane

@wp.kernel(enable_backward=False)
def reset_init_state_pendulum(
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
        # random reset to guarantee there is no penetration to the ground
        state = wp.rand_init(seed, env_id)
        joint_q[env_id * dof_q_per_env] = wp.randf(state, -wp.pi, wp.pi)
        joint_q[env_id * dof_q_per_env + 1] = wp.randf(
            state,
            -joint_q[env_id * dof_q_per_env],
            wp.pi - joint_q[env_id * dof_q_per_env],
        )
        if joint_q[env_id * dof_q_per_env + 1] > wp.pi:
            joint_q[env_id * dof_q_per_env + 1] -= wp.pi * 2.0

        joint_qd[env_id * dof_qd_per_env] = wp.randf(state, -2.0, 2.0)
        joint_qd[env_id * dof_qd_per_env + 1] = wp.randf(state, -2.0, 2.0)
    else:
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[i]
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[i]


class PendulumWithContactEnvironment(Environment):
    robot_name = "Pendulum"
    sim_name = "env_pendulum_with_contact"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_featherstone = 5

    integrator_type = IntegratorType.FEATHERSTONE

    activate_ground_plane = True

    controllable_dofs = [0, 1]
    control_gains = [1500.0, 1500.0]
    control_limits = [(-1.0, 1.0)] * 2

    handle_collisions_once_per_step = True

    show_rigid_contact_points = False
    contact_points_radius = 0.11

    # contact configuration
    CONTACT_CONFIG = 0 # 0: no contact, 1-6: contact with ground
    
    def __init__(self, seed=42, random_reset=True, **kwargs):
        self.seed = seed
        self.random_reset = random_reset
        super().__init__(**kwargs)

    def create_articulation(self, builder):
        self.joint_type = wp.sim.JOINT_REVOLUTE
        self.chain_length = 2
        self.chain_width = 1.5

        self.lower = -2 * np.pi
        self.upper = 2 * np.pi
        self.limitd_ke = 0.0
        self.limitd_kd = 0.0

        shape_ke = 1.0e4
        shape_kd = 1.0e3
        shape_kf = 1.0e4

        builder.set_ground_plane(
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
        )

        for i in range(self.chain_length):
            if i == 0:
                parent = -1
                parent_joint_xform = wp.transform([0.0, 2.0, 1.0], wp.quat_identity())
            else:
                parent = builder.joint_count - 1
                parent_joint_xform = wp.transform(
                    [self.chain_width, 0.0, 0.0], wp.quat_identity()
                )

            # NOTE: the origin of the body will be ignored in an articulation
            # Reference of the articulaion in warp:
            # https://nvidia.github.io/warp/modules/sim.html#forward-inverse-kinematics

            # create body
            b = builder.add_body(
                origin=wp.transform([i, 0.0, 1.0], wp.quat_identity()), armature=0.1
            )

            # create shape
            builder.add_shape_capsule(
                pos=(self.chain_width * 0.5, 0.0, 0.0),
                half_height=self.chain_width * 0.5,
                radius=0.1,
                up_axis=0,
                density=500.0,
                body=b,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
            )

            if i == 0:
                builder.add_joint_revolute(
                    parent=parent,
                    child=b,
                    axis=(0.0, 0.0, 1.0),
                    parent_xform=parent_joint_xform,
                    limit_lower=self.lower,
                    limit_upper=self.upper,
                    limit_ke=self.limitd_ke,
                    limit_kd=self.limitd_kd,
                )
            else:
                builder.add_joint_revolute(
                    parent=parent,
                    child=b,
                    axis=(0.0, 0.0, 1.0),
                    parent_xform=parent_joint_xform,
                    limit_lower=self.lower,
                    limit_upper=self.upper,
                    limit_ke=self.limitd_ke,
                    limit_kd=self.limitd_kd,
                )

        builder.joint_q[:] = [0.0, 0.0]

        # 7 contact configurations used in the paper
        if self.CONTACT_CONFIG == 0:
            # contact-free
            offset = -15.5
            rot_xyz = np.array([0., 0., 0.])
        elif self.CONTACT_CONFIG == 1:
            # config 1
            offset = 0.0
            rot_xyz = np.array([0., 0., 0.])
        elif self.CONTACT_CONFIG == 2:
            # config 2
            offset = 0.2
            rot_xyz = np.array([np.pi / 8., np.pi / 16, np.pi / 16.])
        elif self.CONTACT_CONFIG == 3:
            # config 3
            offset = 0.5
            rot_xyz = np.array([0., 0., 0.])
        elif self.CONTACT_CONFIG == 4:
            # config 4
            offset = -0.5
            rot_xyz = np.array([0., 0., 0.])
        elif self.CONTACT_CONFIG == 5:
            # config 5
            offset = -0.3
            rot_xyz = np.array([0., 0., 0.])
        elif self.CONTACT_CONFIG == 6:
            # config 6
            offset = 0.0
            rot_xyz = np.array([np.pi / 8., 0., 0.])
        else:
            raise ValueError(f"Invalid contact configuration: {self.CONTACT_CONFIG}")
        
        rot = Rotation.from_euler('xyz', rot_xyz).as_quat()
        update_ground_plane(
            builder,
            pos=[0.0, offset, 0.0],
            rot=rot,
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
        )

    def reset_envs(self, env_ids: wp.array = None):
        wp.launch(
            reset_init_state_pendulum,
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

        # update maximal coordinates after reset
        if env_ids is None or env_ids.numpy().any():
            if self.uses_generalized_coordinates:
                wp.sim.eval_fk(
                    self.model,
                    self.state.joint_q,
                    self.state.joint_qd,
                    None,
                    self.state,
                )
