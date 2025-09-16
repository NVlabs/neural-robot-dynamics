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
import numpy as np

from envs.warp_sim_envs import Environment, IntegratorType

ZERO_GRAVITY = False

@wp.kernel
def anymal_forward_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    contact_depths: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    dof_q: int,
    dof_qd: int,
    num_contacts: int,
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool)
):
    env_id = wp.tid()
    
    torso_pos = wp.vec3(
        joint_q[dof_q * env_id + 0],
        joint_q[dof_q * env_id + 1],
        joint_q[dof_q * env_id + 2]
    )
    lin_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 3],
        joint_qd[dof_qd * env_id + 4],
        joint_qd[dof_qd * env_id + 5]
    )
    ang_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 0],
        joint_qd[dof_qd * env_id + 1],
        joint_qd[dof_qd * env_id + 2]
    )
    
    # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
    lin_vel = lin_vel - wp.cross(torso_pos, ang_vel)
    
    target_x_vel = 1.0
    target_z_vel = 0.0
    target_yaw_vel = 0.0
    lin_vel_error = (lin_vel[0] - target_x_vel) ** 2. + (lin_vel[2] - target_z_vel) ** 2.
    ang_vel_error = (ang_vel[1] - target_yaw_vel) ** 2.

    rew_lin_coef = 1.
    rew_ang_coef = 0.5
    rew_torque_coef = 2.5e-5

    rew_lin_vel = wp.exp(-lin_vel_error) * rew_lin_coef
    rew_ang_vel = wp.exp(-ang_vel_error) * rew_ang_coef
    
    rew_torque = 0.0
    for i in range(12):
        rew_torque -= (joint_act[env_id * 12 + i]) ** 2.0 * rew_torque_coef

    c = - rew_lin_vel - rew_ang_vel - rew_torque
    if c > 0.:
        c = 0.
        
    wp.atomic_add(cost, env_id, c)

    if terminated:
        # torso collision termination
        if contact_depths[num_contacts * env_id + 0] < 0.11 or contact_depths[num_contacts * env_id + 1] < 0.11:
            terminated[env_id] = True
        # knee collision termination
        if contact_depths[num_contacts * env_id + 2] < 0.07 or contact_depths[num_contacts * env_id + 3] < 0.07 or \
           contact_depths[num_contacts * env_id + 5] < 0.07 or contact_depths[num_contacts * env_id + 6] < 0.07 or \
           contact_depths[num_contacts * env_id + 8] < 0.07 or contact_depths[num_contacts * env_id + 9] < 0.07 or \
           contact_depths[num_contacts * env_id + 11] < 0.07 or contact_depths[num_contacts * env_id + 12] < 0.07:
            terminated[env_id] = True
        if torso_pos[1] < 0.4:
            terminated[env_id] = True

@wp.kernel
def anymal_side_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    contact_depths: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    dof_q: int,
    dof_qd: int,
    num_contacts: int,
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool)
):
    env_id = wp.tid()
    
    torso_pos = wp.vec3(
        joint_q[dof_q * env_id + 0],
        joint_q[dof_q * env_id + 1],
        joint_q[dof_q * env_id + 2]
    )
    lin_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 3],
        joint_qd[dof_qd * env_id + 4],
        joint_qd[dof_qd * env_id + 5]
    )
    ang_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 0],
        joint_qd[dof_qd * env_id + 1],
        joint_qd[dof_qd * env_id + 2]
    )
    
    # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
    lin_vel = lin_vel - wp.cross(torso_pos, ang_vel)
    
    target_x_vel = 0.0
    target_z_vel = 1.0
    target_yaw_vel = 0.0
    lin_vel_error = (lin_vel[0] - target_x_vel) ** 2. + (lin_vel[2] - target_z_vel) ** 2.
    ang_vel_error = (ang_vel[1] - target_yaw_vel) ** 2.

    rew_lin_coef = 1.
    rew_ang_coef = 0.5
    rew_torque_coef = 2.5e-5

    rew_lin_vel = wp.exp(-lin_vel_error) * rew_lin_coef
    rew_ang_vel = wp.exp(-ang_vel_error) * rew_ang_coef

    rew_torque = 0.0
    for i in range(12):
        rew_torque -= (joint_act[env_id * 12 + i]) ** 2.0 * rew_torque_coef

    c = - rew_lin_vel - rew_ang_vel - rew_torque
    if c > 0.:
        c = 0.
        
    wp.atomic_add(cost, env_id, c)

    if terminated:
        # torso collision termination
        if contact_depths[num_contacts * env_id + 0] < 0.11 or contact_depths[num_contacts * env_id + 1] < 0.11:
            terminated[env_id] = True
        # knee collision termination
        if contact_depths[num_contacts * env_id + 2] < 0.07 or contact_depths[num_contacts * env_id + 3] < 0.07 or \
           contact_depths[num_contacts * env_id + 5] < 0.07 or contact_depths[num_contacts * env_id + 6] < 0.07 or \
           contact_depths[num_contacts * env_id + 8] < 0.07 or contact_depths[num_contacts * env_id + 9] < 0.07 or \
           contact_depths[num_contacts * env_id + 11] < 0.07 or contact_depths[num_contacts * env_id + 12] < 0.07:
            terminated[env_id] = True
        if torso_pos[1] < 0.4:
            terminated[env_id] = True

@wp.kernel
def compute_observations_anymal_simple(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    # outputs
    obs: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    obs[tid, 0] = joint_q[tid * dof_q_per_env + 1]
    for i in range(3, dof_q_per_env):
        obs[tid, i - 2] = joint_q[tid * dof_q_per_env + i]
    for i in range(dof_qd_per_env):
        obs[tid, i + dof_q_per_env - 2] = joint_qd[tid * dof_qd_per_env + i]

@wp.kernel
def compute_observations_anymal_dflex(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    basis_vec0: wp.vec3,
    basis_vec1: wp.vec3,
    dof_q: int,
    dof_qd: int,
    # outputs
    obs: wp.array(dtype=float, ndim=2),
):
    env_id = wp.tid()

    torso_pos = wp.vec3(
        joint_q[dof_q * env_id + 0],
        joint_q[dof_q * env_id + 1],
        joint_q[dof_q * env_id + 2],
    )
    torso_quat = wp.quat(
        joint_q[dof_q * env_id + 3],
        joint_q[dof_q * env_id + 4],
        joint_q[dof_q * env_id + 5],
        joint_q[dof_q * env_id + 6],
    )
    lin_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 3],
        joint_qd[dof_qd * env_id + 4],
        joint_qd[dof_qd * env_id + 5],
    )
    ang_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 0],
        joint_qd[dof_qd * env_id + 1],
        joint_qd[dof_qd * env_id + 2],
    )

    # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
    lin_vel = lin_vel - wp.cross(torso_pos, ang_vel)

    up_vec = wp.quat_rotate(torso_quat, basis_vec1)
    heading_vec = wp.quat_rotate(torso_quat, basis_vec0)

    obs[env_id, 0] = torso_pos[1]  # 0
    for i in range(4):  # 1:5
        obs[env_id, 1 + i] = torso_quat[i]
    for i in range(3):  # 5:8
        obs[env_id, 5 + i] = lin_vel[i]
    for i in range(3):  # 8:11
        obs[env_id, 8 + i] = ang_vel[i]
    for i in range(12):  # 11:23
        obs[env_id, 11 + i] = joint_q[dof_q * env_id + 7 + i]
    for i in range(12):  # 23:35
        obs[env_id, 23 + i] = joint_qd[dof_qd * env_id + 6 + i]
    obs[env_id, 35] = up_vec[1]  # 35
    obs[env_id, 36] = heading_vec[0]  # 36

@wp.kernel(enable_backward=False)
def reset_anymal_dataset(
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

    for i in range(dof_q_per_env):
        joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[
            env_id * dof_q_per_env + i
        ]
    for i in range(dof_qd_per_env):
        joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[
            env_id * dof_qd_per_env + i
        ]

    if random_reset:
        random_state = wp.rand_init(seed, env_id)

        # randomize base position
        base_position_perturbation = wp.vec3(0.2, 0.1, 0.2)
        for i in range(3):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[
                env_id * dof_q_per_env + i
            ] + base_position_perturbation[i] * wp.randf(random_state, -1.0, 1.0)
        # separately do for height
        joint_q[env_id * dof_q_per_env + 1] = wp.randf(random_state, 0.65, 1.0)
        
        # randomize base orientation
        angle = wp.randf(random_state, -1.0, 1.0) * wp.pi / 6.0
        axis = wp.vec3(
            wp.randf(random_state, -1.0, 1.0),
            wp.randf(random_state, -1.0, 1.0),
            wp.randf(random_state, -1.0, 1.0),
        )
        axis = wp.normalize(axis)
        default_quat = wp.quat(
            default_joint_q_init[3],
            default_joint_q_init[4],
            default_joint_q_init[5],
            default_joint_q_init[6],
        )
        delta_quat = wp.quat_from_axis_angle(axis, angle)
        quat_base = default_quat * delta_quat
        for i in range(4):
            joint_q[env_id * dof_q_per_env + i + 3] = quat_base[i]

        # randomize joint angles
        for i in range(12):
            joint_q[env_id * dof_q_per_env + i + 7] = default_joint_q_init[
                env_id * dof_q_per_env + 7 + i
            ] + wp.randf(random_state, -0.2, 0.2)

        # randoimze base angular and linear velocities
        pos_base = wp.vec3(
            joint_q[dof_q_per_env * env_id + 0],
            joint_q[dof_q_per_env * env_id + 1],
            joint_q[dof_q_per_env * env_id + 2],
        )

        ang_vel_base_body = wp.vec3(0.0, 0.0, 0.0)
        lin_vel_base_body = wp.vec3(0.0, 0.0, 0.0)
        ang_vel_base_body = wp.vec3(
            0.25 * wp.randf(random_state, -1., 1.),
            0.25 * wp.randf(random_state, -1., 1.),
            0.25 * wp.randf(random_state, -1., 1.)
        )
        lin_vel_base_body = wp.vec3(
            0.1 * wp.randf(random_state, -1., 1.),
            0.1 * wp.randf(random_state, -1., 1.),
            0.1 * wp.randf(random_state, -1., 1.)
        )

        ang_vel_base_world = wp.quat_rotate(quat_base, ang_vel_base_body)
        lin_vel_base_world = wp.cross(
            pos_base, wp.quat_rotate(quat_base, ang_vel_base_body)
        ) + wp.quat_rotate(quat_base, lin_vel_base_body)
        for i in range(3):
            joint_qd[env_id * dof_qd_per_env + i] = ang_vel_base_world[i]
        for i in range(3):
            joint_qd[env_id * dof_qd_per_env + 3 + i] = lin_vel_base_world[i]

        # randomize joint velocities
        for i in range(7, dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = \
                default_joint_qd_init[env_id * dof_qd_per_env + i] + 0.25 * wp.randf(random_state, -1., 1.)

@wp.kernel(enable_backward=False)
def reset_anymal(
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

    for i in range(dof_q_per_env):
        joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[env_id * dof_q_per_env + i]
    for i in range(dof_qd_per_env):
        joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[env_id * dof_qd_per_env + i]
    
    if random_reset:
        random_state = wp.rand_init(seed, env_id)

        # randomize joint angles
        for i in range(12):
            joint_q[env_id * dof_q_per_env + i + 7] = \
                default_joint_q_init[env_id * dof_q_per_env + 7 + i] * wp.randf(random_state, 0.5, 1.5)
        
        # randomize joint velocities
        for i in range(7, dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = \
                default_joint_qd_init[env_id * dof_qd_per_env + i] + 0.1 * wp.randf(random_state, -1., 1.)
            
class AnymalEnvironment(Environment):
    robot_name = "AnyMAL"
    sim_name = "env_anymal"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=0.5)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_featherstone = 10
    sim_substeps_xpbd = 8

    xpbd_settings = dict(iterations=2)

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    integrator_type = IntegratorType.FEATHERSTONE

    # separate handling of ground contacts
    separate_ground_contacts = True

    handle_collisions_once_per_step = True
    
    use_graph_capture = False

    num_envs = 1

    activate_ground_plane = not ZERO_GRAVITY

    action_strength = 150.0

    controllable_dofs = np.arange(12)
    control_gains = (
        np.array(
            [
                50.0,  # LF_HAA
                40.0,  # LF_HFE
                8.0,  # LF_KFE
                50.0,  # RF_HAA
                40.0,  # RF_HFE
                8.0,  # RF_KFE
                50.0,  # LH_HAA
                40.0,  # LH_HFE
                8.0,  # LH_KFE
                50.0,  # RH_HAA
                40.0,  # RH_HFE
                8.0,  # RH_KFE
            ]
        )
        * action_strength / 100.0
    )

    control_limits = [(-1.0, 1.0)] * 12

    show_rigid_contact_points = False
    contact_points_radius = 0.05

    def __init__(
        self,
        seed=42,
        random_reset=True,
        task="forward",
        obs_type="dflex",
        camera_tracking=False,
        **kwargs
    ):
        self.seed = seed
        self.random_reset = random_reset
        self.obs_type = obs_type
        self.camera_tracking = camera_tracking
        self.task = task
        super().__init__(**kwargs)
        self.after_init()

    def create_articulation(self, builder):
        urdf_filename = "anymal_joint_limits.urdf"
        self.num_contacts_per_env = 14
            
        wp.sim.parse_urdf(
            os.path.join(
                os.path.dirname(__file__),
                f"assets/anymal/urdf/{urdf_filename}",
            ),
            builder,
            floating=True,
            stiffness=85.0,
            damping=2.0,
            armature=0.06,
            contact_ke=2.0e3,
            contact_kd=5.0e2,
            contact_kf=1.0e2,
            contact_mu=0.75,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False
        )

        self.start_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)
        self.inv_start_rot = wp.quat_inverse(self.start_rot)

        builder.joint_q[:7] = [
            0.0,
            0.7,
            0.0,
            *self.start_rot,
        ]

        builder.joint_q[7:] = [
            0.03,  # LF_HAA
            0.4,  # LF_HFE
            -0.8,  # LF_KFE
            -0.03,  # RF_HAA
            0.4,  # RF_HFE
            -0.8,  # RF_KFE
            0.03,  # LH_HAA
            -0.4,  # LH_HFE
            0.8,  # LH_KFE
            -0.03,  # RH_HAA
            -0.4,  # RH_HFE
            0.8,  # RH_KFE
        ]

        for i in range(builder.joint_axis_count):
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_FORCE

        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.device)

        if ZERO_GRAVITY:
            builder.gravity = 0.0


        self.torques = wp.zeros(
            self.num_envs * builder.joint_axis_count,
            dtype=wp.float32,
            device=self.device,
        )

        builder.separate_ground_contacts = self.separate_ground_contacts

    def after_init(self):
        # create additional variables
        self.start_torso_pos = wp.array(
            self.model.joint_q.numpy().reshape(self.num_envs, -1)[:, 0:3].reshape(-1).copy()
        )
        
        self.start_rot = wp.quat(self.model.joint_q.numpy()[3:7])
        self.inv_start_rot = wp.quat_inverse(self.start_rot)

        self.basis_vec0 = wp.vec3(1., 0., 0.)
        self.basis_vec1 = wp.vec3(0., 0., 1.)
    
    def reset_envs(self, env_ids: wp.array = None):
        """Reset environments where env_ids buffer indicates True. Resets all envs if env_ids is None."""
        if self.task == "dataset":
            wp.launch(
                reset_anymal_dataset,
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
                outputs=[
                    self.state.joint_q,
                    self.state.joint_qd,
                ],
                device=self.device,
            )    
        else:
            wp.launch(
                reset_anymal,
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
                outputs=[
                    self.state.joint_q,
                    self.state.joint_qd,
                ],
                device=self.device,
            )
        self.seed += self.num_envs

        # update maximal coordinates after reset
        if env_ids is None or env_ids.numpy().any():
            wp.sim.eval_fk(
                self.model, 
                self.state.joint_q, 
                self.state.joint_qd, 
                None, 
                self.state
            )

    def compute_cost_termination(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        if not self.uses_generalized_coordinates:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        if self.task == "forward":
            wp.launch(
                anymal_forward_cost,
                dim=self.num_envs,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.model.rigid_contact_depth,
                    control.joint_act,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                    self.num_contacts_per_env
                ],
                outputs=[cost, terminated],
                device=self.device,
            )
        elif self.task == "side":
            wp.launch(
                anymal_side_cost,
                dim=self.num_envs,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.model.rigid_contact_depth,
                    control.joint_act,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                    self.num_contacts_per_env
                ],
                outputs=[cost, terminated],
                device=self.device,
            )
        else:
            raise NotImplementedError
            
    @property
    def observation_dim(self):
        if self.obs_type == "simple":
            # joint q, joint qd
            return self.dof_q_per_env + self.dof_qd_per_env - 2
        elif self.obs_type == "dflex":
            return 37
        else:
            raise NotImplementedError

    def compute_observations(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        if not self.uses_generalized_coordinates:
            # evaluate generalized coordinates
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        if self.obs_type == "simple":
            wp.launch(
                compute_observations_anymal_simple,
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
        elif self.obs_type == "dflex":
            wp.launch(
                compute_observations_anymal_dflex,
                dim=self.num_envs,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.basis_vec0,
                    self.basis_vec1,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                ],
                outputs=[observations],
                device=self.device,
            )
        else:
            raise NotImplementedError

        
    def assign_control(
        self,
        actions: wp.array,
        control: wp.sim.Control,
        state: wp.sim.State,
    ):
        super().assign_control(actions, control, state)
        self.raw_joint_act = wp.from_torch(wp.to_torch(control.joint_act).clone())
        # apply joint stiffness/damping
        self.apply_pd_control(
            control_out=control.joint_act,
            joint_q=state.joint_q,
            joint_qd=state.joint_qd,
            body_q=state.body_q,
        )
    
    def custom_render(self, render_state, renderer):
        if self.camera_tracking and hasattr(renderer, "_scaling"):
            robot_pos = wp.to_torch(self.state.body_q)[0, :3]

            offset = np.array([0., -2.5, -7.]) * 0.8
            cam_pos = wp.vec3(
                robot_pos[0] - offset[0], 
                robot_pos[1] - offset[1], 
                robot_pos[2] - offset[2]
            ) * renderer._scaling
            view_direction = offset / np.linalg.norm(offset)
            cam_front = wp.vec3(view_direction[0], view_direction[1], view_direction[2])
            with wp.ScopedTimer("update_view_matrix", color=0x663300, active=self.enable_timers):
                self.renderer.update_view_matrix(cam_pos=cam_pos, cam_front=cam_front)