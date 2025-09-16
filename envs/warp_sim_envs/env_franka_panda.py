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
import numpy as np
import warp as wp
import warp.sim
import copy
import torch

from envs.warp_sim_envs import Environment, IntegratorType, RenderMode

@wp.kernel
def apply_joint_position_pd_control(
    actions: wp.array(dtype=wp.float32, ndim=1),
    action_scale: wp.float32,
    baseline_joint_q: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_torque_limit: wp.array(dtype=wp.float32),
    Kp: wp.array(dtype=wp.float32),
    Kd: wp.array(dtype=wp.float32),
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
        
        tq = Kp[aj] * (tq - q) - Kd[aj] * qd
        
        joint_act[aj] = wp.clamp(
            tq, -joint_torque_limit[aj], joint_torque_limit[aj]
        )

@wp.kernel
def franka_reaching_cost(
    body_q: wp.array(dtype=wp.transform),
    ee_id: int,
    ee_offset: wp.transform,
    targets: wp.array(dtype=wp.vec3),
    env_offsets: wp.array(dtype=wp.vec3),
    num_bodies: int,
    # outputs
    cost: wp.array(dtype=wp.float32),
    distance: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    ee_tf = body_q[num_bodies * env_id + ee_id]
    ee_pos = wp.transform_get_translation(ee_tf * ee_offset)

    c = float(wp.length(ee_pos - targets[env_id] - env_offsets[env_id]))

    distance[env_id] = c

    alive_bonus = 2. * 0.


    dist = wp.length(ee_pos - targets[env_id] - env_offsets[env_id])

    reward_exp = 0.
    a, b = (50., 0.0001)
    reward_exp += 1. / (wp.exp(a * dist) + b + wp.exp(-a * dist))
    a, b = (300., 0.0001)
    reward_exp += 1. / (wp.exp(a * dist) + b + wp.exp(-a * dist))

    wp.atomic_add(cost, env_id, c - alive_bonus - reward_exp)
    
    terminated[env_id] = False

@wp.kernel
def compute_observations_franka(
    include_joint_vel: bool,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    ee_index: int,
    ee_offset: wp.transform,
    targets: wp.array(dtype=wp.vec3),
    env_offsets: wp.array(dtype=wp.vec3),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    bodies_per_env: int,
    # outputs
    obs: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    for i in range(dof_q_per_env):
        obs[tid, i] = joint_q[tid * dof_q_per_env + i]
    if include_joint_vel:
        for i in range(dof_qd_per_env):
            obs[tid, i + dof_q_per_env] = joint_qd[tid * dof_qd_per_env + i]
    else:
        dof_qd_per_env = 0
    ee_tf = body_q[tid * bodies_per_env + ee_index]
    ee_pos = wp.transform_get_translation(ee_tf * ee_offset)
    for i in range(3):
        obs[tid, i + dof_q_per_env + dof_qd_per_env] = ee_pos[i] - env_offsets[tid][i]
    for i in range(3):
        obs[tid, i + dof_q_per_env + dof_qd_per_env + 3] = targets[tid][i]

@wp.kernel(enable_backward=False)
def resample_targets(
    randomize_targets: bool,
    kernel_seed: int,
    env_to_resample: wp.array(dtype=wp.bool),
    env_offsets: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    targets_render: wp.array(dtype=wp.vec3),
    target_pos_bounds: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()
    if env_to_resample:
        if not env_to_resample[env_id]:
            return

    if randomize_targets:
        random_state = wp.rand_init(kernel_seed, env_id)
        # sample target within target_pos_bounds
        target_x = wp.randf(random_state, 0., 1.)
        target_y = wp.randf(random_state, 0., 1.)
        target_z = wp.randf(random_state, 0., 1.)
        
        targets[env_id] = wp.vec3(
            target_x * (target_pos_bounds[0, 1] - target_pos_bounds[0, 0]) + target_pos_bounds[0, 0],
            target_y * (target_pos_bounds[1, 1] - target_pos_bounds[1, 0]) + target_pos_bounds[1, 0],
            target_z * (target_pos_bounds[2, 1] - target_pos_bounds[2, 0]) + target_pos_bounds[2, 0],
        )

    if targets_render:
        targets_render[env_id] = targets[env_id] + env_offsets[env_id]
        
@wp.kernel(enable_backward=False)
def reset_init_state_franka(
    reset: wp.array(dtype=wp.bool),
    seed: int,
    random_reset: bool,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    default_joint_q_init: wp.array(dtype=wp.float32),
    default_joint_qd_init: wp.array(dtype=wp.float32),
    joint_q_range_min: wp.array(dtype=wp.float32),
    joint_q_range_max: wp.array(dtype=wp.float32),
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
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = wp.randf(
                random_state, joint_q_range_min[i], joint_q_range_max[i]
            )
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[i]
    else:
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[i]
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[i]

class FrankaPandaEnvironment(Environment):
    robot_name = 'Franka'
    sim_name = "env_franka_panda"
    
    env_offset = (2.0, 2.0, 0.0)
    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    up_axis: str = "Z"

    frame_dt = 1.0 / 60.0 / 2

    sim_substeps_euler = 32
    sim_substeps_featherstone = 9
    sim_substeps_xpbd = 10

    num_rigid_contacts_per_env = 0

    activate_ground_plane = False
    include_joint_velocities = False

    integrator_type = IntegratorType.FEATHERSTONE

    show_target_point = True
    show_endeffector_point = True

    controllable_dofs = [0, 1, 2, 3, 4, 5, 6]
    control_gains = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    control_limits = [(-1.0, 1.0)] * 7
    
    has_gripper = False

    num_envs = 1

    gravity = 0.0 # disable gravity for franka

    # bounds within which to select target position for subsequent movement of 
    # gripper using RL policy ([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
    target_pos_bounds = wp.array(
        [[0.3, 0.7], [-0.35, 0.35], [0.05, 0.45]], dtype=wp.float32
    ) 

    initial_joint_pos = np.array(
        [0, -0.7853, 0, -2.3575, 0, 1.5703, 0.7857], dtype=np.float32
    )
    
    # randomization range for initial joint position
    initial_joint_pos_range = np.array(
        [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], dtype=np.float32
    )

    def __init__(
        self, 
        seed=42, 
        control_mode="joint_position_control",
        random_reset=False, 
        random_target=True,
        **kwargs
    ):
        self.seed = seed
        self.random_reset = random_reset
        self.randomize_targets = random_target
        assert control_mode in ["joint_position_control", "joint_torque_control"]
        self.control_mode = control_mode
        super().__init__(**kwargs)
        self.distance = wp.zeros(
            self.num_envs, dtype=wp.float32, device=self.device
        )
        
        if control_mode == "joint_position_control":
            self.target_joint_q = wp.empty(
                (self.num_envs * self.control_dim), 
                dtype=wp.float32,
                device=self.device
            )
            self.joint_torque_limits = wp.from_torch(
                torch.tensor(
                    [
                        gain * control_limit[1] for gain, control_limit in zip(
                            self.control_gains, self.control_limits
                        )
                    ], 
                    dtype=torch.float32, 
                    device=wp.device_to_torch(self.device)
                ).repeat((self.num_envs))
            )
            self.Kp = wp.from_torch(
                torch.tensor(
                    [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0],
                    dtype=torch.float32,
                    device=wp.device_to_torch(self.device)
                ).repeat((self.num_envs))
            )
            self.Kd = wp.from_torch(
                torch.tensor(
                    [5.0, 5.0, 5.0, 5.0, 3.0, 2.5, 1.5],
                    dtype=torch.float32,
                    device=wp.device_to_torch(self.device)
                ).repeat((self.num_envs))
            )

    def create_articulation(self, builder):
        urdf_file = (
            "frankaEmikaPanda.urdf"
            if self.has_gripper
            else "frankaEmikaPanda_fixed_gripper.urdf"
        )
        path = os.path.join(
            os.path.dirname(__file__),
            f"assets/franka_description/robots/{urdf_file}",
        )
        wp.sim.parse_urdf(
            path,
            builder,
            xform=wp.transform(
                (0.0, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.0),
            ),
            floating=False,
            armature=0.01,
            density=3500,
            limit_ke=0.,
            limit_kd=0.,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        if self.has_gripper:
            self.endeffector_id = builder.body_count - 3
            self.endeffector_offset = wp.transform([0.0, 0.0, 0.22], wp.quat_identity())
        else:
            self.endeffector_id = builder.body_count - 1
            self.endeffector_offset = wp.transform(
                [0.0, 0.0, 2.1207e-01], wp.quat_identity()
            )

        builder.joint_q[:7] = [0, -0.7853, 0, -2.3575, 0, 1.5703, 0.7857]
        
        self.task_joint_limit_lower = wp.array(
            copy.deepcopy(builder.joint_limit_lower), 
            dtype=wp.float32, 
            device=self.device
        )
        self.task_joint_limit_upper = wp.array(
            copy.deepcopy(builder.joint_limit_upper), 
            dtype=wp.float32, 
            device=self.device
        )

        # set joints to be continuous joints since limit_ke and limit_kd = 0
        builder.joint_limit_lower[:7] = [-2. * np.pi] * 7
        builder.joint_limit_upper[:7] = [2. * np.pi] * 7
        
        if self.show_endeffector_point:
            builder.add_shape_sphere(
                self.endeffector_id,
                pos=self.endeffector_offset.p,
                radius=0.025,
                density=0.0,
                has_shape_collision=False,
            )

        self.env_offsets_wp = wp.array(self.env_offsets, device=self.device, dtype=wp.vec3)

        # Update body mass to match Isaac Lab's model
        builder.body_mass = [2.9281, 2.8215, 2.8565, 2.1814, 2.2329, 3.2655, 2.4965]

        self.targets = wp.array(
            np.tile([0.0, 0.0, 0.5], (self.num_envs, 1)), dtype=wp.vec3, device=self.device
        )
        self.targets_render = wp.empty_like(self.targets)
        self.target_pos_bounds = self.target_pos_bounds.to(self.device)

        for i in range(builder.joint_count):
            builder.joint_act[i] = 0.0
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_FORCE
    
    def sample_targets(self, env_ids: wp.array = None):
        wp.launch(
            resample_targets,
            dim=self.num_envs,
            inputs=[
                self.randomize_targets,
                self.seed,
                env_ids,
                self.env_offsets_wp,
                self.targets,
                self.targets_render,
                self.target_pos_bounds,
            ],
            device=self.device,
        )
        self.seed += self.num_envs

    def reset_envs(self, env_ids: wp.array = None):
        """Print distance for the envs to be reset. """
        if env_ids is not None and env_ids.numpy().any():
            self.extras['episode'] = {
                'final distance': np.mean(self.distance.numpy()[env_ids.numpy()])
            }
            
        """Reset environments where env_ids buffer indicates True. """
        """Resets all envs if env_ids is None. """
        """None only appears in the first reset call before sim."""
        wp.launch(
            reset_init_state_franka,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.seed,
                self.random_reset,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.model.joint_q,
                self.model.joint_qd,
                wp.from_numpy(
                    self.initial_joint_pos - self.initial_joint_pos_range, 
                    device = self.device
                ),
                wp.from_numpy(
                    self.initial_joint_pos + self.initial_joint_pos_range, 
                    device = self.device
                ),
            ],
            outputs=[
                self.state.joint_q,
                self.state.joint_qd,
            ],
            device=self.device,
        )
        self.seed += self.num_envs
        self.sample_targets(env_ids)

        # update maximal coordinates after reset
        if env_ids is None or env_ids.numpy().any():
            if self.uses_generalized_coordinates:
                wp.sim.eval_fk(
                    self.model, 
                    self.state.joint_q, 
                    self.state.joint_qd, 
                    None, 
                    self.state
                )

    def custom_render(self, render_state, renderer):
        if self.show_target_point:
            if self.render_mode == RenderMode.OPENGL:
                renderer.render_points(
                    "targets", self.targets_render, radius=0.025, colors=(1.0, 0.0, 0.0)
                )
            elif self.render_mode == RenderMode.USD:
                renderer.render_points(
                    "targets",
                    self.targets_render.numpy(),
                    radius=0.025,
                    colors=(1.0, 0.0, 0.0),
                )

    def compute_cost_termination(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        step: int,
        max_episode_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        if not self.uses_generalized_coordinates:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)

        wp.launch(
            franka_reaching_cost,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                self.endeffector_id,
                self.endeffector_offset,
                self.targets,
                self.env_offsets_wp,
                self.bodies_per_env,
            ],
            outputs=[cost, self.distance, terminated],
            device=self.device,
        )

    @property
    def observation_dim(self):
        if self.include_joint_velocities:
            # joint q, joint qd, target, endeffector
            return self.dof_q_per_env + self.dof_qd_per_env + 3 + 3
        else:
            # joint q, target, endeffector
            return self.dof_q_per_env + 3 + 3

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

        wp.launch(
            compute_observations_franka,
            dim=self.num_envs,
            inputs=[
                self.include_joint_velocities,
                state.joint_q,
                state.joint_qd,
                state.body_q,
                self.endeffector_id,
                self.endeffector_offset,
                self.targets,
                self.env_offsets_wp,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.bodies_per_env,
            ],
            outputs=[observations],
            device=self.device,
        )
        
    def assign_control(
        self,
        actions: wp.array,
        control: wp.sim.Control,
        state: wp.sim.State,
    ):
        if self.control_mode == "joint_torque_control":
            super().assign_control(actions, control, state)
        elif self.control_mode == "joint_position_control":
            action_scale = 0.1250 / 4.0
            
            wp.launch(
                kernel=apply_joint_position_pd_control,
                dim=self.model.joint_count,
                inputs=[
                    wp.from_torch(wp.to_torch(actions).reshape(-1)),
                    action_scale,
                    state.joint_q,
                    state.joint_q,
                    state.joint_qd,
                    self.joint_torque_limits,
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