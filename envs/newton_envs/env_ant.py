# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Ant environment
#
# Shows how to set up a simulation of a rigid-body Ant articulation based on
# the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import os
import math
import numpy as np

import warp as wp

import newton

from envs.newton_envs import Environment, SolverType

ZERO_GRAVITY = False

@wp.kernel
def ant_running_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    basis_vec0: wp.vec3,
    basis_vec1: wp.vec3,
    dof_q: int,
    dof_qd: int,
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()
    
    torso_pos = wp.vec3(joint_q[dof_q * env_id + 0],
                        joint_q[dof_q * env_id + 1],
                        joint_q[dof_q * env_id + 2])
    torso_quat = wp.quat(joint_q[dof_q * env_id + 3],
                        joint_q[dof_q * env_id + 4],
                        joint_q[dof_q * env_id + 5],
                        joint_q[dof_q * env_id + 6])
    lin_vel = wp.vec3(joint_qd[dof_qd * env_id + 3],
                      joint_qd[dof_qd * env_id + 4],
                      joint_qd[dof_qd * env_id + 5])
    ang_vel = wp.vec3(joint_qd[dof_qd * env_id + 0],
                      joint_qd[dof_qd * env_id + 1],
                      joint_qd[dof_qd * env_id + 2])
    
    # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
    lin_vel = lin_vel - wp.cross(torso_pos, ang_vel)

    up_vec = wp.quat_rotate(torso_quat, basis_vec1)
    heading_vec = wp.quat_rotate(torso_quat, basis_vec0)

    up_reward = up_vec[2] * 0.1
    heading_reward = heading_vec[1]
    # height_reward = torso_pos[1] - 0.3
    progress_reward = lin_vel[1]

    c = -progress_reward - up_reward - heading_reward #- height_reward

    wp.atomic_add(cost, env_id, c)

    if terminated:
        if torso_pos[2] < 0.3:
            terminated[env_id] = True

@wp.kernel
def ant_spinning_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    basis_vec0: wp.vec3,
    basis_vec1: wp.vec3,
    dof_q: int,
    dof_qd: int,
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()
    
    torso_pos = wp.vec3(joint_q[dof_q * env_id + 0],
                        joint_q[dof_q * env_id + 1],
                        joint_q[dof_q * env_id + 2])
    torso_quat = wp.quat(joint_q[dof_q * env_id + 3],
                        joint_q[dof_q * env_id + 4],
                        joint_q[dof_q * env_id + 5],
                        joint_q[dof_q * env_id + 6])
    ang_vel = wp.vec3(joint_qd[dof_qd * env_id + 0],
                      joint_qd[dof_qd * env_id + 1],
                      joint_qd[dof_qd * env_id + 2])

    up_vec = wp.quat_rotate(torso_quat, basis_vec1)

    spin_reward = ang_vel[2]
    up_reward = up_vec[2]

    c = -spin_reward - up_reward

    wp.atomic_add(cost, env_id, c)

    if terminated:
        if torso_pos[2] < 0.3:
            terminated[env_id] = True

@wp.kernel
def ant_spinning_tracking_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    basis_vec0: wp.vec3,
    basis_vec1: wp.vec3,
    dof_q: int,
    dof_qd: int,
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()
    
    torso_pos = wp.vec3(joint_q[dof_q * env_id + 0],
                        joint_q[dof_q * env_id + 1],
                        joint_q[dof_q * env_id + 2])
    torso_quat = wp.quat(joint_q[dof_q * env_id + 3],
                        joint_q[dof_q * env_id + 4],
                        joint_q[dof_q * env_id + 5],
                        joint_q[dof_q * env_id + 6])
    ang_vel = wp.vec3(joint_qd[dof_qd * env_id + 0],
                      joint_qd[dof_qd * env_id + 1],
                      joint_qd[dof_qd * env_id + 2])

    up_vec = wp.quat_rotate(torso_quat, basis_vec1)

    spin_reward = wp.exp(-(ang_vel[2] - 5.) ** 2. - (ang_vel[0] ** 2.) * 0.1 - (ang_vel[1] ** 2.) * 0.1) * 5.
    up_reward = up_vec[2] * 0.1

    c = -spin_reward - up_reward

    wp.atomic_add(cost, env_id, c)

    if terminated:
        if torso_pos[2] < 0.3:
            terminated[env_id] = True

@wp.kernel
def compute_observations_ant_simple(
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
def compute_observations_ant_dflex(
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

    obs[env_id, 0] = torso_pos[2]  # 0
    for i in range(4):  # 1:5
        obs[env_id, 1 + i] = torso_quat[i]
    for i in range(3):  # 5:8
        obs[env_id, 5 + i] = lin_vel[i]
    for i in range(3):  # 8:11
        obs[env_id, 8 + i] = ang_vel[i]
    for i in range(8):  # 11:19
        obs[env_id, 11 + i] = joint_q[dof_q * env_id + 7 + i]
    for i in range(8):  # 19:27
        obs[env_id, 19 + i] = joint_qd[dof_qd * env_id + 6 + i]
    obs[env_id, 27] = up_vec[2]  # 27
    obs[env_id, 28] = heading_vec[1]  # 28

@wp.kernel(enable_backward=False)
def reset_ant(
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

        # randomize base position
        base_position_perturbation = wp.vec3(0.2, 0.2, 0.1)
        for i in range(3):
            joint_q[env_id * dof_q_per_env + i] = \
                default_joint_q_init[env_id * dof_q_per_env + i] + \
                base_position_perturbation[i] * wp.randf(random_state, -1., 1.)
        # randomize base orientation
        angle = wp.randf(random_state, -1., 1.) * wp.pi / 6.
        axis = wp.vec3(wp.randf(random_state, -1., 1.), wp.randf(random_state, -1., 1.), wp.randf(random_state, -1., 1.))
        axis = wp.normalize(axis)
        default_quat = wp.quat(default_joint_q_init[3], default_joint_q_init[4], default_joint_q_init[5], default_joint_q_init[6])
        delta_quat = wp.quat_from_axis_angle(axis, angle)
        quat_base = default_quat * delta_quat
        for i in range(4):
            joint_q[env_id * dof_q_per_env + i + 3] = quat_base[i]

        # randomize joint angles
        for i in range(8):
            joint_q[env_id * dof_q_per_env + i + 7] = \
                default_joint_q_init[env_id * dof_q_per_env + 7 + i] + wp.randf(random_state, -0.2, 0.2)
        
        # randoimze base angular and linear velocities
        pos_base = wp.vec3(joint_q[dof_q_per_env * env_id + 0],
                           joint_q[dof_q_per_env * env_id + 1],
                           joint_q[dof_q_per_env * env_id + 2])
        
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
        lin_vel_base_world = wp.cross(pos_base, wp.quat_rotate(quat_base, ang_vel_base_body)) + wp.quat_rotate(quat_base, lin_vel_base_body)
        for i in range(3):
            joint_qd[env_id * dof_qd_per_env + i] = ang_vel_base_world[i]
        for i in range(3):
            joint_qd[env_id * dof_qd_per_env + 3 + i] = lin_vel_base_world[i]
        
        # randomize joint velocities
        for i in range(7, dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = \
                default_joint_qd_init[env_id * dof_qd_per_env + i] + 0.25 * wp.randf(random_state, -1., 1.)

class AntEnvironment(Environment):
    robot_name = "Ant"
    sim_name = "env_ant"
    env_offset = (2.5, 2.5, 0.0)

    sim_substeps_euler = 32
    sim_substeps_featherstone = 16
    sim_substeps_xpbd = 8
    sim_substeps_mujoco = 10

    xpbd_settings = dict(iterations=1)
    mujoco_settings = dict(
        njmax=300,
        ncon_per_env=100
    )
    
    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 100.0

    solver_type = SolverType.MUJOCO

    separate_ground_contacts = solver_type != SolverType.XPBD
    activate_ground_plane = not ZERO_GRAVITY
    controllable_dofs = np.arange(6, 14)
    control_gains = np.ones(8) * 400.0
    control_limits = [(-1.0, 1.0)] * 8

    show_rigid_contact_points = False
    contact_points_radius = 0.05

    def __init__(
        self,
        seed=42,
        random_reset=True,
        task="run",
        obs_type="dflex",
        camera_tracking=False,
        **kwargs
    ):
        self.seed = seed
        self.random_reset = random_reset
        self.obs_type = obs_type
        self.task = task
        self.camera_tracking = camera_tracking
        super().__init__(**kwargs)
        self.after_init()

    def create_articulation(self, builder: newton.ModelBuilder):
        # Setting for featherstone
        # update default joint config
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.05,
            limit_ke=1.0e3,
            limit_kd=1.0e2,
        )
        # update default shape config
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=4.e4,
            kd=1.e2,
            kf=3.e3,
            mu=0.75,
        )
        # parse mjcf file
        builder.add_mjcf(
            os.path.join(
                os.path.dirname(__file__), "assets", "ant.xml"
            ),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_names=["floor", "ground"],
        )

        builder.joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
        builder.joint_q[:3] = [0.0, 0.0, 1.0]

        if ZERO_GRAVITY:
            builder.gravity = 0.0

    def after_init(self):
        # create additional variables
        self.basis_vec0 = wp.vec3(0., 1., 0.)
        self.basis_vec1 = wp.vec3(0., 0., 1.)

    def reset_envs(self, env_ids: wp.array = None):
        """Print distance for the envs to be reset. """
        if env_ids is not None and env_ids.numpy().any():
            self.extras['episode'] = {}
            
        """Reset environments where env_ids buffer indicates True. Resets all envs if env_ids is None."""
        wp.launch(
            reset_ant,
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
            newton.eval_fk(
                model=self.model,
                joint_q=self.state.joint_q,
                joint_qd=self.state.joint_qd,
                state=self.state,
                mask=None
            )

    def compute_cost_termination(
        self,
        state: newton.State,
        control: newton.Control,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        if not self.uses_generalized_coordinates:
            newton.eval_ik(
                model=self.model, 
                state=state, 
                joint_q=state.joint_q, 
                joint_qd=state.joint_qd
            )
            
        if self.task == "run":
            wp.launch(
                ant_running_cost,
                dim=self.num_envs,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.basis_vec0,
                    self.basis_vec1,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                ],
                outputs=[cost, terminated],
                device=self.device,
            )
        elif self.task == "spin":
            wp.launch(
                ant_spinning_cost,
                dim=self.num_envs,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.basis_vec0,
                    self.basis_vec1,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                ],
                outputs=[cost, terminated],
                device=self.device,
            )
        elif self.task == "spin_track":
            wp.launch(
                ant_spinning_tracking_cost,
                dim=self.num_envs,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.basis_vec0,
                    self.basis_vec1,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
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
            return 29
        else:
            raise NotImplementedError

    def compute_observations(
        self,
        state: newton.State,
        control: newton.Control,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        if not self.uses_generalized_coordinates:
            # get generalized coordinates
            newton.eval_ik(
                model=self.model, 
                state=state, 
                joint_q=state.joint_q, 
                joint_qd=state.joint_qd
            )
            
        if self.obs_type == "simple":
            wp.launch(
                compute_observations_ant_simple,
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
                compute_observations_ant_dflex,
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

    def custom_render(self, render_state, viewer):
        if self.camera_tracking and hasattr(viewer, "_scaling"):
            robot_pos = wp.to_torch(self.state.body_q)[0, :3]

            cam_pos = wp.vec3(robot_pos[0], 3., 12) * viewer._scaling

            with wp.ScopedTimer("update_view_matrix", color=0x663300, active=self.enable_timers):
                self.viewer.update_view_matrix(cam_pos=cam_pos)
