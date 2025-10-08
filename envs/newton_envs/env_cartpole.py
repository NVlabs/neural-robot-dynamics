# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Cartpole environment
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a URDF using the Environment class.
# Note this example does not include a trained policy.
#
###########################################################################

import os

import warp as wp

import newton

from envs.newton_envs import Environment, SolverType
from envs.newton_envs.utils import angle_normalize

@wp.kernel
def single_cartpole_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_f: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    x = joint_q[env_id * 2 + 0]
    th = joint_q[env_id * 2 + 1]
    xdot = joint_qd[env_id * 2 + 0]
    thdot = joint_qd[env_id * 2 + 1]
    u = joint_f[env_id * 2]

    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
    angle = angle_normalize(th)
    c = angle**2.0 + 0.05 * x**2.0 + 0.1 * thdot**2.0 + 0.1 * xdot**2.0

    wp.atomic_add(cost, env_id, c)

    if terminated:
        terminated[env_id] = abs(x) > 4.0 or abs(thdot) > 10.0 or abs(xdot) > 10.0

@wp.kernel(enable_backward=False)
def reset_init_state(
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
def compute_observations(
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
    env_offset = (2.0, 2.0, 0.0)

    sim_substeps_featherstone = 5
    sim_substeps_mujoco = 10
    sim_substeps_euler = 16
    sim_substeps_xpbd = 5

    fps = 60
    frame_dt = 1.0 / fps
    
    activate_ground_plane = False

    # solver_type = SolverType.FEATHERSTONE
    solver_type = SolverType.MUJOCO

    controllable_dofs = [0]
    control_gains = [1500.0]
    control_limits = [(-1.0, 1.0)]

    def __init__(self, seed = 42, random_reset = True, **kwargs):
        self.seed = seed
        self.random_reset = random_reset
        super().__init__(**kwargs)

    def create_articulation(self, articulation_builder):
        path = "cartpole_single.urdf"

        # update default joint config
        articulation_builder.default_joint_cfg = \
            newton.ModelBuilder.JointDofConfig(
                armature=0.01,
                limit_ke=1.0e4,
                limit_kd=1.0e1,
            )
        
        articulation_builder.add_urdf(
            os.path.join(os.path.dirname(__file__), "assets", path),
            xform=wp.transform(
                p=(0.0, 0.0, 0.0),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            ),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True
        )
        
        articulation_builder.joint_q[:] = [0.0, 0.0]
        
    def compute_cost_termination(
        self,
        state: newton.State,
        control: newton.Control,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        if self.solver_type != SolverType.FEATHERSTONE:
            newton.eval_ik(
                model=self.model, 
                state=state, 
                joint_q=state.joint_q, 
                joint_qd=state.joint_qd
            )
        wp.launch(
            single_cartpole_cost,
            dim=self.num_envs,
            inputs=[state.joint_q, state.joint_qd, control.joint_f],
            outputs=[cost, terminated],
            device=self.device,
        )

    def compute_observations(
        self,
        state: newton.State,
        control: newton.Control,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        if not self.uses_generalized_coordinates:
            # evaluate generalized coordinates
            newton.eval_ik(
                model=self.model, 
                state=state, 
                joint_q=state.joint_q, 
                joint_qd=state.joint_qd
            )
        wp.launch(
            compute_observations,
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

    def reset_envs(self, env_ids: wp.array = None):
        if self.uses_generalized_coordinates:
            wp.launch(
                reset_init_state,
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
                newton.eval_fk(
                    model=self.model, 
                    joint_q=self.state.joint_q, 
                    joint_qd=self.state.joint_qd, 
                    state=self.state,
                    mask=None
                )
