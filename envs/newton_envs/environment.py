# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from enum import Enum
from typing import Tuple, Callable

import numpy as np

import warp as wp
import newton
import newton.solvers
import newton.viewer
from envs.newton_envs.utils import assign_controls

from tqdm import trange

from typing import List


class RenderMode(Enum):
    NONE = "none"
    OPENGL = "opengl"
    USD = "usd"

    def __str__(self):
        return self.value

class SolverType(Enum):
    FEATHERSTONE = "featherstone"
    MUJOCO = "mujoco"
    EULER = "euler"
    XPBD = "xpbd"
    NEURAL = "neural"

    def __str__(self):
        return self.value


def compute_env_offsets(
    num_envs, env_offset=(5.0, 0.0, 5.0), up_axis="Y"
) -> List[np.ndarray]:
    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))
    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i * env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
            d0 = i // (side_length * side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)
    env_offsets = np.array(env_offsets)
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0  # ensure the envs are not shifted below the ground plane
    env_offsets -= correction
    return env_offsets

@wp.kernel(enable_backward=False)
def reset_maximal_coords(
    reset: wp.array(dtype=wp.bool),
    body_q_init: wp.array(dtype=wp.transform),
    body_qd_init: wp.array(dtype=wp.spatial_vector),
    num_bodies_per_env: int,
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    env_id = wp.tid()
    if reset:
        if not reset[env_id]:
            return
    for i in range(num_bodies_per_env):
        j = env_id * num_bodies_per_env + i
        body_q[j] = body_q_init[j]
        body_qd[j] = body_qd_init[j]


@wp.kernel(enable_backward=False)
def reset_generalized_coords(
    reset: wp.array(dtype=wp.bool),
    joint_q_init: wp.array(dtype=wp.float32),
    joint_qd_init: wp.array(dtype=wp.float32),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    if reset:
        if not reset[env_id]:
            return
    for i in range(dof_q_per_env):
        j = env_id * dof_q_per_env + i
        joint_q[j] = joint_q_init[j]
    for i in range(dof_qd_per_env):
        j = env_id * dof_qd_per_env + i
        joint_qd[j] = joint_qd_init[j]


class Environment:
    sim_name: str = "Environment"

    fps = 60
    frame_dt = 1.0 / fps

    episode_duration = 5.0  # seconds
    episode_frames = (
        None  # number of steps per episode, if None, use episode_duration / frame_dt
    )

    # whether to play the simulation indefinitely when using the OpenGL renderer
    continuous_opengl_render: bool = True

    sim_substeps_featherstone: int = 10
    featherstone_update_mass_matrix_once_per_step: bool = True

    sim_substeps_xpbd: int = 5 ##
    sim_substeps_mujoco: int = 10
    sim_substeps_euler: int = 16

    euler_settings = dict()
    featherstone_settings = dict()
    xpbd_settings = dict()
    mujoco_settings = dict()

    num_envs = 100
    env_offset = (1.0, 0.0, 1.0)

    render_mode: RenderMode = RenderMode.OPENGL
    opengl_render_settings = dict()
    usd_render_settings = dict()
    show_rigid_contact_points = False
    contact_points_radius = 1e-3

    # whether to apply model.joint_q, joint_qd to bodies before simulating
    eval_fk: bool = True

    use_graph_capture: bool = wp.get_preferred_device().is_cuda

    activate_ground_plane: bool = True

    solver_type: SolverType = SolverType.XPBD

    up_axis: str = "Z"
    gravity: float = -9.81

    # stiffness and damping for joint attachment dynamics used by Euler
    joint_attach_ke: float = 32000.0
    joint_attach_kd: float = 50.0

    # maximum number of rigid contact points to generate per mesh
    rigid_mesh_contact_max: int = 0  # (0 = unlimited)

    # distance threshold at which contacts are generated
    rigid_contact_margin: float = 0.05
    # whether to iterate over mesh vertices for box/capsule collision
    rigid_contact_iterate_mesh_vertices: bool = True
    # number of rigid contact points to allocate in the model during self.finalize() per environment
    # if setting is None, the number of worst-case number of contacts will be calculated in self.finalize()
    num_rigid_contacts_per_env: int | None = None

    # whether to call warp.sim.collide() just once per update
    handle_collisions_once_per_step: bool = False

    # number of search iterations for finding closest contact points between edges and SDF
    edge_sdf_iter: int = 10

    plot_body_coords: bool = False
    plot_joint_coords: bool = False
    plot_joint_coords_qd: bool = False

    # custom dynamics function to be called instead of the default simulation step
    # signature: custom_dynamics(model, state, next_state, sim_dt, control)
    custom_dynamics: Callable[
        [newton.Model, newton.State, newton.State, newton.Control, newton.Contacts, float], None
    ] | None = None

    # control-related definitions, to be updated by derived classes
    controllable_dofs = []
    named_control_gains = None
    control_gains = []
    control_gain_scale = 1.0
    control_limits = []

    def __init__(
        self,
        num_envs: int = None,
        episode_frames: int = None,
        solver_type: SolverType = None,
        render_mode: RenderMode = None,
        env_offset: tuple[float, float, float] | None = None,
        device: wp.context.Devicelike = None,
        requires_grad: bool = False,
        profile: bool = False,
        enable_timers: bool = False,
        use_graph_capture: bool = None,
        setup_viewer: bool = True
    ):
        if num_envs is not None:
            self.num_envs = num_envs
        if episode_frames is not None:
            self.episode_frames = episode_frames
        if solver_type is not None:
            self.solver_type = solver_type
        if render_mode is not None:
            self.render_mode = render_mode
        if use_graph_capture is not None:
            self.use_graph_capture = use_graph_capture
        self.device = wp.get_device(device)
        self.requires_grad = requires_grad
        self.profile = profile
        self.enable_timers = enable_timers

        if env_offset is not None:
            self.env_offset = env_offset
        
        builder = newton.ModelBuilder(
            up_axis=newton.Axis.from_string(self.up_axis), 
            gravity=self.gravity
        )
        
        builder.rigid_mesh_contact_max = self.rigid_mesh_contact_max
        builder.rigid_contact_margin = self.rigid_contact_margin
        builder.num_rigid_contacts_per_env = self.num_rigid_contacts_per_env
        
        self.env_offsets = compute_env_offsets(
            self.num_envs, self.env_offset, self.up_axis
        )
        self.env_offsets_wp = wp.array(
            self.env_offsets, dtype=wp.vec3, device=self.device
        )
        
        try:
            articulation_builder = newton.ModelBuilder(
                up_axis=newton.Axis.from_string(self.up_axis), 
                gravity=self.gravity
            )
            self.create_articulation(articulation_builder)
            for i in trange(
                self.num_envs, desc=f"Creating {self.num_envs} environments"
            ):
                xform = wp.transform(self.env_offsets[i], wp.quat_identity())
                builder.add_builder(
                    articulation_builder,
                    xform,
                )
            self.bodies_per_env = articulation_builder.body_count
            self.dof_q_per_env = articulation_builder.joint_coord_count
            self.dof_qd_per_env = articulation_builder.joint_dof_count
        except NotImplementedError:
            # custom simulation setup where something other than an articulation is used
            self.setup(builder)
            self.bodies_per_env = builder.body_count
            self.dof_q_per_env = builder.joint_coord_count
            self.dof_qd_per_env = builder.joint_dof_count

        if self.activate_ground_plane:
            builder.add_ground_plane()
    
        self.model = builder.finalize(requires_grad=self.requires_grad, device=self.device)
        self.customize_model(self.model)
        
        self.model.ground = self.activate_ground_plane

        self.device = self.model.device
        if not self.model.device.is_cuda:
            self.use_graph_capture = False

        if self.solver_type == SolverType.EULER:
            self.sim_substeps = self.sim_substeps_euler
            self.solver = newton.solvers.SolverSemiImplicit(self.model, **self.euler_settings)
        elif self.solver_type == SolverType.FEATHERSTONE:
            self.sim_substeps = self.sim_substeps_featherstone
            if self.featherstone_update_mass_matrix_once_per_step:
                self.featherstone_settings["update_mass_matrix_interval"] = (
                    self.sim_substeps
                )
            self.solver = newton.solvers.SolverFeatherstone(self.model, **self.featherstone_settings)
        elif self.solver_type == SolverType.MUJOCO:
            self.sim_substeps = self.sim_substeps_mujoco
            self.solver = newton.solvers.SolverMuJoCo(self.model, **self.mujoco_settings)
        elif self.solver_type == SolverType.XPBD:
            self.sim_substeps = self.sim_substeps_xpbd
            self.solver = newton.solvers.SolverXPBD(self.model, **self.xpbd_settings)
        else:
            raise ValueError(f"Solver type {self.solver_type} not supported")
        
        if self.episode_frames is None:
            self.episode_frames = int(self.episode_duration / self.frame_dt)
        self.sim_steps = self.episode_frames * self.sim_substeps
        self.sim_dt = self.frame_dt / max(1, self.sim_substeps)
        self.sim_step = 0
        self.sim_time = 0.0

        self.controls = []
        self.num_controls = self.sim_steps if self.requires_grad else 1
        for _ in range(self.num_controls):
            control = self.model.control()
            self.customize_control(control)
            self.controls.append(control)

        if self.named_control_gains is not None:
            # convert named control gains to control_gains
            self.control_gains, self.controllable_dofs, self.control_limits = [], [], []
            joint_qd_start_np = self.model.joint_qd_start.numpy()
            for joint_name, gain in self.named_control_gains.items():
                self.control_gains.append(gain * self.control_gain_scale)
                joint_idx = self.model.joint_key.index(joint_name)
                self.controllable_dofs.append(joint_qd_start_np[joint_idx].item())
                self.control_limits.append((-1.0, 1.0))

        assert len(self.controllable_dofs) == len(self.control_gains)
        assert len(self.controllable_dofs) == len(self.control_limits)

        self.controllable_dofs_wp = wp.array(self.controllable_dofs, dtype=int, device=self.device)
        self.control_gains_wp = wp.array(
            np.array(self.control_gains) * self.control_gain_scale, 
            dtype=float, 
            device=self.device
        )
        self.control_limits_wp = wp.array(self.control_limits, dtype=float, device=self.device)

        self.contacts = None
        if self.requires_grad:
            self.states = []
            for _ in range(self.sim_steps + 1):
                state = self.model.state()
                self.customize_state(state)
                self.states.append(state)
            self.simulate = self.simulate_grad
        else:
            # set up current and next state to be used by the integrator
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.customize_state(self.state_0)
            self.customize_state(self.state_1)
            self.simulate = self.simulate_nograd

            newton.eval_fk(
                self.model, 
                self.model.joint_q, 
                self.model.joint_qd, 
                self.state_0
            )
            if self.solver_type != SolverType.MUJOCO:
                self.contacts = self.model.collide(self.state_0)

        if self.use_graph_capture:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        self.viewer = None
        if self.profile:
            self.render_mode = RenderMode.NONE

        self._kinetic_energy = None
        self._potential_energy = None

        if setup_viewer:
            self.setup_viewer()
        
        self.extras = {}

    def setup_viewer(self):
        if self.render_mode == RenderMode.OPENGL:
            self.viewer = newton.viewer.ViewerGL(**self.opengl_render_settings)
        elif self.render_mode == RenderMode.USD:
            filename = os.path.join(
                os.path.dirname(__file__), "..", "outputs", self.sim_name + ".usd"
            )
            self.viewer = newton.viewer.ViewerUSD(
                filename,
                **self.usd_render_settings,
            )
        else:
            self.viewer = newton.viewer.ViewerNull()
        self.viewer.set_model(self.model)

    @property
    def uses_generalized_coordinates(self):
        # whether the model uses generalized or maximal coordinates (joint q/qd vs body q/qd) in the state
        return (
            self.solver_type == SolverType.FEATHERSTONE or 
            self.solver_type == SolverType.NEURAL or
            self.solver_type == SolverType.MUJOCO
        )

    def create_articulation(self, builder: newton.ModelBuilder):
        raise NotImplementedError

    def setup(self, builder):
        pass

    def before_simulate(self):
        pass

    def after_simulate(self):
        pass

    def before_step(
        self,
        state: newton.State,
        next_state: newton.State,
        control: newton.Control,
        eval_collisions: bool = True,
    ):
        pass

    def after_step(
        self,
        state: newton.State,
        next_state: newton.State,
        control: newton.Control,
        eval_collisions: bool = True,
    ):
        pass

    def before_simulate(self):
        pass

    def after_simulate(self):
        pass

    def custom_render(self, render_state, viewer):
        pass

    @property
    def state(self) -> newton.State:
        """
        Shortcut to the current state
        """
        if self.requires_grad:
            return self.states[self.sim_step]
        return self.state_0

    @property
    def control(self):
        return self.controls[
            min(len(self.controls) - 1, max(0, self.sim_step % self.sim_steps))
        ]

    @property
    def control_input(self):
        # points to the actuation input of the control
        return self.control.joint_f
    
    @property
    def joint_f(self):
        return self.control.joint_f
    
    @property
    def joint_f_dim(self):
        return len(self.control_input) // self.num_envs

    @property
    def control_dim(self):
        return len(self.controllable_dofs)

    @property
    def observation_dim(self):
        # default observation consists of generalized joint positions and velocities
        return self.dof_q_per_env + self.dof_qd_per_env

    def assign_control(
        self,
        actions: wp.array,
        control: newton.Control,
        state: newton.State,
    ):
        assert actions.ndim == 2
        assert actions.shape[0] == self.num_envs
        control_full_dim = len(self.control_input) // self.num_envs
        wp.launch(
            assign_controls,
            dim=self.num_envs,
            inputs=[
                actions,
                self.control_gains_wp,
                self.control_limits_wp,
                self.controllable_dofs_wp,
                control_full_dim,
            ],
            outputs=[
                control.joint_f,
            ],
            device=self.device,
        )

    def customize_state(self, state: newton.State):
        pass

    def customize_control(self, control: newton.Control):
        pass

    def customize_model(self, model):
        pass

    def compute_cost_termination(
        self,
        state: newton.State,
        control: newton.Control,
        step: int,
        max_episode_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        pass

    def get_extras(
        self,
        extras: dict
    ):
        for k, v in self.extras.items():
            if isinstance(v, dict):
                if k not in extras:
                    extras[k] = {}
                for k2, v2 in v.items():
                    extras[k][k2] = v2
            else:
                extras[k] = v

    def step(
        self,
        state: newton.State,
        next_state: newton.State,
        control: newton.Control,
        contacts: newton.Contacts = None,
        eval_collisions: bool = True,
    ):
        self.extras = {}
        state.clear_forces()
        self.before_step(state=state, next_state=next_state, control=control, eval_collisions=eval_collisions)
        
        if contacts is None:
            contacts = self.contacts
            
        if self.custom_dynamics is not None:
            self.custom_dynamics(self.model, state, next_state, control, self.sim_dt)
        else:
            if eval_collisions and self.solver_type != SolverType.MUJOCO:
                with wp.ScopedTimer(
                    "collision_handling", color="orange", active=self.enable_timers
                ):
                    self.contacts = self.model.collide(state)
                    contacts = self.contacts

            with wp.ScopedTimer("simulation", color="red", active=self.enable_timers):
                self.solver.step(
                    state_in=state, 
                    state_out=next_state, 
                    control=control,
                    contacts=contacts,
                    dt=self.sim_dt, 
                )
        self.after_step(state, next_state, control, eval_collisions=eval_collisions)
        self.sim_time += self.sim_dt
        self.sim_step += 1

    def simulate_nograd(self):
        self.before_simulate()

        for i in range(self.sim_substeps):
            if self.handle_collisions_once_per_step and i > 0:
                eval_collisions = False
            else:
                eval_collisions = True
            self.step(
                self.state_0,
                self.state_1,
                self.control,
                eval_collisions=eval_collisions,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
 
        self.after_simulate()

    def update(self):
        with wp.ScopedTimer("simulate", active=self.enable_timers):
            if self.use_graph_capture:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        return self.state_0

    def render(self, state=None):
        if self.viewer is not None:
            with wp.ScopedTimer("render", color="yellow", active=self.enable_timers):
                self.viewer.begin_frame(self.sim_time)
                if self.requires_grad:
                    # ensure we do not render beyond the last state
                    render_state = (
                        state or self.states[min(self.sim_steps, self.sim_step)]
                    )
                else:
                    render_state = state or self.state
                if self.uses_generalized_coordinates:
                    newton.eval_fk(
                        model=self.model,
                        joint_q=render_state.joint_q,
                        joint_qd=render_state.joint_qd,
                        state=render_state,
                        mask=None
                    )
                with wp.ScopedTimer(
                    "custom_render", color="orange", active=self.enable_timers
                ):
                    self.custom_render(render_state, viewer=self.viewer)
                self.viewer.log_state(render_state)
                self.viewer.end_frame()

    def reset(self):
        if self.render_mode != RenderMode.USD:
            self.sim_time = 0.0
            self.sim_step = 0

        if self.eval_fk:
            newton.eval_fk(
                model=self.model,
                joint_q=self.model.joint_q,
                joint_qd=self.model.joint_qd,
                state=self.state,
                mask=None
            )
            self.model.body_q.assign(self.state.body_q)
            self.model.body_qd.assign(self.state.body_qd)

        if self.model.particle_count > 1:
            self.model.particle_grid.build(
                self.state.particle_q,
                self.model.particle_max_radius * 2.0,
            )

        self.reset_envs()

    def reset_envs(self, env_ids: wp.array = None, state=None):
        """Reset environments where env_ids buffer indicates True. Resets all envs if env_ids is None."""
        if state is None:
            state = self.state
        if self.uses_generalized_coordinates:
            wp.launch(
                reset_generalized_coords,
                dim=self.num_envs,
                inputs=[
                    env_ids,
                    self.model.joint_q,
                    self.model.joint_qd,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                ],
                outputs=[
                    state.joint_q,
                    state.joint_qd,
                ],
                device=self.device,
                record_tape=False,
            )
        else:
            wp.launch(
                reset_maximal_coords,
                dim=self.num_envs,
                inputs=[
                    env_ids,
                    self.model.body_q,
                    self.model.body_qd,
                    self.bodies_per_env,
                ],
                outputs=[
                    state.body_q,
                    state.body_qd,
                ],
                device=self.device,
                record_tape=False,
            )

    def after_reset(self):
        pass

    def close(self):
        pass