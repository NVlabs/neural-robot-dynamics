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

import json
import math
import sys, os
import csv
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import time
import torch
from pathlib import Path
import shutil
import cv2
from typing import Optional

import warp as wp

from envs.warp_sim_envs import RenderMode
from envs.warp_sim_envs.environment import IntegratorType

from integrators.integrator_neural import NeuralIntegrator
from integrators.integrator_neural_stateful import StatefulNeuralIntegrator
from integrators.integrator_neural_transformer import TransformerNeuralIntegrator
from integrators.integrator_neural_rnn import RNNNeuralIntegrator

from utils import warp_utils
from utils.python_utils import print_info, print_ok, print_warning
from utils.env_utils import create_abstract_contact_env

class NeuralEnvironment():
    """
        Simulation environment wrapper that uses Neural Robot Dynamics Integrator.
    """
    def __init__(
        self,
        # warp environment arguments
        env_name,
        num_envs,
        warp_env_cfg = None,
        # neural integrator arguments
        neural_integrator_cfg = None,
        neural_model = None,
        # neural environment arguments
        default_env_mode = 'neural',
        device = 'cuda:0',
        render = False
    ):

        # Handle dict arguments
        if neural_integrator_cfg is None:
            neural_integrator_cfg = {}

        if warp_env_cfg is None:
            warp_env_cfg = {}

        # create abstract contact environment
        print_info(f'[NeuralEnvironment] Creating abstract contact environment: {env_name}.')
        self.env = create_abstract_contact_env(
                        env_name = env_name, 
                        num_envs = num_envs, 
                        requires_grad = False, 
                        device = device,
                        render = render, 
                        **warp_env_cfg
                    )
        self.integrator_gt = self.env.integrator
        self.sim_substeps_gt = self.env.sim_substeps
        self.integrator_type_gt = self.env.integrator_type


        neural_integrator_type = neural_integrator_cfg.get('name', 'NeuralIntegrator')
        self.sim_substeps_neural = 1
        if neural_integrator_type == 'NeuralIntegrator':
            self.integrator_neural = NeuralIntegrator(
                    model = self.env.model,
                    neural_model = neural_model,
                    **neural_integrator_cfg
                )
        elif neural_integrator_type == 'StatefulNeuralIntegrator':
            self.integrator_neural = StatefulNeuralIntegrator(
                model = self.env.model,
                neural_model = neural_model,
                **neural_integrator_cfg
            )
        elif neural_integrator_type == 'TransformerNeuralIntegrator':
            self.integrator_neural = TransformerNeuralIntegrator(
                model = self.env.model,
                neural_model = neural_model,
                **neural_integrator_cfg
            )
        elif neural_integrator_type == 'RNNNeuralIntegrator':
            self.integrator_neural = RNNNeuralIntegrator(
                model = self.env.model,
                neural_model = neural_model,
                **neural_integrator_cfg
            )
        else:
            raise NotImplementedError
        
        if neural_model is not None:
            print_info('[NeuralEnvironment] Created a Neural Integrator.')
        else:
            print_warning('[NeuralEnvironment] Created a DUMMY Neural Integrator.')

        # default env mode
        assert default_env_mode in ['ground-truth', 'neural']
        self.default_env_mode = default_env_mode
        self.set_env_mode(default_env_mode)

        # states in generalized coordinates
        self.states = torch.zeros(
            (self.num_envs, self.state_dim), 
            device = self.torch_device
        )
        self.joint_acts = torch.zeros(
            (self.num_envs, self.joint_act_dim), 
            device = self.torch_device
        )

        # root body q (used for dataset generation)
        self.root_body_q = wp.to_torch(
            self.sim_states.body_q
        )[0::self.bodies_per_env, :].view(self.num_envs, 7)

        # variables to be used by rlgames wrapper
        self.use_graph_capture = False
        self.render_mode = RenderMode.NONE

        # logging for debug
        self.visited_state_min = torch.full(
            (self.state_dim,), 
            torch.inf, 
            device = self.torch_device
        )
        self.visited_state_max = torch.full(
            (self.state_dim,), 
            -torch.inf, 
            device = self.torch_device
        )
        self._trace_step_env = None

        # video writer
        self.export_video = False
        self.video_export_filename = None
        self.video_tmp_folder = None
        self.video_frame_cnt = 0
        self._rollout_log_enabled = False
        self._rollout_log_env = 0
        self._rollout_log_step = 0
        self._rollout_log_path = None
        self._rollout_log_file = None
        self._rollout_log_writer = None

    """ Expose functions in warp env """
    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def dof_q_per_env(self):
        return self.env.dof_q_per_env
    
    @property
    def dof_qd_per_env(self):
        return self.env.dof_qd_per_env
    
    @property
    def state_dim(self):
        return self.env.dof_q_per_env + self.env.dof_qd_per_env
    
    @property
    def bodies_per_env(self):
        return self.env.bodies_per_env

    @property
    def joint_limit_lower(self):
        return self.env.model.joint_limit_lower

    @property
    def joint_limit_upper(self):
        return self.env.model.joint_limit_upper
        
    @property
    def joint_act_dim(self):
        return self.env.joint_act_dim
    
    @property
    def action_dim(self):
        return self.env.control_dim

    @property
    def action_limits(self):
        return self.env.control_limits

    @property
    def control_limits(self):
        return self.action_limits
    
    @property
    def observation_dim(self):
        return self.env.observation_dim

    @property
    def joint_types(self):
        return self.integrator_neural.joint_types
    
    @property
    def device(self):
        return self.env.device
    
    @property
    def torch_device(self):
        return wp.device_to_torch(self.env.device)

    @property
    def robot_name(self):
        return self.env.robot_name

    # properties for abstract contact info
    @property
    def abstract_contacts(self):
        return self.env.abstract_contacts

    @property
    def sim_states(self):
        return self.env.state

    # joint_control is the applied torque for all joints
    @property
    def joint_control(self):
        return self.env.control
    
    @property
    def controllable_dofs(self):
        return self.env.controllable_dofs
    
    @property
    def control_gains(self):
        return self.env.control_gains
    
    @property
    def model(self):
        return self.env.model

    @property
    def eval_collisions(self):
        return self.env.eval_collisions
    
    @property
    def num_contacts_per_env(self):
        return self.env.abstract_contacts.num_contacts_per_env
    
    @property
    def frame_dt(self):
        return self.env.frame_dt
    
    def setup_renderer(self):
        self.env.setup_renderer()

    def compute_observations(
        self,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        self.env.compute_observations(
            self.sim_states, 
            self.joint_control, 
            observations, 
            step, 
            horizon_length
        )

    def compute_cost_termination(
        self,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        self.env.compute_cost_termination(
            self.sim_states, 
            self.joint_control, 
            step, 
            traj_length, 
            cost, 
            terminated
        )

    def get_extras(
        self,
        extras: dict
    ):
        self.env.get_extras(extras)

    def close(self):
        self.env.close()

    """ Expose functions in neural integrator. """
    def init_rnn(self, batch_size):
        self.integrator_neural.init_rnn(batch_size)

    def wrap2PI(self, states):
        self.integrator_neural.wrap2PI(states)

    """ Functions of Neural Environment """
    def set_neural_model(self, neural_model):
        self.integrator_neural.set_neural_model(neural_model)

    def set_env_mode(self, env_mode):
        self.env_mode = env_mode
        if self.env_mode == 'ground-truth':
            self.env.integrator = self.integrator_gt
            self.env.sim_substeps = self.sim_substeps_gt
            self.env.sim_dt = self.env.frame_dt / self.env.sim_substeps
            self.env.integrator_type = self.integrator_type_gt
        elif self.env_mode  == 'neural':

            self.env.integrator = self.integrator_neural
            self.env.sim_substeps = self.sim_substeps_neural
            self.env.sim_dt = self.env.frame_dt / self.env.sim_substeps
            self.env.integrator_type = IntegratorType.NEURAL
        else:
            raise NotImplementedError

    def set_eval_collisions(self, eval_collisions):
        self.env.set_eval_collisions(eval_collisions)

    def enable_one_step_trace(self, env_id=0):
        """Print one state -> control -> next-state transition."""
        if self.robot_name not in ("AnyMAL", "Ant"):
            raise ValueError("One-step tracing currently supports ANYmal and ANT.")
        if not 0 <= env_id < self.num_envs:
            raise ValueError(f"env_id must be in [0, {self.num_envs - 1}]")
        self._trace_step_env = env_id
        self.integrator_neural.print_model_io_env = env_id

    @staticmethod
    def _quat_conjugate(quat):
        result = quat.clone()
        result[:3] = -result[:3]
        return result

    @staticmethod
    def _quat_multiply(lhs, rhs):
        lx, ly, lz, lw = lhs
        rx, ry, rz, rw = rhs
        return torch.stack(
            (
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            )
        )

    @classmethod
    def _quat_rotate(cls, quat, vector):
        vector_quat = torch.cat(
            (vector, torch.zeros(1, dtype=vector.dtype, device=vector.device))
        )
        return cls._quat_multiply(
            cls._quat_multiply(quat, vector_quat),
            cls._quat_conjugate(quat),
        )[:3]

    def _trace_anymal_transition(self, state, actions, joint_acts, next_state):
        """Format the quantities used by ANYmal's policy and dynamics."""
        env_id = self._trace_step_env
        q_dim = self.dof_q_per_env
        dt = self.frame_dt

        q = state[:q_dim]
        qd = state[q_dim:]
        next_q = next_state[:q_dim]
        next_qd = next_state[q_dim:]

        heading_yaw = float(self.env.heading_yaws[env_id])
        half_yaw = 0.5 * heading_yaw
        heading_quat = q.new_tensor(
            [0.0, math.sin(half_yaw), 0.0, math.cos(half_yaw)]
        )
        heading_inv = self._quat_conjugate(heading_quat)

        def physical_velocity(root_q, root_qd):
            angular_world = root_qd[:3]
            linear_world = root_qd[3:6] - torch.cross(
                root_q[:3], angular_world, dim=0
            )
            return angular_world, linear_world

        angular, linear = physical_velocity(q, qd)
        next_angular, next_linear = physical_velocity(next_q, next_qd)
        local_angular = self._quat_rotate(heading_inv, angular)
        local_linear = self._quat_rotate(heading_inv, linear)

        if self.env.task == "forward":
            target_linear = q.new_tensor([1.0, 0.0, 0.0])
        elif self.env.task == "side":
            target_linear = q.new_tensor([0.0, 0.0, 1.0])
        else:
            target_linear = q.new_zeros(3)
        target_yaw_rate = q.new_tensor(0.0)

        trace = {
            "environment": env_id,
            "mode": self.env_mode,
            "dt_seconds": dt,
            "target": {
                "meaning": "velocity target in configured heading frame",
                "heading_yaw_degrees": math.degrees(heading_yaw),
                "linear_velocity": target_linear.tolist(),
                "yaw_rate": target_yaw_rate.item(),
            },
            "current_pose_world": {
                "position": q[:3].tolist(),
                "quaternion_xyzw": q[3:7].tolist(),
            },
            "current_velocity_heading_frame": {
                "linear": local_linear.tolist(),
                "angular": local_angular.tolist(),
            },
            "tracking_error_target_minus_current": {
                "linear_velocity": (target_linear - local_linear).tolist(),
                "yaw_rate": (target_yaw_rate - local_angular[1]).item(),
            },
            "control_output": {
                "policy_action": actions.tolist(),
                "applied_joint_torque": joint_acts.tolist(),
            },
            "acceleration_world_finite_difference": {
                "linear": ((next_linear - linear) / dt).tolist(),
                "angular": ((next_angular - angular) / dt).tolist(),
                "joint": ((next_qd[6:] - qd[6:]) / dt).tolist(),
            },
            "next_state": {
                "position_world": next_q[:3].tolist(),
                "quaternion_xyzw_world": next_q[3:7].tolist(),
                "joint_positions": next_q[7:].tolist(),
                "generalized_velocity": next_qd.tolist(),
            },
        }
        if hasattr(self.env, "target_joint_q"):
            trace["control_output"]["target_joint_position"] = (
                wp.to_torch(self.env.target_joint_q)
                .view(self.num_envs, -1)[env_id]
                .detach()
                .cpu()
                .tolist()
            )
        print("\n[ONE-STEP TRACE]")
        print(json.dumps(trace, indent=2))
        print("[END ONE-STEP TRACE]\n")
        self._trace_step_env = None

    def _trace_ant_transition(self, state, actions, joint_acts, next_state):
        """Format the quantities used by ANT's policy and dynamics."""
        env_id = self._trace_step_env
        q_dim = self.dof_q_per_env
        dt = self.frame_dt

        q = state[:q_dim]
        qd = state[q_dim:]
        next_q = next_state[:q_dim]
        next_qd = next_state[q_dim:]

        def physical_velocity(root_q, root_qd):
            angular_world = root_qd[:3]
            linear_world = root_qd[3:6] - torch.cross(
                root_q[:3], angular_world, dim=0
            )
            return angular_world, linear_world

        angular, linear = physical_velocity(q, qd)
        next_angular, next_linear = physical_velocity(next_q, next_qd)
        up = self._quat_rotate(q[3:7], q.new_tensor([0.0, 0.0, 1.0]))
        heading = self._quat_rotate(q[3:7], q.new_tensor([1.0, 0.0, 0.0]))

        if self.env.task == "run":
            target = {
                "meaning": "maximize world +X velocity while upright and facing +X",
                "finite_setpoint": False,
                "direction_world": [1.0, 0.0, 0.0],
            }
            tracking_error = {
                "meaning": "no finite velocity target; reward uses current forward speed",
                "forward_velocity": linear[0].item(),
            }
        elif self.env.task == "spin":
            target = {
                "meaning": "maximize positive world-Y angular velocity while upright",
                "finite_setpoint": False,
                "axis_world": [0.0, 1.0, 0.0],
            }
            tracking_error = {
                "meaning": "no finite angular-velocity target",
                "spin_rate": angular[1].item(),
            }
        elif self.env.task == "spin_track":
            target_angular = q.new_tensor([0.0, 5.0, 0.0])
            target = {
                "meaning": "track world-Y angular velocity while upright",
                "finite_setpoint": True,
                "angular_velocity_world": target_angular.tolist(),
            }
            tracking_error = {
                "angular_velocity_target_minus_current": (
                    target_angular - angular
                ).tolist(),
            }
        else:
            target = {"meaning": f"unknown ANT task: {self.env.task}"}
            tracking_error = None

        trace = {
            "environment": env_id,
            "robot": "Ant",
            "task": self.env.task,
            "mode": self.env_mode,
            "dt_seconds": dt,
            "target": target,
            "current_pose_world": {
                "position": q[:3].tolist(),
                "quaternion_xyzw": q[3:7].tolist(),
                "up_alignment_world_y": up[1].item(),
                "heading_alignment_world_x": heading[0].item(),
            },
            "current_velocity_world": {
                "linear": linear.tolist(),
                "angular": angular.tolist(),
            },
            "tracking_error": tracking_error,
            "control_output": {
                "meaning": "torque = clamp(policy_action, limits) * control_gain",
                "policy_action": actions.tolist(),
                "control_gain": self.env.control_gains.tolist(),
                "applied_joint_torque": joint_acts.tolist(),
            },
            "acceleration_world_finite_difference": {
                "linear": ((next_linear - linear) / dt).tolist(),
                "angular": ((next_angular - angular) / dt).tolist(),
                "joint": ((next_qd[6:] - qd[6:]) / dt).tolist(),
            },
            "next_state": {
                "position_world": next_q[:3].tolist(),
                "quaternion_xyzw_world": next_q[3:7].tolist(),
                "joint_positions": next_q[7:].tolist(),
                "generalized_velocity": next_qd.tolist(),
            },
        }
        print("\n[ONE-STEP TRACE]")
        print(json.dumps(trace, indent=2))
        print("[END ONE-STEP TRACE]\n")
        self._trace_step_env = None

    def _trace_transition(self, state, actions, joint_acts, next_state):
        if self.robot_name == "AnyMAL":
            self._trace_anymal_transition(
                state, actions, joint_acts, next_state
            )
        elif self.robot_name == "Ant":
            self._trace_ant_transition(
                state, actions, joint_acts, next_state
            )

    @staticmethod
    def _quat_to_yaw_y_up(quat):
        x, y, z, w = quat
        sin_yaw = 2.0 * (w * y + x * z)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(sin_yaw, cos_yaw)

    def enable_rollout_logging(self, path, env_id=0):
        if not 0 <= env_id < self.num_envs:
            raise ValueError(f"env_id must be in [0, {self.num_envs - 1}]")
        output_path = Path(path)
        if output_path.suffix.lower() != ".csv":
            raise ValueError("Rollout streaming currently supports only CSV output.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._rollout_log_file is not None:
            self.disable_rollout_logging()

        self._rollout_log_file = output_path.open("w", newline="", encoding="utf-8")
        self._rollout_log_writer = csv.DictWriter(
            self._rollout_log_file,
            fieldnames=[
                "step",
                "env_id",
                "x",
                "y",
                "z",
                "body_yaw_world_rad",
                "heading_yaw_rad",
                "current_waypoint_id",
                "target_x",
                "target_y",
                "target_z",
                "quat_x",
                "quat_y",
                "quat_z",
                "quat_w",
            ],
        )
        self._rollout_log_writer.writeheader()
        self._rollout_log_file.flush()
        self._rollout_log_enabled = True
        self._rollout_log_env = env_id
        self._rollout_log_step = 0
        self._rollout_log_path = output_path

    def disable_rollout_logging(self):
        self._rollout_log_enabled = False
        if self._rollout_log_file is not None:
            self._rollout_log_file.close()
            self._rollout_log_file = None
            self._rollout_log_writer = None

    def _build_rollout_record(self, env_id):
        state = self.states[env_id].detach().cpu()
        q_dim = self.dof_q_per_env
        q = state[:q_dim]
        qd = state[q_dim:]
        quat_xyzw = q[3:7].tolist()

        record = {
            "step": self._rollout_log_step,
            "env_id": env_id,
            "position_world": {
                "x": float(q[0]),
                "y": float(q[1]),
                "z": float(q[2]),
            },
            "quaternion_xyzw_world": quat_xyzw,
            "body_yaw_world_rad": self._quat_to_yaw_y_up(quat_xyzw),
            "generalized_velocity": qd.tolist(),
        }

        if hasattr(self.env, "heading_yaws"):
            record["heading_yaw_rad"] = float(self.env.heading_yaws[env_id])

        if hasattr(self.env, "current_waypoint_ids"):
            waypoint_id = int(self.env.current_waypoint_ids[env_id])
            record["current_waypoint_id"] = waypoint_id
            if getattr(self.env, "waypoints", None) is not None:
                target = self.env.waypoints[waypoint_id]
                record["target_waypoint"] = {
                    "x": float(target[0]),
                    "y": float(target[1]),
                    "z": float(target[2]),
                }

        return record

    def _append_rollout_record(self):
        if not self._rollout_log_enabled:
            return
        record = self._build_rollout_record(self._rollout_log_env)
        quat = record["quaternion_xyzw_world"]
        target = record.get("target_waypoint") or {}
        self._rollout_log_writer.writerow(
            {
                "step": record["step"],
                "env_id": record["env_id"],
                "x": record["position_world"]["x"],
                "y": record["position_world"]["y"],
                "z": record["position_world"]["z"],
                "body_yaw_world_rad": record["body_yaw_world_rad"],
                "heading_yaw_rad": record.get("heading_yaw_rad"),
                "current_waypoint_id": record.get("current_waypoint_id"),
                "target_x": target.get("x"),
                "target_y": target.get("y"),
                "target_z": target.get("z"),
                "quat_x": quat[0],
                "quat_y": quat[1],
                "quat_z": quat[2],
                "quat_w": quat[3],
            }
        )
        self._rollout_log_file.flush()
        self._rollout_log_step += 1

    def save_rollout_log(self, path):
        if self._rollout_log_path is None:
            raise ValueError("Rollout logging was not enabled.")
        if Path(path) != self._rollout_log_path:
            raise ValueError(
                f"Rollout log is streaming to {self._rollout_log_path}, not {path}."
            )
        self.disable_rollout_logging()
        print_ok(f"Saved rollout log to {path}")
        
    '''
    Update states in neural env and keep the states in warp env synchronized.
    This states are mainly used by RL or other applications.
    If argument states is not specified (None), update states by obtaining states from warp env.
    [Attention] Forward kinematics needs to be applied by the caller function.
    '''
    def _update_states(self, states: Optional[torch.Tensor] = None):
        if states is None:
            if not self.env.uses_generalized_coordinates:
                warp_utils.eval_ik(self.env.model, self.env.state)
            warp_utils.acquire_states_to_torch(self.env, self.states)
        else:
            self.states.copy_(states)
        
        self.integrator_neural.wrap2PI(self.states)
        
        if states is not None:
            # update states in warp
            warp_utils.assign_states_from_torch(self.env, self.states)
            # update the maximal coordinates in warp
            warp_utils.eval_fk(self.env.model, self.env.state)

    """
    Step forward the environment with the action defined in the environment.
    Primarily used by RL.
    """
    def step(
        self, 
        actions: torch.Tensor, 
        env_mode = None
    ) -> torch.Tensor:
        
        assert env_mode in [None, 'neural', 'ground-truth']
        assert actions.shape[0] == self.num_envs
        assert actions.shape[1] == self.action_dim
        assert actions.device == self.torch_device or \
            str(actions.device) == self.torch_device

        if env_mode is None:
            env_mode = self.default_env_mode

        trace_env = self._trace_step_env
        state_before = None
        if trace_env is not None:
            state_before = self.states[trace_env].detach().cpu().clone()

        # Update env mode
        self.set_env_mode(env_mode)
        # Convert actions to real values and copy to joint_act array in warp_env
        if self.action_dim > 0:
            self.env.assign_control(
                wp.from_torch(actions), 
                self.env.control,
                self.env.state
            )
            # store converted joint_acts 
            self.joint_acts.copy_(
                wp.to_torch(self.env.control.joint_act).view(
                    self.num_envs,
                    self.joint_act_dim
                )
            )
        
        # Step forward the environment
        self.env.update()

        # Update states
        self._update_states()
        self._append_rollout_record()

        if trace_env is not None:
            self._trace_transition(
                state_before,
                actions[trace_env].detach().cpu(),
                self.joint_acts[trace_env].detach().cpu(),
                self.states[trace_env].detach().cpu(),
            )
        
        # update debug info
        self.visited_state_min = torch.minimum(
            self.visited_state_min, 
            self.states.min(dim = 0).values
        )
        self.visited_state_max = torch.maximum(
            self.visited_state_max, 
            self.states.max(dim = 0).values
        )

        return self.states

    """
    Step forward the environment with the joint torques.
    """
    def step_with_joint_act(
        self, 
        joint_acts: torch.Tensor, 
        env_mode = None
    ) -> torch.Tensor:
        
        assert env_mode in [None, 'neural', 'ground-truth']
        assert joint_acts.shape[0] == self.num_envs
        assert joint_acts.shape[1] == self.joint_act_dim
        assert joint_acts.device == self.torch_device or \
            str(joint_acts.device) == self.torch_device

        if env_mode is None:
            env_mode = self.default_env_mode

        # Update env mode
        self.set_env_mode(env_mode)

        # Assign joint_act to warp
        if self.joint_act_dim > 0:
            self.env.joint_act.assign(wp.array(joint_acts.view(-1)))
            self.joint_acts.copy_(
                wp.to_torch(self.env.control.joint_act).view(
                    self.num_envs,
                    self.joint_act_dim
                )
            )

        # Step forward the environment
        self.env.update()

        # Update states
        self._update_states()

        return self.states

    def reset(
        self, 
        initial_states: Optional[torch.Tensor] = None
    ):
        if initial_states is not None:
            assert initial_states.shape[0] == self.num_envs
            assert initial_states.device == self.torch_device or \
                str(initial_states.device) == self.torch_device

            self._update_states(initial_states)
        else:
            self.env.reset()
            self._update_states()
        self._append_rollout_record()
        
        # special reset for neural integrator (e.g. clear states history)            
        self.integrator_neural.reset()

    def reset_envs(
        self, 
        env_ids: Optional[wp.array] = None
    ):
        """Reset environments where env_ids buffer indicates True."""
        """Resets all envs if env_ids is None."""
        self.env.reset_envs(env_ids)
        self._update_states()
        if env_ids is None or env_ids.numpy()[self._rollout_log_env]:
            self._append_rollout_record()
        # special reset for neural integrator (e.g. clear states history)  
        # TODO[Jie]: now reset for all envs together, need to be fixed.
        self.integrator_neural.reset()

    def start_video_export(self, video_export_filename):
        self.export_video = True
        self.video_export_filename = os.path.join(
            "gifs",
            video_export_filename
        )
        self.video_tmp_folder = os.path.join(
            Path(video_export_filename).parent, 
            'tmp'
        )
        os.makedirs(self.video_tmp_folder, exist_ok = False)
        self.video_frame_cnt = 0
    
    def end_video_export(self):
        self.export_video = False
        frame_rate = round(1. / self.env.frame_dt)
        images_path = os.path.join(self.video_tmp_folder, r"%d.png")
        
        if not os.path.exists(os.path.dirname(self.video_export_filename)):
            os.makedirs(os.path.dirname(self.video_export_filename), exist_ok = False)
            
        os.system("ffmpeg -i {} -vf palettegen palette.png".format(images_path))
        os.system("ffmpeg -framerate {} -i {} "
                  "-i palette.png -lavfi paletteuse {}".format(
                      frame_rate, 
                      images_path, 
                      self.video_export_filename
        ))
        
        os.remove("palette.png")
        shutil.rmtree(self.video_tmp_folder)
        print_ok("Export video to {}".format(self.video_export_filename))

        self.video_export_filename = None
        self.video_tmp_folder = None
        self.video_frame_cnt = 0
        
    def render(self):
        self.env.render()
        if self.export_video:
            img = wp.zeros(
                (self.env.renderer.screen_height, self.env.renderer.screen_width, 3), 
                dtype=wp.uint8
            )
            self.env.renderer.get_pixels(
                img, 
                split_up_tiles=False, 
                mode="rgb", 
                use_uint8=True
            )
            cv2.imwrite(
                os.path.join(
                    self.video_tmp_folder, 
                    '{}.png'.format(self.video_frame_cnt)
                ), 
                img.numpy()[:, :, ::-1]
            )    
            self.video_frame_cnt += 1
        time.sleep(self.env.frame_dt)
    
    def save_usd(self):
        self.env.renderer.save()



