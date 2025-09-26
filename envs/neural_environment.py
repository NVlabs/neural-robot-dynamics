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

# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE.md for details].

import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import time
import torch
from typing import Optional

import warp as wp

from envs.newton_envs import RenderMode
from envs.newton_envs.environment import SolverType
from solvers import (
    NeuralSolver,
    StatefulNeuralSolver,
    TransformerNeuralSolver,
    RNNNeuralSolver,
)
from utils import warp_utils
from utils.python_utils import print_info, print_warning
from utils.env_utils import create_fixed_contact_env

class NeuralEnvironment():
    def __init__(
        self,
        # newton environment arguments
        env_name,
        num_envs,
        newton_env_cfg = None,
        # neural solver arguments
        neural_solver_cfg = None,
        neural_model = None,
        # neural environment arguments
        default_env_mode = 'neural',
        use_graph_capture = False,
        device = 'cuda:0',
        render = False
    ):

        # Handle dict arguments
        if neural_solver_cfg is None:
            neural_solver_cfg = {}

        if newton_env_cfg is None:
            newton_env_cfg = {}

        # create abstract contact environment
        print_info(f'[NeuralEnvironment] Creating abstract contact environment: {env_name}.')

        self.env = create_fixed_contact_env(
                        env_name = env_name, 
                        num_envs = num_envs, 
                        use_graph_capture=use_graph_capture,
                        requires_grad = False, 
                        device = device,
                        render = render, 
                        **newton_env_cfg
                    )

        self.solver_gt = self.env.solver
        self.sim_substeps_gt = self.env.sim_substeps
        self.solver_type_gt = self.env.solver_type

        # create neural solver
        neural_solver_type = neural_solver_cfg.get('name', 'NeuralSolver')
        self.sim_substeps_neural = 1
        if neural_solver_type == 'NeuralSolver':
            self.solver_neural = NeuralSolver(
                    model = self.env.model,
                    contacts = self.env.contacts_neural_solver,
                    neural_model = neural_model,
                    **neural_solver_cfg
                )
        elif neural_solver_type == 'StatefulNeuralSolver':
            self.solver_neural = StatefulNeuralSolver(
                model = self.env.model,
                contacts = self.env.contacts_neural_solver,
                neural_model = neural_model,
                **neural_solver_cfg
            )
        elif neural_solver_type == 'TransformerNeuralSolver':
            self.solver_neural = TransformerNeuralSolver(
                model = self.env.model,
                contacts = self.env.contacts_neural_solver,
                neural_model = neural_model,
                **neural_solver_cfg
            )
        elif neural_solver_type == 'RNNNeuralSolver':
            self.solver_neural = RNNNeuralSolver(
                model = self.env.model,
                contacts = self.env.contacts_neural_solver,
                neural_model = neural_model,
                **neural_solver_cfg
            )
        else:
            raise NotImplementedError
        
        if neural_model is not None:
            print_info('[NeuralEnvironment] Created a Neural Solver.')
        else:
            print_warning('[NeuralEnvironment] Created a DUMMY Neural Solver.')

        # default env mode
        assert default_env_mode in ['ground-truth', 'neural']
        self.default_env_mode = default_env_mode
        self.env_mode = default_env_mode
        self.set_env_mode(default_env_mode)

        # states in generalized coordinates
        self.states = torch.zeros(
            (self.num_envs, self.state_dim), 
            device = self.torch_device
        )
        self.joint_f = torch.zeros(
            (self.num_envs, self.joint_f_dim), 
            device = self.torch_device
        )

        # root body q (used for dataset generation)
        self.root_body_q = wp.to_torch(
            self.sim_states.body_q
        )[0::self.bodies_per_env, :].view(self.num_envs, 7)

        # variables to be used by rlgames wrapper
        self.use_graph_capture = self.env.use_graph_capture
        self.render_mode = RenderMode.NONE

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
    def joint_f_dim(self):
        return self.env.joint_f_dim
    
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
        return self.solver_neural.joint_types
    
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
        return self.env.num_contacts_per_env
    
    @property
    def frame_dt(self):
        return self.env.frame_dt
    
    def setup_viewer(self):
        self.env.setup_viewer()

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

    """ Expose functions in neural solver. """
    def init_rnn(self, batch_size):
        self.solver_neural.init_rnn(batch_size)

    def wrap2PI(self, states):
        self.solver_neural.wrap2PI(states)

    """ Functions of Neural Environment """
    def set_neural_model(self, neural_model):
        self.solver_neural.set_neural_model(neural_model)

    def set_env_mode(self, env_mode):
        if self.env_mode != env_mode:
            recapture_graph = True
        else:
            recapture_graph = False
            
        self.env_mode = env_mode
        if self.env_mode == 'ground-truth':
            self.env.solver = self.solver_gt
            self.env.sim_substeps = self.sim_substeps_gt
            self.env.sim_dt = self.env.frame_dt / self.env.sim_substeps
            self.env.solver_type = self.solver_type_gt
        elif self.env_mode  == 'neural':
            self.env.solver = self.solver_neural
            self.env.sim_substeps = self.sim_substeps_neural
            self.env.sim_dt = self.env.frame_dt / self.env.sim_substeps
            self.env.solver_type = SolverType.NEURAL
        else:
            raise NotImplementedError
        
        if recapture_graph:
            self.env.recapture_graph()

    def set_eval_collisions(self, eval_collisions):
        self.env.set_eval_collisions(eval_collisions)
        
    '''
    Update states in neural env and keep the states in warp env synchronoused.
    This states are mainly used by RL or other applications.
    If argument states is not specified (None), update states by obtaining states from warp env.
    Attension: Forward kinematics needs to be applied by the caller function.
    '''
    def _update_states(self, states: Optional[torch.Tensor] = None):
        if states is None:
            if not self.env.uses_generalized_coordinates:
                warp_utils.eval_ik(self.env.model, self.env.state)
            warp_utils.acquire_states_to_torch(self.env, self.states)
        else:
            self.states.copy_(states)
        
        self.solver_neural.wrap2PI(self.states)
        
        if states is not None:
            # update states in warp
            warp_utils.assign_states_from_torch(self.env, self.states)
            # update the maximal coordinates in warp
            warp_utils.eval_fk(self.env.model, self.env.state)

    def step(
        self, 
        actions: torch.Tensor, 
        env_mode = None
    ) -> torch.Tensor:
        
        assert env_mode in [None, 'neural', 'ground-truth', 'copy']
        assert actions.shape[0] == self.num_envs
        assert actions.shape[1] == self.action_dim
        assert actions.device == self.torch_device or \
            str(actions.device) == self.torch_device

        if env_mode is None:
            env_mode = self.default_env_mode

        # Update env mode
        self.set_env_mode(env_mode)
        # Convert actions to real values and copy to joint_f array in newton_env
        if self.action_dim > 0:
            self.env.assign_control(
                wp.from_torch(actions), 
                self.env.control,
                self.env.state
            )
            # store converted joint_f 
            self.joint_f.copy_(wp.to_torch(self.env.control.joint_f).view(
                self.num_envs,
                self.joint_f_dim
            ))
        
        # Step forward the environment
        self.env.update()

        # Update states
        self._update_states()

        return self.states

    # joint_f are the raw values
    def step_with_joint_f(
        self, 
        joint_f: torch.Tensor, 
        env_mode = None
    ) -> torch.Tensor:
        
        assert env_mode in [None, 'neural', 'ground-truth', 'copy']
        assert joint_f.shape[0] == self.num_envs
        assert joint_f.shape[1] == self.joint_f_dim
        assert joint_f.device == self.torch_device or \
            str(joint_f.device) == self.torch_device

        if env_mode is None:
            env_mode = self.default_env_mode

        # Update env mode
        self.set_env_mode(env_mode)

        # Assign joint_f to warp
        self.env.joint_f.assign(wp.array(joint_f.reshape(-1)))
        self.joint_f.copy_(wp.to_torch(self.env.control.joint_f).view(
            self.num_envs,
            self.joint_f_dim
        ))

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
        
        # special reset for neural solver (e.g. clear states history)            
        self.solver_neural.reset()

    def reset_envs(
        self, 
        env_ids: Optional[wp.array] = None
    ):
        """Reset environments where env_ids buffer indicates True."""
        """Resets all envs if env_ids is None."""
        self.env.reset_envs(env_ids)
        self._update_states()
        # special reset for neural solver (e.g. clear states history)  
        # TODO[Jie]: now reset for all envs together, need to be fixed.
        self.solver_neural.reset()
        
    def render(self):
        self.env.render()
        time.sleep(self.env.frame_dt)
