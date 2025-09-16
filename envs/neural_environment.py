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

import sys, os
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

        # create neural integrator
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

        # video writer
        self.export_video = False
        self.video_export_filename = None
        self.video_tmp_folder = None
        self.video_frame_cnt = 0

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



