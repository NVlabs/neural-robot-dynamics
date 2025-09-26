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

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
)
sys.path.append(base_dir)

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase
from newton import Control, Model, State, Contacts
from newton import JointType

from utils import warp_utils
from utils.commons import CONTACT_DEPTH_UPPER_RATIO, MIN_CONTACT_EVENT_THRESHOLD
from utils import torch_utils

import torch

from typing import Optional

@wp.kernel
def determine_angular_dofs(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    # outputs
    joint_q_end: wp.array(dtype=int),
    is_angular: wp.array(dtype=bool),
    is_continuous: wp.array(dtype=bool),
):
    joint_id = wp.tid()
    q_start = joint_q_start[joint_id]
    limit_start = joint_qd_start[joint_id] # NOTE: joint limit is in joint_dof_count shape
    type = joint_type[joint_id]
    if type == JointType.FREE or type == JointType.DISTANCE:
        joint_q_end[joint_id] = q_start + 7
        for i in range(7):
            # quaternions do not count as angular dofs
            is_angular[q_start + i] = False
    elif type == JointType.BALL:
        joint_q_end[joint_id] = q_start + 4
        for i in range(4):
            # quaternions do not count as angular dofs
            is_angular[q_start + i] = False
    elif type == JointType.PRISMATIC:
        joint_q_end[joint_id] = q_start + 1
        is_angular[q_start] = False
    elif type == JointType.REVOLUTE:
        joint_q_end[joint_id] = q_start + 1
        is_angular[q_start] = True
        lower = joint_limit_lower[limit_start]
        upper = joint_limit_upper[limit_start]
        if upper - lower > 2.0 * wp.pi:
            is_continuous[q_start] = True

class NeuralSolver(SolverBase):
    """
    An integrator that uses a neural network to predict the next state.
    This integrator only handles articulated rigid body dynamics.
    The state is represented in generalized coordinates, i.e. joint_q, joint_qd.
    """

    def __init__(
        self,
        name = 'NeuralSolver',
        model: Model = None,
        contacts: Contacts = None,
        neural_model: Optional[torch.nn.Module] = None,
        states_frame: Optional[str] = 'body',
        anchor_frame_step: Optional[str] = 'every',
        states_embedding_type: Optional[str] = None,
        prediction_type: str = "relative",
        orientation_prediction_parameterization: str = "quaternion"
    ):
        """
        Args:
            model (Model): The Warp sim model to use for simulation which stores static
                information about the system to be simulated.
            neural_model (torch.nn.Module): The neural network to use for prediction.
            states_frame (str, optional): The frame type to express the states, can be
                "world" or "body". Defaults to "world".
            anchor_frame_step (str, optional): The step to use for the anchor frame if
                using body frame as states_frame, can be "first", "last", or "every".
                Defaults to "first".
            states_embedding_type (Optional[str], optional): The type of embedding to use
                for the states, can be None ("identical") or "sinusoidal". Defaults to
                None which means no state embedding is used.
            prediction_type (str, optional): The type of prediction to use, can be
                "absolute" or "relative". Defaults to "absolute".
            orientation_prediction_parameterization (str, optional): The type of prediction
                to use for quaternion, can be "quaternion" or "exponential" or "naive".
                Defaults to "quaternion".
        """
        self.solver_name = name
        self.torch_device = wp.device_to_torch(model.device)
        self.model = model
        self.neural_model = neural_model
        if neural_model is not None:
            self.neural_model.to(self.torch_device)

        self.use_target_prediction = False # for debugging only purpose

        if model.articulation_count == 0:
            raise ValueError(
                "NeuralSolver only supports articulated rigid body dynamics, "
                "so there has to be at least one articulation in the provided Warp sim model."
            )

        # assume there is one articulation or a duplicated articulations only
        art_starts = model.articulation_start.numpy()
        q_starts = model.joint_q_start.numpy()
        qd_starts = model.joint_qd_start.numpy()
        i0 = art_starts[0]
        i1 = art_starts[1]
        self.dof_q_per_env = int(q_starts[i1] - q_starts[i0])
        self.dof_qd_per_env = int(qd_starts[i1] - qd_starts[i0])
        self.state_dim = self.dof_q_per_env + self.dof_qd_per_env
        self.num_envs = model.articulation_count
        self.num_joints_per_env = model.joint_count // self.num_envs
        self.num_bodies_per_env = model.body_count // self.num_envs
        self.joint_f_dim = self.model.joint_f.shape[0] // self.num_envs
        
        self.num_contacts_per_env = model.num_contacts_per_env
        
        # verify that all articulations in the Warp model are the same
        # (at least in terms of state dimensionality)
        for i, j in zip(art_starts[1:], q_starts[:: self.num_joints_per_env]):
            assert q_starts[i] - j == self.dof_q_per_env

        # initialize model input variables
        self.root_body_q = torch.empty(
            (self.num_envs, 7), device=self.torch_device
        )
        self.states = torch.empty(
            (self.num_envs, self.state_dim), device=self.torch_device
        )
        self.joint_f = torch.empty(
            (self.num_envs, self.joint_f_dim), device=self.torch_device
        )
        self.contacts = self.get_abstract_contacts(contacts)
        
        self.gravity_dir = torch.zeros(
            (self.num_envs, 3), device=self.torch_device
        )
        self.gravity_dir[:, self.model.up_axis] = -1.0

        self.joint_f_wp = None # shape (num_envs * joint_f_dim, )

        self._build_dof_types()

        # remove base free joint's joint_f from model's input
        if self.joint_types[0] == JointType.FREE:
            self.model_joint_f_dim = self.joint_f_dim - 6
        else:
            self.model_joint_f_dim = self.joint_f_dim

        assert states_frame in ["world", "body", "body_translation_only"]
        self.states_frame = states_frame
        if states_frame == "body" or states_frame == "body_translation_only":
            assert anchor_frame_step in ["first", "last", "every"]
            self.anchor_frame_step = anchor_frame_step

        assert states_embedding_type in [None, "identical", "sinusoidal"]
        self.states_embedding_type = states_embedding_type
        self._init_state_embedding()

        assert prediction_type in ["absolute", "relative"]
        self.prediction_type = prediction_type
        
        assert orientation_prediction_parameterization in ["quaternion", "exponential", "naive"]
        self.orientation_prediction_parameterization = orientation_prediction_parameterization
        
        self._init_prediction()
    
    def _build_dof_types(self):
        # compute information about the joint dofs
        self.joint_q_end_wp = wp.empty(
            self.num_joints_per_env,
            dtype=int,
            device=self.device
        )
        self.is_angular_dof_wp = wp.empty(
            self.dof_q_per_env, 
            dtype=bool, 
            device=self.device
        )
        self.is_continuous_dof_wp = wp.zeros(
            self.state_dim, 
            dtype=bool, 
            device=self.device
        )
        # determine which dofs are angular and continuous
        wp.launch(
            determine_angular_dofs,
            dim=self.num_joints_per_env,
            inputs=[
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
            ],
            outputs=[
                self.joint_q_end_wp,
                self.is_angular_dof_wp, 
                self.is_continuous_dof_wp],
            device=self.model.device,
        )
        
        self.joint_q_start = self.model.joint_q_start.numpy()[:self.num_joints_per_env]
        self.joint_q_end = self.joint_q_end_wp.numpy()
        self.joint_types = self.model.joint_type.numpy()[:self.num_joints_per_env]
        self.is_angular_dof = self.is_angular_dof_wp.numpy()
        self.is_continuous_dof = self.is_continuous_dof_wp.numpy()
                          
    """Compute the dimension of the input state embedding. """
    def _init_state_embedding(self):
        if self.states_embedding_type is None or self.states_embedding_type == "identical":
            self.state_embedding_dim = self.state_dim
        elif self.states_embedding_type == "sinusoidal":
            self.state_embedding_dim = self.state_dim + (self.is_angular_dof).sum().item()
        else:
            raise NotImplementedError

        self.states_embedding = torch.zeros(
            (self.num_envs, self.state_embedding_dim), device=self.torch_device
        )
    
    """Compute the dimension of the prediction output."""
    def _init_prediction(self):
        if self.prediction_type == 'absolute' or self.prediction_type == 'relative':
            num_regular_dofs, num_sperical_joints = 0, 0
            for i in range(self.num_joints_per_env):
                if self.joint_types[i] == JointType.FREE:
                    num_regular_dofs += 3
                    num_sperical_joints += 1
                elif self.joint_types[i] == JointType.BALL:
                    num_sperical_joints += 1
                else:
                    num_regular_dofs += self.joint_q_end[i] - self.joint_q_start[i]
            if self.orientation_prediction_parameterization == 'quaternion':
                self.prediction_dim = num_regular_dofs + num_sperical_joints * 4 + self.dof_qd_per_env
            elif self.orientation_prediction_parameterization == 'exponential':
                self.prediction_dim = num_regular_dofs + num_sperical_joints * 3 + self.dof_qd_per_env
            elif self.orientation_prediction_parameterization == 'naive':
                self.prediction_dim = num_regular_dofs + num_sperical_joints * 4 + self.dof_qd_per_env
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
    def set_neural_model(self, neural_model):
        self.neural_model = neural_model

    # NOTE[Jie]: need to be called when using sequence model
    def init_rnn(self, batch_size):
        if self.neural_model is not None:
            self.neural_model.init_rnn(batch_size)

    def reset(self):
        pass
    
    def before_model_forward(self):
        pass

    # TODO: re-use contacts from the inputs
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control = None,
        contacts: Contacts = None,
        dt: float = 0.01,
    ):
        assert self.neural_model is not None, (
            "Cannot simulate via neural integrator as "
            "a neural model has not been setup yet."
        )
        self._update_states(state_in, contacts, control.joint_f)

        torch_stream = wp.stream_to_torch(self.device)
        # NOTE[Jie]: need to remove no_grad if we want the differentiability
        with torch.no_grad(), torch.cuda.stream(
            torch_stream
        ): 
            self.before_model_forward()
            # get the inputs for neural model
            model_inputs = self.get_neural_model_inputs()
            
            # compute the prediction using neural model, shape (num_envs, 1, dim)
            if not self.use_target_prediction:
                prediction = self.neural_model.evaluate(model_inputs)
            else: # for debugging only purpose
                prediction = self.target_prediction
            
            # convert the prediction to next states
            cur_states = model_inputs["states"][:, -1, :]
            next_states = self.convert_prediction_to_next_states(
                cur_states, prediction.squeeze(1), dt
            )
            
            next_states_world = self.convert_states_back_to_world(
                model_inputs["root_body_q"], 
                next_states
            )
            
            # copy next states to state_out
            self.wrap2PI(next_states_world)

            self.assign_states_from_torch(state_out, next_states_world)
        
        # update maximal coordinates
        warp_utils.eval_fk(self.model, state_out)

    """
    Update the states, joint_f, and contacts in neural solver from a warp states.
    """

    def _update_states(self, warp_states: State, contacts: Contacts, joint_f):
        self.acquire_states_to_torch(warp_states, self.states)
        self.wrap2PI(self.states)
        self.root_body_q = wp.to_torch(
            warp_states.body_q
        )[0::self.num_bodies_per_env, :]
        if self.joint_f_dim > 0:
            self.joint_f = wp.to_torch(joint_f).view(
                self.num_envs, self.joint_f_dim)
        self.contacts = self.get_abstract_contacts(contacts)

    def get_contact_masks(
        self, 
        contact_depths, # (num_envs, (T), num_contacts_per_env)
        contact_thickness0, # (num_envs, (T), num_contacts_per_env)
        contact_thickness1 # (num_envs, (T), num_contacts_per_env)
    ):
        # compute the threhold to a detect contact event
        contact_event_threshold = CONTACT_DEPTH_UPPER_RATIO * (contact_thickness0 + contact_thickness1)
        contact_event_threshold = torch.where(
            contact_event_threshold < MIN_CONTACT_EVENT_THRESHOLD,
            MIN_CONTACT_EVENT_THRESHOLD,
            contact_event_threshold
        )
        
        contact_masks = (contact_depths < contact_event_threshold) # (num_envs, (T), num_contacts_per_env)
        
        return contact_masks
        
    """
    Get abstract contact representation for neural network input
    """
    def get_abstract_contacts(self, contacts: Contacts):
        contact_normals = wp.to_torch(
            contacts.rigid_contact_normal
        ).view(self.num_envs, self.num_contacts_per_env * 3).clone()
        
        contact_depths = wp.to_torch(
            contacts.rigid_contact_depth
        ).view(self.num_envs, self.num_contacts_per_env).clone()
        
        contact_thickness0 = wp.to_torch(
            contacts.rigid_contact_thickness0
        ).view(self.num_envs, self.num_contacts_per_env).clone()
        
        contact_thickness1 = wp.to_torch(
            contacts.rigid_contact_thickness1
        ).view(self.num_envs, self.num_contacts_per_env).clone()
        
        contact_points_0 = wp.to_torch(
            contacts.rigid_contact_point0
        ).view(self.num_envs, self.num_contacts_per_env * 3).clone()
        
        contact_points_1 = wp.to_torch(
            contacts.rigid_contact_point1
        ).view(self.num_envs, self.num_contacts_per_env * 3).clone()
        
        contact_masks = self.get_contact_masks(
            contact_depths,
            contact_thickness0,
            contact_thickness1
        )
        
        return {
            "contact_masks": contact_masks,
            "contact_normals": contact_normals,
            "contact_depths": contact_depths,
            "contact_thicknesses_0": contact_thickness0,
            "contact_thicknesses_1": contact_thickness1,
            "contact_points_0": contact_points_0,
            "contact_points_1": contact_points_1
        }
    
    def process_neural_model_inputs(self, model_inputs):
        # convert frame
        (
            model_inputs["states"],
            model_inputs["next_states"],
            model_inputs["contact_points_1"],
            model_inputs["contact_normals"],
            model_inputs["gravity_dir"]
        ) = self.convert_coordinate_frame(
            model_inputs["root_body_q"],
            model_inputs["states"],
            model_inputs.get("next_states", None),
            model_inputs.get("contact_points_1", None),
            model_inputs.get("contact_normals", None),
            model_inputs.get("gravity_dir", None)
        )

        # post processing
        self.wrap2PI(model_inputs["states"])
        if model_inputs["next_states"] is not None:
            self.wrap2PI(model_inputs["next_states"])
        # [TESTING] special handling for unique quaternion representation
        if self.joint_types[0] == JointType.FREE:
            flip_mask = (
                model_inputs["states"][..., 6] < 0.
            ).unsqueeze(-1).expand_as(model_inputs["states"][..., 3:7])
            model_inputs["states"][..., 3:7][flip_mask] *= -1.
        elif self.joint_types[0] == JointType.BALL:
            flip_mask = (
                model_inputs["states"][..., 3] < 0.
            ).unsqueeze(-1).expand_as(model_inputs["states"][..., 0:4])
            model_inputs["states"][..., 0:4][flip_mask] *= -1.

        if "states_embedding" in model_inputs:
            self.embed_states(
                model_inputs["states"], 
                model_inputs["states_embedding"]
            )
        else:
            model_inputs["states_embedding"] = self.embed_states(
                model_inputs["states"]
            )
        
        # apply contact mask
        if model_inputs["contact_points_1"] is not None:
            for key in model_inputs.keys():
                if key.startswith('contact_'):
                    model_inputs[key] = torch.where(
                        model_inputs['contact_masks'].unsqueeze(-1) < 1e-5,
                        0.,
                        model_inputs[key].view(
                            model_inputs[key].shape[0], # num_envs 
                            model_inputs[key].shape[1], # T
                            self.num_contacts_per_env, 
                            -1
                        )
                    ).view(model_inputs[key].shape)
        
        return model_inputs
    
    """
    Prepare the inputs for the neural model inference.
    """

    def get_neural_model_inputs(self):
        # assemble the model inputs in world frame
        model_inputs = {
            "root_body_q": self.root_body_q,
            "states": self.states,
            "states_embedding": self.states_embedding,
            "joint_f": self.joint_f[..., -self.model_joint_f_dim:],
            "gravity_dir": self.gravity_dir,
            **self.contacts
        }
        for k in model_inputs.keys():
            model_inputs[k] = model_inputs[k].unsqueeze(1) # (num_envs, T, dim)
        
        processed_model_inputs = self.process_neural_model_inputs(model_inputs)

        return processed_model_inputs
        
    """
    Fix continuous angular dofs in the states vector (in-place operation).
    """

    def wrap2PI(self, states, is_continuous_dof = None):
        if is_continuous_dof is None:
            is_continuous_dof = self.is_continuous_dof
        if not is_continuous_dof.any():
            return
        assert states.shape[-1] == is_continuous_dof.shape[0]
        wrap_delta = torch.floor(
            (states[..., is_continuous_dof] + np.pi) / (2 * np.pi)
        ) * (2 * np.pi)
        states[..., is_continuous_dof] -= wrap_delta

    def acquire_states_to_torch(self, warp_states: State, torch_states: torch.Tensor):
        wp.launch(
            warp_utils._acquire_states,
            dim=self.num_envs,
            inputs=[
                warp_states.joint_q,
                warp_states.joint_qd,
                self.dof_q_per_env,
                self.dof_qd_per_env,
            ],
            outputs=[wp.from_torch(torch_states)],
            device=self.device,
        )

    def assign_states_from_torch(self, warp_state: State, torch_states: torch.Tensor):
        wp.launch(
            warp_utils._assign_states,
            dim=self.num_envs,
            inputs=[
                wp.from_torch(torch_states),
                self.dof_q_per_env,
                self.dof_qd_per_env,
            ],
            outputs=[warp_state.joint_q, warp_state.joint_qd],
            device=self.device,
        )
        
    """
    Converts the prediction tensor to the next states tensor according to prediction_type.

    Args:
        states (torch.Tensor): The current states tensor (num_envs, state_dim).
        prediction (torch.Tensor): The prediction tensor (num_envs, pred_dim).

    Returns:
        torch.Tensor: The next states tensor (num_envs, state_dim).

    Raises:
        NotImplementedError: If the prediction type is not supported.
    """
    def convert_prediction_to_next_states(self, states, prediction, dt = None):
        next_states = torch.empty_like(states)
        
        if self.prediction_type in ["absolute", "relative"]:
            """ full state prediction: absolute or relative """
            prediction_dof_offset = 0

            # Compute position components of the next states for each joint individually
            for joint_id in range(self.num_joints_per_env):
                joint_dof_start = self.joint_q_start[joint_id]
                if self.joint_types[joint_id] == JointType.FREE:
                    # position dofs
                    prediction_dof_offset += \
                        self.convert_prediction_to_next_states_regular_dofs(
                            states[..., joint_dof_start:joint_dof_start + 3],
                            prediction[..., prediction_dof_offset:],
                            next_states[..., joint_dof_start:joint_dof_start + 3]
                        )
                    # 3d orientation dofs
                    prediction_dof_offset += \
                        self.convert_prediction_to_next_states_orientation_dofs(
                            states[..., joint_dof_start + 3:joint_dof_start + 7],
                            prediction[..., prediction_dof_offset:],
                            next_states[..., joint_dof_start + 3:joint_dof_start + 7]
                        )
                elif self.joint_types[joint_id] == JointType.BALL:
                    prediction_dof_offset += \
                        self.convert_prediction_to_next_states_orientation_dofs(
                            states[..., joint_dof_start + 3:joint_dof_start + 7],
                            prediction[..., prediction_dof_offset:],
                            next_states[..., joint_dof_start + 3:joint_dof_start + 7]
                        )
                else:
                    joint_dof_end = self.joint_q_end[joint_id]
                    prediction_dof_offset += \
                        self.convert_prediction_to_next_states_regular_dofs(
                            states[..., joint_dof_start:joint_dof_end],
                            prediction[..., prediction_dof_offset:],
                            next_states[..., joint_dof_start:joint_dof_end]
                        )
                    
            # Compute velocity components of the next states
            if self.prediction_type == "absolute":
                next_states[..., self.dof_q_per_env:].copy_(
                    prediction[..., prediction_dof_offset:]
                )
            elif self.prediction_type == "relative":
                next_states[..., self.dof_q_per_env:] = (
                    states[..., self.dof_q_per_env:] + 
                    prediction[..., prediction_dof_offset:]
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        return next_states

    """
    Converts the prediction to the next states for regular degrees-of-freedom.

    Params:
        states: The current states.
        prediction: The prediction.
        next_states: The next states.
    
    Return:
        The number of corresponding degrees-of-freedom of prediction.
    
    Assume the dofs in the states and next_states are all regular dofs.
    Assume prediction is index from zero, prediction.shape[-1] might be longer 
        than states.shape[-1], but only the first prediction_dims will be used. 
    """
    def convert_prediction_to_next_states_regular_dofs(
        self,
        states,
        prediction,
        next_states
    ):
        assert states.shape[-1] == next_states.shape[-1]
        dofs = states.shape[-1]
        if self.prediction_type == 'absolute':
            next_states.copy_(prediction[..., :dofs])
            return dofs
        elif self.prediction_type == 'relative':
            next_states.copy_(states + prediction[..., :dofs])
            return dofs
        else:
            raise NotImplementedError
    
    """
    Converts the prediction to the next states for orientation degrees-of-freedom.

    Params:
        states: The current states.
        prediction: The prediction.
        next_states: The next states.
    
    Return:
        The number of corresponding degrees-of-freedom of prediction.
    
    Assume the dofs in the states and next_states are all regular dofs.
    Assume prediction is index from zero, prediction.shape[-1] might be longer 
        than states.shape[-1], but only the first prediction_dim will be used. 
    """
    def convert_prediction_to_next_states_orientation_dofs(
        self,
        states,
        prediction,
        next_states
    ):
        assert states.shape[-1] == 4 and next_states.shape[-1] == 4

        # Parse the prediction into quaternion
        prediction_dofs = None
        if self.orientation_prediction_parameterization == 'naive':
            predicted_quaternion = prediction[..., :4]
            prediction_dofs = 4
        elif self.orientation_prediction_parameterization == 'quaternion':
            predicted_quaternion = prediction[..., :4]
            predicted_quaternion = torch_utils.normalize(predicted_quaternion)
            prediction_dofs = 4
        elif self.orientation_prediction_parameterization == 'exponential':
            predicted_quaternion = torch_utils.exponential_coord_to_quat(prediction[..., :3])
            prediction_dofs = 3
        else:
            raise NotImplementedError
        
        # Apply quaternion/delta quaternion to the states to acquire next_states
        if self.prediction_type == 'absolute':
            raw_next_quaternion = predicted_quaternion
        elif self.prediction_type == 'relative':
            if self.orientation_prediction_parameterization == 'naive':
                raw_next_quaternion = states + predicted_quaternion
            else:
                # raw_next_quaternion = torch_utils.quat_mul(states, predicted_quaternion)
                raw_next_quaternion = torch_utils.quat_mul(predicted_quaternion, states)
        else:
            raise NotImplementedError
        
        # Normalize the next_states quaternion
        next_states.copy_(torch_utils.normalize(raw_next_quaternion))

        return prediction_dofs
    
    def convert_next_states_to_prediction(
        self, 
        states, # (B, (T), dof_states)
        next_states, # (B, (T), dof_states)
        dt = None
    ):  
        prediction = torch.empty(
            (*states.shape[:-1], self.prediction_dim),
            dtype = states.dtype,
            device = self.torch_device
        )
        
        if self.prediction_type in ["absolute", "relative"]:
            prediction_dof_offset = 0
            
            # Compute position components of the prediction for each joint individually
            for joint_id in range(self.num_joints_per_env):
                joint_dof_start = self.joint_q_start[joint_id]
                if self.joint_types[joint_id] == JointType.FREE:
                    prediction_dof_offset += \
                        self.convert_next_states_to_prediction_regular_dofs(
                            states[..., joint_dof_start:joint_dof_start + 3],
                            next_states[..., joint_dof_start:joint_dof_start + 3],
                            self.is_continuous_dof[joint_dof_start:joint_dof_start + 3],
                            prediction[..., prediction_dof_offset:]
                        )
                    prediction_dof_offset += \
                        self.convert_next_states_to_prediction_orientation_dofs(
                            states[..., joint_dof_start + 3:joint_dof_start + 7],
                            next_states[..., joint_dof_start + 3:joint_dof_start + 7],
                            prediction[..., prediction_dof_offset:]
                        )
                elif self.joint_types[joint_id] == JointType.BALL:
                    prediction_dof_offset += \
                        self.convert_next_states_to_prediction_orientation_dofs(
                            states[..., joint_dof_start:joint_dof_start + 4],
                            next_states[..., joint_dof_start:joint_dof_start + 4],
                            prediction[..., prediction_dof_offset:]
                        )
                else:
                    joint_dof_end = self.joint_q_end[joint_id]
                    prediction_dof_offset += \
                        self.convert_next_states_to_prediction_regular_dofs(
                            states[..., joint_dof_start:joint_dof_end],
                            next_states[..., joint_dof_start:joint_dof_end],
                            self.is_continuous_dof[joint_dof_start:joint_dof_end],
                            prediction[..., prediction_dof_offset:]
                        )

            # Compute velocity components of the prediction
            if self.prediction_type == "absolute":
                prediction[..., prediction_dof_offset:].copy_(
                    next_states[..., self.dof_q_per_env:]
                )
            elif self.prediction_type == "relative":
                prediction[..., prediction_dof_offset:] = (
                    next_states[..., self.dof_q_per_env:] - 
                    states[..., self.dof_q_per_env:]
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        return prediction
    
    """
    Converts the next states to the prediction for regular degrees-of-freedom.

    Params:
        states: The current states.
        next_states: The next states.
        is_continuous_dof: Indicates whether the degrees-of-freedom are continuous.
        prediction: The prediction.
    
    Return:
        The number of degrees-of-freedom of converted prediction.
    
    Assume the dofs in the states and next_states are all regular dofs.
    Assume is_continuous_dof is index from zero and has the same length as states.shape[-1].
    Assume prediction is index from zero, prediction.shape[-1] might be longer 
        than states.shape[-1], but only the first prediction_dim will be used. 
    The converted prediction is saved in predction[0:prediction_dim]
    """
    def convert_next_states_to_prediction_regular_dofs(
        self, 
        states, 
        next_states, 
        is_continuous_dof,
        prediction
    ):
        assert states.shape[-1] == next_states.shape[-1]
        dofs = states.shape[-1]
        if self.prediction_type == 'absolute':
            prediction[..., :dofs].copy_(next_states)
            return dofs
        elif self.prediction_type == 'relative':
            prediction[..., :dofs].copy_(next_states - states)
            self.wrap2PI(prediction[..., :dofs], is_continuous_dof)
            return dofs
        else:
            raise NotImplementedError
    
    """
    Converts the next states to the prediction for orientation degrees-of-freedom.

    Params:
        states: The current states of shape (..., 4).
        next_states: The next states of shape (..., 4).
        prediction: The prediction.
        
    Return:
        The number of degrees-of-freedom of converted prediction.

    Assume the dofs in the states and next_states are in correct sizes.
    Assume prediction is index from zero. 
    The converted prediction is saved in predction[0:prediction_dim]
    """
    def convert_next_states_to_prediction_orientation_dofs(
        self, 
        states, 
        next_states, 
        prediction
    ):
        assert states.shape[-1] == 4 and next_states.shape[-1] == 4
        if self.prediction_type == 'absolute':  
            target_quaternion = next_states
        elif self.prediction_type == 'relative':
            if self.orientation_prediction_parameterization == 'naive':
                target_quaternion = next_states - states
            else:
                target_quaternion = torch_utils.delta_quat(
                    states, 
                    next_states, 
                    frame='world'
                )

        # [TESTING] special handling for unique quaternion representation
        flip_mask = (
            target_quaternion[..., 3] < 0.
        ).unsqueeze(-1).expand_as(target_quaternion[..., :4])
        target_quaternion[..., :4][flip_mask] *= -1.
        
        if self.orientation_prediction_parameterization == 'naive':
            prediction[..., :4].copy_(target_quaternion)
            return 4
        elif self.orientation_prediction_parameterization == 'quaternion':
            prediction[..., :4].copy_(target_quaternion)
            return 4
        elif self.orientation_prediction_parameterization == 'exponential':
            prediction[..., :3].copy_(
                torch_utils.quat_to_exponential_coord(target_quaternion)
            )
            return 3
        else:
            raise NotImplementedError

    def _convert_contacts_w2b(
        self,
        root_body_q, # (B, T, num_contacts, 7)
        contact_points_1, # (B, T, num_contacts * 3)
        contact_normals, # (B, T, num_contacts * 3)
        translation_only
    ):
        shape = contact_points_1.shape
        root_body_q = root_body_q.reshape(-1, 7)
        contact_points_1 = contact_points_1.reshape(-1, 3)
        contact_normals = contact_normals.reshape(-1, 3)

        body_frame_pos = root_body_q[:, :3]
        if translation_only:
            body_frame_quat = torch.zeros_like(root_body_q[:, 3:7])
            body_frame_quat[:, 3] = 1.
        else:
            body_frame_quat = root_body_q[:, 3:7]

        assert contact_points_1.shape[0] == root_body_q.shape[0]
        contact_points_1_body = torch_utils.transform_point_inverse(
            body_frame_pos, body_frame_quat, contact_points_1).view(*shape)
        
        assert contact_normals.shape[0] == root_body_q.shape[0]
        if translation_only:
            contact_normals_body = contact_normals.view(*shape)
        else:
            contact_normals_body = torch_utils.quat_rotate_inverse(
                body_frame_quat, contact_normals).view(*shape)        
        
        return contact_points_1_body, contact_normals_body

    """
    Convert the states from world frame to body frame defined by root_body_q. 
    Only convert the root joint states if applicable.
    """
    def _convert_states_w2b(
        self,
        root_body_q, # (B, T, 7)
        states, # (B, T, dof_states)
        translation_only
    ):
        shape = states.shape
        root_body_q = root_body_q.reshape(-1, 7)
        states = states.reshape(-1, self.state_dim)

        body_frame_pos = root_body_q[:, :3]
        if translation_only:
            body_frame_quat = torch.zeros_like(root_body_q[:, 3:7])
            body_frame_quat[:, 3] = 1.
        else:
            body_frame_quat = root_body_q[:, 3:7]

        assert states.shape[0] == root_body_q.shape[0]
        states_body = states.clone()
        if self.joint_types[0] == JointType.FREE:
            (
                states_body[:, 0:3], 
                states_body[:, 3:7], 
                states_body[:, self.dof_q_per_env:self.dof_q_per_env + 3], 
                states_body[:, self.dof_q_per_env + 3:self.dof_q_per_env + 6]
            ) = torch_utils.convert_states_w2b(
                    body_frame_pos,
                    body_frame_quat,
                    p = states[:, 0:3],
                    quat = states[:, 3:7],
                    omega = states[:, self.dof_q_per_env:self.dof_q_per_env + 3],
                    nu = states[:, self.dof_q_per_env + 3:self.dof_q_per_env + 6]
                )
            
        return states_body.view(*shape)
    
    def _convert_gravity_w2b(
        self,
        root_body_q, # (B, T, 7)
        gravity_dir, # (B, T, 3)
        translation_only
    ):
        if translation_only:
            return gravity_dir
        
        shape = gravity_dir.shape
        root_body_q = root_body_q.reshape(-1, 7)
        gravity_dir = gravity_dir.reshape(-1, 3)

        body_frame_quat = root_body_q[:, 3:7]

        assert gravity_dir.shape[0] == body_frame_quat.shape[0]
        gravity_dir_body = torch_utils.quat_rotate_inverse(
            body_frame_quat, gravity_dir).view(*shape)    
        
        return gravity_dir_body
    
    def convert_coordinate_frame(
        self, 
        root_body_q, # (B, T, 7)
        states, # (B, T, dof_states)
        next_states, # (B, T, dof_states), can be None
        contact_points_1, # (B, T, num_contacts * 3)
        contact_normals, # (B, T, num_contacts * 3)
        gravity_dir, # (B, T, 3)
    ):
        assert len(states.shape) == 3

        if self.states_frame == 'world':
            return states, next_states, contact_points_1, contact_normals, gravity_dir
        elif self.states_frame == 'body' or self.states_frame == 'body_translation_only':
            B, T = states.shape[0], states.shape[1]

            if self.anchor_frame_step == "first":
                anchor_frame_body_q = root_body_q[:, 0:1, :].expand(B, T, 7)
            elif self.anchor_frame_step == "last":
                anchor_frame_body_q = root_body_q[:, -1:, :].expand(B, T, 7)
            elif self.anchor_frame_step == "every":
                anchor_frame_body_q = root_body_q
            else:
                raise NotImplementedError

            # convert contacts
            if contact_points_1 is not None:
                contact_points_1_body, contact_normals_body = \
                    self._convert_contacts_w2b(
                        anchor_frame_body_q.view(B, T, 1, 7).expand(
                            B, T, self.num_contacts_per_env, 7
                        ), 
                        contact_points_1, 
                        contact_normals,
                        translation_only = (self.states_frame == "body_translation_only")
                    )
            else:
                contact_points_1_body = None
                contact_normals_body = None
            
            # convert states
            states_body = self._convert_states_w2b(
                anchor_frame_body_q, 
                states,
                translation_only = (self.states_frame == "body_translation_only")
            )
            if next_states is not None:
                next_states_body = self._convert_states_w2b(
                    anchor_frame_body_q, 
                    next_states,
                    translation_only = (self.states_frame == "body_translation_only")
                )
            else:
                next_states_body = None
            
            # convert gravity
            if gravity_dir is not None:
                gravity_dir_body = self._convert_gravity_w2b(
                    anchor_frame_body_q, 
                    gravity_dir,
                    translation_only = (self.states_frame == "body_translation_only")
                )
            else:
                gravity_dir_body = None

            return (
                states_body, 
                next_states_body, 
                contact_points_1_body, 
                contact_normals_body, 
                gravity_dir_body
            )
        else:
            raise NotImplementedError

    def convert_states_back_to_world(
        self,
        root_body_q, # (B, T, 7)
        states # (B, dof_states)
    ):
        if self.states_frame == "world":
            return states
        elif self.states_frame == "body" or self.states_frame == "body_translation_only":
            if self.anchor_frame_step == "first":
                anchor_step = 0
            elif self.anchor_frame_step == "last" or self.anchor_frame_step == "every":
                anchor_step = -1
            else:
                raise NotImplementedError
            
            shape = states.shape

            anchor_frame_q = root_body_q[:, anchor_step, :]

            anchor_frame_pos = anchor_frame_q[:, :3]
            if self.states_frame == "body":
                anchor_frame_quat = anchor_frame_q[:, 3:7]
            elif self.states_frame == "body_translation_only":
                anchor_frame_quat = torch.zeros_like(anchor_frame_q[:, 3:7])
                anchor_frame_quat[:, 3] = 1.

            assert states.shape[0] == anchor_frame_q.shape[0]
            states_world = states.clone()
            # only need to convert the states of the first joint in the articulation
            if self.joint_types[0] == JointType.FREE:
                (
                    states_world[:, 0:3], 
                    states_world[:, 3:7], 
                    states_world[:, self.dof_q_per_env:self.dof_q_per_env + 3], 
                    states_world[:, self.dof_q_per_env + 3:self.dof_q_per_env + 6] 
                ) = torch_utils.convert_states_b2w(
                        anchor_frame_pos,
                        anchor_frame_quat,
                        p = states[:, 0:3],
                        quat = states[:, 3:7],
                        omega = states[:, self.dof_q_per_env:self.dof_q_per_env + 3],
                        nu = states[:, self.dof_q_per_env + 3:self.dof_q_per_env + 6]
                    )
            return states_world.view(*shape)
        
    """
    Embeds the given states into a new representation based on states_embedding_type.

    Args:
        states (torch.Tensor): The input states to be embedded.
        states_embedding (torch.Tensor, optional): The tensor to store the embedded states.
            If None, a new tensor will be created.

    Returns:
        torch.Tensor: The embedded states.

    Raises:
        NotImplementedError: If the states_embedding_type is not supported.
    """
    def embed_states(self, states, states_embedding=None):
        if (
            self.states_embedding_type is None 
            or self.states_embedding_type == "identical"
        ):
            if states_embedding is not None:
                states_embedding.copy_(states)
            else:
                return states.clone()
        elif self.states_embedding_type == "sinusoidal":
            if states_embedding is None:
                states_embedding = torch.zeros(
                    (*states.shape[:-1], self.state_embedding_dim), 
                    device = states.device
                )
            idx = 0
            for dof_idx in range(len(self.is_angular_dof)):
                if not self.is_angular_dof[dof_idx]:
                    states_embedding[..., idx] = states[..., dof_idx].clone()
                    idx += 1
                else:
                    states_embedding[..., idx] = torch.sin(states[..., dof_idx])
                    states_embedding[..., idx + 1] = torch.cos(states[..., dof_idx])
                    idx += 2
            states_embedding[..., idx:] = states[..., self.dof_q_per_env :].clone()
            if states_embedding is None:
                return states_embedding
        else:
            raise NotImplementedError