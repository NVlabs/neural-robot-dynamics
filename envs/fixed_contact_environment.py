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
import sys

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
)
sys.path.append(base_dir)

import warp as wp
import newton
from newton import GeoType, ShapeFlags
from newton import Contacts
from newton._src.geometry.kernels import get_box_vertex

from envs.newton_envs import Environment
from envs.abstract_contact import AbstractContact
from envs.newton_envs.environment import SolverType
from utils import warp_utils
from utils.time_report import TimeReport, TimeProfiler

@wp.kernel(enable_backward=False)
def generate_contact_points(
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_thickness: wp.array(dtype=float),
    shape_flags: wp.array(dtype=wp.int32),
    num_shapes_per_env: int,
    num_contacts_per_env: int,
    ground_shape_index: int,
    up_vector: wp.vec3,
    # outputs
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_thickness0: wp.array(dtype=float),
    contact_thickness1: wp.array(dtype=float),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=float),
):
    env_id = wp.tid()

    shape_offset = num_shapes_per_env * env_id
    contact_idx = num_contacts_per_env * env_id
    for i in range(num_shapes_per_env):
        body = shape_body[shape_offset + i]

        if body == -1:
            # static shapes are ignored, e.g. ground
            continue

        if shape_flags[shape_offset + i] & ShapeFlags.COLLIDE_SHAPES == 0:
            # filter out visual meshes
            continue

        geo_type = shape_type[shape_offset + i]
        geo_scale = shape_scale[shape_offset + i]
        geo_thickness = shape_thickness[shape_offset + i]
        shape_tf = shape_transform[shape_offset + i]

        if geo_type == GeoType.SPHERE:
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_get_translation(shape_tf)
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness0[contact_idx] = geo_thickness + geo_scale[0]
            contact_thickness1[contact_idx] = 0.0
            contact_idx += 1

        if geo_type == GeoType.CAPSULE:
            # add points at the two ends of the capsule
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, 0.0,geo_scale[1])
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness0[contact_idx] = geo_thickness + geo_scale[0]
            contact_thickness1[contact_idx] = 0.0
            contact_idx += 1

            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, 0.0, -geo_scale[1])
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness0[contact_idx] = geo_thickness + geo_scale[0]
            contact_thickness1[contact_idx] = 0.0
            contact_idx += 1

        if geo_type == GeoType.BOX:
            # add box corner points
            for j in range(8):
                p = get_box_vertex(j, geo_scale)
                contact_shape0[contact_idx] = shape_offset + i
                contact_shape1[contact_idx] = ground_shape_index
                contact_point0[contact_idx] = wp.transform_point(shape_tf, p)
                contact_point1[contact_idx] = wp.vec3(0.0)
                contact_normal[contact_idx] = up_vector
                contact_depth[contact_idx] = 1000.0
                contact_thickness0[contact_idx] = geo_thickness
                contact_thickness1[contact_idx] = 0.0
                contact_idx += 1

        # TODO: temporary hack for mesh/cylinder bodies
        if (
            geo_type != GeoType.BOX
            and geo_type != GeoType.CAPSULE
            and geo_type != GeoType.SPHERE
        ):
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, 0.0, 0.0)
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness0[contact_idx] = 0.0
            contact_thickness1[contact_idx] = 0.0
            contact_idx += 1

@wp.kernel(enable_backward=False)
def collision_detection_ground(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_thickness: wp.array(dtype=float),
    ground_shape_index: int,
    contact_shape0: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    # outputs
    contact_shape1: wp.array(dtype=int),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=float),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
):
    contact_id = wp.tid()
    shape = contact_shape0[contact_id]
    ground_tf = shape_transform[ground_shape_index]
    body = shape_body[shape]
    point_world = wp.transform_point(body_q[body], contact_point0[contact_id])

    # contact shape1 is always ground
    contact_shape1[contact_id] = ground_shape_index

    # get contact normal in world frame
    ground_up_vec = wp.vec3(0.0, 0.0, 1.0)
    contact_normal[contact_id] = wp.transform_vector(ground_tf, ground_up_vec)

    # transform point to ground shape frame
    T_world_to_ground = wp.transform_inverse(ground_tf)
    point_plane = wp.transform_point(T_world_to_ground, point_world)

    # get contact depth
    contact_depth[contact_id] = wp.dot(point_plane, ground_up_vec)

    # project to plane
    projected_point = point_plane - contact_depth[contact_id] * ground_up_vec

    # transform to world frame (applying the shape transform)
    contact_point1[contact_id] = wp.transform_point(ground_tf, projected_point)

    # compute contact offsets
    T_world_to_body0 = wp.transform_inverse(body_q[body])
    thickness_body0 = shape_thickness[shape]
    thickness_ground = shape_thickness[ground_shape_index]
    contact_offset0[contact_id] = wp.transform_vector(
        T_world_to_body0, -thickness_body0 * contact_normal[contact_id]
    )
    contact_offset1[contact_id] = wp.transform_vector(
        T_world_to_ground, thickness_ground * contact_normal[contact_id]
    )


class FixedContactEnvironment:
    """
    Implements an environment where the set of contact points are fixed.
    A fixed set of possible contact pair geometries are determined at the beginning
    of the simulation and remain fixed throughout the simulation. This is useful for
    ensuring that the number of contact points does not change. 
    
    It can have two modes:
    - Collision-detection mode: the contact information is filled by running a 
      collision detection at the beginning of the frame.
    - Manual mode: the contact information is filled by a manual set function.
    
    Each contact is represented by:
    - contact_shape0 (fixed): a shape from robot
    - contact_point0 (fixed): the contact point in the shape0's body's frame (if body doesn't exist (e.g. ground), it's in the world frame)
    - contact_shape1 (variable, fixed for now): a shape from the external object
    - contact_point1 (variable): the contact point in the shape1's body's frame (if body doesn't exist (e.g. ground), it's in the world frame)
    - contact_normal (variable, fixed for now): the contact normal in world frame
    - contact_depth (variable): the contact depth
    
    Note: the computed contacts are not complete to work with the XPBDIntegrator, since the
    contact point offsets need to be recompute at every step. This is not implemented here.
    """
    def __init__(self, env: Environment, use_graph_capture: bool = False):
        # create wrapper
        super().__setattr__('_wrapped_env', env)
        
        self.eval_collisions = True
    
        self.initialize_contacts(self.model)

        self.time_report = TimeReport(cuda_synchronize = False)
        self.time_report.add_timers(
            ["collision_detection", "dynamics"]
        )

        self.use_graph_capture = use_graph_capture
        if self.use_graph_capture:
            with wp.ScopedCapture() as capture:
                self.simulate_nograd()
            self.graph = capture.graph
    
    # inherit all methods from the wrapped environment
    def __getattr__(self, name):
        return getattr(self._wrapped_env, name)
    
    # inherit the setting function from the wrapped environment
    def __setattr__(self, name, value):
        # If the attribute is already defined on the wrapped object,
        # or if it is a part of its class, delegate the assignment.
        if hasattr(self._wrapped_env, name):
            setattr(self._wrapped_env, name, value)
        else:
            # Otherwise, set it on the wrapper instance itself.
            super().__setattr__(name, value)
    
    def initialize_contacts(self, model: newton.Model):
        num_shapes_per_env = (model.shape_count - 1) // model.num_envs
        self.num_contacts_per_env = 0
        geo_types = model.shape_type.numpy()
        shape_body = model.shape_body.numpy()
        shape_flags = model.shape_flags.numpy()

        for i in range(num_shapes_per_env):
            # static shapes are ignored, e.g. ground
            if shape_body[i] == -1:
                continue
            # filter out visual meshes
            if (
                shape_flags[i] & int(ShapeFlags.COLLIDE_SHAPES) == 0
            ):
                continue
            
            geo_type = geo_types[i]
            if geo_type == GeoType.SPHERE:
                self.num_contacts_per_env += 1
            elif geo_type == GeoType.CAPSULE:
                self.num_contacts_per_env += 2
            elif geo_type == GeoType.BOX:
                self.num_contacts_per_env += 8
            else: # NOTE: temporary use COM for for mesh and cylinder shapes
                self.num_contacts_per_env += 1
        
        model.num_contacts_per_env = self.num_contacts_per_env

        # create abstract_contacts (for torch access)
        self.abstract_contacts = AbstractContact(
            num_contacts_per_env = self.num_contacts_per_env,
            num_envs = model.num_envs,
            device = warp_utils.device_to_torch(model.device)
        )

        # create contacts (substitute the contacts in Newton's collide.py)
        self.contacts_neural_solver = self.create_newton_contacts(model)
        
        # Generate the fixed contact points once at the beginning of the simulation
        wp.launch(
            generate_contact_points,
            dim=model.num_envs,
            inputs=[
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_thickness,
                model.shape_flags,
                num_shapes_per_env,
                self.num_contacts_per_env,
                model.shape_count - 1,  # ground plane index
                model.up_vector,
            ],
            outputs=[
                self.contacts_neural_solver.rigid_contact_shape0,
                self.contacts_neural_solver.rigid_contact_shape1,
                self.contacts_neural_solver.rigid_contact_point0,
                self.contacts_neural_solver.rigid_contact_point1,
                self.contacts_neural_solver.rigid_contact_thickness0,
                self.contacts_neural_solver.rigid_contact_thickness1,
                self.contacts_neural_solver.rigid_contact_normal,
                self.contacts_neural_solver.rigid_contact_depth,
            ],
            device=model.device,
        )


    def create_newton_contacts(self, model: newton.Model):
        contacts = Contacts(
            rigid_contact_max = self.abstract_contacts.num_total_contacts,
            soft_contact_max = 0,
            requires_grad = False,
            device=model.device
        )

        contacts.rigid_contact_count = wp.array(
            [self.abstract_contacts.num_total_contacts], 
            dtype=wp.int32, 
            device=model.device
        )

        contacts.rigid_contact_shape0 = wp.from_torch(
            self.abstract_contacts.contact_shape0
        )
        contacts.rigid_contact_point0 = wp.from_torch(
            self.abstract_contacts.contact_point0,
            dtype = wp.vec3
        )
        contacts.rigid_contact_shape1 = wp.from_torch(
            self.abstract_contacts.contact_shape1
        )
        contacts.rigid_contact_point1 = wp.from_torch(
            self.abstract_contacts.contact_point1,
            dtype = wp.vec3
        )
        contacts.rigid_contact_normal = wp.from_torch(
            self.abstract_contacts.contact_normal,
            dtype = wp.vec3
        )
        contacts.rigid_contact_depth = wp.from_torch(
            self.abstract_contacts.contact_depth
        )
        contacts.rigid_contact_thickness0 = wp.from_torch(
            self.abstract_contacts.contact_thickness0
        )
        contacts.rigid_contact_thickness1 = wp.from_torch(
            self.abstract_contacts.contact_thickness1
        )
        contacts.rigid_contact_offset0 = wp.from_torch(
            self.abstract_contacts.contact_offset0,
            dtype = wp.vec3
        )
        contacts.rigid_contact_offset1 = wp.from_torch(
            self.abstract_contacts.contact_offset1,
            dtype = wp.vec3
        )

        return contacts

    # change contact mode
    def set_eval_collisions(self, eval_collisions):
        self.eval_collisions = eval_collisions

    # NOTE: only implemented for ground plane for now
    # results stored in self.contacts_neural_solver
    def collision_detection(self, model: newton.Model, state: newton.State):
        # project ground contact points at every step
        if model.ground:
            wp.launch(
                collision_detection_ground,
                dim=self.contacts_neural_solver.rigid_contact_max,
                inputs=[
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_thickness,
                    model.shape_count - 1,  # ground plane index
                    self.contacts_neural_solver.rigid_contact_shape0,
                    self.contacts_neural_solver.rigid_contact_point0,
                ],
                outputs=[
                    self.contacts_neural_solver.rigid_contact_shape1,
                    self.contacts_neural_solver.rigid_contact_point1,
                    self.contacts_neural_solver.rigid_contact_normal,
                    self.contacts_neural_solver.rigid_contact_depth,
                    self.contacts_neural_solver.rigid_contact_offset0,
                    self.contacts_neural_solver.rigid_contact_offset1
                ],
                device=model.device,
            )

    # New simulate function with customized collision detection and fill the contact information for integrator
    def simulate_nograd(self):
        self.before_simulate()

        with TimeProfiler(self.time_report, 'collision_detection'):
            if self.eval_collisions:
                self.collision_detection(self.model, self.state_0)
                
        with TimeProfiler(self.time_report, 'dynamics'):
            for _ in range(self.sim_substeps):
                self.step(
                    state = self.state_0,
                    next_state = self.state_1,
                    control = self.control,
                    contacts=self.contacts_neural_solver,
                    eval_collisions=False,
                )
                self.state_0, self.state_1 = self.state_1, self.state_0

        self.after_simulate()
    
    def recapture_graph(self):
        # TODO: graph capture somehow does not work for neural solver
        if self.use_graph_capture and self.solver_type != SolverType.NEURAL:
            with wp.ScopedCapture() as capture:
                self.simulate_nograd()
            self.graph = capture.graph
        
    # new update function
    def update(self):
        with wp.ScopedTimer("simulate", active=self.enable_timers):
            if self.use_graph_capture and self.solver_type != SolverType.NEURAL:
                wp.capture_launch(self.graph)
            else:
                self.simulate_nograd()
        return self.state_0
        
    def reset_timer(self):
        self.time_report.reset_timer()