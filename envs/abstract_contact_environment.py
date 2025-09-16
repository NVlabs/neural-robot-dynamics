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

import numpy as np

import warp as wp
import warp.sim
from warp.sim.collide import get_box_vertex
from envs.warp_sim_envs import Environment
from envs.abstract_contact import AbstractContact
from utils import warp_utils
from utils.time_report import TimeReport, TimeProfiler

@wp.kernel(enable_backward=False)
def generate_contact_pairs(
    geo: warp.sim.ModelShapeGeometry,
    shape_shape_collision: wp.array(dtype=bool),
    num_shapes_per_env: int,
    num_contacts_per_env: int,
    ground_shape_index: int,
    shape_body: wp.array(dtype=int),
    up_vector: wp.vec3,
    shape_X_bs: wp.array(dtype=wp.transform),
    # outputs
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=float),
    contact_thickness: wp.array(dtype=float),
):
    env_id = wp.tid()

    shape_offset = num_shapes_per_env * env_id
    contact_idx = num_contacts_per_env * env_id
    for i in range(num_shapes_per_env):
        body = shape_body[shape_offset + i]

        if body == -1:
            # static shapes are ignored, e.g. ground
            continue
        
        if shape_shape_collision[shape_offset + i] == False:
            # filter out visual meshes
            continue

        geo_type = geo.type[shape_offset + i]
        geo_scale = geo.scale[shape_offset + i]
        geo_thickness = geo.thickness[shape_offset + i]
        shape_tf = shape_X_bs[shape_offset + i]

        if geo_type == wp.sim.GEO_SPHERE:
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_get_translation(shape_tf)
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness[contact_idx] = geo_scale[0]
            contact_idx += 1
        
        if geo_type == wp.sim.GEO_CAPSULE:
            # add points at the two ends of the capsule
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, geo_scale[1], 0.0)
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness[contact_idx] = geo_scale[0]
            contact_idx += 1
            
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, -geo_scale[1], 0.0)
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness[contact_idx] = geo_scale[0]
            contact_idx += 1

        if geo_type == wp.sim.GEO_BOX:
            # add box corner points
            for j in range(8):
                p = get_box_vertex(j, geo_scale)
                contact_shape0[contact_idx] = shape_offset + i
                contact_shape1[contact_idx] = ground_shape_index
                contact_point0[contact_idx] = wp.transform_point(shape_tf, p)
                contact_point1[contact_idx] = wp.vec3(0.0)
                contact_normal[contact_idx] = up_vector
                contact_depth[contact_idx] = 1000.0
                contact_thickness[contact_idx] = geo_thickness
                contact_idx += 1
        
        # TODO: temporary fix for mesh body
        if geo_type != wp.sim.GEO_BOX and geo_type != wp.sim.GEO_CAPSULE and geo_type != wp.sim.GEO_SPHERE:
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, 0.0, 0.0)
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness[contact_idx] = 0.0
            contact_idx += 1

@wp.kernel(enable_backward=False)
def collision_detection_ground(
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geo: warp.sim.ModelShapeGeometry,
    ground_shape_index: int,
    contact_shape0: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    # outputs
    contact_shape1: wp.array(dtype=int),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=float),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3)
):
    contact_id = wp.tid()
    shape = contact_shape0[contact_id]
    ground_tf = shape_X_bs[ground_shape_index]
    body = shape_body[shape]
    point_world = wp.transform_point(body_q[body], contact_point0[contact_id])

    # contact shape1 is always ground
    contact_shape1[contact_id] = ground_shape_index
    
    # get contact normal in world frame
    ground_up_vec = wp.vec3(0., 1., 0.)
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
    thickness_body0 = geo.thickness[shape]
    thickness_ground = geo.thickness[ground_shape_index]
    contact_offset0[contact_id] = wp.transform_vector(
        T_world_to_body0, 
        -thickness_body0 * contact_normal[contact_id]
    )
    contact_offset1[contact_id] = wp.transform_vector(
        T_world_to_ground,
        thickness_ground * contact_normal[contact_id]
    )


class AbstractContactEnvironment():
    """
    Implements an environment where the contacts are represented as an abstracted representation.
    A fixed set of possible contact pairs are determined at the beginning
    of the simulation and remain fixed throughout the simulation. This is useful for
    ensuring that the number of contact points does not change.
    
    Each contact is represented by:
    - contact_shape0: a shape index from robot
    - contact_point0: the contact point in the shape0's body's frame 
                             (if body doesn't exist (e.g. ground), it's in the world frame)
    - contact_shape1: a shape from the external object
    - contact_point1: the contact point in the shape1's body's frame 
                                 (if body doesn't exist (e.g. ground), it's in the world frame)
    - contact_normal: the contact normal in world frame
    - contact_depth: the penetration depth between the two shapes
    - other contact information: contact thickness, contact offset0, contact offset1
    """
    def __init__(self, env: Environment):
        # create wrapper
        super().__setattr__('_wrapped_env', env)
        
        self.eval_collisions = True
    
        self.initialize_contacts(self.model)

        self.time_report = TimeReport(cuda_synchronize = False)
        self.time_report.add_timers(
            ["collision_detection", "dynamics"]
        )
    
    # Inherit all methods from the wrapped environment
    def __getattr__(self, name):
        return getattr(self._wrapped_env, name)
    
    # Inherit the setting function from the wrapped environment
    def __setattr__(self, name, value):
        # If the attribute is already defined on the wrapped object,
        # or if it is a part of its class, delegate the assignment.
        if hasattr(self._wrapped_env, name):
            setattr(self._wrapped_env, name, value)
        else:
            # Otherwise, set it on the wrapper instance itself.
            super().__setattr__(name, value)

    # Construct the ordered list of contact pairs
    def initialize_contacts(self, model: wp.sim.Model):
        # compute number of contact pairs per env
        # NOTE: only work for ground env for now
        num_shapes_per_env = (model.shape_count - 1) // model.num_envs
        num_contacts_per_env = 0
        geo_types = model.shape_geo.type.numpy()
        shape_body = model.shape_body.numpy()
        for i in range(num_shapes_per_env):
            # static shapes are ignored, e.g. ground
            if shape_body[i] == -1:
                continue
            # filter out visual meshes
            if model.shape_shape_collision[i] == False:
                continue
            
            geo_type = geo_types[i]
            if geo_type == wp.sim.GEO_SPHERE:
                num_contacts_per_env += 1
            elif geo_type == wp.sim.GEO_CAPSULE:
                num_contacts_per_env += 2
            elif geo_type == wp.sim.GEO_BOX:
                num_contacts_per_env += 8
            else: # TODO: temporary fix for mesh body
                num_contacts_per_env += 1
        
        self.abstract_contacts = AbstractContact(
            num_contacts_per_env = num_contacts_per_env,
            num_envs = model.num_envs,
            model = model, 
            device = warp_utils.device_to_torch(model.device)
        )

        # Generate contact points once at the beginning of the simulation
        wp.launch(
            generate_contact_pairs,
            dim=model.num_envs,
            inputs=[
                model.shape_geo,
                wp.from_numpy(np.array(model.shape_shape_collision, dtype=bool)),
                num_shapes_per_env,
                num_contacts_per_env,
                model.shape_count - 1,  # ground plane index
                model.shape_body,
                model.up_vector,
                model.shape_transform,
            ],
            outputs=[
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point0,
                model.rigid_contact_point1,
                model.rigid_contact_normal,
                model.rigid_contact_depth,
                model.rigid_contact_thickness,
            ],
            device=model.device,
        )

        model.rigid_contact_max = self.abstract_contacts.num_total_contacts
        model.rigid_contact_count = wp.array(
            [model.rigid_contact_max], 
            dtype=wp.int32, 
            device=model.device
        )

    # Change contact mode
    def set_eval_collisions(self, eval_collisions):
        self.eval_collisions = eval_collisions

    # NOTE: only implemented for ground plane for now
    def collision_detection(self, model: wp.sim.Model, state: wp.sim.State):
        # project ground contact points at every step
        if model.ground:
            wp.launch(
                collision_detection_ground,
                dim=model.rigid_contact_max,
                inputs=[
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_geo,
                    model.shape_count - 1,  # ground plane index
                    model.rigid_contact_shape0,
                    model.rigid_contact_point0,
                ],
                outputs=[
                    model.rigid_contact_shape1,
                    model.rigid_contact_point1,
                    model.rigid_contact_normal,
                    model.rigid_contact_depth,
                    model.rigid_contact_offset0,
                    model.rigid_contact_offset1
                ],
                device=model.device,
            )

    # New simulation update function using customized collision detection 
    # to fill the contact information for integrator
    def update(self):
        self.before_update()

        with TimeProfiler(self.time_report, 'collision_detection'):
            if self.eval_collisions:
                self.collision_detection(self.model, self.state_0)
        
        with TimeProfiler(self.time_report, 'dynamics'):
            for _ in range(self.sim_substeps):
                # regular substepping of the integrator
                self.step(
                    self.state_0,
                    self.state_1,
                    self.control,
                    # avoid duplicated collision detection in the simulator
                    eval_collisions=False, 
                )
                # swap the state references
                self.state_0, self.state_1 = self.state_1, self.state_0
                
            self.after_update()

    def update_grad(self):
        raise NotImplementedError(
            "AbstractContactEnvironment does not support gradients at the moment"
        )
    
    def reset_timer(self):
        self.time_report.reset_timer()