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

import warp as wp
import numpy as np
import warp.sim
from typing import Union
from scipy.spatial.transform import Rotation
from warp.sim.integrator_featherstone import dense_index, transform_twist


@wp.func
def mod(n: float, M: float):
    return ((n % M) + M) % M


@wp.func
def angle_normalize(x: float):
    return (mod(x + wp.pi, 2.0 * wp.pi)) - wp.pi


@wp.func
def kinetic_energy(
    body_v: wp.array(dtype=wp.spatial_vector),
    joint_start: int,
    joint_count: int,
    M_start: int,
    M: wp.array(dtype=float),
):
    energy = float(0.0)
    stride = joint_count * 6
    for ji in range(joint_count):
        qd = body_v[joint_start + ji]
        for i in range(6):
            for j in range(6):
                m = M[M_start + dense_index(stride, ji * 6 + i, ji * 6 + j)]
                energy += qd[j] * m * qd[i]
    return energy * 0.5


@wp.kernel
def eval_kinetic_energy(
    articulation_start: wp.array(dtype=int),
    articulation_M_start: wp.array(dtype=int),
    body_v: wp.array(dtype=wp.spatial_vector),
    M: wp.array(dtype=float),
    # outputs
    energy: wp.array(dtype=float),
):
    # one thread per-articulation
    env_id = wp.tid()

    joint_start = articulation_start[env_id]
    joint_end = articulation_start[env_id + 1]
    joint_count = joint_end - joint_start

    M_offset = articulation_M_start[env_id]

    energy[env_id] = kinetic_energy(body_v, joint_start, joint_count, M_offset, M)


@wp.kernel
def eval_potential_energy(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    bodies_per_env: int,
    # outputs
    energy: wp.array(dtype=float),
):
    # one thread per-articulation
    env_id = wp.tid()

    e = float(0.0)
    for i in range(bodies_per_env):
        bi = env_id * bodies_per_env + i
        com = wp.transform_point(body_q[bi], body_com[bi])
        mass = body_mass[bi]
        e += mass * wp.dot(gravity, com)
    energy[env_id] = e


@wp.kernel
def eval_center_of_mass(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    bodies_per_env: int,
    # outputs
    coms: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
):
    # one thread per-articulation
    env_id = wp.tid()

    total_com = wp.vec3(0.0)
    total_mass = float(0.0)
    for i in range(bodies_per_env):
        bi = env_id * bodies_per_env + i
        com = wp.transform_point(body_q[bi], body_com[bi])
        mass = body_mass[bi]
        total_com += mass * com
        total_mass += mass
    coms[env_id] = total_com / total_mass
    if masses:
        masses[env_id] = total_mass


def tetrahedralize(
    vertices, faces, stop_quality=10, max_its=50, edge_length_r=0.1, epsilon=0.01
):
    """
    Tetrahedralizes a 3D triangular surface mesh using "Fast Tetrahedral Meshing in the Wild" (fTetWild).

    This function requires that wildmeshing is installed, see
    https://wildmeshing.github.io/python/ for installation instructions.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces.
        stop_quality: The maximum AMIPS energy for stopping mesh optimization.
        max_its: The maximum number of mesh optimization iterations.
        edge_length_r: The relative target edge length as a fraction of the bounding box diagonal.
        epsilon: The relative envelope size as a fraction of the bounding box diagonal.

    Returns:
        A tuple (vertices, elements) containing the tetrahedralized mesh.
    """
    import wildmeshing as wm

    tetra = wm.Tetrahedralizer(
        stop_quality=stop_quality,
        max_its=max_its,
        edge_length_r=edge_length_r,
        epsilon=epsilon,
    )
    tetra.set_mesh(vertices, np.array(faces).reshape(-1, 3))
    tetra.tetrahedralize()
    res = tetra.get_tet_mesh()
    tet_vertices, tet_indices = list(res)[:2]
    return tet_vertices, tet_indices


def remesh_ftetwild(
    vertices, faces, stop_quality=10, max_its=50, edge_length_r=0.1, epsilon=0.01
):
    """
    Remeshes a 3D triangular surface mesh using "Fast Tetrahedral Meshing in the Wild" (fTetWild).
    This is useful for improving the quality of the mesh, and for ensuring that the mesh is
    watertight. This function first tetrahedralizes the mesh, then extracts the surface mesh.
    The resulting mesh is guaranteed to be watertight and may have a different topology than the
    input mesh.

    This function requires that wildmeshing is installed, see
    https://wildmeshing.github.io/python/ for installation instructions.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces.
        stop_quality: The maximum AMIPS energy for stopping mesh optimization.
        max_its: The maximum number of mesh optimization iterations.
        edge_length_r: The relative target edge length as a fraction of the bounding box diagonal.
        epsilon: The relative envelope size as a fraction of the bounding box diagonal.
        visualize: If True, visualize the input mesh next to the remeshed result using matplotlib.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh. Returns the original vertices and faces
        if the remeshing fails.
    """
    from collections import defaultdict

    tet_vertices, tet_indices = tetrahedralize(
        vertices, faces, stop_quality, max_its, edge_length_r, epsilon
    )

    def face_indices(tet):
        face1 = (tet[0], tet[2], tet[1])
        face2 = (tet[1], tet[2], tet[3])
        face3 = (tet[0], tet[1], tet[3])
        face4 = (tet[0], tet[3], tet[2])
        return (
            (face1, tuple(sorted(face1))),
            (face2, tuple(sorted(face2))),
            (face3, tuple(sorted(face3))),
            (face4, tuple(sorted(face4))),
        )

    # determine surface faces
    elements_per_face = defaultdict(set)
    unique_faces = {}
    for e, tet in enumerate(tet_indices):
        for face, key in face_indices(tet):
            elements_per_face[key].add(e)
            unique_faces[key] = face
    surface_faces = [
        face for key, face in unique_faces.items() if len(elements_per_face[key]) == 1
    ]

    new_vertices = np.array(tet_vertices)
    new_faces = np.array(surface_faces, dtype=np.int32)

    if len(new_vertices) == 0 or len(new_faces) == 0:
        import warnings

        warnings.warn(
            "Remeshing failed, the optimized mesh has no vertices or faces; return previous mesh."
        )
        return vertices, faces

    return new_vertices, new_faces


def remesh_alphashape(vertices, faces=None, alpha=3.0):
    """
    Remeshes a 3D triangular surface mesh using the alpha shape algorithm.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces (not needed).
        alpha: The alpha shape parameter.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh.
    """
    import alphashape

    alpha_shape = alphashape.alphashape(vertices, alpha)
    return np.array(alpha_shape.vertices), np.array(alpha_shape.faces, dtype=np.int32)


def remesh(vertices, faces, method="ftetwild", visualize=False, **remeshing_kwargs):
    """
    Remeshes a 3D triangular surface mesh using the specified method.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces.
        method: The remeshing method to use. One of "ftetwild" or "alphashape".
        visualize: Whether to render the input and output meshes using matplotlib.
        **remeshing_kwargs: Additional keyword arguments passed to the remeshing function.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh.
    """
    if method == "ftetwild":
        new_vertices, new_faces = remesh_ftetwild(vertices, faces, **remeshing_kwargs)
    elif method == "alphashape":
        new_vertices, new_faces = remesh_alphashape(vertices, faces, **remeshing_kwargs)
    # TODO add poisson sampling (trimesh has implementation at https://trimsh.org/trimesh.sample.html)
    else:
        raise ValueError(f"Unknown remeshing method: {method}")

    if visualize:
        # side-by-side visualization of the input and output meshes
        wp.sim.visualize_meshes(
            [(vertices, faces), (new_vertices, new_faces)],
            titles=["Original", "Remeshed"],
        )
    return new_vertices, new_faces


def remesh(mesh: wp.sim.Mesh, recompute_inertia=True, **remeshing_kwargs):
    mesh.vertices, mesh.indices = wp.sim.remesh(
        mesh.vertices, mesh.indices.reshape(-1, 3), **remeshing_kwargs
    )
    mesh.indices = mesh.indices.flatten()
    if recompute_inertia:
        mesh.mass, mesh.com, mesh.I, _ = wp.sim.compute_mesh_inertia(
            1.0, mesh.vertices, mesh.indices, is_solid=mesh.is_solid
        )


def plot_graph(vertices, edges, edge_labels=[]):
    """
    Plots a graph using matplotlib.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        edges: A numpy array of shape (M, 2) containing the vertex indices of the edges.
        edge_labels: A list of edge labels.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    name_to_index = {}
    for i, name in enumerate(vertices):
        G.add_node(i)
        name_to_index[name] = i
    g_edge_labels = {}
    for i, (a, b) in enumerate(edges):
        a = a if isinstance(a, int) else name_to_index[a]
        b = b if isinstance(b, int) else name_to_index[b]
        label = None
        if i < len(edge_labels):
            label = edge_labels[i]
            g_edge_labels[(a, b)] = label
        G.add_edge(a, b, label=label)

    # try:
    #     pos = nx.nx_agraph.graphviz_layout(
    #         G, prog='neato', args='-Gnodesep="10" -Granksep="10"')
    # except:
    #     print(
    #         "Warning: could not use graphviz to layout graph. Falling back to spring layout.")
    #     print("To get better layouts, install graphviz and pygraphviz.")
    #     pos = nx.spring_layout(G, k=3.5, iterations=200)
    #     # pos = nx.kamada_kawai_layout(G, scale=1.5)
    #     # pos = nx.spectral_layout(G, scale=1.5)
    pos = nx.nx_agraph.graphviz_layout(
        G, prog="neato", args='-Gnodesep="20" -Granksep="20"'
    )

    default_draw_args = dict(alpha=0.9, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_nodes(G, pos, **default_draw_args)
    nx.draw_networkx_labels(
        G,
        pos,
        labels={i: v for i, v in enumerate(vertices)},
        font_size=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.5),
    )

    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(), arrows=True, edge_color="black", node_size=1000
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=g_edge_labels,
        font_color="darkslategray",
        font_size=8,
    )
    plt.axis("off")
    plt.show()


@wp.kernel
def assign_controls(
    actions: wp.array(dtype=wp.float32, ndim=2),
    gains: wp.array(dtype=wp.float32),
    limits: wp.array(dtype=wp.float32, ndim=2),
    controllable_dofs: wp.array(dtype=wp.int32),
    control_full_dim: int,
    # outputs
    controls: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    num_controls = gains.shape[0]
    for i in range(num_controls):
        lo = limits[i, 0]
        hi = limits[i, 1]
        idx = controllable_dofs[i]
        controls[env_id * control_full_dim + idx] = (
            wp.clamp(actions[env_id, i], lo, hi) * gains[i]
        )


@wp.kernel
def convert_joint_torques(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_act: wp.array(dtype=float),
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    type = joint_type[tid]
    if type == wp.sim.JOINT_FIXED:
        return
    if type == wp.sim.JOINT_FREE:
        return
    if type == wp.sim.JOINT_DISTANCE:
        return
    if type == wp.sim.JOINT_BALL:
        return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]

    X_wp = X_pj
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c
    com_c = body_com[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)

    # # local joint rotations
    # q_p = wp.transform_get_rotation(X_wp)
    # q_c = wp.transform_get_rotation(X_wc)

    # joint properties (for 1D joints)
    axis_start = joint_axis_start[tid]
    lin_axis_count = joint_axis_dim[tid, 0]
    ang_axis_count = joint_axis_dim[tid, 1]

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    # handle angular constraints
    if type == wp.sim.JOINT_REVOLUTE:
        axis = joint_axis[axis_start]
        act = joint_act[axis_start]
        a_p = wp.transform_vector(X_wp, axis)
        t_total += act * a_p
    elif type == wp.sim.JOINT_PRISMATIC:
        axis = joint_axis[axis_start]
        act = joint_act[axis_start]
        a_p = wp.transform_vector(X_wp, axis)
        f_total += act * a_p
    elif type == wp.sim.JOINT_COMPOUND:
        # q_off = wp.transform_get_rotation(X_cj)
        # q_pc = wp.quat_inverse(q_off)*wp.quat_inverse(q_p)*q_c*q_off
        # # decompose to a compound rotation each axis
        # angles = quat_decompose(q_pc)

        # # reconstruct rotation axes
        # axis_0 = wp.vec3(1.0, 0.0, 0.0)
        # q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        # axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        # q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        # axis_2 = wp.quat_rotate(q_1*q_0, wp.vec3(0.0, 0.0, 1.0))

        # q_w = q_p*q_off
        # t_total += joint_act[qd_start+0] * wp.quat_rotate(q_w, axis_0)
        # t_total += joint_act[qd_start+1] * wp.quat_rotate(q_w, axis_1)
        # t_total += joint_act[qd_start+2] * wp.quat_rotate(q_w, axis_2)

        axis_0 = joint_axis[axis_start + 0]
        t_total += joint_act[axis_start + 0] * wp.transform_vector(X_wp, axis_0)

        axis_1 = joint_axis[axis_start + 1]
        t_total += joint_act[axis_start + 1] * wp.transform_vector(X_wp, axis_1)

        axis_2 = joint_axis[axis_start + 2]
        t_total += joint_act[axis_start + 2] * wp.transform_vector(X_wp, axis_2)

    elif type == wp.sim.JOINT_UNIVERSAL:
        # q_off = wp.transform_get_rotation(X_cj)
        # q_pc = wp.quat_inverse(q_off)*wp.quat_inverse(q_p)*q_c*q_off

        # # decompose to a compound rotation each axis
        # angles = quat_decompose(q_pc)

        # # reconstruct rotation axes
        # axis_0 = wp.vec3(1.0, 0.0, 0.0)
        # q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        # axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        # q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        # axis_2 = wp.quat_rotate(q_1*q_0, wp.vec3(0.0, 0.0, 1.0))

        # q_w = q_p*q_off

        # free axes
        # t_total += joint_act[qd_start+0] * wp.quat_rotate(q_w, axis_0)
        # t_total += joint_act[qd_start+1] * wp.quat_rotate(q_w, axis_1)

        axis_0 = joint_axis[axis_start + 0]
        t_total += joint_act[axis_start + 0] * wp.transform_vector(X_wp, axis_0)

        axis_1 = joint_axis[axis_start + 1]
        t_total += joint_act[axis_start + 1] * wp.transform_vector(X_wp, axis_1)

    elif type == wp.sim.JOINT_D6:
        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a dynamic for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            act = joint_act[axis_start + 0]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += act * a_p
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            act = joint_act[axis_start + 1]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += act * a_p
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            act = joint_act[axis_start + 2]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += act * a_p

        if ang_axis_count > 0:
            axis = joint_axis[axis_start + lin_axis_count + 0]
            act = joint_act[axis_start + lin_axis_count + 0]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += act * a_p
        if ang_axis_count > 1:
            axis = joint_axis[axis_start + lin_axis_count + 1]
            act = joint_act[axis_start + lin_axis_count + 1]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += act * a_p
        if ang_axis_count > 2:
            axis = joint_axis[axis_start + lin_axis_count + 2]
            act = joint_act[axis_start + lin_axis_count + 2]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += act * a_p

    else:
        print("joint type not handled in apply_joint_actions")

    # write forces
    if id_p >= 0:
        wp.atomic_sub(
            body_f, id_p, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total)
        )
    wp.atomic_add(
        body_f, id_c, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total)
    )


def convert_joint_torques_to_body_forces(
    model: wp,
    body_q: wp.array,
    joint_torques: wp.array,
    body_f: wp.array,
):
    """
    Converts generalized joint torques `tau` to body forces `f` acting on the rigid bodies (to be applied as `wp.sim.State.body_f`).

    Args:
        model (wp.sim.Model): The model object.
        body_q (wp.array): The array of body world transforms (e.g. `wp.sim.State.body_q`).
        joint_torques (wp.array): The array of joint torques to be converted (dimension matched `joint_qd`, dtype is `float32`).
        body_f (wp.array): The array of body forces (dimension matches `body_q`, dtype is `spatial_vector`).
    """
    assert (
        len(body_q) == model.body_count
    ), "body_q must have the same length as the number of bodies"
    assert (
        len(joint_torques) == model.joint_axis_count
    ), "joint_torques must have the same length as the number of joint axes"
    assert (
        len(body_f) == model.body_count
    ), "body_f must have the same length as the number of bodies"

    if model.joint_count:
        wp.launch(
            kernel=convert_joint_torques,
            dim=model.joint_count,
            inputs=[
                body_q,
                model.body_com,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_axis_start,
                model.joint_axis_dim,
                model.joint_axis,
                joint_torques,
            ],
            outputs=[body_f],
            device=model.device,
        )


@wp.kernel
def generate_pd_control_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    target_q: wp.array(dtype=wp.float32),
    target_qd: wp.array(dtype=wp.float32),
    target_ke: wp.array(dtype=wp.float32),
    target_kd: wp.array(dtype=wp.float32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_axis_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_axis_start: wp.array(dtype=wp.int32),
    # outputs
    torques: wp.array(dtype=wp.float32),
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
        if target_q:
            tq = target_q[qj]
        else:
            tq = 0.0
        if target_qd:
            tqd = target_qd[qdj]
        else:
            tqd = 0.0
        tq = target_ke[aj] * (tq - q) + target_kd[aj] * (tqd - qd)
        wp.atomic_add(torques, aj, tq)


def generate_pd_control(
    model: wp.sim.Model,
    joint_torques: wp.array,
    joint_q: wp.array = None,
    joint_qd: wp.array = None,
    target_q: wp.array = None,
    target_qd: wp.array = None,
    target_ke: wp.array = None,
    target_kd: wp.array = None,
):
    """
    Generates PD control torques for the joints of a model.

    Args:
        model (wp.sim.Model): The model object.
        torques (wp.array): The array of joint torques to be generated (dimension matched `joint_qd`, dtype is `float32`).
        joint_q (wp.array): The array of joint positions (dimension matched `joint_qd`, dtype is `float32`).
        joint_qd (wp.array): The array of joint velocities (dimension matched `joint_q`, dtype is `float32`).
        target_q (wp.array): The array of target joint positions (dimension matched `joint_q`, dtype is `float32`).
        target_qd (wp.array): The array of target joint velocities (dimension matched `joint_qd`, dtype is `float32`).
        target_ke (wp.array): The array of proportional gains for the PD controller (dimension matched `joint_axis`, dtype is `float32`).
        target_kd (wp.array): The array of derivative gains for the PD controller (dimension matched `joint_axis`, dtype is `float32`).
    """
    if model.joint_count == 0:
        return
    if target_q is None:
        target_q = model.joint_q
    if target_qd is None:
        target_qd = model.joint_qd
    if target_ke is None:
        target_ke = model.joint_target_ke
    if target_kd is None:
        target_kd = model.joint_target_kd
    assert (
        len(joint_torques) == model.joint_axis_count
    ), "joint_torques must have the same length as the number of joint axes"
    wp.launch(
        kernel=generate_pd_control_kernel,
        dim=model.joint_count,
        inputs=[
            joint_q,
            joint_qd,
            target_q,
            target_qd,
            target_ke,
            target_kd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_axis_dim,
            model.joint_axis_start,
        ],
        outputs=[joint_torques],
        device=model.device,
    )

    # print("target-q", target_q.numpy() - joint_q.numpy())
    # print("control ", joint_torques.numpy())


def compute_body_jacobian(
    model: wp.sim.Model,
    joint_q: wp.array,
    joint_qd: wp.array,
    body_id: Union[int, str],
    offset: wp.transform = None,
    velocity=True,
    include_rotation=False,
):
    if isinstance(body_id, str):
        body_id = model.body_name.get(body_id)
    if offset is None:
        offset = wp.transform_identity()

    joint_q.requires_grad = True
    joint_qd.requires_grad = True

    if velocity:

        @wp.kernel
        def compute_body_out(
            body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)
        ):
            # TODO verify transform twist
            mv = transform_twist(offset, body_qd[body_id])
            if wp.static(include_rotation):
                for i in range(6):
                    body_out[i] = mv[i]
            else:
                for i in range(3):
                    body_out[i] = mv[3 + i]

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3
    else:

        @wp.kernel
        def compute_body_out(
            body_q: wp.array(dtype=wp.transform), body_out: wp.array(dtype=float)
        ):
            tf = body_q[body_id] * offset
            if wp.static(include_rotation):
                for i in range(7):
                    body_out[i] = tf[i]
            else:
                for i in range(3):
                    body_out[i] = tf[i]

        in_dim = model.joint_coord_count
        out_dim = 7 if include_rotation else 3

    out_state = model.state(requires_grad=True)
    body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.sim.eval_fk(model, joint_q, joint_qd, None, out_state)
        wp.launch(
            compute_body_out,
            1,
            inputs=[out_state.body_qd if velocity else out_state.body_q],
            outputs=[body_out],
            device=model.device,
        )

    def onehot(i):
        x = np.zeros(out_dim, dtype=np.float32)
        x[i] = 1.0
        return wp.array(x, device=model.device)

    J = np.empty((out_dim, in_dim), dtype=wp.float32)
    for i in range(out_dim):
        tape.backward(grads={body_out: onehot(i)})
        J[i] = joint_qd.grad.numpy() if velocity else joint_q.grad.numpy()
        tape.zero()
    return J.astype(np.float32)

def update_ground_plane(
    builder,
    pos,
    rot,
    ke: float = None,
    kd: float = None,
    kf: float = None,
    mu: float = None,
    restitution: float = None,
):
    normal = Rotation.from_quat(rot).as_matrix() @ np.array([0., 1., 0.])
    d = np.dot(pos, normal)
    # print(normal, d)
    builder._ground_params = {
        'plane': [*normal, d],
        'pos': pos,
        'rot': rot,
        'width': 0.0,
        'length': 0.0,
        'ke': ke if ke is not None else builder.default_shape_ke,
        'kd': kd if kd is not None else builder.default_shape_kd,
        'kf': kf if kf is not None else builder.default_shape_kf,
        'mu': mu if mu is not None else builder.default_shape_mu,
        'restitution': restitution if restitution is not None else builder.default_shape_restitution
    }