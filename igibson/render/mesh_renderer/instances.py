import numpy as np


from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.mesh_util import mat2xyz, safemat2quat, transform_vertex, xyz2mat


class InstanceGroup(object):
    """
    InstanceGroup is a set of visual objects.
    It is grouped together because they are kinematically connected.
    Robots and articulated objects are represented as instance groups.
    """

    def __init__(
        self,
        objects,
        id,
        link_ids,
        pybullet_uuid,
        ig_object,
        class_id,
        poses_trans,
        poses_rot,
        dynamic,
        softbody,
        use_pbr=True,
        use_pbr_mapping=True,
        shadow_caster=True,
    ):
        """
        :param objects: visual objects
        :param id: id of this instance_group
        :param link_ids: link_ids in pybullet
        :param pybullet_uuid: body id in pybullet
        :param class_id: class_id to render semantics
        :param poses_trans: initial translations for each visual object
        :param poses_rot: initial rotation matrix for each visual object
        :param dynamic: whether the instance group is dynamic
        :param softbody: whether the instance group is for a soft body
        :param use_pbr: whether to use PBR
        :param use_pbr_mapping: whether to use PBR mapping
        :param shadow_caster: whether to cast shadow
        """
        self.objects = objects
        self.poses_trans = poses_trans
        self.poses_rot = poses_rot
        self.id = id
        self.link_ids = link_ids
        self.class_id = class_id
        if len(objects) > 0:
            self.renderer = objects[0].renderer
        else:
            self.renderer = None

        self.pybullet_uuid = pybullet_uuid
        self.ig_object = ig_object
        self.dynamic = dynamic
        self.tf_tree = None
        self.softbody = softbody
        self.use_pbr = use_pbr
        self.use_pbr_mapping = use_pbr_mapping
        self.shadow_caster = shadow_caster
        self.roughness = 1
        self.metalness = 0
        # Determines whether object will be rendered
        self.hidden = False
        self.highlight = False
        # Indices into optimized buffers such as color information and transformation buffer
        # These values are used to set buffer information during simulation
        self.or_buffer_indices = None
        self.last_trans = [np.copy(item) for item in poses_trans]
        self.last_rot = [np.copy(item) for item in poses_rot]

    def set_highlight(self, highlight):
        self.highlight = highlight

    def render(self, shadow_pass=0):
        """
        Render this instance group
        shadow_pass = 0: normal rendering mode, disable shadow
        shadow_pass = 1: enable_shadow, rendering depth map from light space
        shadow_pass = 2: use rendered depth map to calculate shadow

        :param shadow_pass: shadow pass mode"""
        if self.renderer is None:
            return

        self.renderer.r.initvar(
            self.renderer.shaderProgram,
            self.renderer.V,
            self.renderer.last_V,
            self.renderer.lightV,
            shadow_pass,
            self.renderer.P,
            self.renderer.lightP,
            self.renderer.camera,
            self.renderer.lightpos,
            self.renderer.lightcolor,
        )

        for i, visual_object in enumerate(self.objects):
            for object_idx in visual_object.VAO_ids:
                if self.softbody:
                    # construct new vertex position into shape format
                    vertices = p.getMeshData(self.pybullet_uuid)[1]
                    vertices_flattened = [item for sublist in vertices for item in sublist]
                    vertex_position = np.array(vertices_flattened).reshape((len(vertices_flattened) // 3, 3))
                    shape = self.renderer.shapes[object_idx]
                    n_indices = len(shape.mesh.indices)
                    np_indices = shape.mesh.numpy_indices().reshape((n_indices, 3))
                    shape_vertex_index = np_indices[:, 0]
                    shape_vertex = vertex_position[shape_vertex_index]

                    # update new vertex position in buffer data
                    new_data = self.renderer.vertex_data[object_idx]
                    new_data[:, 0 : shape_vertex.shape[1]] = shape_vertex
                    new_data = new_data.astype(np.float32)

                    # transform and rotation already included in mesh data
                    self.pose_trans = np.eye(4)
                    self.pose_rot = np.eye(4)
                    self.last_trans = np.eye(4)
                    self.last_rot = np.eye(4)

                    # update buffer data into VBO
                    self.renderer.r.render_softbody_instance(
                        self.renderer.VAOs[object_idx], self.renderer.VBOs[object_idx], new_data
                    )

                self.renderer.r.init_pos_instance(
                    self.renderer.shaderProgram,
                    self.poses_trans[i],
                    self.poses_rot[i],
                    self.last_trans[i],
                    self.last_rot[i],
                )
                current_material = self.renderer.material_idx_to_material_instance_mapping[
                    self.renderer.shape_material_idx[object_idx]
                ]
                self.renderer.r.init_material_instance(
                    self.renderer.shaderProgram,
                    float(self.class_id) / MAX_CLASS_COUNT,
                    float(self.id) / MAX_INSTANCE_COUNT,
                    current_material.kd,
                    float(current_material.is_texture()),
                    float(self.use_pbr),
                    float(self.use_pbr_mapping),
                    float(self.metalness),
                    float(self.roughness),
                    current_material.transform_param,
                )

                try:
                    texture_id = current_material.texture_id
                    metallic_texture_id = current_material.metallic_texture_id
                    roughness_texture_id = current_material.roughness_texture_id
                    normal_texture_id = current_material.normal_texture_id

                    if texture_id is None:
                        texture_id = -1
                    if metallic_texture_id is None:
                        metallic_texture_id = -1
                    if roughness_texture_id is None:
                        roughness_texture_id = -1
                    if normal_texture_id is None:
                        normal_texture_id = -1

                    if self.renderer.msaa:
                        buffer = self.renderer.fbo_ms
                    else:
                        buffer = self.renderer.fbo
                    self.renderer.r.draw_elements_instance(
                        self.renderer.material_idx_to_material_instance_mapping[
                            self.renderer.shape_material_idx[object_idx]
                        ].is_texture(),
                        texture_id,
                        metallic_texture_id,
                        roughness_texture_id,
                        normal_texture_id,
                        self.renderer.depth_tex_shadow,
                        self.renderer.VAOs[object_idx],
                        self.renderer.faces[object_idx].size,
                        self.renderer.faces[object_idx],
                        buffer,
                    )
                finally:
                    self.renderer.r.cglBindVertexArray(0)
        self.renderer.r.cglUseProgram(0)

    def get_pose_in_camera(self):
        """
        Get instance group pose in camera reference frame
        """
        mat = self.renderer.V.dot(self.pose_trans.T).dot(self.pose_rot).T
        pose = np.concatenate([mat2xyz(mat), safemat2quat(mat[:3, :3].T)])
        return pose

    def set_position(self, pos):
        """
        Set positions for each part of this InstanceGroup

        :param pos: positions
        """

        self.last_trans = [np.copy(item) for item in self.poses_trans]
        self.poses_trans = pos

    def set_rotation(self, rot):
        """
        Set rotations for each part of this InstanceGroup

        :param rot: rotation matrix
        """

        self.last_rot = [np.copy(item) for item in self.poses_rot]
        self.poses_rot = rot

    def set_position_for_part(self, pos, j):
        """
        Set positions for one part of this InstanceGroup

        :param pos: position
        :param j: part index
        """

        self.last_trans[j] = np.copy(self.poses_trans[j])
        self.poses_trans[j] = pos

    def set_rotation_for_part(self, rot, j):
        """
        Set rotations for one part of this InstanceGroup

        :param rot: rotation matrix
        :param j: part index
        """

        self.last_rot[j] = np.copy(self.poses_rot[j])
        self.poses_rot[j] = rot

    def dump(self):
        """
        Dump vertex and face information
        """
        vertices_info = []
        faces_info = []
        for i, visual_obj in enumerate(self.objects):
            for vertex_data_index, face_data_index in zip(visual_obj.vertex_data_indices, visual_obj.face_indices):
                vertices_info.append(
                    transform_vertex(
                        self.renderer.vertex_data[vertex_data_index],
                        pose_trans=self.poses_trans[i],
                        pose_rot=self.poses_rot[i],
                    )
                )
                faces_info.append(self.renderer.faces[face_data_index])
        return vertices_info, faces_info

    def __str__(self):
        return "InstanceGroup({}) -> Objects({})".format(self.id, ",".join([str(object.id) for object in self.objects]))

    def __repr__(self):
        return self.__str__()
