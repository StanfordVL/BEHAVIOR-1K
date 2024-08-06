import asyncio
import os

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.prims.prim_base import BasePrim
from omnigibson.utils.physx_utils import bind_material
from omnigibson.utils.usd_utils import absolute_prim_path_to_scene_relative


class MaterialPrim(BasePrim):
    """
    Provides high level functions to deal with a material prim and its attributes/ properties.

    If there is a material prim present at the path, it will use it. Otherwise, a new material prim at
    the specified prim path will be created.

    Args:
        relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. Note that this is only needed if the prim does not already exist at
            @relative_prim_path -- it will be ignored if it already exists. Subclasses should define the exact keys expected
            for their class. For this material prim, the below values can be specified:

            mdl_name (None or str): If specified, should be the name of the mdl preset to load (including .mdl).
                None results in default, "OmniPBR.mdl"
            mtl_name (None or str): If specified, should be the name of the mtl preset to load.
                None results in default, "OmniPBR"
    """

    # Persistent dictionary of materials, mapped from prim_path to MaterialPrim
    MATERIALS = dict()

    @classmethod
    def get_material(cls, scene, name, prim_path, load_config=None):
        """
        Get a material prim from the persistent dictionary of materials, or create a new one if it doesn't exist.

        Args:
            name (str): Name for the object.
            prim_path (str): prim path of the MaterialPrim.
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime. Note that this is only needed if the prim does not already exist at
                @prim_path -- it will be ignored if it already exists.
        Returns:
            MaterialPrim: Material prim at the specified path
        """
        # If the material already exists, return it
        if prim_path in cls.MATERIALS:
            return cls.MATERIALS[prim_path]

        # Otherwise, create a new one and return it
        relative_prim_path = absolute_prim_path_to_scene_relative(scene, prim_path)
        new_material = cls(relative_prim_path=relative_prim_path, name=name, load_config=load_config)
        new_material.load(scene)
        assert (
            new_material.prim_path == prim_path
        ), f"Material prim path {new_material.prim_path} does not match {prim_path}"
        cls.MATERIALS[prim_path] = new_material
        return new_material

    def __init__(
        self,
        relative_prim_path,
        name,
        load_config=None,
    ):
        # Other values that will be filled in at runtime
        self._shader = None

        # Users of this material: should be a set of BaseObject and BaseSystem
        self._users = set()

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _load(self):
        # We create a new material at the specified path
        mtl_created = []
        lazy.omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name=(
                "OmniPBR.mdl" if self._load_config.get("mdl_name", None) is None else self._load_config["mdl_name"]
            ),
            mtl_name="OmniPBR" if self._load_config.get("mtl_name", None) is None else self._load_config["mtl_name"],
            mtl_created_list=mtl_created,
        )
        material_path = mtl_created[0]

        # Move prim to desired location
        lazy.omni.kit.commands.execute("MovePrim", path_from=material_path, path_to=self.prim_path)

        # Return generated material
        return lazy.omni.isaac.core.utils.prims.get_prim_at_path(self.prim_path)

    @classmethod
    def clear(cls):
        cls.MATERIALS = dict()

    @property
    def users(self):
        """
        Users of this material: should be a list of BaseObject and BaseSystem
        """
        return self._users

    def add_user(self, user):
        """
        Adds a user to the material. This can be a BaseObject or BaseSystem.

        Args:
            user (BaseObject or BaseSystem): User to add to the material
        """
        self._users.add(user)

    def remove_user(self, user):
        """
        Removes a user from the material. This can be a BaseObject or BaseSystem.
        If there are no users left, the material will be removed.

        Args:
            user (BaseObject or BaseSystem): User to remove from the material
        """
        self._users.remove(user)
        if len(self._users) == 0:
            self.remove()

    def remove(self):
        # Remove from global sensors dictionary
        self.MATERIALS.pop(self.prim_path)

        # Run super
        super().remove()

    def _post_load(self):
        # run super first
        super()._post_load()

        # Add this material to the list of global materials
        self.MATERIALS[self.prim_path] = self

        # Generate shader reference
        self._shader = lazy.omni.usd.get_shader_from_material(self._prim)

    def bind(self, target_prim_path):
        """
        Bind this material to an arbitrary prim (usually a visual mesh prim)

        Args:
            target_prim_path (str): prim path of the Prim to bind to
        """
        bind_material(prim_path=target_prim_path, material_path=self.prim_path)

    async def _load_mdl_parameters(self, render=True):
        """
        Loads MDL parameters internally so they can be accessed by our class instance

        Args:
            render (bool): If True, takes a rendering step before loading the mdl parameters.
                Note that a rendering step is necessary to load these parameters, though if a step has already
                occurred externally, no additional rendering step is needed
        """
        if render:
            og.sim.render()
        await lazy.omni.usd.get_context().load_mdl_parameters_for_prim_async(self._shader)

    def shader_force_populate(self, render=True):
        """
        Force populate inputs and outputs of the shader

        Args:
            render (bool): If True, takes a rendering step before force populating the inputs and outputs.
                Note that a rendering step is necessary to load these I/Os, though if a step has already
                occurred externally, no additional rendering step is needed
        """
        # TODO: Consider optimizing this somehow.
        assert self._shader is not None
        asyncio.run(self._load_mdl_parameters(render=render))

    def shader_update_asset_paths_with_root_path(self, root_path):
        """
        Similar to @shader_update_asset_paths, except in this case, root_path is explicitly provided by the caller.

        Args:
            root_path (str): root to be pre-appended to the original asset paths
        """

        for inp_name in self.shader_input_names_by_type("SdfAssetPath"):
            inp = self.get_input(inp_name)
            # If the input doesn't have any path, skip
            if inp is None:
                continue

            original_path = inp.path if inp.resolvedPath == "" else inp.resolvedPath
            # If the input has an empty path, skip
            if original_path == "":
                continue

            new_path = os.path.join(root_path, original_path)
            self.set_input(inp_name, new_path)

    def get_input(self, inp):
        """
        Grabs the input with corresponding name @inp associated with this material and shader

        Args:
            inp (str): Name of the shader input whose value will be grabbed

        Returns:
            any: value of the requested @inp
        """
        return self._shader.GetInput(inp).Get()

    def set_input(self, inp, val):
        """
        Sets the input with corresponding name @inp associated with this material and shader

        Args:
            inp (str): Name of the shader input whose value will be set
            val (any): Value to set for the input. This should be the valid type for that attribute.
        """
        # Make sure the input exists first, so we avoid segfaults with "invalid null prim"
        assert (
            inp in self.shader_input_names
        ), f"Got invalid shader input to set! Current inputs are: {self.shader_input_names}. Got: {inp}"
        self._shader.GetInput(inp).Set(val)

    @property
    def is_glass(self):
        """
        Returns:
            bool: Whether this material is a glass material or not
        """
        return "glass_color" in self.shader_input_names

    @property
    def shader(self):
        """
        Returns:
            Usd.Shade: Shader associated with this material
        """
        return self._shader

    @property
    def shader_input_names(self):
        """
        Returns:
            set: All the shader input names associated with this material
        """
        return {inp.GetBaseName() for inp in self._shader.GetInputs()}

    def shader_input_names_by_type(self, input_type):
        """
        Args:
            input_type (str): input type

        Returns:
            set: All the shader input names associated with this material that match the given input type
        """
        return {inp.GetBaseName() for inp in self._shader.GetInputs() if inp.GetTypeName().cppTypeName == input_type}

    @property
    def diffuse_color_constant(self):
        """
        Returns:
            3-array: this material's applied (R,G,B) color
        """
        diffuse_color_constant = self.get_input(inp="diffuse_color_constant")
        return th.tensor(diffuse_color_constant, dtype=th.float32) if diffuse_color_constant is not None else None

    @diffuse_color_constant.setter
    def diffuse_color_constant(self, color):
        """
        Args:
             color (3-array): this material's applied (R,G,B) color
        """
        self.set_input(inp="diffuse_color_constant", val=lazy.pxr.Gf.Vec3f(*color.tolist()))

    @property
    def diffuse_texture(self):
        """
        Returns:
            str: this material's applied diffuse_texture filepath
        """
        return self.get_input(inp="diffuse_texture").resolvedPath

    @diffuse_texture.setter
    def diffuse_texture(self, fpath):
        """
        Args:
            str: this material's applied diffuse_texture filepath
        """
        self.set_input(inp="diffuse_texture", val=lazy.pxr.Sdf.AssetPath(fpath))

    @property
    def albedo_desaturation(self):
        """
        Returns:
            float: this material's applied albedo_desaturation
        """
        return self.get_input(inp="albedo_desaturation")

    @albedo_desaturation.setter
    def albedo_desaturation(self, desaturation):
        """
        Args:
             desaturation (float): this material's applied albedo_desaturation
        """
        self.set_input(inp="albedo_desaturation", val=desaturation)

    @property
    def albedo_add(self):
        """
        Returns:
            float: this material's applied albedo_add
        """
        return self.get_input(inp="albedo_add")

    @albedo_add.setter
    def albedo_add(self, add):
        """
        Args:
             add (float): this material's applied albedo_add
        """
        self.set_input(inp="albedo_add", val=add)

    @property
    def albedo_brightness(self):
        """
        Returns:
            float: this material's applied albedo_brightness
        """
        return self.get_input(inp="albedo_brightness")

    @albedo_brightness.setter
    def albedo_brightness(self, brightness):
        """
        Args:
             brightness (float): this material's applied albedo_brightness
        """
        self.set_input(inp="albedo_brightness", val=brightness)

    @property
    def diffuse_tint(self):
        """
        Returns:
            3-array: this material's applied (R,G,B) diffuse_tint
        """
        diffuse_tint = self.get_input(inp="diffuse_tint")
        return th.tensor(diffuse_tint, dtype=th.float32) if diffuse_tint is not None else None

    @diffuse_tint.setter
    def diffuse_tint(self, color):
        """
        Args:
             color (3-array): this material's applied (R,G,B) diffuse_tint
        """
        self.set_input(inp="diffuse_tint", val=lazy.pxr.Gf.Vec3f(*color.tolist()))

    @property
    def reflection_roughness_constant(self):
        """
        Returns:
            float: this material's applied reflection_roughness_constant
        """
        return self.get_input(inp="reflection_roughness_constant")

    @reflection_roughness_constant.setter
    def reflection_roughness_constant(self, roughness):
        """
        Args:
             roughness (float): this material's applied reflection_roughness_constant
        """
        self.set_input(inp="reflection_roughness_constant", val=roughness)

    @property
    def reflection_roughness_texture_influence(self):
        """
        Returns:
            float: this material's applied reflection_roughness_texture_influence
        """
        return self.get_input(inp="reflection_roughness_texture_influence")

    @reflection_roughness_texture_influence.setter
    def reflection_roughness_texture_influence(self, prop):
        """
        Args:
             prop (float): this material's applied reflection_roughness_texture_influence proportion
        """
        self.set_input(inp="reflection_roughness_texture_influence", val=prop)

    @property
    def reflectionroughness_texture(self):
        """
        Returns:
            None or str: this material's applied reflectionroughness_texture fpath if there is a texture applied, else
                None
        """
        inp = self.get_input(inp="reflectionroughness_texture")
        return None if inp is None else inp.resolvedPath

    @reflectionroughness_texture.setter
    def reflectionroughness_texture(self, fpath):
        """
        Args:
             fpath (str): this material's applied reflectionroughness_texture fpath
        """
        self.set_input(inp="reflectionroughness_texture", val=lazy.pxr.Sdf.AssetPath(fpath))

    @property
    def metallic_constant(self):
        """
        Returns:
            float: this material's applied metallic_constant
        """
        return self.get_input(inp="metallic_constant")

    @metallic_constant.setter
    def metallic_constant(self, constant):
        """
        Args:
             constant (float): this material's applied metallic_constant
        """
        self.set_input(inp="metallic_constant", val=constant)

    @property
    def metallic_texture_influence(self):
        """
        Returns:
            float: this material's applied metallic_texture_influence
        """
        return self.get_input(inp="metallic_texture_influence")

    @metallic_texture_influence.setter
    def metallic_texture_influence(self, prop):
        """
        Args:
             prop (float): this material's applied metallic_texture_influence
        """
        self.set_input(inp="metallic_texture_influence", val=prop)

    @property
    def metallic_texture(self):
        """
        Returns:
            None or str: this material's applied metallic_texture fpath if there is a texture applied, else
                None
        """
        inp = self.get_input(inp="metallic_texture")
        return None if inp is None else inp.resolvedPath

    @metallic_texture.setter
    def metallic_texture(self, fpath):
        """
        Args:
             fpath (str): this material's applied metallic_texture fpath
        """
        self.set_input(inp="metallic_texture", val=lazy.pxr.Sdf.AssetPath(fpath))

    @property
    def specular_level(self):
        """
        Returns:
            float: this material's applied specular_level
        """
        return self.get_input(inp="specular_level")

    @specular_level.setter
    def specular_level(self, level):
        """
        Args:
             level (float): this material's applied specular_level
        """
        self.set_input(inp="specular_level", val=level)

    @property
    def enable_ORM_texture(self):
        """
        Returns:
            bool: this material's applied enable_ORM_texture
        """
        return self.get_input(inp="enable_ORM_texture")

    @enable_ORM_texture.setter
    def enable_ORM_texture(self, enabled):
        """
        Args:
             enabled (bool): this material's applied enable_ORM_texture
        """
        self.set_input(inp="enable_ORM_texture", val=enabled)

    @property
    def ORM_texture(self):
        """
        Returns:
            None or str: this material's applied ORM_texture fpath if there is a texture applied, else
                None
        """
        inp = self.get_input(inp="ORM_texture")
        return None if inp is None else inp.resolvedPath

    @ORM_texture.setter
    def ORM_texture(self, fpath):
        """
        Args:
             fpath (str): this material's applied ORM_texture fpath
        """
        self.set_input(inp="ORM_texture", val=lazy.pxr.Sdf.AssetPath(fpath))

    @property
    def ao_to_diffuse(self):
        """
        Returns:
            float: this material's applied ao_to_diffuse
        """
        return self.get_input(inp="ao_to_diffuse")

    @ao_to_diffuse.setter
    def ao_to_diffuse(self, val):
        """
        Args:
             val (float): this material's applied ao_to_diffuse
        """
        self.set_input(inp="ao_to_diffuse", val=val)

    @property
    def ao_texture(self):
        """
        Returns:
            None or str: this material's applied ao_texture fpath if there is a texture applied, else
                None
        """
        inp = self.get_input(inp="ao_texture")
        return None if inp is None else inp.resolvedPath

    @ao_texture.setter
    def ao_texture(self, fpath):
        """
        Args:
             fpath (str): this material's applied ao_texture fpath
        """
        self.set_input(inp="ao_texture", val=lazy.pxr.Sdf.AssetPath(fpath))

    @property
    def enable_emission(self):
        """
        Returns:
            bool: this material's applied enable_emission
        """
        return self.get_input(inp="enable_emission")

    @enable_emission.setter
    def enable_emission(self, enabled):
        """
        Args:
             enabled (bool): this material's applied enable_emission
        """
        self.set_input(inp="enable_emission", val=enabled)

    @property
    def emissive_color(self):
        """
        Returns:
            3-array: this material's applied (R,G,B) emissive_color
        """
        color = self.get_input(inp="emissive_color")
        return th.tensor(color, dtype=th.float32) if color is not None else None

    @emissive_color.setter
    def emissive_color(self, color):
        """
        Args:
             color (3-array): this material's applied emissive_color
        """
        self.set_input(inp="emissive_color", val=lazy.pxr.Gf.Vec3f(*color))

    @property
    def emissive_color_texture(self):
        """
        Returns:
            None or str: this material's applied emissive_color_texture fpath if there is a texture applied, else
                None
        """
        inp = self.get_input(inp="emissive_color_texture")
        return None if inp is None else inp.resolvedPath

    @emissive_color_texture.setter
    def emissive_color_texture(self, fpath):
        """
        Args:
             fpath (str): this material's applied emissive_color_texture fpath
        """
        self.set_input(inp="emissive_color_texture", val=lazy.pxr.Sdf.AssetPath(fpath))

    @property
    def emissive_mask_texture(self):
        """
        Returns:
            None or str: this material's applied emissive_mask_texture fpath if there is a texture applied, else
                None
        """
        inp = self.get_input(inp="emissive_mask_texture")
        return None if inp is None else inp.resolvedPath

    @emissive_mask_texture.setter
    def emissive_mask_texture(self, fpath):
        """
        Args:
             fpath (str): this material's applied emissive_mask_texture fpath
        """
        self.set_input(inp="emissive_mask_texture", val=lazy.pxr.Sdf.AssetPath(fpath))

    @property
    def emissive_intensity(self):
        """
        Returns:
            float: this material's applied emissive_intensity
        """
        return self.get_input(inp="emissive_intensity")

    @emissive_intensity.setter
    def emissive_intensity(self, intensity):
        """
        Args:
             intensity (float): this material's applied emissive_intensity
        """
        self.set_input(inp="emissive_intensity", val=intensity)

    @property
    def enable_opacity(self):
        """
        Returns:
            bool: this material's applied enable_opacity
        """
        return self.get_input(inp="enable_opacity")

    @enable_opacity.setter
    def enable_opacity(self, enabled):
        """
        Args:
             enabled (bool): this material's applied enable_opacity
        """
        self.set_input(inp="enable_opacity", val=enabled)

    @property
    def enable_opacity_texture(self):
        """
        Returns:
            bool: this material's applied enable_opacity_texture
        """
        return self.get_input(inp="enable_opacity_texture")

    @enable_opacity_texture.setter
    def enable_opacity_texture(self, enabled):
        """
        Args:
             enabled (bool): this material's applied enable_opacity_texture
        """
        self.set_input(inp="enable_opacity_texture", val=enabled)

    @property
    def opacity_constant(self):
        """
        Returns:
            float: this material's applied opacity_constant
        """
        return self.get_input(inp="opacity_constant")

    @opacity_constant.setter
    def opacity_constant(self, constant):
        """
        Args:
             constant (float): this material's applied opacity_constant
        """
        self.set_input(inp="opacity_constant", val=constant)

    @property
    def opacity_texture(self):
        """
        Returns:
            None or str: this material's applied opacity_texture fpath if there is a texture applied, else
                None
        """
        inp = self.get_input(inp="opacity_texture")
        return None if inp is None else inp.resolvedPath

    @opacity_texture.setter
    def opacity_texture(self, fpath):
        """
        Args:
             fpath (str): this material's applied opacity_texture fpath
        """
        self.set_input(inp="opacity_texture", val=lazy.pxr.Sdf.AssetPath(fpath))

    @property
    def opacity_mode(self):
        """
        Returns:
            int: this material's applied opacity_mode
        """
        return self.get_input(inp="opacity_mode")

    @opacity_mode.setter
    def opacity_mode(self, mode):
        """
        Args:
             mode (int): this material's applied opacity_mode
        """
        self.set_input(inp="opacity_mode", val=mode)

    @property
    def opacity_threshold(self):
        """
        Returns:
            float: this material's applied opacity_threshold
        """
        return self.get_input(inp="opacity_threshold")

    @opacity_threshold.setter
    def opacity_threshold(self, threshold):
        """
        Args:
             threshold (float): this material's applied opacity_threshold
        """
        self.set_input(inp="opacity_threshold", val=threshold)

    @property
    def bump_factor(self):
        """
        Returns:
            float: this material's applied bump_factor
        """
        return self.get_input(inp="bump_factor")

    @bump_factor.setter
    def bump_factor(self, factor):
        """
        Args:
             factor (float): this material's applied bump_factor
        """
        self.set_input(inp="bump_factor", val=factor)

    @property
    def normalmap_texture(self):
        """
        Returns:
            None or str: this material's applied normalmap_texture fpath if there is a texture applied, else
                None
        """
        inp = self.get_input(inp="normalmap_texture")
        return None if inp is None else inp.resolvedPath

    @normalmap_texture.setter
    def normalmap_texture(self, fpath):
        """
        Args:
             fpath (str): this material's applied normalmap_texture fpath
        """
        self.set_input(inp="normalmap_texture", val=lazy.pxr.Sdf.AssetPath(fpath))

    @property
    def detail_bump_factor(self):
        """
        Returns:
            float: this material's applied detail_bump_factor
        """
        return self.get_input(inp="detail_bump_factor")

    @detail_bump_factor.setter
    def detail_bump_factor(self, factor):
        """
        Args:
             factor (float): this material's applied detail_bump_factor
        """
        self.set_input(inp="detail_bump_factor", val=factor)

    @property
    def detail_normalmap_texture(self):
        """
        Returns:
            None or str: this material's applied detail_normalmap_texture fpath if there is a texture applied, else
                None
        """
        inp = self.get_input(inp="detail_normalmap_texture")
        return None if inp is None else inp.resolvedPath

    @detail_normalmap_texture.setter
    def detail_normalmap_texture(self, fpath):
        """
        Args:
             fpath (str): this material's applied detail_normalmap_texture fpath
        """
        self.set_input(inp="detail_normalmap_texture", val=lazy.pxr.Sdf.AssetPath(fpath))

    @property
    def flip_tangent_u(self):
        """
        Returns:
            bool: this material's applied flip_tangent_u
        """
        return self.get_input(inp="flip_tangent_u")

    @flip_tangent_u.setter
    def flip_tangent_u(self, flipped):
        """
        Args:
             flipped (bool): this material's applied flip_tangent_u
        """
        self.set_input(inp="flip_tangent_u", val=flipped)

    @property
    def flip_tangent_v(self):
        """
        Returns:
            bool: this material's applied flip_tangent_v
        """
        return self.get_input(inp="flip_tangent_v")

    @flip_tangent_v.setter
    def flip_tangent_v(self, flipped):
        """
        Args:
             flipped (bool): this material's applied flip_tangent_v
        """
        self.set_input(inp="flip_tangent_v", val=flipped)

    @property
    def project_uvw(self):
        """
        Returns:
            bool: this material's applied project_uvw
        """
        return self.get_input(inp="project_uvw")

    @project_uvw.setter
    def project_uvw(self, projected):
        """
        Args:
             projected (bool): this material's applied project_uvw
        """
        self.set_input(inp="project_uvw", val=projected)

    @property
    def world_or_object(self):
        """
        Returns:
            bool: this material's applied world_or_object
        """
        return self.get_input(inp="world_or_object")

    @world_or_object.setter
    def world_or_object(self, val):
        """
        Args:
             val (bool): this material's applied world_or_object
        """
        self.set_input(inp="world_or_object", val=val)

    @property
    def uv_space_index(self):
        """
        Returns:
            int: this material's applied uv_space_index
        """
        return self.get_input(inp="uv_space_index")

    @uv_space_index.setter
    def uv_space_index(self, index):
        """
        Args:
             index (int): this material's applied uv_space_index
        """
        self.set_input(inp="uv_space_index", val=index)

    @property
    def texture_translate(self):
        """
        Returns:
            2-array: this material's applied texture_translate
        """
        return th.tensor(self.get_input(inp="texture_translate"))

    @texture_translate.setter
    def texture_translate(self, translate):
        """
        Args:
             translate (2-array): this material's applied (x,y) texture_translate
        """
        self.set_input(inp="texture_translate", val=lazy.pxr.Gf.Vec2f(*th.tensor(translate, dtype=th.float32)))

    @property
    def texture_rotate(self):
        """
        Returns:
            float: this material's applied texture_rotate
        """
        return self.get_input(inp="texture_rotate")

    @texture_rotate.setter
    def texture_rotate(self, rotate):
        """
        Args:
             rotate (float): this material's applied texture_rotate
        """
        self.set_input(inp="texture_rotate", val=rotate)

    @property
    def texture_scale(self):
        """
        Returns:
            2-array: this material's applied texture_scale
        """
        return th.tensor(self.get_input(inp="texture_scale"))

    @texture_scale.setter
    def texture_scale(self, scale):
        """
        Args:
             scale (2-array): this material's applied (x,y) texture_scale
        """
        self.set_input(inp="texture_scale", val=lazy.pxr.Gf.Vec2f(*th.tensor(scale, dtype=th.float32)))

    @property
    def detail_texture_translate(self):
        """
        Returns:
            2-array: this material's applied detail_texture_translate
        """
        detail_texture_translate = self.get_input(inp="detail_texture_translate")
        return th.tensor(detail_texture_translate, dtype=th.float32) if detail_texture_translate is not None else None

    @detail_texture_translate.setter
    def detail_texture_translate(self, translate):
        """
        Args:
             translate (2-array): this material's applied detail_texture_translate
        """
        self.set_input(inp="detail_texture_translate", val=lazy.pxr.Gf.Vec2f(*th.tensor(translate, dtype=th.float32)))

    @property
    def detail_texture_rotate(self):
        """
        Returns:
            float: this material's applied detail_texture_rotate
        """
        return self.get_input(inp="detail_texture_rotate")

    @detail_texture_rotate.setter
    def detail_texture_rotate(self, rotate):
        """
        Args:
             rotate (float): this material's applied detail_texture_rotate
        """
        self.set_input(inp="detail_texture_rotate", val=rotate)

    @property
    def detail_texture_scale(self):
        """
        Returns:
            2-array: this material's applied detail_texture_scale
        """
        detail_texture_scale = self.get_input(inp="detail_texture_scale")
        return th.tensor(detail_texture_scale, dtype=th.float32) if detail_texture_scale is not None else None

    @detail_texture_scale.setter
    def detail_texture_scale(self, scale):
        """
        Args:
             scale (2-array): this material's applied detail_texture_scale
        """
        self.set_input(inp="detail_texture_scale", val=lazy.pxr.Gf.Vec2f(*th.tensor(scale, dtype=th.float32)))

    @property
    def exclude_from_white_mode(self):
        """
        Returns:
            bool: this material's applied excludeFromWhiteMode
        """
        return self.get_input(inp="excludeFromWhiteMode")

    @exclude_from_white_mode.setter
    def exclude_from_white_mode(self, exclude):
        """
        Args:
             exclude (bool): this material's applied excludeFromWhiteMode
        """
        self.set_input(inp="excludeFromWhiteMode", val=exclude)

    @property
    def diffuse_reflection_weight(self):
        """
        Returns:
            float: this material's applied diffuse_reflection_weight
        """
        return self.get_input(inp="diffuse_reflection_weight")

    @diffuse_reflection_weight.setter
    def diffuse_reflection_weight(self, weight):
        """
        Args:
             weight (float): this material's applied diffuse_reflection_weight
        """
        self.set_input(inp="diffuse_reflection_weight", val=weight)

    @property
    def enable_specular_transmission(self):
        """
        Returns:
            bool: this material's applied enable_specular_transmission
        """
        return self.get_input(inp="enable_specular_transmission")

    @enable_specular_transmission.setter
    def enable_specular_transmission(self, enabled):
        """
        Args:
             enabled (bool): this material's applied enable_specular_transmission
        """
        self.set_input(inp="enable_specular_transmission", val=enabled)

    @property
    def specular_transmission_weight(self):
        """
        Returns:
            float: this material's applied specular_transmission_weight
        """
        return self.get_input(inp="specular_transmission_weight")

    @specular_transmission_weight.setter
    def specular_transmission_weight(self, weight):
        """
        Args:
             weight (float): this material's applied specular_transmission_weight
        """
        self.set_input(inp="specular_transmission_weight", val=weight)

    @property
    def diffuse_reflection_color(self):
        """
        Returns:
            3-array: this material's diffuse_reflection_color in (R,G,B)
        """
        diffuse_reflection_color = self.get_input(inp="diffuse_reflection_color")
        return th.tensor(diffuse_reflection_color, dtype=th.float32) if diffuse_reflection_color is not None else None

    @diffuse_reflection_color.setter
    def diffuse_reflection_color(self, color):
        """
        Args:
             color (3-array): this material's diffuse_reflection_color in (R,G,B)
        """
        self.set_input(inp="diffuse_reflection_color", val=lazy.pxr.Gf.Vec3f(*color))

    @property
    def specular_reflection_color(self):
        """
        Returns:
            3-array: this material's specular_reflection_color in (R,G,B)
        """
        specular_reflection_color = self.get_input(inp="specular_reflection_color")
        return th.tensor(specular_reflection_color, dtype=th.float32) if specular_reflection_color is not None else None

    @specular_reflection_color.setter
    def specular_reflection_color(self, color):
        """
        Args:
             color (3-array): this material's specular_reflection_color in (R,G,B)
        """
        self.set_input(inp="specular_reflection_color", val=lazy.pxr.Gf.Vec3f(*color))

    @property
    def specular_transmission_color(self):
        """
        Returns:
            3-array: this material's specular_transmission_color in (R,G,B)
        """
        specular_transmission_color = self.get_input(inp="specular_transmission_color")
        return (
            th.tensor(specular_transmission_color, dtype=th.float32)
            if specular_transmission_color is not None
            else None
        )

    @specular_transmission_color.setter
    def specular_transmission_color(self, color):
        """
        Args:
             color (3-array): this material's specular_transmission_color in (R,G,B)
        """
        self.set_input(inp="specular_transmission_color", val=lazy.pxr.Gf.Vec3f(*color))

    @property
    def specular_transmission_scattering_color(self):
        """
        Returns:
            3-array: this material's specular_transmission_scattering_color in (R,G,B)
        """
        specular_transmission_scattering_color = self.get_input(inp="specular_transmission_scattering_color")
        return (
            th.tensor(specular_transmission_scattering_color, dtype=th.float32)
            if specular_transmission_scattering_color is not None
            else None
        )

    @specular_transmission_scattering_color.setter
    def specular_transmission_scattering_color(self, color):
        """
        Args:
             color (3-array): this material's specular_transmission_scattering_color in (R,G,B)
        """
        self.set_input(inp="specular_transmission_scattering_color", val=lazy.pxr.Gf.Vec3f(*color))

    @property
    def specular_reflection_ior_preset(self):
        """
        Returns:
            int: this material's specular_reflection_ior_preset (int corresponding to enum)
        """
        return self.get_input(inp="specular_reflection_ior_preset")

    @specular_reflection_ior_preset.setter
    def specular_reflection_ior_preset(self, preset):
        """
        Args:
             preset (int): this material's specular_reflection_ior_preset (int corresponding to enum)
        """
        self.set_input(inp="specular_reflection_ior_preset", val=preset)

    @property
    def enable_diffuse_transmission(self):
        """
        Returns:
            float: this material's applied enable_diffuse_transmission
        """
        return self.get_input(inp="enable_diffuse_transmission")

    @enable_diffuse_transmission.setter
    def enable_diffuse_transmission(self, val):
        """
        Args:
             val (bool): this material's applied enable_diffuse_transmission
        """
        self.set_input(inp="enable_diffuse_transmission", val=val)

    @property
    def glass_color(self):
        """
        Returns:
            3-array: this material's applied (R,G,B) glass color (only applicable to OmniGlass materials)
        """
        assert self.is_glass, (
            f"Tried to query glass_color shader input, "
            f"but material at {self.prim_path} is not an OmniGlass material!"
        )
        glass_color = self.get_input(inp="glass_color")
        return th.tensor(glass_color, dtype=th.float32) if glass_color is not None else None

    @glass_color.setter
    def glass_color(self, color):
        """
        Args:
             color (3-array): this material's applied (R,G,B) glass color (only applicable to OmniGlass materials)
        """
        assert self.is_glass, (
            f"Tried to set glass_color shader input, " f"but material at {self.prim_path} is not an OmniGlass material!"
        )
        self.set_input(inp="glass_color", val=lazy.pxr.Gf.Vec3f(*color))
