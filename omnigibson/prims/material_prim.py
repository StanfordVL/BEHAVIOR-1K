import os

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.prims.prim_base import BasePrim
from omnigibson.utils.physx_utils import bind_material
from omnigibson.utils.usd_utils import absolute_prim_path_to_scene_relative, get_sdf_value_type_name


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
        self._shader_node = None

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
        og.sim.update_handles()

        # Return generated material
        return lazy.isaacsim.core.utils.prims.get_prim_at_path(self.prim_path)

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
        self._shader_node = lazy.usd.mdl.RegistryUtils.GetShaderNodeForPrim(self._shader.GetPrim())

    def bind(self, target_prim_path):
        """
        Bind this material to an arbitrary prim (usually a visual mesh prim)

        Args:
            target_prim_path (str): prim path of the Prim to bind to
        """
        bind_material(prim_path=target_prim_path, material_path=self.prim_path)

    def shader_update_asset_paths_with_root_path(self, root_path, relative=False):
        """
        Similar to @shader_update_asset_paths, except in this case, root_path is explicitly provided by the caller.

        Args:
            root_path (str): root directory from which to update shader paths
            relative (bool): If set, all paths will be updated as relative paths with respect to @root_path.
                Otherwise, @root_path will be pre-appended to the original asset paths
        """

        for inp_name in self.get_shader_input_names_by_type("SdfAssetPath", include_default=True):
            inp = self.get_input(inp_name)
            # If the input doesn't have any path, skip
            if inp is None:
                continue

            original_path = inp.path if inp.resolvedPath == "" else inp.resolvedPath
            # If the input has an empty path, skip
            if original_path == "":
                continue

            new_path = (
                f"./{os.path.relpath(original_path, root_path)}" if relative else os.path.join(root_path, original_path)
            )
            self.set_input(inp_name, new_path)

    def get_input(self, inp):
        """
        Grabs the input with corresponding name @inp associated with this material and shader

        Args:
            inp (str): Name of the shader input whose value will be grabbed

        Returns:
            any: value of the requested @inp
        """
        non_default_inp = self._shader.GetInput(inp).Get()
        if non_default_inp is not None:
            return non_default_inp

        return self._shader_node.GetInput(inp).GetDefaultValue()

    def set_input(self, inp, val):
        """
        Sets the input with corresponding name @inp associated with this material and shader

        Args:
            inp (str): Name of the shader input whose value will be set
            val (any): Value to set for the input. This should be the valid type for that attribute.
        """
        # Make sure the input exists first, so we avoid segfaults with "invalid null prim"
        if inp in self.shader_input_names:
            self._shader.GetInput(inp).Set(val)
        elif inp in self.shader_default_input_names:
            input_type = get_sdf_value_type_name(val)
            self._shader.CreateInput(inp, input_type).Set(val)
        else:
            raise ValueError(f"Got invalid shader input to set! Got: {inp}")

    @property
    def is_glass(self):
        """
        Returns:
            bool: Whether this material is a glass material or not
        """
        return "glass_color" in self.shader_input_names | self.shader_default_input_names

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

    @property
    def shader_default_input_names(self):
        """
        Returns:
            set: All the shader input names associated with this material that have default values
        """
        return set(self._shader_node.GetInputNames())

    def get_shader_input_names_by_type(self, input_type, include_default=False):
        """
        Args:
            input_type (str): input type
            include_default (bool): whether to include default inputs
        Returns:
            set: All the shader input names associated with this material that match the given input type
        """
        shader_input_names = {
            inp.GetBaseName() for inp in self._shader.GetInputs() if inp.GetTypeName().cppTypeName == input_type
        }
        if not include_default:
            return shader_input_names
        shader_default_input_names = {
            inp_name
            for inp_name in self.shader_default_input_names
            if self._shader_node.GetInput(inp_name).GetType() == input_type
        }
        return shader_input_names | shader_default_input_names

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
    def albedo_add(self):
        """
        Returns:
            float: this material's applied albedo_add
        """
        # TODO: Implement this with the V-Ray material
        return 0.0  # self.get_input(inp="albedo_add")

    @albedo_add.setter
    def albedo_add(self, add):
        """
        Args:
             add (float): this material's applied albedo_add
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="albedo_add", val=add)

    @property
    def diffuse_tint(self):
        """
        Returns:
            3-array: this material's applied (R,G,B) diffuse_tint
        """
        # TODO: Implement this with the V-Ray material
        return th.zeros(3)
        diffuse_tint = self.get_input(inp="diffuse_tint")
        return th.tensor(diffuse_tint, dtype=th.float32) if diffuse_tint is not None else None

    @diffuse_tint.setter
    def diffuse_tint(self, color):
        """
        Args:
             color (3-array): this material's applied (R,G,B) diffuse_tint
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="diffuse_tint", val=lazy.pxr.Gf.Vec3f(*color.tolist()))

    @property
    def reflection_roughness_constant(self):
        """
        Returns:
            float: this material's applied reflection_roughness_constant
        """
        # TODO: Implement this with the V-Ray material
        return 0.0
        return self.get_input(inp="reflection_roughness_constant")

    @reflection_roughness_constant.setter
    def reflection_roughness_constant(self, roughness):
        """
        Args:
             roughness (float): this material's applied reflection_roughness_constant
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="reflection_roughness_constant", val=roughness)

    @property
    def reflection_roughness_texture_influence(self):
        """
        Returns:
            float: this material's applied reflection_roughness_texture_influence
        """
        # TODO: Implement this with the V-Ray material
        return 0.0
        return self.get_input(inp="reflection_roughness_texture_influence")

    @reflection_roughness_texture_influence.setter
    def reflection_roughness_texture_influence(self, prop):
        """
        Args:
             prop (float): this material's applied reflection_roughness_texture_influence proportion
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="reflection_roughness_texture_influence", val=prop)

    @property
    def enable_emission(self):
        """
        Returns:
            bool: this material's applied enable_emission
        """
        # TODO: Implement this with the V-Ray material
        return False
        return self.get_input(inp="enable_emission")

    @enable_emission.setter
    def enable_emission(self, enabled):
        """
        Args:
             enabled (bool): this material's applied enable_emission
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="enable_emission", val=enabled)

    @property
    def emissive_color(self):
        """
        Returns:
            3-array: this material's applied (R,G,B) emissive_color
        """
        # TODO: Implement this with the V-Ray material
        return th.zeros(3)
        color = self.get_input(inp="emissive_color")
        return th.tensor(color, dtype=th.float32) if color is not None else None

    @emissive_color.setter
    def emissive_color(self, color):
        """
        Args:
             color (3-array): this material's applied emissive_color
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="emissive_color", val=lazy.pxr.Gf.Vec3f(*color))

    @property
    def emissive_intensity(self):
        """
        Returns:
            float: this material's applied emissive_intensity
        """
        # TODO: Implement this with the V-Ray material
        return 0.0
        return self.get_input(inp="emissive_intensity")

    @emissive_intensity.setter
    def emissive_intensity(self, intensity):
        """
        Args:
             intensity (float): this material's applied emissive_intensity
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="emissive_intensity", val=intensity)

    @property
    def opacity_constant(self):
        """
        Returns:
            float: this material's applied opacity_constant
        """
        # TODO: Implement this with the V-Ray material
        return 1.0
        return self.get_input(inp="opacity_constant")

    @opacity_constant.setter
    def opacity_constant(self, constant):
        """
        Args:
             constant (float): this material's applied opacity_constant
        """
        # TODO: Implement this with the V-Ray material
        return
        # TODO: another instance of missing config type checking
        self.set_input(inp="opacity_constant", val=float(constant))

    @property
    def diffuse_reflection_weight(self):
        """
        Returns:
            float: this material's applied diffuse_reflection_weight
        """
        # TODO: Implement this with the V-Ray material
        return 0.0
        return self.get_input(inp="diffuse_reflection_weight")

    @diffuse_reflection_weight.setter
    def diffuse_reflection_weight(self, weight):
        """
        Args:
             weight (float): this material's applied diffuse_reflection_weight
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="diffuse_reflection_weight", val=weight)

    @property
    def enable_specular_transmission(self):
        """
        Returns:
            bool: this material's applied enable_specular_transmission
        """
        # TODO: Implement this with the V-Ray material
        return False
        return self.get_input(inp="enable_specular_transmission")

    @enable_specular_transmission.setter
    def enable_specular_transmission(self, enabled):
        """
        Args:
             enabled (bool): this material's applied enable_specular_transmission
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="enable_specular_transmission", val=enabled)

    @property
    def specular_transmission_weight(self):
        """
        Returns:
            float: this material's applied specular_transmission_weight
        """
        # TODO: Implement this with the V-Ray material
        return 0.0
        return self.get_input(inp="specular_transmission_weight")

    @specular_transmission_weight.setter
    def specular_transmission_weight(self, weight):
        """
        Args:
             weight (float): this material's applied specular_transmission_weight
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="specular_transmission_weight", val=weight)

    @property
    def diffuse_reflection_color(self):
        """
        Returns:
            3-array: this material's diffuse_reflection_color in (R,G,B)
        """
        # TODO: Implement this with the V-Ray material
        return th.zeros(3)
        diffuse_reflection_color = self.get_input(inp="diffuse_reflection_color")
        return th.tensor(diffuse_reflection_color, dtype=th.float32) if diffuse_reflection_color is not None else None

    @diffuse_reflection_color.setter
    def diffuse_reflection_color(self, color):
        """
        Args:
             color (3-array): this material's diffuse_reflection_color in (R,G,B)
        """
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="diffuse_reflection_color", val=lazy.pxr.Gf.Vec3f(*color))

    @property
    def specular_transmission_color(self):
        """
        Returns:
            3-array: this material's specular_transmission_color in (R,G,B)
        """
        # TODO: Implement this with the V-Ray material
        return th.zeros(3)
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
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="specular_transmission_color", val=lazy.pxr.Gf.Vec3f(*color))

    @property
    def specular_transmission_scattering_color(self):
        """
        Returns:
            3-array: this material's specular_transmission_scattering_color in (R,G,B)
        """
        # TODO: Implement this with the V-Ray material
        return th.zeros(3)
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
        # TODO: Implement this with the V-Ray material
        return
        self.set_input(inp="specular_transmission_scattering_color", val=lazy.pxr.Gf.Vec3f(*color))

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
