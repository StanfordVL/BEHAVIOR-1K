import cv2
from omnigibson.utils.ui_utils import create_module_logger
import torch as th

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.prims.prim_base import BasePrim
from omnigibson.utils.physx_utils import bind_material
from omnigibson.utils.usd_utils import absolute_prim_path_to_scene_relative, get_sdf_value_type_name


log = create_module_logger(module_name=__name__)


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
    def get_material(cls, scene, name, prim_path, **kwargs):
        """
        Get a material prim from the persistent dictionary of materials, or create a new one if it doesn't exist.

        Args:
            scene (Scene): Scene to which this material belongs.
            name (str): Name for the object.
            prim_path (str): prim path of the MaterialPrim.
            **kwargs (dict): Additional keyword arguments to pass to the MaterialPrim or subclass constructor.

        Returns:
            MaterialPrim: Material prim at the specified path
        """
        # If the material already exists, return it
        if prim_path in MaterialPrim.MATERIALS:
            return MaterialPrim.MATERIALS[prim_path]

        # Otherwise, create a new one and return it
        material_class = cls
        if lazy.isaacsim.core.utils.prims.is_prim_path_valid(prim_path):
            # If the prim already exists, infer its type.
            material_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path)
            shader_prim = lazy.omni.usd.get_shader_from_material(material_prim)
            assert shader_prim is not None, (
                f"Material prim at {prim_path} exists, but does not have a shader associated with it! "
                f"Please make sure the material is created correctly."
            )

            # If we are able to obtain the asset path and sub-identifier, we can determine the material class.
            mdl_asset = shader_prim.GetSourceAsset("mdl")
            if mdl_asset is not None:
                asset_path = mdl_asset.path
                asset_sub_identifier = shader_prim.GetSourceAssetSubIdentifier("mdl")

                if cls == MaterialPrim:
                    # If this function is getting called on MaterialPrim, then we try to pick a compatible subclass.
                    compatible_classes = [
                        subclass
                        for subclass in MaterialPrim.__subclasses__()
                        if subclass.supports_material(asset_path, asset_sub_identifier)
                    ]
                    assert len(compatible_classes) <= 1, (
                        "Found multiple compatible material prim classes for "
                        f"material at {prim_path} with asset path {asset_path} and sub-identifier {asset_sub_identifier}: "
                        f"{compatible_classes}"
                    )
                    if len(compatible_classes) == 1:
                        # Use the only found compatible class
                        material_class = compatible_classes[0]
                    else:
                        # Fall back to MaterialPrim
                        log.warning(
                            f"No compatible material prim class found for material at {prim_path} with "
                            f"asset path {asset_path} and sub-identifier {asset_sub_identifier}. "
                            f"Using MaterialPrim as a fallback."
                        )
                else:
                    # If this function is called on a subclass of MaterialPrim, then we check if the subclass supports the material.
                    assert material_class.supports_material(asset_path, asset_sub_identifier), (
                        f"MaterialPrim subclass {material_class.__name__} does not support material at {prim_path} "
                        f"with asset path {asset_path} and sub-identifier {asset_sub_identifier}!"
                    )
            else:
                # If the material prim exists, but does not have a shader file we can recognize.
                log.warning(
                    f"Material prim at {prim_path} exists, but does not have a known shader file associated with it! "
                    f"Using MaterialPrim as a fallback. If this is not intended, please make sure the material is created correctly."
                )

        relative_prim_path = absolute_prim_path_to_scene_relative(scene, prim_path)
        new_material = material_class(relative_prim_path=relative_prim_path, name=name, **kwargs)
        new_material.load(scene)
        assert (
            new_material.prim_path == prim_path
        ), f"Material prim path {new_material.prim_path} does not match {prim_path}"
        MaterialPrim.MATERIALS[prim_path] = new_material
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

    @classmethod
    def supports_material(cls, asset_path, asset_sub_identifier):
        """
        Checks if this material prim supports the given asset path and sub-identifier.

        Args:
            asset_path (str): The asset path of the MDL file.
            asset_sub_identifier (str): The sub-identifier of the MDL file.

        Returns:
            bool: True if this material prim supports the given asset path and sub-identifier, False otherwise.
        """
        raise NotImplementedError(f"MaterialPrim subclass {cls.__name__} does not implement supports_material method!")

    @property
    def mdl_name(self):
        """
        The name of the MDL file to load for this material.
        This is expected to be defined in the load_config dictionary for this base class which is used
        for generic material prims. The PBR and V-Ray material prims will override this method
        to return their own MDL names.
        """
        assert "mdl_name" in self._load_config, (
            f"MaterialPrim at {self.prim_path} does not have a 'mdl_name' in its load_config! "
            f"Please specify it in the load_config dictionary or use one of the subclasses that "
            f"already define it (e.g. OmniPBRMaterialPrim, VRayMaterialPrim, OmniGlassMaterialPrim, OmniSurfaceMaterialPrim)"
        )
        return self._load_config["mdl_name"]

    @property
    def mtl_name(self):
        """
        The name of the material from the MDL file to load for this material.
        This is expected to be defined in the load_config dictionary for this base class which is used
        for generic material prims. The PBR and V-Ray material prims will override this method
        to return their own MDL names.
        """
        assert "mtl_name" in self._load_config, (
            f"MaterialPrim at {self.prim_path} does not have a 'mtl_name' in its load_config! "
            f"Please specify it in the load_config dictionary or use one of the subclasses that "
            f"already define it (e.g. OmniPBRMaterialPrim, VRayMaterialPrim, OmniGlassMaterialPrim)"
        )
        return self._load_config["mtl_name"]

    def _load(self):
        # We create a new material at the specified path
        mtl_created = []
        lazy.omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name=self.mdl_name,
            mtl_name=self.mtl_name,
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
    def average_diffuse_color(self):
        return th.zeros(3)

    @property
    def albedo_add(self):
        """
        Returns:
            float: this material's applied albedo_add
        """
        return 0.0

    @albedo_add.setter
    def albedo_add(self, add):
        """
        Args:
             add (float): this material's applied albedo_add
        """
        return

    @property
    def diffuse_tint(self):
        """
        Returns:
            3-array: this material's applied (R,G,B) diffuse_tint
        """
        return th.zeros(3)

    @diffuse_tint.setter
    def diffuse_tint(self, color):
        """
        Args:
             color (3-array): this material's applied (R,G,B) diffuse_tint
        """
        return

    def enable_highlight(self, highlight_color, highlight_intensity):
        """
        Enables highlight for this material with the specified color and intensity.

        Args:
            highlight_color (3-array): Color of the highlight in (R,G,B)
            highlight_intensity (float): Intensity of the highlight
        """
        pass

    def disable_highlight(self):
        """
        Disables highlight for this material.
        """
        pass


class OmniPBRMaterialPrim(MaterialPrim):
    """
    A MaterialPrim that uses the OmniPBR material preset.
    """

    @classmethod
    def supports_material(cls, asset_path, asset_sub_identifier):
        return asset_path.endswith("OmniPBR.mdl") and asset_sub_identifier == "OmniPBR"

    @property
    def mdl_name(self):
        return "OmniPBR.mdl"

    @property
    def mtl_name(self):
        return "OmniPBR"

    def _post_load(self):
        # Run super first
        super()._post_load()

        # Apply any forced roughness updates
        self.set_input(inp="reflection_roughness_texture_influence", val=0.0)
        self.set_input(inp="reflection_roughness_constant", val=gm.FORCE_ROUGHNESS)

        # TODO: Check that it does not use emission by default.

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
        return self.get_input(inp="albedo_add")

    @albedo_add.setter
    def albedo_add(self, add):
        """
        Args:
             add (float): this material's applied albedo_add
        """
        self.set_input(inp="albedo_add", val=add)

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
    def average_diffuse_color(self):
        diffuse_texture = self.diffuse_texture
        return cv2.imread(diffuse_texture).mean(axis=(0, 1)) if diffuse_texture else self.diffuse_color_constant

    def enable_highlight(self, highlight_color, highlight_intensity):
        """
        Enables highlight for this material with the specified color and intensity.

        Args:
            highlight_color (3-array): Color of the highlight in (R,G,B)
            highlight_intensity (float): Intensity of the highlight
        """
        # Set emissive properties to enable highlight
        self.set_input(inp="enable_emission", val=True)
        self.set_input(inp="emissive_color", val=lazy.pxr.Gf.Vec3f(*highlight_color))
        self.set_input(inp="emissive_intensity", val=highlight_intensity)

    def disable_highlight(self):
        self.set_input(inp="enable_emission", val=False)


class VRayMaterialPrim(MaterialPrim):
    """
    A MaterialPrim that uses the V-Ray material preset.
    """

    @classmethod
    def supports_material(cls, asset_path, asset_sub_identifier):
        return asset_path.endswith("omnigibson_vray_mtl.mdl") and asset_sub_identifier == "OmniGibsonVRayMtl"

    @property
    def mdl_name(self):
        return "omnigibson_vray_mtl.mdl"

    @property
    def mtl_name(self):
        return "OmniGibsonVRayMtl"

    @property
    def diffuse_texture(self):
        """
        Returns:
            str: this material's applied diffuse_texture filepath
        """
        return self.get_input(inp="diffuse_texture").resolvedPath

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
    def average_diffuse_color(self):
        """
        Returns:
            3-array: this material's average diffuse color.
        """
        return cv2.imread(self.diffuse_texture).mean(axis=(0, 1))

    def enable_highlight(self, highlight_color, highlight_intensity):
        """
        Enables highlight for this material with the specified color and intensity.

        Args:
            highlight_color (3-array): Color of the highlight in (R,G,B)
            highlight_intensity (float): Intensity of the highlight
        """
        # Set emissive properties to enable highlight
        self.set_input(inp="emission_color", val=lazy.pxr.Gf.Vec3f(*highlight_color))
        self.set_input(inp="emission_intensity", val=highlight_intensity)

    def disable_highlight(self):
        self.set_input(inp="emission_color", val=lazy.pxr.Gf.Vec3f(0, 0, 0))
        self.set_input(inp="emission_intensity", val=0.0)


class OmniGlassMaterialPrim(MaterialPrim):
    """
    A MaterialPrim that uses the OmniGlass material preset.
    """

    @classmethod
    def supports_material(cls, asset_path, asset_sub_identifier):
        return asset_path.endswith("OmniGlass.mdl") and asset_sub_identifier == "OmniGlass"

    @property
    def mdl_name(self):
        return "OmniGlass.mdl"

    @property
    def mtl_name(self):
        return "OmniGlass"

    @property
    def color(self):
        """
        Returns:
            3-array: this material's applied (R,G,B) glass color (only applicable to OmniGlass materials)
        """
        glass_color = self.get_input(inp="glass_color")
        return th.tensor(glass_color, dtype=th.float32) if glass_color is not None else None

    @color.setter
    def color(self, color):
        """
        Args:
             color (3-array): this material's applied (R,G,B) glass color (only applicable to OmniGlass materials)
        """
        self.set_input(inp="glass_color", val=lazy.pxr.Gf.Vec3f(*color))

    @property
    def average_diffuse_color(self):
        """
        Returns:
            3-array: this material's average diffuse color - we pretend this is the same as the glass color.
        """
        return self.color


class OmniSurfaceMaterialPrim(MaterialPrim):
    """
    A MaterialPrim that uses the OmniSurface material preset.
    """

    def __init__(self, relative_prim_path, name, preset_name, load_config=None):
        """
        Args:
            relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
            name (str): Name for the object. Names need to be unique per scene.
            preset_name (str): Name of the preset to use for this material. If None, defaults to "OmniSurface".
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime. Note that this is only needed if the prim does not already exist at
                @relative_prim_path -- it will be ignored if it already exists.
        """
        self.preset_name = preset_name
        super().__init__(relative_prim_path=relative_prim_path, name=name, load_config=load_config)

    @classmethod
    def supports_material(cls, asset_path, asset_sub_identifier):
        return (asset_path.endswith("OmniSurface.mdl") and asset_sub_identifier == "OmniSurface") or (
            asset_path.endswith("OmniSurfacePresets.mdl") and asset_sub_identifier.startswith("OmniSurface_")
        )

    @property
    def mdl_name(self):
        return "OmniSurfacePresets.mdl" if self.preset_name else "OmniSurface.mdl"

    @property
    def mtl_name(self):
        return f"OmniSurface_{self.preset_name}" if self.preset_name else "OmniSurface"

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
    def specular_transmission_color(self):
        """
        Returns:
            3-array: this material's specular_transmission_color in (R,G,B)
        """
        specular_transmission_color = self.get_input(inp="specular_transmission_color")
        return th.tensor(specular_transmission_color, dtype=th.float32)

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
        return th.tensor(specular_transmission_scattering_color, dtype=th.float32)

    @specular_transmission_scattering_color.setter
    def specular_transmission_scattering_color(self, color):
        """
        Args:
             color (3-array): this material's specular_transmission_scattering_color in (R,G,B)
        """
        self.set_input(inp="specular_transmission_scattering_color", val=lazy.pxr.Gf.Vec3f(*color))

    @property
    def average_diffuse_color(self):
        base_color_weight = self.diffuse_reflection_weight
        transmission_weight = self.enable_specular_transmission * self.specular_transmission_weight
        total_weight = base_color_weight + transmission_weight

        # If the fluid doesn't have any color, we add a "blue" tint by default
        if total_weight == 0.0:
            return th.tensor([0.0, 0.0, 1.0])

        base_color_weight /= total_weight
        transmission_weight /= total_weight
        # Weighted sum of base color and transmission color
        return base_color_weight * self.diffuse_reflection_color + transmission_weight * (
            0.5 * self.specular_transmission_color + 0.5 * self.specular_transmission_scattering_color
        )
