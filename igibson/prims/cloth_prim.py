# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from igibson.utils.usd_utils import array_to_vtarray
from pxr import UsdPhysics, Gf
from pxr.Sdf import ValueTypeNames as VT

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.usd import get_shader_from_material
from omni.physx.scripts import particleUtils

from igibson.prims.geom_prim import GeomPrim
from igibson.utils.usd_utils import update_shader_asset_paths
import numpy as np


class ClothPrim(GeomPrim):
    """
    Provides high level functions to deal with a cloth prim and its attributes/ properties.
    If there is an prim present at the path, it will use it. Otherwise, a new XForm prim at
    the specified prim path will be created.

    Notes: if the prim does not already have a cloth api applied to it before it is loaded,
        it will apply it.

    Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str): Name for the object. Names need to be unique per scene.
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime. Note that this is only needed if the prim does not already exist at
                @prim_path -- it will be ignored if it already exists. For this joint prim, the below values can be
                specified:

                scale (None or float or 3-array): If specified, sets the scale for this object. A single number corresponds
                    to uniform scaling along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
                mass (None or float): If specified, mass of this body in kg
    """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):
        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # run super first
        super()._post_load()

        self._mass_api = UsdPhysics.MassAPI(self._prim) if self._prim.HasAPI(UsdPhysics.MassAPI) else \
            UsdPhysics.MassAPI.Apply(self._prim)

        # Possibly set the mass / density
        if "mass" in self._load_config and self._load_config["mass"] is not None:
            self.mass = self._load_config["mass"]

        # TODO (eric): customize stiffness
        stretch_stiffness = 10000.0
        bend_stiffness = 200.0
        shear_stiffness = 100.0
        damping = 0.2
        particleUtils.add_physx_particle_cloth(
            stage=get_current_stage(),
            path=self._prim.GetPath(),
            dynamic_mesh_path=None,
            particle_system_path=f"/World/ClothSystem",
            spring_stretch_stiffness=stretch_stiffness,
            spring_bend_stiffness=bend_stiffness,
            spring_shear_stiffness=shear_stiffness,
            spring_damping=damping,
            self_collision=True,
            self_collision_filter=True,
        )

        # Possibly update the shader associated with this geom if specified
        # This may need to happen if we're initializing a geom on a machine that was NOT the original
        # machine that loaded the USD
        # This is an omni limitation that we need to work around
        if "material:binding" in self._prim.GetPropertyNames():
            for target in self._prim.GetProperty("material:binding").GetTargets():
                shader = get_shader_from_material(prim=get_prim_at_path(target))
                update_shader_asset_paths(shader=shader)



    def _initialize(self):
        super()._initialize()
        # TODO (eric): hacky way to get cloth rendering to work (otherwise, there exist some rendering artifacts).
        self._prim.CreateAttribute("primvars:isVolume", VT.Bool, False).Set(True)
        self._prim.GetAttribute("primvars:isVolume").Set(False)

    @property
    def particle_positions(self):
        """
        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
        """
        return np.array(self.get_attribute(attr="points")) + self.get_position()

    @particle_positions.setter
    def particle_positions(self, pos):
        """
        Set the particle positions for this instancer

        Args:
            np.array: (N, 3) numpy array, where each of the N particles' desired positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
        """
        assert pos.shape[0] == self.particle_positions.shape[0], \
            f"Got mismatch in particle setting size: {pos.shape[0]}, vs. number of particles {self.particle_positions.shape[0]}!"
        pos = (pos - self.get_position()).astype(float)
        self.set_attribute(attr="points", val=array_to_vtarray(arr=pos, element_type=Gf.Vec3f))

    def update_handles(self):
        """
        Updates all internal handles for this prim, in case they change since initialization
        """
        pass

    @property
    def volume(self):
        raise NotImplementedError("Cannot get volume for ClothPrim")

    @volume.setter
    def volume(self, volume):
        raise NotImplementedError("Cannot set volume for ClothPrim")

    @property
    def mass(self):
        """
        Returns:
            float: mass of the rigid body in kg.
        """
        return self._mass_api.GetMassAttr().Get()

    @mass.setter
    def mass(self, mass):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        self._mass_api.GetMassAttr().Set(mass)

    @property
    def density(self):
        raise NotImplementedError("Cannot get density for ClothPrim")

    @density.setter
    def density(self, density):
        raise NotImplementedError("Cannot set density for ClothPrim")

    def set_linear_velocity(self, velocity):
        """Sets the linear velocity of the prim in stage.

        Args:
            velocity (np.ndarray): linear velocity to set the rigid prim to. Shape (3,).
        """
        # TODO (eric): Just a pass through for now.
        return

    def set_angular_velocity(self, velocity):
        """Sets the angular velocity of the prim in stage.

        Args:
            velocity (np.ndarray): angular velocity to set the rigid prim to. Shape (3,).
        """
        # TODO (eric): Just a pass through for now.
        return


    def wake(self):
        """
        Enable physics for this rigid body
        """
        # TODO (eric): Just a pass through for now.
        return