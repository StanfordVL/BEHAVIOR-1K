# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omnigibson.utils.usd_utils import array_to_vtarray
from pxr import UsdPhysics, Gf
from pxr.Sdf import ValueTypeNames as VT

from omni.isaac.core.utils.stage import get_current_stage
from omni.physx.scripts import particleUtils

from omnigibson.macros import create_module_macros
from omnigibson.prims.geom_prim import GeomPrim
import omnigibson.utils.transform_utils as T

import numpy as np


# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Tune these default values!
m.CLOTH_STRETCH_STIFFNESS = 10000.0
m.CLOTH_BEND_STIFFNESS = 200.0
m.CLOTH_SHEAR_STIFFNESS = 100.0
m.CLOTH_DAMPING = 0.2


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

        particleUtils.add_physx_particle_cloth(
            stage=get_current_stage(),
            path=self._prim.GetPath(),
            dynamic_mesh_path=None,
            particle_system_path=f"/World/ClothSystem",
            spring_stretch_stiffness=m.CLOTH_STRETCH_STIFFNESS,
            spring_bend_stiffness=m.CLOTH_BEND_STIFFNESS,
            spring_shear_stiffness=m.CLOTH_SHEAR_STIFFNESS,
            spring_damping=m.CLOTH_DAMPING,
            self_collision=True,
            self_collision_filter=True,
        )

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
        r = T.quat2mat(self.get_orientation())
        t = self.get_position()
        s = self.scale

        p_local = np.array(self.get_attribute(attr="points"))
        p_world = (r @ (p_local * s).T).T + t

        return p_world

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

        r = T.quat2mat(self.get_orientation())
        t = self.get_position()
        s = self.scale

        p_local = (r.T @ (pos - t).T).T / s
        self.set_attribute(attr="points", val=array_to_vtarray(arr=p_local, element_type=Gf.Vec3f))

    def update_handles(self):
        # no handles to update
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
        # We have to read the mass directly in the cloth prim
        return self._mass_api.GetMassAttr().Get()

    @mass.setter
    def mass(self, mass):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        # We have to set the mass directly in the cloth prim
        self._mass_api.GetMassAttr().Set(mass)

    @property
    def density(self):
        raise NotImplementedError("Cannot get density for ClothPrim")

    @density.setter
    def density(self, density):
        raise NotImplementedError("Cannot set density for ClothPrim")

    def set_linear_velocity(self, velocity):
        # TODO (eric): Just a pass through for now.
        return

    def set_angular_velocity(self, velocity):
        # TODO (eric): Just a pass through for now.
        return

    def wake(self):
        # TODO (eric): Just a pass through for now.
        return
