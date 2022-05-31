import logging
import numpy as np
from igibson.objects.stateful_object import StatefulObject

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path


class USDObject(StatefulObject):
    """
    USDObjects are instantiated from a USD file. They can be composed of one
    or more links and joints. They may or may not be passive.
    """

    def __init__(
        self,
        prim_path,
        usd_path,
        name=None,
        category="object",
        class_id=None,
        scale=None,
        rendering_params=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        load_config=None,
        abilities=None,
        **kwargs,
    ):
        """
        @param prim_path: str, global path in the stage to this object
        @param usd_path: str, global path to the USD file to load
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
        self_collisions (bool): Whether to enable self collisions for this object
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
        @param abilities: dict in the form of {ability: {param: value}} containing
            object abilities and parameters.
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        super().__init__(
            prim_path=prim_path,
            name=name,
            category=category,
            class_id=class_id,
            scale=scale,
            rendering_params=rendering_params,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            **kwargs,
        )
        self._usd_path = usd_path

    def _load(self, simulator=None):
        """
        Load the object into pybullet and set it to the correct pose
        """
        logging.info(f"Loading the following USD: {self._usd_path}")
        # Add reference to stage and grab prim
        add_reference_to_stage(usd_path=self._usd_path, prim_path=self._prim_path)
        prim = get_prim_at_path(self._prim_path)

        return prim

    def _create_prim_with_same_kwargs(self, prim_path, name, load_config):
        # Add additional kwargs
        return self.__class__(
            prim_path=prim_path,
            usd_path=self._usd_path,
            name=name,
            category=self.category,
            class_id=self.class_id,
            scale=self.scale,
            rendering_params=self.rendering_params,
            visible=self.visible,
            fixed_base=self.fixed_base,
            visual_only=self._visual_only,
            load_config=load_config,
            abilities=self._abilities,
        )

    @property
    def bbox(self):
        # Override this function to pull directly from the native_bbox property if it exists
        return super().bbox if self.native_bbox is not None else self.native_bbox * self.scale

    @property
    def native_bbox(self):
        """
        Get this object's native bounding box

        Returns:
            None or 3-array: (x,y,z) bounding box if it exists, else None
        """
        return np.array(self.get_attribute(attr="ig:nativeBB")) if "ig:nativeBB" in self.property_names else None
