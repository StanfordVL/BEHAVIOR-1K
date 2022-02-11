import logging
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
        load_config=None,
        abilities=None,
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
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
        @param abilities: dict in the form of {ability: {param: value}} containing
            object abilities and parameters.
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
            load_config=load_config,
            abilities=abilities,
        )
        self._usd_path = usd_path

        logging.info(f"Category: {self.category}")

    def _load(self, simulator=None):
        """
        Load the object into pybullet and set it to the correct pose
        """
        logging.info(f"Loading the following USD: {self._usd_path}")
        # Add reference to stage and grab prim
        add_reference_to_stage(usd_path=self._usd_path, prim_path=self._prim_path)
        prim = get_prim_at_path(self._prim_path)

        return prim
