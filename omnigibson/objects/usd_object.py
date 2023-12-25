import os
import tempfile

import omnigibson as og
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.utils.constants import PrimType
from omnigibson.utils.usd_utils import add_asset_to_stage
from omnigibson.utils.asset_utils import decrypt_file


class USDObject(StatefulObject):
    """
    USDObjects are instantiated from a USD file. They can be composed of one
    or more links and joints. They may or may not be passive.
    """

    def __init__(
        self,
        name,
        usd_path,
        encrypted=False,
        prim_path=None,
        category="object",
        class_id=None,
        uuid=None,
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        kinematic_only=None,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        load_config=None,
        abilities=None,
        include_default_states=True,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            usd_path (str): global path to the USD file to load
            encrypted (bool): whether this file is encrypted (and should therefore be decrypted) or not
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            category (str): Category for the object. Defaults to "object".
            class_id (None or int): What class ID the object should be assigned in semantic segmentation rendering mode.
                If None, the ID will be inferred from this object's category.
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            kinematic_only (None or bool): Whether this object should be kinematic only (and not get affected by any
                collisions). If None, then this value will be set to True if @fixed_base is True and some other criteria
                are satisfied (see object_base.py post_load function), else False.
            self_collisions (bool): Whether to enable self collisions for this object
            prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            include_default_states (bool): whether to include the default object states from @get_default_states
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
                Note that this base object does NOT pass kwargs down into the Prim-type super() classes, and we assume
                that kwargs are only shared between all SUBclasses (children), not SUPERclasses (parents).
        """
        self._usd_path = usd_path
        self._encrypted = encrypted
        super().__init__(
            prim_path=prim_path,
            name=name,
            category=category,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            kinematic_only=kinematic_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            include_default_states=include_default_states,
            load_config=load_config,
            abilities=abilities,
            **kwargs,
        )

    def _load(self):
        """
        Load the object into pybullet and set it to the correct pose
        """
        usd_path = self._usd_path
        if self._encrypted:
            # Create a temporary file to store the decrytped asset, load it, and then delete it
            encrypted_filename = self._usd_path.replace(".usd", ".encrypted.usd")
            decrypted_fd, usd_path = tempfile.mkstemp(os.path.basename(self._usd_path), dir=og.tempdir)
            decrypt_file(encrypted_filename, usd_path)

        prim = add_asset_to_stage(asset_path=usd_path, prim_path=self._prim_path)

        if self._encrypted:
            os.close(decrypted_fd)
            # On Windows, Isaac Sim won't let go of the file until the prim is removed, so we can't delete it.
            if os.name == "posix":
                os.remove(usd_path)

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
            visible=self.visible,
            fixed_base=self.fixed_base,
            visual_only=self._visual_only,
            prim_type=self._prim_type,
            load_config=load_config,
            abilities=self._abilities,
        )

    @property
    def usd_path(self):
        """
        Returns:
            str: absolute path to this model's USD file. By default, this is the loaded usd path
                passed in as an argument
        """
        return self._usd_path
