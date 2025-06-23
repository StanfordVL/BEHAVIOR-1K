import string
from abc import ABC

import omnigibson.lazy as lazy
from omnigibson.utils.python_utils import Recreatable, Serializable
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import (
    delete_or_deactivate_prim,
    scene_relative_prim_path_to_absolute,
    get_sdf_value_type_name,
    activate_prim_and_children,
)

# Create module logger
log = create_module_logger(module_name=__name__)


class BasePrim(Serializable, Recreatable, ABC):
    """
    Provides high level functions to deal with a basic prim and its attributes/ properties.
    If there is an Xform prim present at the path, it will use it. Otherwise, a new XForm prim at
    the specified prim path will be created.

    Note: the prim will have "xformOp:orient", "xformOp:translate" and "xformOp:scale" only post init,
        unless it is a non-root articulation link.

    Args:
        relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. Note that this is only needed if the prim does not already exist at
            @relative_prim_path -- it will be ignored if it already exists. Subclasses should define the exact keys expected
            for their class.
    """

    def __init__(
        self,
        relative_prim_path,
        name,
        load_config=None,
    ):
        self._relative_prim_path = relative_prim_path
        assert relative_prim_path.startswith("/"), f"Relative prim path {relative_prim_path} must start with a '/'!"
        assert all(
            component[0] in string.ascii_letters for component in relative_prim_path[1:].split("/")
        ), f"Each component of relative prim path {relative_prim_path} must start with a letter!"

        self._name = name
        self._load_config = dict() if load_config is None else load_config

        # Other values that will be filled in at runtime
        self._scene = None
        self._scene_assigned = False
        self._applied_visual_material = None
        self._loaded = False  # Whether this prim exists in the stage or not
        self._initialized = False  # Whether this prim has its internal handles / info initialized or not (occurs AFTER and INDEPENDENTLY from loading!)
        self._prim = None
        self._n_duplicates = 0  # Simple counter for keeping track of duplicates for unique name indexing

        # Check if this prim was created manually. This member will be automatically set for prims
        # that get created during the _load phase of this class, but sometimes we create prims using
        # alternative methods and then create this class - in that case too we need to make sure we
        # add the right xform properties, so callers will just pass in the created manually flag.
        self._xform_props_pre_loaded = self._load_config.get("xform_props_pre_loaded", False)
        # Run super init
        super().__init__()

    def _initialize(self):
        """
        Initializes state of this object and sets up any references necessary post-loading. Should be implemented by
        sub-class for extended utility
        """
        pass

    def initialize(self):
        """
        Initializes state of this object and sets up any references necessary post-loading. Subclasses should
        implement / extend the _initialize() method.
        """
        assert (
            not self._initialized
        ), f"Prim {self.name} at prim_path {self.prim_path} can only be initialized once! (It is already initialized)"
        self._initialize()

        self._initialized = True

        # Cache state size (note that we are doing this after initialized is set to True because
        # dump_state asserts that the prim is initialized for some prims).
        self._state_size = len(self.dump_state(serialized=True))

    def load(self, scene):
        """
        Load this prim into omniverse, and return loaded prim reference.

        Returns:
            Usd.Prim: Prim object loaded into the simulator
        """
        # Load the prim if it doesn't exist yet.
        assert (
            not self._loaded
        ), f"Prim {self.name} at prim_path {self.prim_path} can only be loaded once! (It is already loaded)"

        # Assign the scene first.
        self._scene = scene
        self._scene_assigned = True

        # Check if the prim path exists in the stage
        if lazy.isaacsim.core.utils.prims.is_prim_path_valid(prim_path=self.prim_path):
            existing_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path=self.prim_path)

            # Note: A prim path can be valid but the prim itself may be inactive.
            # This commonly occurs after transition rules when scene prims get deleted -
            # the prim paths remain valid but the prims become inactive.
            # In such cases, we need to activate the prim to make it usable again.
            if not existing_prim.IsActive():
                log.debug(f"Prim '{self.name}' exists but is inactive/invalid, activating it")
                # Recursively find all prims under it, even those that are deactivated
                activate_prim_and_children(self.prim_path)
            self._prim = existing_prim
        else:
            # Prim path doesn't exist - load it for the first time
            log.debug(f"Prim '{self.name}' doesn't exist, loading")
            self._prim = self._load()

        # Mark the prim as loaded.
        self._loaded = True

        # Run any post-loading logic
        self._post_load()

        return self._prim

    def _post_load(self):
        """
        Any actions that should be taken (e.g.: modifying the object's properties such as scale, visibility, additional
        joints, etc.) that should be taken after loading the raw object into omniverse but BEFORE we initialize the
        object and grab its handles and internal references. By default, this is a no-op.
        """
        pass

    def remove(self):
        """
        Removes this prim from omniverse stage.
        """
        if not self._loaded:
            raise ValueError("Cannot remove a prim that was never loaded.")

        # Remove or deactivate prim if it's possible
        if not delete_or_deactivate_prim(self.prim_path):
            log.warning(f"Prim {self.name} at prim_path {self.prim_path} could not be deleted or deactivated.")

    def _load(self):
        """
        Loads the raw prim into the simulator. Any post-processing should be done in @self._post_load()
        """
        raise NotImplementedError()

    @property
    def loaded(self):
        return self._loaded

    @property
    def initialized(self):
        return self._initialized

    @property
    def scene(self):
        """
        Returns:
            Scene or None: Scene object that this prim is loaded into
        """
        assert self._scene_assigned, "Scene has not been assigned to this prim yet!"
        return self._scene

    @property
    def state_size(self):
        # This is the cached value
        return self._state_size

    @property
    def prim_path(self):
        """
        Returns:
            str: prim path in the stage.
        """
        return scene_relative_prim_path_to_absolute(self.scene, self._relative_prim_path)

    @property
    def name(self):
        """
        Returns:
            str: unique name assigned to this prim
        """
        return self._name

    @property
    def prim(self):
        """
        Returns:
            Usd.Prim: USD Prim object that this object holds.
        """
        return self._prim

    @property
    def property_names(self):
        """
        Returns:
            set of str: Set of property names that this prim has (e.g.: visibility, proxyPrim, etc.)
        """
        return set(self._prim.GetPropertyNames())

    @property
    def visible(self):
        """
        Returns:
            bool: true if the prim is visible in stage. false otherwise.
        """
        return (
            lazy.pxr.UsdGeom.Imageable(self.prim).ComputeVisibility(lazy.pxr.Usd.TimeCode.Default())
            != lazy.pxr.UsdGeom.Tokens.invisible
        )

    @visible.setter
    def visible(self, visible):
        """
        Sets the visibility of the prim in stage.

        Args:
            visible (bool): flag to set the visibility of the usd prim in stage.
        """
        imageable = lazy.pxr.UsdGeom.Imageable(self.prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
        return

    def is_valid(self):
        """
        Returns:
            bool: True is the current prim path corresponds to a valid prim in stage. False otherwise.
        """
        return lazy.isaacsim.core.utils.prims.is_prim_path_valid(self.prim_path)

    def is_attribute_valid(self, attr):
        """
        Check if the attribute is valid for this prim.

        Args:
            attr (str): Attribute to check

        Returns:
            bool: True if the attribute is valid for this prim. False otherwise.
        """
        return self._prim.GetAttribute(attr).IsValid()

    def get_attribute(self, attr):
        """
        Get this prim's attribute. Should be a valid attribute under self._prim.GetAttributes()

        Returns:
            any: value of the requested @attribute
        """
        return self._prim.GetAttribute(attr).Get()

    def set_attribute(self, attr, val):
        """
        Set this prim's attribute. Should be a valid attribute under self._prim.GetAttributes()

        Args:
            attr (str): Attribute to set
            val (any): Value to set for the attribute. This should be the valid type for that attribute.
        """
        self._prim.GetAttribute(attr).Set(val)

    def create_attribute(self, attr, val):
        """
        Create a new attribute for this prim. Should be a valid attribute under self._prim.GetAttributes()

        Args:
            attr (str): Attribute to create
            val (any): Value to set for the attribute. This should be the valid type for that attribute.
        """
        self._prim.CreateAttribute(attr, get_sdf_value_type_name(val))

    def get_property(self, prop):
        """
        Sets property @prop with value @val

        Args:
            prop (str): Name of the property to get. See Raw USD Properties in the GUI for examples of property names

        Returns:
            any: Property value
        """
        self._prim.GetProperty(prop).Get()

    def set_property(self, prop, val):
        """
        Sets property @prop with value @val

        Args:
            prop (str): Name of the property to set. See Raw USD Properties in the GUI for examples of property names
            val (any): Value to set for the property. Should be valid for that property
        """
        self._prim.GetProperty(prop).Set(val)

    def get_custom_data(self):
        """
        Get custom data associated with this prim

        Returns:
            dict: Dictionary of any custom information
        """
        return self._prim.GetCustomData()

    def _create_prim_with_same_kwargs(self, relative_prim_path, name, load_config):
        """
        Generates a new instance of this prim's class with specified @relative_prim_path, @name, and @load_config, but otherwise
        all other kwargs should be identical to this instance's values.

        Args:
            relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
            name (str): Name for the newly created prim
            load_config (dict): Keyword-mapped kwargs to use to set specific attributes for the created prim's instance

        Returns:
            BasePrim: Generated prim object (not loaded, and not initialized!)
        """
        return self.__class__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )
