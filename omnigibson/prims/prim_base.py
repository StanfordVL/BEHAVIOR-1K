from abc import ABC, abstractmethod
from pxr import Gf, Usd, UsdGeom, UsdShade
from omni.isaac.core.utils.prims import (
    get_prim_at_path,
    move_prim,
    query_parent_path,
    is_prim_path_valid,
    define_prim,
    get_prim_parent,
    get_prim_object_type,
)
import omnigibson as og
from omni.isaac.core.utils.prims import delete_prim
from omnigibson.utils.python_utils import Serializable, UniquelyNamed, Recreatable
from omnigibson.utils.sim_utils import check_deletable_prim
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


class BasePrim(Serializable, UniquelyNamed, Recreatable, ABC):
    """
    Provides high level functions to deal with a basic prim and its attributes/ properties.
    If there is an Xform prim present at the path, it will use it. Otherwise, a new XForm prim at
    the specified prim path will be created.

    Note: the prim will have "xformOp:orient", "xformOp:translate" and "xformOp:scale" only post init,
        unless it is a non-root articulation link.

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. Note that this is only needed if the prim does not already exist at
            @prim_path -- it will be ignored if it already exists. Subclasses should define the exact keys expected
            for their class.
    """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):
        self._prim_path = prim_path
        self._name = name
        self._load_config = dict() if load_config is None else load_config

        # Other values that will be filled in at runtime
        self._applied_visual_material = None
        self._loaded = False                                # Whether this prim exists in the stage or not
        self._initialized = False                           # Whether this prim has its internal handles / info initialized or not (occurs AFTER and INDEPENDENTLY from loading!)
        self._prim = None
        self._state_size = None
        self._n_duplicates = 0                              # Simple counter for keeping track of duplicates for unique name indexing

        # Run super init
        super().__init__()

        # Run some post-loading steps if this prim has already been loaded
        if is_prim_path_valid(prim_path=self._prim_path):
            log.debug(f"prim {name} already exists, skipping load")
            self._prim = get_prim_at_path(prim_path=self._prim_path)
            self._loaded = True
            # Run post load.
            self._post_load()

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
        assert not self._initialized, \
            f"Prim {self.name} at prim_path {self._prim_path} can only be initialized once! (It is already initialized)"
        self._initialize()

        # Cache state size
        self._state_size = len(self.dump_state(serialized=True))

        self._initialized = True

    def load(self):
        """
        Load this prim into omniverse, and return loaded prim reference.

        Returns:
            Usd.Prim: Prim object loaded into the simulator
        """
        if self._loaded:
            raise ValueError(f"Cannot load prim {self.name} multiple times.")

        # Load prim
        self._prim = self._load()
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
        Removes this prim from omniverse stage
        """
        if not self._loaded:
            raise ValueError("Cannot remove a prim that was never loaded.")

        # Remove prim if it can be deleted
        if check_deletable_prim(self.prim_path):
            delete_prim(self.prim_path)

        # Also clear the name so we can reuse this later
        self.remove_names(include_all_owned=True)

    @abstractmethod
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
    def state_size(self):
        # This is the cached value
        return self._state_size

    @property
    def prim_path(self):
        """
        Returns:
            str: prim path in the stage.
        """
        return self._prim_path

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
        return UsdGeom.Imageable(self.prim).ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible

    @visible.setter
    def visible(self, visible):
        """
        Sets the visibility of the prim in stage.

        Args:
            visible (bool): flag to set the visibility of the usd prim in stage.
        """
        imageable = UsdGeom.Imageable(self.prim)
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
        return is_prim_path_valid(self.prim_path)

    def change_prim_path(self, new_prim_path):
        """
        Moves prim from the old path to a new one.

        Args:
            new_prim_path (str): new path of the prim to be moved to.
        """
        move_prim(path_from=self.prim_path, path_to=new_prim_path)
        self._prim_path = new_prim_path
        self._prim = get_prim_at_path(self._prim_path)
        return

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

    def _create_prim_with_same_kwargs(self, prim_path, name, load_config):
        """
        Generates a new instance of this prim's class with specified @prim_path, @name, and @load_config, but otherwise
        all other kwargs should be identical to this instance's values.

        Args:
            prim_path (str): Absolute path to the newly generated prim
            name (str): Name for the newly created prim
            load_config (dict): Keyword-mapped kwargs to use to set specific attributes for the created prim's instance

        Returns:
            BasePrim: Generated prim object (not loaded, and not initialized!)
        """
        return self.__class__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def duplicate(self, prim_path):
        """
        Duplicates this object, and generates a new instance at @prim_path.
        Note that the created object is automatically loaded into the simulator, but is NOT initialized
        until a sim step occurs!

        Args:
            prim_path (str): Absolute path to the newly generated prim

        Returns:
            BasePrim: Generated prim object
        """
        new_prim = self._create_prim_with_same_kwargs(
            prim_path=prim_path,
            name=f"{self.name}_copy{self._n_duplicates}",
            load_config=self._load_config,
        )
        og.sim.import_object(new_prim, register=False)

        # Increment duplicate count
        self._n_duplicates += 1

        # Set visibility
        new_prim.visible = self.visible

        return new_prim
