import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_all_system_categories
from omnigibson.utils.python_utils import classproperty, assert_valid_key, get_uuid, camel_case_to_snake_case, \
    snake_case_to_camel_case, subclass_factory, SerializableNonInstance, UniquelyNamedNonInstance
from omnigibson.utils.registry_utils import SerializableRegistry
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


class BaseSystem(SerializableNonInstance, UniquelyNamedNonInstance):
    """
    Base class for all systems. These are non-instance objects that should be used globally for a given environment.
    This is useful for items in a scene that are non-discrete / cannot be distinguished into individual instances,
    e.g.: water, particles, etc. While we keep the python convention of the system class name being camel case
    (e.g. StrawberrySmoothie), we adopt the snake case for the system registry to unify with the category of BaseObject.
    For example, get_system("strawberry_smoothie") will return the StrawberrySmoothie class.
    """
    def __init_subclass__(cls, **kwargs):
        # While class names are camel case, we convert them to snake case to be consistent with object categories.
        cls._snake_case_name = camel_case_to_snake_case(cls.__name__)

        # Run super init
        super().__init_subclass__(**kwargs)

        # Register this system if requested
        if cls._register_system:
            global REGISTERED_SYSTEMS
            REGISTERED_SYSTEMS[cls._snake_case_name] = cls
            cls._uuid = get_uuid(cls._snake_case_name)

    initialized = False
    _uuid = None
    _snake_case_name = None

    @classproperty
    def name(cls):
        # Class name is the unique name assigned
        return cls._snake_case_name

    @classproperty
    def uuid(cls):
        return cls._uuid

    @classproperty
    def prim_path(cls):
        """
        Returns:
            str: Path to this system's prim in the scene stage
        """
        return f"/World/{cls.name}"

    @classproperty
    def _register_system(cls):
        """
        Returns:
            bool: True if this system should be registered (i.e.: it is not an intermediate class but a "final" subclass
                representing a system we'd actually like to use, e.g.: water, dust, etc. Should be set by the subclass
        """
        # We assume we aren't registering by default
        return False

    @classmethod
    def initialize(cls):
        """
        Initializes this system
        """
        assert not cls.initialized, f"Already initialized system {cls.name}!"
        og.sim.stage.DefinePrim(cls.prim_path, "Scope")
        cls.initialized = True

    @classmethod
    def clear(cls):
        """
        Clears this system, so that it may possibly be re-initialized. Useful for, e.g., when loading from a new
        scene during the same sim instance
        """
        if cls.initialized:
            cls.reset()
            cls.initialized = False

    @classmethod
    def reset(cls):
        """
        Reset this system
        """
        pass

    @classmethod
    def create(cls, name, **kwargs):
        """
        Helper function to programmatically generate systems

        Args:
            name (str): Name of the visual particles, in snake case.
            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @classproperty or @classmethod decorators!)


        Returns:
            BaseSystem: Generated system class given input arguments
        """
        # Create and return the class
        return subclass_factory(name=snake_case_to_camel_case(name), base_classes=cls, **kwargs)


    @classmethod
    def get_active_systems(cls):
        """
        Returns:
            dict: Mapping from system name to system for all systems that are subclasses of this system AND active (initialized)
        """
        return {system.name: system for system in SYSTEM_REGISTRY.objects if issubclass(system, cls)}

    def __init__(self):
        raise ValueError("System classes should not be created!")

# Global dict that contains mappings of all the systems
REGISTERED_SYSTEMS = dict()

# Serializable registry of systems that are active on the stage (initialized)
SYSTEM_REGISTRY = SerializableRegistry(
    name="system_registry",
    class_types=BaseSystem,
    default_key="name",
    unique_keys=["name", "prim_path", "uuid"],
)


def _create_system_from_metadata(system_name):
    """
    Internal helper function to programmatically create a system from dataset metadata

    NOTE: This only creates the system, and does NOT initialize the system

    Args:
        system_name (str): Name of the system to create, e.g.: "water", "stain", etc.

    Returns:
        BaseSystem: Created system class
    """
    # Search for the appropriate system, if not found, fallback
    # TODO: Once dataset is fully constructed, DON'T fallback, and assert False instead
    all_systems = set(get_all_system_categories())
    if system_name not in all_systems:
        # Avoid circular imports
        from omnigibson import systems
        # Use default config -- assume @system_name is a fluid that uses the same params as water
        return systems.__dict__["FluidSystem"].create(
            name=system_name,
            particle_contact_offset=0.012,
            particle_density=500.0,
            material_mtl_name="DeepWater",
        )
    else:
        """
        This is not defined yet, but one proposal:
        
        Metadata = .json dict, with format:
        {
            "type": one of {"visual", "fluid", "granular"},
        }
        
        if visual or granular, also includes:
            "KWARG_0" : ... (e.g.: stain kwargs)
            
            --> note: create_particle_template should be deterministic, configured via:
                lambda prim_path, name: og.objects.DatasetObject(
                    prim_path=prim_path,
                    name=name,
                    usd_path=os.path.join(gm.DATASET_PATH, "systems", system_name, f"{system_name}.usd"),
                    category=system_name,
                    visible=False,
                    fixed_base=False,
                    visual_only=True,
                    include_default_states=False,
                    abilities={},
                )
        
        if fluid / granular, also include:
            "particle_contact_offset": ...,
            "particle_density": ...,
        
        if fluid, also include:
            "material_mtl_name": ...,       # Base material config to use
            "customize_particle_kwargs": {  # Maps property/ies from @MaterialPrim to value to set
                "opacity_constant": ...,
                "albedo_add": ...,
                "diffuse_color_constant": ...,
                ...,
            }
            
            --> This will be programmatically constructed into a function:
                def _customize_particle_material(mat: MaterialPrim): --> None
                    for attr, val in metadata["customize_particle_kwargs"].items():
                        mat.__setattr__(attr, val)
                        
        Then, compile the necessary kwargs and generate the requested system
        """
        raise ValueError("Metadata format for loading system not defined yet!")


def is_system_active(system_name):
    assert system_name in REGISTERED_SYSTEMS, f"System {system_name} not in REGISTERED_SYSTEMS."
    system = REGISTERED_SYSTEMS[system_name]
    return system.initialized


def get_system(system_name):
    # Make sure scene exists
    assert og.sim.scene is not None, "Cannot get systems until scene is imported!"
    # If system_name is not in REGISTERED_SYSTEMS, create from metadata
    system = REGISTERED_SYSTEMS[system_name] if system_name in REGISTERED_SYSTEMS \
        else _create_system_from_metadata(system_name=system_name)
    if not system.initialized:
        system.initialize()
        SYSTEM_REGISTRY.add(obj=system)
        # Make sure to refresh any transition rules that require this system
        # Import now to avoid circular imports
        from omnigibson.transition_rules import TransitionRuleAPI, RULES_REGISTRY
        system_rules = RULES_REGISTRY("required_systems", system.name, default_val=[])
        TransitionRuleAPI.refresh_rules(rules=system_rules)
    return system
