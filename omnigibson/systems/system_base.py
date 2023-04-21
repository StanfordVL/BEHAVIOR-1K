import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.python_utils import classproperty, assert_valid_key, get_uuid, camel_case_to_snake_case, \
    SerializableNonInstance, UniquelyNamedNonInstance
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


def is_system_active(system_name):
    assert system_name in REGISTERED_SYSTEMS, f"System {system_name} not in REGISTERED_SYSTEMS."
    system = REGISTERED_SYSTEMS[system_name]
    return system.initialized


def get_system(system_name):
    assert system_name in REGISTERED_SYSTEMS, f"System {system_name} not in REGISTERED_SYSTEMS."
    system = REGISTERED_SYSTEMS[system_name]
    if not system.initialized:
        system.initialize()
        SYSTEM_REGISTRY.add(obj=system)
        # Make sure to refresh any transition rules that require this system
        # Import now to avoid circular imports
        from omnigibson.transition_rules import TransitionRuleAPI, RULES_REGISTRY
        system_rules = RULES_REGISTRY("required_systems", system.name, default_val=[])
        TransitionRuleAPI.refresh_rules(rules=system_rules, objects=og.sim.scene.objects)
    return system
