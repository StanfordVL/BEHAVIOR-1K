from omnigibson.macros import gm
from omnigibson.utils.python_utils import classproperty, assert_valid_key, \
    SerializableNonInstance, UniquelyNamedNonInstance
from omnigibson.utils.registry_utils import SerializableRegistry
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Global dicts that will contain mappings
REGISTERED_SYSTEMS = dict()


def get_system_from_element_name(name):
    """
    Grabs system based on its element name @name that it controls (e.g.: Water, Dust, Stain, etc....)

    Args:
        name (str): element name corresponding to the desired system to grab

    Returns:
        BaseSystem: Corresponding system singleton
    """
    systems = SYSTEMS_REGISTRY.get_dict("name")
    system_name = f"{name}System"
    assert_valid_key(key=system_name, valid_keys=systems, name="system name")
    return systems[system_name]


def get_element_name_from_system(system):
    """
    Grabs system element name representing the element being controlled by system @system

    Args:
        system (BaseSystem): system from which to grab element name

    Returns:
        BaseSystem: Corresponding system singleton
    """
    systems = {v: k for k, v in SYSTEMS_REGISTRY.get_dict("name").items()}
    assert_valid_key(key=system, valid_keys=systems, name="system")
    return systems[system].split("System")[0]


def refresh_systems_registry():
    """
    Updates the global systems registry based on whether GPU dynamics are enabled

    Returns:
        SerializableRegistry: Updated global systems registry, also mapped to the SYSTEMS_REGISTRY variable
    """
    global SYSTEMS_REGISTRY
    SYSTEMS_REGISTRY.clear()
    # Note that we possibly filter out systems that require GPU dynamics if we're not using GPU dynamics!
    # In that case, we also notify the user to warn them that those systems will not be accessible
    if not gm.USE_GPU_DYNAMICS:
        log.warning("Omniverse-based particle systems (e.g. fluid, cloth) require gm.USE_GPU_DYNAMICS flag "
                    "to be enabled. These systems will not be initialized.")

    for system in REGISTERED_SYSTEMS.values():
        if gm.USE_GPU_DYNAMICS or not system.requires_gpu_dynamics:
            SYSTEMS_REGISTRY.add(obj=system)

    return SYSTEMS_REGISTRY


class BaseSystem(SerializableNonInstance, UniquelyNamedNonInstance):
    """
    Base class for all systems. These are non-instance objects that should be used globally for a given environment.
    This is useful for items in a scene that are non-discrete / cannot be distinguished into individual instances,
    e.g.: water, particles, etc.
    """
    def __init_subclass__(cls, **kwargs):
        # Run super init
        super().__init_subclass__(**kwargs)

        global REGISTERED_SYSTEMS
        # Register this system if requested
        if cls._register_system:
            REGISTERED_SYSTEMS[cls.__name__] = cls

    # Simulator reference
    simulator = None

    @classproperty
    def name(cls):
        # Class name is the unique name assigned
        return cls.__name__

    @classproperty
    def _register_system(cls):
        """
        Returns:
            bool: True if this system should be registered (i.e.: it is not an intermediate class but a "final" subclass
                representing a system we'd actually like to use, e.g.: WaterSystem, DustSystem, etc. Should be set by
                the subclass
        """
        # We assume we aren't registering by default
        return False

    @classproperty
    def initialized(cls):
        """
        Returns:
            bool: True if this system has been initialized via cls.initialize(...), else False
        """
        # We are initialized if we have an internal simulator reference
        return cls.simulator is not None

    @classproperty
    def requires_gpu_dynamics(cls):
        """
        Returns:
            bool: Whether this system requires GPU dynamics to function or not
        """
        raise NotImplementedError()

    @classmethod
    def initialize(cls, simulator):
        """
        Initializes this system. Default behavior is to simply store the @simulator reference internally
        """
        assert not cls.initialized, f"Already initialized system {cls.name}!"
        cls.simulator = simulator

    @classmethod
    def clear(cls):
        """
        Clears this system, so that it may possibly be re-initialized. Useful for, e.g., when loading from a new
        scene during the same sim instance
        """
        if cls.initialized:
            cls.reset()
            cls.simulator = None

    @classmethod
    def cache(cls):
        """
        Cache any necessary system level state info used by the object state system.
        """
        pass

    @classmethod
    def update(cls):
        """
        Conduct any necessary internal updates after a simulation step
        """
        pass

    @classmethod
    def reset(cls):
        """
        Reset this system
        """
        pass

    @classmethod
    def get_systems(cls):
        """
        Returns:
            dict: Mapping from system name to system for all systems that are subclasses of this system
        """
        return {system.name: system for system in SYSTEMS_REGISTRY.objects if issubclass(system, cls)}

    def __init__(self):
        raise ValueError("System classes should not be created!")


# Serializable registry of systems -- note this may be a subset of all registered systems!
SYSTEMS_REGISTRY = SerializableRegistry(
    name="system_registry",
    class_types=BaseSystem,
    default_key="name",
)
