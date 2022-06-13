from abc import ABCMeta
from igibson.utils.python_utils import classproperty, assert_valid_key, \
    SerializableNonInstance, UniquelyNamedNonInstance
from igibson.utils.registry_utils import SerializableRegistry


def get_system_from_element_name(name):
    """
    Grabs system based on its element name @name that it controls (e.g.: Water, Dust, Stain, etc....)

    Args:
        name (str): element name corresponding to the desired system to grab

    Returns:
        BaseSystem: Corresponding system singleton
    """
    systems = SYSTEMS_REGISTRY.get_dict("__name__")
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
    systems = {v: k for k, v in SYSTEMS_REGISTRY.get_dict("__name__").items()}
    assert_valid_key(key=system, valid_keys=systems, name="system")
    return systems[system].split("System")[0]


class BaseSystem(SerializableNonInstance, UniquelyNamedNonInstance):
    """
    Base class for all systems. These are non-instance objects that should be used globally for a given environment.
    This is useful for items in a scene that are non-discrete / cannot be distinguished into individual instances,
    e.g.: water, particles, etc.
    """
    def __init_subclass__(cls, **kwargs):
        # Run super init
        super().__init_subclass__(**kwargs)

        global SYSTEMS_REGISTRY
        # Register this system if requested
        if cls._register_system:
            print(f"registering system: {cls.name}")
            SYSTEMS_REGISTRY.add(obj=cls)

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

    def __init__(self):
        raise ValueError("System classes should not be created!")


# Because we don't instantiate individual systems, we store the classes themselves in a global registry
SYSTEMS_REGISTRY = SerializableRegistry(
    name="system_registry",
    class_types=BaseSystem,
    # Ideally "name" would work, but for some reason __getattribute__()
    # is not equivalent to .attr during the __init_subclass__ call, which is when we add these systems
    default_key="__name__",
)
