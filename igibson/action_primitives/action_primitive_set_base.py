import inspect
from abc import ABCMeta, abstractmethod
from enum import IntEnum

from future.utils import with_metaclass

from igibson.robots import BaseRobot
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.tasks.task_base import BaseTask

REGISTERED_PRIMITIVE_SETS = {}


class ActionPrimitiveError(ValueError):
    class Reason(IntEnum):
        # The planning for a primitive was successfully completed, but an error occurred during execution.
        EXECUTION_ERROR = 0

        # The planning for a primitive failed possibly due to not being able to find a path.
        PLANNING_ERROR = 1

        # A primitive could not be executed because a precondition was not satisfied, e.g. PLACE was called without an
        # object currently in hand.f
        PRE_CONDITION_ERROR = 2

        # A sampling error occurred: a position to place an object could not be found.
        SAMPLING_ERROR = 3

    def __init__(self, reason: Reason, message, metadata=None):
        self.reason = reason
        self.metadata = metadata if metadata is not None else {}
        super().__init__(f"{reason}: {message}. Additional info: f{metadata}")


class BaseActionPrimitiveSet(with_metaclass(ABCMeta, object)):
    def __init_subclass__(cls, **kwargs):
        """
        Registers all subclasses as part of this registry. This is useful to decouple internal codebase from external
        user additions. This way, users can add their custom robot by simply extending this Robot class,
        and it will automatically be registered internally. This allows users to then specify their robot
        directly in string-from in e.g., their config files, without having to manually set the str-to-class mapping
        in our code.
        """
        if not inspect.isabstract(cls):
            REGISTERED_PRIMITIVE_SETS[cls.__name__] = cls

    def __init__(self, env, task, scene, robot):
        self.env = env
        self.task: BaseTask = task
        self.scene: InteractiveTraversableScene = scene
        self.robot: BaseRobot = robot

    @abstractmethod
    def get_action_space(self):
        """Get the higher-level action space as an OpenAI Gym Space object."""
        pass

    @abstractmethod
    def apply(self, action):
        """Given a higher-level action, generates a sequence of lower level actions (or raise ActionPrimitiveError)"""
        pass
