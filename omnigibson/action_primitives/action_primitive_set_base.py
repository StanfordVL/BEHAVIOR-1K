import inspect
from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import List

from future.utils import with_metaclass

REGISTERED_PRIMITIVE_SETS = {}


class ActionPrimitiveError(ValueError):
    class Reason(IntEnum):
        # A primitive could not be executed because a precondition was not satisfied, e.g. PLACE was called without an
        # object currently in hand.
        PRE_CONDITION_ERROR = 0

        # A sampling error occurred: e.g. a position to place an object could not be found, or the robot could not
        # find a pose near the object to navigate to.
        SAMPLING_ERROR = 1

        # The planning for a primitive failed possibly due to not being able to find a path.
        PLANNING_ERROR = 2

        # The planning for a primitive was successfully completed, but an error occurred during execution.
        EXECUTION_ERROR = 3

        # The execution of the primitive happened correctly, but while checking post-conditions, an error was found.
        POST_CONDITION_ERROR = 4

    def __init__(self, reason: Reason, message, metadata=None):
        self.reason = reason
        self.metadata = metadata if metadata is not None else {}
        super().__init__(f"{reason.name}: {message}. Additional info: {metadata}")


class ActionPrimitiveErrorGroup(ValueError):
    def __init__(self, exceptions: List[ActionPrimitiveError]) -> None:
        self._exceptions = tuple(exceptions)
        submessages = [f"Attempt {i}: {e}" for i, e in enumerate(exceptions)]
        submessages = "\n\n".join(submessages)
        message = "An error occurred during each attempt of this action.\n\n" + submessages
        super().__init__(message)

    @property
    def exceptions(self):
        return self._exceptions


class BaseActionPrimitiveSet(with_metaclass(ABCMeta, object)):
    def __init_subclass__(cls, **kwargs):
        """
        Registers all subclasses as part of this registry. This is useful to decouple internal codebase from external
        user additions. This way, users can add their custom primitive set by simply extending this class,
        and it will automatically be registered internally. This allows users to then specify their primitive set
        directly in string-from in e.g., their config files, without having to manually set the str-to-class mapping
        in our code.
        """
        if not inspect.isabstract(cls):
            REGISTERED_PRIMITIVE_SETS[cls.__name__] = cls

    def __init__(self, env):
        self.env = env

    @property
    def robot(self):
        # Currently returns the first robot in the environment, but can be scaled to multiple robots
        # by creating multiple action generators and passing in a robot index etc.
        return self.env.robots[0]

    @abstractmethod
    def get_action_space(self):
        """Get the higher-level action space as an OpenAI Gym Space object."""
        pass

    @abstractmethod
    def apply(self, action):
        """
        Apply a primitive action.

        Given a higher-level action in the same format as the action space (e.g. as a number),
        generates a sequence of lower level actions (or raise ActionPrimitiveError). The action
        will get resolved and passed into apply_ref.
        """
        pass

    @abstractmethod
    def apply_ref(self, action, *args):
        """
        Apply a primitive action by reference.

        Given a higher-level action from the corresponding action set enum and any necessary arguments,
        generates a sequence of lower level actions (or raise ActionPrimitiveError)
        """
        pass
