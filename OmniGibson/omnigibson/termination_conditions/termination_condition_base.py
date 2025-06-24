from abc import ABCMeta, abstractmethod

from omnigibson.utils.python_utils import Registerable, classproperty

REGISTERED_TERMINATION_CONDITIONS = dict()
REGISTERED_SUCCESS_CONDITIONS = dict()
REGISTERED_FAILURE_CONDITIONS = dict()


def register_success_condition(cls):
    if cls.__name__ not in REGISTERED_SUCCESS_CONDITIONS:
        REGISTERED_SUCCESS_CONDITIONS[cls.__name__] = cls


def register_failure_condition(cls):
    if cls.__name__ not in REGISTERED_FAILURE_CONDITIONS:
        REGISTERED_FAILURE_CONDITIONS[cls.__name__] = cls


class BaseTerminationCondition(Registerable, metaclass=ABCMeta):
    """
    Base TerminationCondition class
    Condition-specific _step() method is implemented in subclasses
    """

    def __init__(self):
        # Initialize internal vars that will be filled in at runtime
        self._done = None

    @abstractmethod
    def _step(self, task, env, action):
        """
        Step the termination condition and return whether the episode should terminate. Overwritten by subclasses.

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment

        Returns:
            bool: whether environment should terminate or not
        """
        raise NotImplementedError()

    def step(self, task, env, action):
        """
        Step the termination condition and return whether the episode should terminate.

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment

        Returns:
            2-tuple:
                - bool: whether environment should terminate or not
                - bool: whether a success was reached under this termination condition
        """
        # Step internally and store the done state internally as well
        self._done = self._step(task=task, env=env, action=action)

        # We are successful if done is True AND this is a success condition
        success = self._done and self._terminate_is_success

        return self._done, success

    def reset(self, task, env):
        """
        Termination condition-specific reset

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
        """
        # Reset internal vars
        self._done = None

    @property
    def done(self):
        """
        Returns:
            bool: Whether this termination condition has triggered or not
        """
        assert self._done is not None, "At least one step() must occur before done can be calculated!"
        return self._done

    @property
    def success(self):
        """
        Returns:
            bool: Whether this termination condition has been evaluated as a success or not
        """
        assert self._done is not None, "At least one step() must occur before success can be calculated!"
        return self._done and self._terminate_is_success

    @classproperty
    def _terminate_is_success(cls):
        """
        Returns:
            bool: Whether this termination condition corresponds to a success
        """
        raise NotImplementedError()

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseTerminationCondition")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_TERMINATION_CONDITIONS
        return REGISTERED_TERMINATION_CONDITIONS


class SuccessCondition(BaseTerminationCondition):
    """
    Termination condition corresponding to a success
    """

    def __init_subclass__(cls, **kwargs):
        # Register as part of locomotion controllers
        super().__init_subclass__(**kwargs)
        register_success_condition(cls)

    @classproperty
    def _terminate_is_success(cls):
        # Done --> success
        return True

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("SuccessCondition")
        return classes


class FailureCondition(BaseTerminationCondition):
    """
    Termination condition corresponding to a failure
    """

    def __init_subclass__(cls, **kwargs):
        # Register as part of locomotion controllers
        super().__init_subclass__(**kwargs)
        register_failure_condition(cls)

    @classproperty
    def _terminate_is_success(cls):
        # Done --> not success
        return False

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("FailureCondition")
        return classes
