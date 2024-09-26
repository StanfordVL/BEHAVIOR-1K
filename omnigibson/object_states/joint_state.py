import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import AbsoluteObjectState

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.POSITIONAL_VALIDATION_EPSILON = 1e-10


class Joint(AbsoluteObjectState):

    def _get_value(self):
        return self.obj.get_joint_positions() if self.obj.n_joints > 0 else th.tensor([])

    def _has_changed(self, get_value_args, value, info):
        # Only changed if the squared distance between old and current q has changed above some threshold
        old_q = value
        # Get current joint values
        cur_q = self.get_value()
        dist_squared = th.sum(th.square(cur_q - old_q))
        return dist_squared > m.POSITIONAL_VALIDATION_EPSILON
