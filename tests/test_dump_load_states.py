import pytest
from utils import SYSTEM_EXAMPLES, og_test

import omnigibson as og
from omnigibson.object_states import Covered
from omnigibson.systems import *


@og_test
def test_dump_load(env):
    breakfast_table = env.scene.object_registry("name", "breakfast_table")
    for system_name, system_class in SYSTEM_EXAMPLES.items():
        system = env.scene.get_system(system_name)
        assert isinstance(system, system_class)
        if issubclass(system_class, VisualParticleSystem):
            assert breakfast_table.states[Covered].set_value(system, True)
        else:
            system.generate_particles(positions=th.tensor([[0, 0, 1]]))
        assert system.n_particles > 0
        system.remove_all_particles()

    state = og.sim.dump_state()
    og.sim.load_state(state)

    for system_name, system_class in SYSTEM_EXAMPLES.items():
        system = env.scene.get_system(system_name)
        system.clear()


@og_test
def test_dump_load_serialized(env):
    breakfast_table = env.scene.object_registry("name", "breakfast_table")
    for system_name, system_class in SYSTEM_EXAMPLES.items():
        system = env.scene.get_system(system_name)
        assert isinstance(system, system_class)
        if issubclass(system_class, VisualParticleSystem):
            assert breakfast_table.states[Covered].set_value(system, True)
        else:
            system.generate_particles(positions=th.tensor([[0, 0, 1]]))
        assert system.n_particles > 0

    state = og.sim.dump_state(serialized=True)
    og.sim.load_state(state, serialized=True)

    for system_name, system_class in SYSTEM_EXAMPLES.items():
        system = env.scene.get_system(system_name)
        system.clear()
