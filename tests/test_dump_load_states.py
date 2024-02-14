import omnigibson as og
from omnigibson.systems import *
from omnigibson.object_states import Covered

from utils import og_test

import pytest

@og_test
def test_dump_load():
    system_examples = {
        "water": FluidSystem,
        "white_rice": GranularSystem,
        "diced__apple": MacroPhysicalParticleSystem,
        "sand": MacroVisualParticleSystem,
    }
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    for system_name, system_class in system_examples.items():
        system = get_system(system_name)
        assert issubclass(system, system_class)
        if system_name == "sand":
            assert breakfast_table.states[Covered].set_value(system, True)
        else:
            system.generate_particles(positions=[[0, 0, 1]])
        assert system.n_particles > 0

    state = og.sim.dump_state()
    og.sim.load_state(state)

    for system_name, system_class in system_examples.items():
        system = get_system(system_name)
        system.clear()

@og_test
def test_dump_load_serialized():
    system_examples = {
        "water": FluidSystem,
        "white_rice": GranularSystem,
        "diced__apple": MacroPhysicalParticleSystem,
        "sand": MacroVisualParticleSystem,
    }
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    for system_name, system_class in system_examples.items():
        system = get_system(system_name)
        assert issubclass(system, system_class)
        if system_name == "sand":
            assert breakfast_table.states[Covered].set_value(system, True)
        else:
            system.generate_particles(positions=[[0, 0, 1]])
        assert system.n_particles > 0

    state = og.sim.dump_state(serialized=True)
    og.sim.load_state(state, serialized=True)

    for system_name, system_class in system_examples.items():
        system = get_system(system_name)
        system.clear()
