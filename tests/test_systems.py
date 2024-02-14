import omnigibson as og
from omnigibson.systems import *
from omnigibson.object_states import Covered

from utils import og_test

import pytest

@og_test
def test_system_clear():
    system_examples = {
        "water": FluidSystem,
        "white_rice": GranularSystem,
        "diced__apple": MacroPhysicalParticleSystem,
        "sand": MacroVisualParticleSystem,
    }
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    for system_name, system_class in system_examples.items():
        for _ in range(3):
            system = get_system(system_name)
            assert issubclass(system, system_class)
            if system_name == "sand":
                assert breakfast_table.states[Covered].set_value(system, True)
            else:
                system.generate_particles(positions=[[0, 0, 1]])
            assert system.n_particles > 0
            og.sim.step()
            system.clear()
