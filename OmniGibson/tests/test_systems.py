import torch as th

from utils import SYSTEM_EXAMPLES, og_test

import omnigibson as og
from omnigibson.object_states import Covered
from omnigibson.systems import VisualParticleSystem, MicroPhysicalParticleSystem


@og_test
def test_system_spawn_and_clear(env):
    og.sim.viewer_camera.set_position_orientation(
        th.tensor([0.1525, -0.2867, 0.0773]), th.tensor([0.6538, 0.1604, 0.1762, 0.7181])
    )
    og.sim.step()
    breakfast_table = env.scene.object_registry("name", "breakfast_table")
    for system_name, system_class in SYSTEM_EXAMPLES.items():
        for _ in range(3):
            system = env.scene.get_system(system_name)
            assert isinstance(system, system_class)
            if issubclass(system_class, VisualParticleSystem):
                assert breakfast_table.states[Covered].set_value(system, True)
            else:
                system.generate_particles(positions=[[0, 0, 0.05]])
                assert th.allclose(
                    system.get_particle_position_orientation(idx=0)[0], th.tensor([0.0, 0.0, 0.05], dtype=th.float32)
                )
                for _ in range(10):
                    og.sim.step()
                if issubclass(system_class, MicroPhysicalParticleSystem):
                    assert th.allclose(
                        system.get_particle_position_orientation(idx=0)[0],
                        th.tensor([0, 0, system.particle_contact_radius], dtype=th.float32),
                        atol=1e-3,
                    )
            assert system.n_particles > 0
            og.sim.step()
            env.scene.clear_system(system_name)
            # This is a bit of a hack to address a very niche situation where we clear a system and immediately reinitialize it
            # If we do not take a physics step here, the system will not be reinitialized properly
            og.sim.step()
