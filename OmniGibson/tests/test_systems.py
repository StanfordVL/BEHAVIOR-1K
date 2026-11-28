import pytest
import torch as th

from utils import SYSTEM_EXAMPLES

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import Covered
from omnigibson.systems import VisualParticleSystem, MicroPhysicalParticleSystem


def test_system_spawn_and_clear(env, breakfast_table):
    if og.sim.viewer_camera is not None:
        og.sim.viewer_camera.set_position_orientation(
            th.tensor([0.1525, -0.2867, 0.0773]), th.tensor([0.6538, 0.1604, 0.1762, 0.7181])
        )
    og.sim.step()
    for system_name, system_class in SYSTEM_EXAMPLES.items():
        for _ in range(3):
            system = env.scene.get_system(system_name)
            assert isinstance(system, system_class)
            if issubclass(system_class, VisualParticleSystem):
                assert breakfast_table.states[Covered].set_value(system, True)
            else:
                system.generate_particles(positions=[[0, 0, 0.05]])
                assert th.allclose(
                    system.get_particle_position_orientation(idx=0)[0],
                    th.tensor([0.0, 0.0, 0.05], dtype=th.float32, device=og.sim.device),
                )
                for _ in range(10):
                    og.sim.step()
                if issubclass(system_class, MicroPhysicalParticleSystem):
                    pos = system.get_particle_position_orientation(idx=0)[0]
                    if gm.PHYSICS_BACKEND == "newton":
                        # Newton's MPM meets colliders at the particle center, so it rests on the floor surface
                        assert th.allclose(pos[:2], th.zeros(2, device=pos.device), atol=1e-3)
                        assert -1e-3 <= pos[2].item() <= system.particle_contact_radius + 1e-3
                    else:
                        assert th.allclose(
                            pos,
                            th.tensor([0, 0, system.particle_contact_radius], dtype=th.float32, device=og.sim.device),
                            atol=1e-3,
                        )
            assert system.n_particles > 0
            og.sim.step()
            env.scene.clear_system(system_name)
            # This is a bit of a hack to address a very niche situation where we clear a system and immediately reinitialize it
            # If we do not take a physics step here, the system will not be reinitialized properly
            og.sim.step()


@pytest.mark.skipif(gm.PHYSICS_BACKEND != "newton", reason="Newton rebuilds its whole model on joint/mass changes")
def test_fluid_particles_survive_rebuild(env):
    water = env.scene.get_system("water")
    water.generate_particles(positions=th.tensor([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]]))
    for _ in range(60):
        og.sim.step()
    settled = water.get_particles_position_orientation()[0].clone()
    assert settled[:, 2].max() < 0.25

    # What an AttachedTo attach/detach sets; the rebuild must keep the settled particles where they are
    og.sim.physics_backend._joints_dirty = True
    og.sim.step()
    assert th.allclose(water.get_particles_position_orientation()[0], settled, atol=0.05)

    env.scene.clear_system("water")
