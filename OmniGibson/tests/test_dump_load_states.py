import tempfile

import torch as th
from utils import SYSTEM_EXAMPLES

import omnigibson as og
from omnigibson.object_states import Covered
from omnigibson.systems import VisualParticleSystem


def test_dump_load(env, breakfast_table):
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
        env.scene.clear_system(system_name)


def test_dump_load_serialized(env, breakfast_table):
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
        env.scene.clear_system(system_name)


def test_save_restore_partial(env, breakfast_table):
    decrypted_fd, tmp_json_path = tempfile.mkstemp("test_save_restore.json", dir=og.tempdir)
    og.sim.save([tmp_json_path])

    # Delete the breakfast table
    env.scene.remove_object(breakfast_table)

    og.sim.step()

    # Restore the saved environment
    og.sim.restore([tmp_json_path])

    # Make sure we still have an object that existed beforehand
    assert og.sim.scenes[0].object_registry("name", "breakfast_table") is not None


def test_save_restore_full(env, breakfast_table):
    decrypted_fd, tmp_json_path = tempfile.mkstemp("test_save_restore.json", dir=og.tempdir)
    og.sim.save([tmp_json_path])

    # Clear the simulator
    og.clear()

    # Restore the saved environment
    og.sim.restore([tmp_json_path])

    # This generates a new scene, so we monkey-patch it into the original env to avoid crashes
    env._scenes[0] = og.sim.scenes[0]

    # Make sure we still have an object that existed beforehand
    assert og.sim.scenes[0].object_registry("name", "breakfast_table") is not None


def test_dump_load_fluid_particles(env):
    water = env.scene.get_system("water")
    water.generate_particles(positions=th.tensor([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]]))
    for _ in range(5):
        og.sim.step()
    positions = water.get_particles_position_orientation()[0].clone()

    for serialized in (False, True):
        state = og.sim.dump_state(serialized=serialized)
        # Move every particle so the load has to bring it back
        water.set_particles_position_orientation(positions=positions + th.tensor([0.0, 0.0, 1.0]).to(positions.device))
        og.sim.load_state(state, serialized=serialized)
        assert th.allclose(water.get_particles_position_orientation()[0], positions, atol=1e-3)

    env.scene.clear_system("water")
