import tempfile

import torch as th
from utils import SYSTEM_EXAMPLES, og_test

import omnigibson as og
from omnigibson.object_states import Covered
from omnigibson.systems import VisualParticleSystem


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
        env.scene.clear_system(system_name)


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
        env.scene.clear_system(system_name)


@og_test
def test_save_restore_partial(env):
    breakfast_table = env.scene.object_registry("name", "breakfast_table")

    decrypted_fd, tmp_json_path = tempfile.mkstemp("test_save_restore.json", dir=og.tempdir)
    og.sim.save([tmp_json_path])

    # Delete the breakfast table
    env.scene.remove_object(breakfast_table)

    og.sim.step()

    # Restore the saved environment
    og.sim.restore([tmp_json_path])

    # Make sure we still have an object that existed beforehand
    assert og.sim.scenes[0].object_registry("name", "breakfast_table") is not None


@og_test
def test_save_restore_full(env):
    decrypted_fd, tmp_json_path = tempfile.mkstemp("test_save_restore.json", dir=og.tempdir)
    og.sim.save([tmp_json_path])

    # Clear the simulator
    og.clear()

    # Restore the saved environment
    og.sim.restore([tmp_json_path])

    # This generates a new scene, so we monkey-patch it into the original env to avoid crashes
    env._scene = og.sim.scenes[0]

    # Make sure we still have an object that existed beforehand
    assert og.sim.scenes[0].object_registry("name", "breakfast_table") is not None
