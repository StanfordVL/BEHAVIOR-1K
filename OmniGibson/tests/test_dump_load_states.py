import tempfile
from unittest.mock import MagicMock

import torch as th
from utils import SYSTEM_EXAMPLES

import omnigibson as og
from omnigibson.object_states import Covered
from omnigibson.scenes.scene_base import Scene
from omnigibson.systems import VisualParticleSystem


def test_scene_restore_stops_before_system_topology_changes(monkeypatch):
    events = []
    sim = MagicMock()
    sim.is_stopped.return_value = False
    sim.is_playing.return_value = True
    sim.stop.side_effect = lambda: events.append("stop")
    sim.play.side_effect = lambda: events.append("play")
    monkeypatch.setattr(og, "sim", sim)

    class RestoreScene:
        pass

    scene = RestoreScene()
    scene._check_versions_compatible = MagicMock()
    scene.active_systems = {"obsolete_system": MagicMock()}
    scene.object_registry = MagicMock()
    # The obsolete system owns a template object. clear_system() removes it; restore must
    # recompute the object delta rather than trying to remove the stale object a second time.
    scene.object_registry.get_dict.side_effect = [{"obsolete_template": MagicMock()}, {}]
    scene.clear_system = lambda name: events.append(f"clear:{name}")
    scene.get_system = MagicMock()
    scene.load_state = lambda state, serialized: events.append("load")
    scene_info = {
        "init_info": {"class_name": "RestoreScene"},
        "state": {
            "pos": [0.0, 0.0, 0.0],
            "ori": [0.0, 0.0, 0.0, 1.0],
            "registry": {"system_registry": {}, "object_registry": {}},
        },
        "objects_info": {"init_info": {}},
    }

    Scene.restore(scene, scene_file=scene_info)

    assert events == ["stop", "clear:obsolete_system", "play", "load"]
    sim.batch_remove_objects.assert_called_once_with([])


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
