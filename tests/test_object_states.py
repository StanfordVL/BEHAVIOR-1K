from omnigibson.macros import macros as m
from omnigibson.object_states import *
from omnigibson.systems import get_system, is_physical_particle_system, is_visual_particle_system
from omnigibson.utils.constants import PrimType
from omnigibson.utils.physx_utils import apply_force_at_pos, apply_torque
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import BoundingBoxAPI
import omnigibson as og

from utils import og_test, get_random_pose, place_objA_on_objB_bbox, place_obj_on_floor_plane

import pytest
import numpy as np


@og_test
def test_on_top():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    place_obj_on_floor_plane(breakfast_table)
    for i, obj in enumerate((bowl, dishtowel)):
        place_objA_on_objB_bbox(obj, breakfast_table)
        for _ in range(5):
            og.sim.step()

        assert obj.states[OnTop].get_value(breakfast_table)

        obj.set_position(np.ones(3) * 10 * (i + 1))
        og.sim.step()

        assert not obj.states[OnTop].get_value(breakfast_table)

    assert bowl.states[OnTop].set_value(breakfast_table, True)
    assert dishtowel.states[OnTop].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        bowl.states[OnTop].set_value(breakfast_table, False)


@og_test
def test_inside():
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    place_obj_on_floor_plane(bottom_cabinet)
    bowl.set_position([0., 0., 0.08])
    dishtowel.set_position([0, 0., 0.5])

    for _ in range(5):
        og.sim.step()

    assert bowl.states[Inside].get_value(bottom_cabinet)
    assert dishtowel.states[Inside].get_value(bottom_cabinet)

    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[Inside].get_value(bottom_cabinet)
    assert not dishtowel.states[Inside].get_value(bottom_cabinet)

    assert bowl.states[Inside].set_value(bottom_cabinet, True)
    assert dishtowel.states[Inside].set_value(bottom_cabinet, True)

    with pytest.raises(NotImplementedError):
        bowl.states[OnTop].set_value(bottom_cabinet, False)


@og_test
def test_under():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    place_obj_on_floor_plane(breakfast_table)
    for i, obj in enumerate((bowl, dishtowel)):
        place_obj_on_floor_plane(obj)
        for _ in range(5):
            og.sim.step()

        assert obj.states[Under].get_value(breakfast_table)

        obj.set_position(np.ones(3) * 10 * (i + 1))
        og.sim.step()

        assert not obj.states[Under].get_value(breakfast_table)

    assert bowl.states[Under].set_value(breakfast_table, True)
    assert dishtowel.states[Under].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        bowl.states[Under].set_value(breakfast_table, False)


@og_test
def test_touching():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    place_obj_on_floor_plane(breakfast_table)
    for i, obj in enumerate((bowl, dishtowel)):
        place_objA_on_objB_bbox(obj, breakfast_table)
        for _ in range(5):
            og.sim.step()

        assert obj.states[Touching].get_value(breakfast_table)
        assert breakfast_table.states[Touching].get_value(obj)

        obj.set_position(np.ones(3) * 10 * (i + 1))
        og.sim.step()

        assert not obj.states[Touching].get_value(breakfast_table)
        assert not breakfast_table.states[Touching].get_value(obj)

    with pytest.raises(NotImplementedError):
        bowl.states[Touching].set_value(breakfast_table, None)


@og_test
def test_contact_bodies():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    place_obj_on_floor_plane(breakfast_table)
    for i, obj in enumerate((bowl, dishtowel)):
        place_objA_on_objB_bbox(obj, breakfast_table)
        for _ in range(5):
            og.sim.step()

        # TODO: rigid body's ContactBodies should include cloth
        if obj.prim_type != PrimType.CLOTH:
            assert obj.root_link in breakfast_table.states[ContactBodies].get_value()
        assert breakfast_table.root_link in obj.states[ContactBodies].get_value()

        obj.set_position(np.ones(3) * 10 * (i + 1))
        og.sim.step()

        assert obj.root_link not in breakfast_table.states[ContactBodies].get_value()
        assert breakfast_table.root_link not in obj.states[ContactBodies].get_value()

    with pytest.raises(NotImplementedError):
        bowl.states[ContactBodies].set_value(None)


@og_test
def test_next_to():
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    place_obj_on_floor_plane(bottom_cabinet)
    for i, (axis, obj) in enumerate(zip(("x", "y"), (bowl, dishtowel))):
        place_obj_on_floor_plane(obj, **{f"{axis}_offset": 0.3})
        for _ in range(5):
            og.sim.step()

        assert obj.states[NextTo].get_value(bottom_cabinet)
        assert bottom_cabinet.states[NextTo].get_value(obj)

        obj.set_position(np.ones(3) * 10 * (i + 1))
        og.sim.step()

        assert not obj.states[NextTo].get_value(bottom_cabinet)
        assert not bottom_cabinet.states[NextTo].get_value(obj)

    with pytest.raises(NotImplementedError):
        bowl.states[NextTo].set_value(bottom_cabinet, None)


@og_test
def test_overlaid():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    carpet = og.sim.scene.object_registry("name", "carpet")

    place_obj_on_floor_plane(breakfast_table)
    place_objA_on_objB_bbox(carpet, breakfast_table)

    for _ in range(5):
        og.sim.step()

    assert carpet.states[Overlaid].get_value(breakfast_table)

    carpet.set_position(np.ones(3) * 20.0)
    og.sim.step()

    assert not carpet.states[Overlaid].get_value(breakfast_table)

    assert carpet.states[Overlaid].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        carpet.states[Overlaid].set_value(breakfast_table, False)


@og_test
def test_pose():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    pos1, orn1 = get_random_pose()
    breakfast_table.set_position_orientation(pos1, orn1)

    pos2, orn2 = get_random_pose()
    dishtowel.set_position_orientation(pos2, orn2)

    assert np.allclose(breakfast_table.states[Pose].get_value()[0], pos1)
    assert np.allclose(breakfast_table.states[Pose].get_value()[1], orn1) or np.allclose(breakfast_table.states[Pose].get_value()[1], -orn1)
    assert np.allclose(dishtowel.states[Pose].get_value()[0], pos2)
    assert np.allclose(dishtowel.states[Pose].get_value()[1], orn2) or np.allclose(dishtowel.states[Pose].get_value()[1], -orn2)

    with pytest.raises(NotImplementedError):
        breakfast_table.states[Pose].set_value(None)


@og_test
def test_aabb():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    pos1, orn1 = get_random_pose()
    breakfast_table.set_position_orientation(pos1, orn1)

    pos2, orn2 = get_random_pose()
    dishtowel.set_position_orientation(pos2, orn2)

    # Need to take one sim step
    og.sim.step()

    assert np.allclose(breakfast_table.states[AABB].get_value(), BoundingBoxAPI.compute_aabb(breakfast_table))
    assert np.all((breakfast_table.states[AABB].get_value()[0] < pos1) & (pos1 < breakfast_table.states[AABB].get_value()[1]))

    pp = dishtowel.root_link.compute_particle_positions()
    offset = dishtowel.root_link.cloth_system.particle_contact_offset
    assert np.allclose(dishtowel.states[AABB].get_value(), (pp.min(axis=0) - offset, pp.max(axis=0) + offset))
    assert np.all((dishtowel.states[AABB].get_value()[0] < pos2) & (pos2 < dishtowel.states[AABB].get_value()[1]))

    with pytest.raises(NotImplementedError):
        breakfast_table.states[AABB].set_value(None)


@og_test
def test_adjacency():
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    place_obj_on_floor_plane(bottom_cabinet)
    for i, (axis, obj) in enumerate(zip(("x", "y"), (bowl, dishtowel))):
        place_obj_on_floor_plane(obj, **{f"{axis}_offset": 0.4})
        og.sim.step()

        assert bottom_cabinet in set.union(
            *(axis.positive_neighbors | axis.negative_neighbors
              for coordinate in obj.states[HorizontalAdjacency].get_value() for axis in coordinate)
        )

    bowl.set_position([0., 0., 1.])
    dishtowel.set_position([0., 0., 2.0])

    # Need to take one sim step
    og.sim.step()

    assert bowl in bottom_cabinet.states[VerticalAdjacency].get_value().positive_neighbors
    # TODO: adjacency relies on raytest, which doesn't take particle systems into account
    # assert dishtowel in bottom_cabinet.states[VerticalAdjacency].get_value().positive_neighbors
    assert bottom_cabinet in bowl.states[VerticalAdjacency].get_value().negative_neighbors
    # TODO: adjacency relies on raytest, which doesn't take particle systems into account
    # assert dishtowel in bowl.states[VerticalAdjacency].get_value().positive_neighbors
    assert bottom_cabinet in dishtowel.states[VerticalAdjacency].get_value().negative_neighbors
    assert bowl in dishtowel.states[VerticalAdjacency].get_value().negative_neighbors

    with pytest.raises(NotImplementedError):
        bottom_cabinet.states[HorizontalAdjacency].set_value(None)
        bottom_cabinet.states[VerticalAdjacency].set_value(None)


@og_test
def test_temperature():
    microwave = og.sim.scene.object_registry("name", "microwave")
    stove = og.sim.scene.object_registry("name", "stove")
    fridge = og.sim.scene.object_registry("name", "fridge")
    plywood = og.sim.scene.object_registry("name", "plywood")
    bagel = og.sim.scene.object_registry("name", "bagel")
    dishtowel = og.sim.scene.object_registry("name", "cookable_dishtowel")

    place_obj_on_floor_plane(microwave)
    place_obj_on_floor_plane(stove, x_offset=1.0)
    place_obj_on_floor_plane(fridge, x_offset=2.0)
    place_obj_on_floor_plane(plywood, x_offset=3.0)

    # Set the objects to be far away
    place_obj_on_floor_plane(bagel, x_offset=-0.5)
    place_obj_on_floor_plane(dishtowel, x_offset=-1.0)

    for _ in range(5):
        og.sim.step()

    # Not affected by any heat source
    assert bagel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE

    # Open the microwave
    microwave.joints["j_link_0"].set_pos(np.pi / 2)

    # Set the objects to be inside the microwave
    bagel.set_position_orientation([0, 0, 0.11], [0, 0, 0, 1])
    dishtowel.set_position_orientation([-0.15, 0, 0.11], [0, 0, 0, 1])

    for _ in range(5):
        og.sim.step()

    # Not affected by any heat source (the microwave is NOT toggled on)
    assert bagel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE

    microwave.states[ToggledOn].set_value(True)

    for _ in range(5):
        og.sim.step()

    # Not affected by any heat source (the microwave is open)
    assert bagel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE

    microwave.joints["j_link_0"].set_pos(0.)

    for _ in range(5):
        og.sim.step()

    # Affected by the microwave
    bagel_new_temp = bagel.states[Temperature].get_value()
    dishtowel_new_temp = dishtowel.states[Temperature].get_value()
    assert bagel.states[Temperature].get_value() > m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() > m.object_states.temperature.DEFAULT_TEMPERATURE

    # Set the objects to be far away
    place_obj_on_floor_plane(bagel, x_offset=-0.5)
    place_obj_on_floor_plane(dishtowel, x_offset=-1.0)
    for _ in range(5):
        og.sim.step()

    # Not affected by any heat source (should cool down by itself towards the default temp)
    assert bagel.states[Temperature].get_value() < bagel_new_temp
    assert dishtowel.states[Temperature].get_value() < dishtowel_new_temp

    # Setter should work
    assert bagel.states[Temperature].set_value(m.object_states.temperature.DEFAULT_TEMPERATURE)
    assert dishtowel.states[Temperature].set_value(m.object_states.temperature.DEFAULT_TEMPERATURE)
    assert bagel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE

    # Set the objects to be on top of the stove
    bagel.set_position_orientation([0.71, 0.11, 0.88], [0, 0, 0, 1])
    dishtowel.set_position_orientation([0.84, 0.11, 0.88], [0, 0, 0, 1])

    for _ in range(5):
        og.sim.step()

    # Not affected by any heat source (the stove is off)
    assert bagel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE

    stove.states[ToggledOn].set_value(True)

    for _ in range(5):
        og.sim.step()

    # Affected by the stove
    assert bagel.states[Temperature].get_value() > m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() > m.object_states.temperature.DEFAULT_TEMPERATURE

    # Reset
    assert bagel.states[Temperature].set_value(m.object_states.temperature.DEFAULT_TEMPERATURE)
    assert dishtowel.states[Temperature].set_value(m.object_states.temperature.DEFAULT_TEMPERATURE)

    # Set the objects to be inside the fridge
    bagel.set_position_orientation([1.9, 0, 0.89], [0, 0, 0, 1])
    dishtowel.set_position_orientation([2.1, 0, 0.89], [0, 0, 0, 1])

    for _ in range(5):
        og.sim.step()

    # Affected by the fridge
    assert bagel.states[Temperature].get_value() < m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() < m.object_states.temperature.DEFAULT_TEMPERATURE

    # Reset temp
    assert bagel.states[Temperature].set_value(m.object_states.temperature.DEFAULT_TEMPERATURE)
    assert dishtowel.states[Temperature].set_value(m.object_states.temperature.DEFAULT_TEMPERATURE)

    # Set the objects to be near the plywood
    place_obj_on_floor_plane(bagel, x_offset=2.9)
    place_obj_on_floor_plane(dishtowel, x_offset=3.1)

    for _ in range(5):
        og.sim.step()

    # Not affected by any heat source (the plywood is NOT onfire)
    assert bagel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE

    plywood.states[OnFire].set_value(True)

    for _ in range(5):
        og.sim.step()

    assert bagel.states[Temperature].get_value() > m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() > m.object_states.temperature.DEFAULT_TEMPERATURE


@og_test
def test_max_temperature():
    bagel = og.sim.scene.object_registry("name", "bagel")
    dishtowel = og.sim.scene.object_registry("name", "cookable_dishtowel")

    assert bagel.states[MaxTemperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[MaxTemperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE

    assert bagel.states[MaxTemperature].set_value(m.object_states.temperature.DEFAULT_TEMPERATURE - 1)
    assert dishtowel.states[MaxTemperature].set_value(m.object_states.temperature.DEFAULT_TEMPERATURE - 1)
    assert bagel.states[MaxTemperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE - 1
    assert dishtowel.states[MaxTemperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE - 1

    bagel.states[Temperature].set_value(m.object_states.temperature.DEFAULT_TEMPERATURE + 1)
    dishtowel.states[Temperature].set_value(m.object_states.temperature.DEFAULT_TEMPERATURE + 1)

    og.sim.step()

    assert bagel.states[MaxTemperature].get_value() > m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[MaxTemperature].get_value() > m.object_states.temperature.DEFAULT_TEMPERATURE


@og_test
def test_heat_source_or_sink():
    microwave = og.sim.scene.object_registry("name", "microwave")
    stove = og.sim.scene.object_registry("name", "stove")
    fridge = og.sim.scene.object_registry("name", "fridge")

    assert microwave.states[HeatSourceOrSink].requires_inside
    assert microwave.states[HeatSourceOrSink].requires_closed
    assert microwave.states[HeatSourceOrSink].requires_toggled_on

    microwave.joints["j_link_0"].set_pos(np.pi / 2)
    microwave.states[ToggledOn].set_value(False)

    og.sim.step()
    assert not microwave.states[HeatSourceOrSink].get_value()

    microwave.joints["j_link_0"].set_pos(0.0)
    og.sim.step()
    assert not microwave.states[HeatSourceOrSink].get_value()

    microwave.states[ToggledOn].set_value(True)
    og.sim.step()
    assert microwave.states[HeatSourceOrSink].get_value()

    assert fridge.states[HeatSourceOrSink].requires_inside
    assert fridge.states[HeatSourceOrSink].requires_closed
    assert not fridge.states[HeatSourceOrSink].requires_toggled_on

    fridge.joints["j_link_0"].set_pos(np.pi / 2)

    og.sim.step()
    assert not fridge.states[HeatSourceOrSink].get_value()

    fridge.joints["j_link_0"].set_pos(0.0)
    og.sim.step()
    assert fridge.states[HeatSourceOrSink].get_value()

    assert not stove.states[HeatSourceOrSink].requires_inside
    assert not stove.states[HeatSourceOrSink].requires_closed
    assert stove.states[HeatSourceOrSink].requires_toggled_on

    stove.states[ToggledOn].set_value(False)

    og.sim.step()
    assert not stove.states[HeatSourceOrSink].get_value()

    stove.states[ToggledOn].set_value(True)
    og.sim.step()
    assert stove.states[HeatSourceOrSink].get_value()


@og_test
def test_cooked():
    bagel = og.sim.scene.object_registry("name", "bagel")
    dishtowel = og.sim.scene.object_registry("name", "cookable_dishtowel")

    assert not bagel.states[Cooked].get_value()
    assert not dishtowel.states[Cooked].get_value()

    bagel.states[MaxTemperature].set_value(bagel.states[Cooked].cook_temperature)
    dishtowel.states[MaxTemperature].set_value(dishtowel.states[Cooked].cook_temperature)
    og.sim.step()
    assert bagel.states[Cooked].get_value()
    assert dishtowel.states[Cooked].get_value()

    assert bagel.states[Cooked].set_value(False)
    assert dishtowel.states[Cooked].set_value(False)
    assert not bagel.states[Cooked].get_value()
    assert not dishtowel.states[Cooked].get_value()
    assert bagel.states[MaxTemperature].get_value() < bagel.states[Cooked].cook_temperature
    assert dishtowel.states[MaxTemperature].get_value() < dishtowel.states[Cooked].cook_temperature

    assert bagel.states[Cooked].set_value(True)
    assert dishtowel.states[Cooked].set_value(True)
    assert bagel.states[Cooked].get_value()
    assert dishtowel.states[Cooked].get_value()
    assert bagel.states[MaxTemperature].get_value() >= bagel.states[Cooked].cook_temperature
    assert dishtowel.states[MaxTemperature].get_value() >= dishtowel.states[Cooked].cook_temperature


@og_test
def test_burnt():
    bagel = og.sim.scene.object_registry("name", "bagel")
    dishtowel = og.sim.scene.object_registry("name", "cookable_dishtowel")

    assert not bagel.states[Burnt].get_value()
    assert not dishtowel.states[Burnt].get_value()

    bagel.states[MaxTemperature].set_value(bagel.states[Burnt].burn_temperature)
    dishtowel.states[MaxTemperature].set_value(dishtowel.states[Burnt].burn_temperature)
    og.sim.step()
    assert bagel.states[Burnt].get_value()
    assert dishtowel.states[Burnt].get_value()

    assert bagel.states[Burnt].set_value(False)
    assert dishtowel.states[Burnt].set_value(False)
    assert not bagel.states[Burnt].get_value()
    assert not dishtowel.states[Burnt].get_value()
    assert bagel.states[MaxTemperature].get_value() < bagel.states[Burnt].burn_temperature
    assert dishtowel.states[MaxTemperature].get_value() < dishtowel.states[Burnt].burn_temperature

    assert bagel.states[Burnt].set_value(True)
    assert dishtowel.states[Burnt].set_value(True)
    assert bagel.states[Burnt].get_value()
    assert dishtowel.states[Burnt].get_value()
    assert bagel.states[MaxTemperature].get_value() >= bagel.states[Burnt].burn_temperature
    assert dishtowel.states[MaxTemperature].get_value() >= dishtowel.states[Burnt].burn_temperature


@og_test
def test_frozen():
    bagel = og.sim.scene.object_registry("name", "bagel")
    dishtowel = og.sim.scene.object_registry("name", "cookable_dishtowel")

    assert not bagel.states[Frozen].get_value()
    assert not dishtowel.states[Frozen].get_value()

    bagel.states[Temperature].set_value(bagel.states[Frozen].freeze_temperature - 1)
    dishtowel.states[Temperature].set_value(dishtowel.states[Frozen].freeze_temperature - 1)
    og.sim.step()
    assert bagel.states[Frozen].get_value()
    assert dishtowel.states[Frozen].get_value()

    assert bagel.states[Frozen].set_value(False)
    assert dishtowel.states[Frozen].set_value(False)
    assert not bagel.states[Frozen].get_value()
    assert not dishtowel.states[Frozen].get_value()
    assert bagel.states[Temperature].get_value() > bagel.states[Frozen].freeze_temperature
    assert dishtowel.states[Temperature].get_value() > dishtowel.states[Frozen].freeze_temperature

    assert bagel.states[Frozen].set_value(True)
    assert dishtowel.states[Frozen].set_value(True)
    assert bagel.states[Frozen].get_value()
    assert dishtowel.states[Frozen].get_value()
    assert bagel.states[Temperature].get_value() <= bagel.states[Frozen].freeze_temperature
    assert dishtowel.states[Temperature].get_value() <= dishtowel.states[Frozen].freeze_temperature


@og_test
def test_heated():
    bagel = og.sim.scene.object_registry("name", "bagel")
    dishtowel = og.sim.scene.object_registry("name", "cookable_dishtowel")

    assert not bagel.states[Heated].get_value()
    assert not dishtowel.states[Heated].get_value()

    bagel.states[Temperature].set_value(bagel.states[Heated].heat_temperature + 1)
    dishtowel.states[Temperature].set_value(dishtowel.states[Heated].heat_temperature + 1)
    og.sim.step()
    assert bagel.states[Heated].get_value()
    assert dishtowel.states[Heated].get_value()

    assert bagel.states[Heated].set_value(False)
    assert dishtowel.states[Heated].set_value(False)
    assert not bagel.states[Heated].get_value()
    assert not dishtowel.states[Heated].get_value()
    assert bagel.states[Temperature].get_value() < bagel.states[Heated].heat_temperature
    assert dishtowel.states[Temperature].get_value() < dishtowel.states[Heated].heat_temperature

    assert bagel.states[Heated].set_value(True)
    assert dishtowel.states[Heated].set_value(True)
    assert bagel.states[Heated].get_value()
    assert dishtowel.states[Heated].get_value()
    assert bagel.states[Temperature].get_value() >= bagel.states[Heated].heat_temperature
    assert dishtowel.states[Temperature].get_value() >= dishtowel.states[Heated].heat_temperature


@og_test
def test_on_fire():
    plywood = og.sim.scene.object_registry("name", "plywood")

    assert not plywood.states[OnFire].get_value()

    plywood.states[Temperature].set_value(plywood.states[OnFire].ignition_temperature + 1)

    og.sim.step()
    assert plywood.states[OnFire].get_value()

    assert plywood.states[OnFire].set_value(False)
    assert not plywood.states[OnFire].get_value()
    assert plywood.states[Temperature].get_value() < plywood.states[OnFire].ignition_temperature

    assert plywood.states[OnFire].set_value(True)
    assert plywood.states[OnFire].get_value()
    assert plywood.states[Temperature].get_value() == plywood.states[OnFire].temperature

    for _ in range(5):
        og.sim.step()

    assert plywood.states[Temperature].get_value() == plywood.states[OnFire].temperature


@og_test
def test_toggled_on():
    stove = og.sim.scene.object_registry("name", "stove")
    robot = og.sim.scene.object_registry("name", "robot0")

    stove.set_position_orientation([1.46, 0.3, 0.45], T.euler2quat([0, 0, -np.pi / 2.0]))
    robot.set_position_orientation([0.01, 0.38, 0.01], [0, 0, 0, 1])

    assert not stove.states[ToggledOn].get_value()

    robot.joints["torso_lift_joint"].set_pos(0.0)
    robot.joints["shoulder_pan_joint"].set_pos(0.0)
    robot.joints["shoulder_lift_joint"].set_pos(np.pi / 15)
    robot.joints["upperarm_roll_joint"].set_pos(0.0)
    robot.joints["elbow_flex_joint"].set_pos(0.0)
    robot.joints["forearm_roll_joint"].set_pos(0.0)
    robot.joints["wrist_flex_joint"].set_pos(0.0)
    robot.joints["wrist_roll_joint"].set_pos(0.0)
    robot.joints["l_gripper_finger_joint"].set_pos(0.0)
    robot.joints["r_gripper_finger_joint"].set_pos(0.0)

    steps = m.object_states.toggle.CAN_TOGGLE_STEPS
    for _ in range(steps):
        og.sim.step()

    # End-effector not close to the button, stays False
    assert not stove.states[ToggledOn].get_value()

    robot.joints["shoulder_pan_joint"].set_pos(0)

    for _ in range(steps - 1):
        og.sim.step()

    # End-effector close to the button, but not enough time has passed, still False
    assert not stove.states[ToggledOn].get_value()

    og.sim.step()

    # Enough time has passed, turns True
    assert stove.states[ToggledOn].get_value()

    # Setter should work
    assert stove.states[ToggledOn].set_value(False)
    assert not stove.states[ToggledOn].get_value()


@og_test
def test_attached_to():
    shelf_back_panel = og.sim.scene.object_registry("name", "shelf_back_panel")
    shelf_shelf = og.sim.scene.object_registry("name", "shelf_shelf")
    shelf_baseboard = og.sim.scene.object_registry("name", "shelf_baseboard")

    shelf_back_panel.set_position_orientation([0, 0, 0.01], [0, 0, 0, 1])
    shelf_back_panel.keep_still()
    shelf_shelf.set_position_orientation([0, 0.03, 0.17], [0, 0, 0, 1])
    shelf_shelf.keep_still()

    # The shelf should not be attached to the back panel (no contact yet)
    assert not shelf_shelf.states[AttachedTo].get_value(shelf_back_panel)

    # Let the shelf fall
    for _ in range(10):
        og.sim.step()

    # The shelf should be attached to the back panel
    assert shelf_shelf.states[AttachedTo].get_value(shelf_back_panel)

    assert shelf_shelf.states[AttachedTo].set_value(shelf_back_panel, True)
    # The shelf should still be attached to the back panel
    assert shelf_shelf.states[AttachedTo].get_value(shelf_back_panel)

    assert shelf_shelf.states[AttachedTo].set_value(shelf_back_panel, False)
    # The shelf should not be attached to the back panel
    assert not shelf_shelf.states[AttachedTo].get_value(shelf_back_panel)

    assert shelf_shelf.states[AttachedTo].set_value(shelf_back_panel, True)
    # shelf should be attached to the back panel
    assert shelf_shelf.states[AttachedTo].get_value(shelf_back_panel)

    force_dir = np.array([0, 0, 1])
    # A small force will not break the attachment
    force_mag = 10
    apply_force_at_pos(shelf_shelf.root_link, force_dir * force_mag, shelf_shelf.get_position())
    og.sim.step()
    assert shelf_shelf.states[AttachedTo].get_value(shelf_back_panel)

    # A large force will break the attachment
    force_mag = 1000
    apply_force_at_pos(shelf_shelf.root_link, force_dir * force_mag, shelf_shelf.get_position())
    og.sim.step()
    assert not shelf_shelf.states[AttachedTo].get_value(shelf_back_panel)

    shelf_shelf.set_position_orientation([0, 0, 10], [0, 0, 0, 1])
    assert not shelf_shelf.states[AttachedTo].set_value(shelf_back_panel, True)
    # The shelf should not be attached to the back panel because the alignment is wrong
    assert not shelf_shelf.states[AttachedTo].get_value(shelf_back_panel)

    assert shelf_shelf.states[AttachedTo].set_value(shelf_back_panel, True, bypass_alignment_checking=True)
    # The shelf should be attached to the back panel because the alignment checking is bypassed
    assert shelf_shelf.states[AttachedTo].get_value(shelf_back_panel)

    # The shelf baseboard should NOT be attached because the attachment has the wrong type
    shelf_baseboard.set_position_orientation([0.37, -0.93,  0.03], [0, 0, 0, 1])
    assert not shelf_baseboard.states[AttachedTo].set_value(shelf_back_panel, True, bypass_alignment_checking=True)
    assert not shelf_baseboard.states[AttachedTo].get_value(shelf_back_panel)


@og_test
def test_particle_source():
    sink = og.sim.scene.object_registry("name", "sink")

    place_obj_on_floor_plane(sink)
    for _ in range(3):
        og.sim.step()

    assert not sink.states[ToggledOn].get_value()
    water_system = get_system("water")
    # Sink is toggled off, no water should be present
    assert water_system.n_particles == 0

    sink.states[ToggledOn].set_value(True)

    for _ in range(sink.states[ParticleSource].n_steps_per_modification):
        og.sim.step()

    # Sink is toggled on, some water should be present
    assert water_system.n_particles > 0

    # Cannot set this state
    with pytest.raises(NotImplementedError):
        sink.states[ParticleSource].set_value(True)


@og_test
def test_particle_sink():
    sink = og.sim.scene.object_registry("name", "sink")
    place_obj_on_floor_plane(sink)
    for _ in range(3):
        og.sim.step()

    water_system = get_system("water")
    # There should be no water particles.
    assert water_system.n_particles == 0

    sink_pos = sink.states[ParticleSink].link.get_position()
    water_system.generate_particles(positions=[sink_pos + np.array([0, 0, 0.05])])
    # There should be exactly 1 water particle.
    assert water_system.n_particles == 1

    for _ in range(sink.states[ParticleSink].n_steps_per_modification):
        og.sim.step()

    # There should be no water particles because the fluid source absorbs them.
    assert water_system.n_particles == 0

    # Cannot set this state
    with pytest.raises(NotImplementedError):
        sink.states[ParticleSink].set_value(True)


@og_test
def test_particle_applier():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    spray_bottle = og.sim.scene.object_registry("name", "spray_bottle")
    applier_dishtowel = og.sim.scene.object_registry("name", "applier_dishtowel")

    # Test projection

    place_obj_on_floor_plane(breakfast_table)
    place_objA_on_objB_bbox(spray_bottle, breakfast_table, z_offset=0.1)
    spray_bottle.set_orientation(np.array([0.707, 0, 0, 0.707]))
    for _ in range(3):
        og.sim.step()

    assert not spray_bottle.states[ToggledOn].get_value()
    water_system = get_system("water")
    # Spray bottle is toggled off, no water should be present
    assert water_system.n_particles == 0

    # Take number of steps for water to be generated, make sure there is still no water
    n_applier_steps = spray_bottle.states[ParticleApplier].n_steps_per_modification
    for _ in range(n_applier_steps):
        og.sim.step()

    assert water_system.n_particles == 0

    # Turn particle applier on, and verify particles are generated after the same number of steps are taken
    spray_bottle.states[ToggledOn].set_value(True)

    for _ in range(n_applier_steps):
        og.sim.step()

    # Some water should be present
    assert water_system.n_particles > 0

    # Test adjacency

    water_system.remove_all_particles()
    spray_bottle.set_position_orientation(position=np.ones(3) * 50.0, orientation=np.array([0, 0, 0, 1.0]))

    place_objA_on_objB_bbox(applier_dishtowel, breakfast_table)
    og.sim.step()

    # no water should be present
    assert water_system.n_particles == 0

    # Take number of steps for water to be generated
    n_applier_steps = applier_dishtowel.states[ParticleApplier].n_steps_per_modification
    for _ in range(n_applier_steps):
        og.sim.step()

    # Some water should be present
    assert water_system.n_particles > 0

    # Cannot set this state
    with pytest.raises(NotImplementedError):
        spray_bottle.states[ParticleApplier].set_value(True)


@og_test
def test_particle_remover():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    vacuum = og.sim.scene.object_registry("name", "vacuum")
    remover_dishtowel = og.sim.scene.object_registry("name", "remover_dishtowel")

    # Test projection

    place_obj_on_floor_plane(breakfast_table)
    place_objA_on_objB_bbox(vacuum, breakfast_table, z_offset=0.02)
    for _ in range(3):
        og.sim.step()

    assert not vacuum.states[ToggledOn].get_value()
    water_system = get_system("water")
    # Place single particle of water on middle of table
    water_system.generate_particles(positions=[np.array([0, 0, breakfast_table.aabb[1][2] + water_system.particle_radius])])
    assert water_system.n_particles > 0

    # Take number of steps for water to be removed, make sure there is still water
    n_remover_steps = vacuum.states[ParticleRemover].n_steps_per_modification
    for _ in range(n_remover_steps):
        og.sim.step()

    assert water_system.n_particles > 0

    # Turn particle remover on, and verify particles are generated after the same number of steps are taken
    vacuum.states[ToggledOn].set_value(True)

    for _ in range(n_remover_steps):
        og.sim.step()

    # No water should be present
    assert water_system.n_particles == 0

    # Test adjacency

    vacuum.set_position(np.ones(3) * 50.0)
    place_objA_on_objB_bbox(remover_dishtowel, breakfast_table, z_offset=0.03)
    og.sim.step()
    # Place single particle of water on middle of table
    water_system.generate_particles(positions=[np.array([0, 0, breakfast_table.aabb[1][2] + water_system.particle_radius])])

    # Water should be present
    assert water_system.n_particles > 0

    # Take number of steps for water to be removed
    n_remover_steps = remover_dishtowel.states[ParticleRemover].n_steps_per_modification
    for _ in range(n_remover_steps):
        og.sim.step()

    # No water should be present
    assert water_system.n_particles == 0

    # Cannot set this state
    with pytest.raises(NotImplementedError):
        vacuum.states[ParticleRemover].set_value(True)


@og_test
def test_saturated():
    remover_dishtowel = og.sim.scene.object_registry("name", "remover_dishtowel")

    place_obj_on_floor_plane(remover_dishtowel)

    for _ in range(5):
        og.sim.step()

    water_system = get_system("water")

    # Place single row of water above dishtowel
    n_particles = 5
    remover_dishtowel.states[Saturated].set_limit(water_system, n_particles)
    water_system.generate_particles(positions=[np.array([0, 0, remover_dishtowel.aabb[1][2] + water_system.particle_radius * (1 + 2 * i)]) for i in range(n_particles)])

    # Take a few steps
    for _ in range(20):
        og.sim.step()

    # Make sure Saturated is True, and no particles exist
    assert water_system.n_particles == 0
    assert remover_dishtowel.states[Saturated].get_value(water_system)

    # Make sure we can toggle saturated to be true and false
    assert remover_dishtowel.states[Saturated].set_value(water_system, False)
    assert remover_dishtowel.states[Saturated].set_value(water_system, True)


@og_test
def test_open():
    microwave = og.sim.scene.object_registry("name", "microwave")
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet")

    # By default, objects should not be open.
    assert not microwave.states[Open].get_value()
    assert not bottom_cabinet.states[Open].get_value()

    # Set the joints to their upper limits.
    microwave.joints["j_link_0"].set_pos(microwave.joints["j_link_0"].upper_limit)
    bottom_cabinet.joints["j_link_2"].set_pos(bottom_cabinet.joints["j_link_2"].upper_limit)

    og.sim.step()

    # The objects should be open.
    assert microwave.states[Open].get_value()
    assert bottom_cabinet.states[Open].get_value()

    # Set the joints to their lower limits.
    microwave.joints["j_link_0"].set_pos(microwave.joints["j_link_0"].lower_limit)
    bottom_cabinet.joints["j_link_2"].set_pos(bottom_cabinet.joints["j_link_2"].lower_limit)

    og.sim.step()

    # The objects should not be open.
    assert not microwave.states[Open].get_value()
    assert not bottom_cabinet.states[Open].get_value()

    # Setters should work.
    assert microwave.states[Open].set_value(True)
    assert bottom_cabinet.states[Open].set_value(True)

    # The objects should be open.
    assert microwave.states[Open].get_value()
    assert bottom_cabinet.states[Open].get_value()

    # Setters should work.
    assert microwave.states[Open].set_value(False)
    assert bottom_cabinet.states[Open].set_value(False)

    # The objects should not be open.
    assert not microwave.states[Open].get_value()
    assert not bottom_cabinet.states[Open].get_value()


@og_test
def test_folded_unfolded():
    carpet = og.sim.scene.object_registry("name", "carpet")

    place_obj_on_floor_plane(carpet)

    for _ in range(5):
        og.sim.step()

    assert not carpet.states[Folded].get_value()
    assert carpet.states[Unfolded].get_value()

    pos = carpet.root_link.compute_particle_positions()
    x_min, x_max = np.min(pos, axis=0)[0], np.max(pos, axis=0)[0]
    x_extent = x_max - x_min
    # Get indices for the bottom 10 percent vertices in the x-axis
    indices = np.argsort(pos, axis=0)[:, 0][:(pos.shape[0] // 10)]
    start = np.copy(pos[indices])

    # lift up a bit
    mid = np.copy(start)
    mid[:, 2] += x_extent * 0.2

    # move towards x_max
    end = np.copy(mid)
    end[:, 0] += x_extent * 0.9

    increments = 25
    for ctrl_pts in np.concatenate([np.linspace(start, mid, increments), np.linspace(mid, end, increments)]):
        carpet.root_link.set_particle_positions(ctrl_pts, idxs=indices)
        og.sim.step()

    assert carpet.states[Folded].get_value()
    assert not carpet.states[Unfolded].get_value()
    assert carpet.states[Unfolded].set_value(True)

    with pytest.raises(NotImplementedError):
        carpet.states[Unfolded].set_value(False)

    with pytest.raises(NotImplementedError):
        carpet.states[Folded].set_value(True)


@og_test
def test_draped():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    carpet = og.sim.scene.object_registry("name", "carpet")

    place_obj_on_floor_plane(breakfast_table)
    place_objA_on_objB_bbox(carpet, breakfast_table)

    for _ in range(10):
        og.sim.step()

    assert carpet.states[Draped].get_value(breakfast_table)

    carpet.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not carpet.states[Draped].get_value(breakfast_table)

    assert carpet.states[Draped].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        carpet.states[Draped].set_value(breakfast_table, False)


@og_test
def test_filled():
    stockpot = og.sim.scene.object_registry("name", "stockpot")

    systems = (
        get_system("water"),
        get_system("raspberry"),
        get_system("diced_apple"),
    )
    for system in systems:
        stockpot.set_position_orientation(position=np.ones(3) * 50.0, orientation=[0, 0, 0, 1.0])
        place_obj_on_floor_plane(stockpot)
        for _ in range(5):
            og.sim.step()

        assert stockpot.states[Filled].set_value(system, True)

        for _ in range(5):
            og.sim.step()

        assert stockpot.states[Filled].get_value(system)

        # Cannot set Filled state False
        with pytest.raises(NotImplementedError):
            stockpot.states[Filled].set_value(system, False)
        system.remove_all_particles()

        for _ in range(5):
            og.sim.step()
        assert not stockpot.states[Filled].get_value(system)

        system.remove_all_particles()


@og_test
def test_contains():
    stockpot = og.sim.scene.object_registry("name", "stockpot")

    systems = (
        get_system("water"),
        get_system("stain"),
        get_system("raspberry"),
        get_system("diced__apple"),
    )
    for system in systems:
        stockpot.set_position_orientation(position=np.ones(3) * 50.0, orientation=[0, 0, 0, 1.0])
        place_obj_on_floor_plane(stockpot)
        for _ in range(5):
            og.sim.step()

        # Sample single particle
        if is_physical_particle_system(system_name=system.name):
            system.generate_particles(positions=[np.array([0, 0, stockpot.aabb[1][2] + system.particle_radius * 1.01])])
            assert not stockpot.states[Contains].get_value(system)
        else:
            if system.get_group_name(stockpot) not in system.groups:
                system.create_attachment_group(stockpot)
            system.generate_group_particles(
                group=system.get_group_name(stockpot),
                positions=np.array([np.array([0, 0, stockpot.aabb[1][2] - 0.1])]),
                link_prim_paths=[stockpot.root_link.prim_path],
            )

        for _ in range(10):
            og.sim.step()

        assert stockpot.states[Contains].get_value(system)

        # Remove all particles and make sure contains returns False
        stockpot.states[Contains].set_value(system, False)
        og.sim.step()
        assert not stockpot.states[Contains].get_value(system)

        # Cannot set Contains state
        with pytest.raises(NotImplementedError):
            stockpot.states[Contains].set_value(system, True)


@og_test
def test_covered():
    bracelet = og.sim.scene.object_registry("name", "bracelet")
    oyster = og.sim.scene.object_registry("name", "oyster")
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")

    systems = (
        get_system("water"),
        get_system("stain"),
        get_system("raspberry"),
        get_system("diced__apple"),
    )
    for obj in (bracelet, oyster, breakfast_table):
        for system in systems:
            sampleable = is_visual_particle_system(system.name) or np.all(obj.aabb_extent > (2 * system.particle_radius))
            obj.set_position_orientation(position=np.ones(3) * 50.0, orientation=[0, 0, 0, 1.0])
            place_obj_on_floor_plane(obj)

            for _ in range(5):
                og.sim.step()

            assert obj.states[Covered].set_value(system, True) == sampleable

            for _ in range(5):
                og.sim.step()

            assert obj.states[Covered].get_value(system) == sampleable
            obj.states[Covered].set_value(system, False)

            for _ in range(5):
                og.sim.step()
            assert not obj.states[Covered].get_value(system)

            system.remove_all_particles()

        obj.set_position_orientation(position=np.ones(3) * 75.0, orientation=[0, 0, 0, 1.0])


def test_clear_sim():
    og.sim.clear()
