from omnigibson.macros import macros as m
from omnigibson.object_states import *
from omnigibson.systems import get_system
from omnigibson.utils.physx_utils import apply_force_at_pos, apply_torque
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import BoundingBoxAPI
import omnigibson as og

from utils import og_test, get_random_pose

import pytest
import numpy as np


@og_test
def test_on_top():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    breakfast_table.set_position([0., 0., 0.53])
    bowl.set_position([0., 0., 0.7])
    dishtowel.set_position([0.5, 0., 0.67])

    for _ in range(5):
        og.sim.step()

    assert bowl.states[OnTop].get_value(breakfast_table)
    assert dishtowel.states[OnTop].get_value(breakfast_table)

    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[OnTop].get_value(breakfast_table)
    assert not dishtowel.states[OnTop].get_value(breakfast_table)

    assert bowl.states[OnTop].set_value(breakfast_table, True)
    assert dishtowel.states[OnTop].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        bowl.states[OnTop].set_value(breakfast_table, False)


@og_test
def test_inside():
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    bottom_cabinet.set_position([0., 0., 0.38])
    bowl.set_position([0., 0., 0.08])
    dishtowel.set_position([0, 0., 0.63])

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

    breakfast_table.set_position([0., 0., 0.53])
    bowl.set_position([0., 0., 0.04])
    dishtowel.set_position([0.3, 0., 0.02])

    for _ in range(5):
        og.sim.step()

    assert bowl.states[Under].get_value(breakfast_table)
    assert dishtowel.states[Under].get_value(breakfast_table)

    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[Under].get_value(breakfast_table)
    assert not dishtowel.states[Under].get_value(breakfast_table)

    assert bowl.states[Under].set_value(breakfast_table, True)
    assert dishtowel.states[Under].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        bowl.states[Under].set_value(breakfast_table, False)


@og_test
def test_touching():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    breakfast_table.set_position([0., 0., 0.53])
    bowl.set_position([0., 0., 0.7])
    dishtowel.set_position([0.5, 0., 0.67])

    for _ in range(5):
        og.sim.step()

    assert bowl.states[Touching].get_value(breakfast_table)
    assert breakfast_table.states[Touching].get_value(bowl)
    assert dishtowel.states[Touching].get_value(breakfast_table)
    assert breakfast_table.states[Touching].get_value(dishtowel)

    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[Touching].get_value(breakfast_table)
    assert not breakfast_table.states[Touching].get_value(bowl)
    assert not dishtowel.states[Touching].get_value(breakfast_table)
    assert not breakfast_table.states[Touching].get_value(dishtowel)

    with pytest.raises(NotImplementedError):
        bowl.states[Touching].set_value(breakfast_table, None)


@og_test
def test_contact_bodies():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    breakfast_table.set_position([0., 0., 0.53])
    bowl.set_position([0., 0., 0.7])
    dishtowel.set_position([0.5, 0., 0.67])

    for _ in range(5):
        og.sim.step()

    assert bowl.root_link in breakfast_table.states[ContactBodies].get_value()
    # TODO: rigid body's ContactBodies should include cloth
    # assert dishtowel.root_link in breakfast_table.states[ContactBodies].get_value()
    assert breakfast_table.root_link in bowl.states[ContactBodies].get_value()
    assert breakfast_table.root_link in dishtowel.states[ContactBodies].get_value()

    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert bowl.root_link not in breakfast_table.states[ContactBodies].get_value()
    # TODO: rigid body's ContactBodies should include cloth
    # assert dishtowel.root_link in breakfast_table.states[ContactBodies].get_value()
    assert breakfast_table.root_link not in bowl.states[ContactBodies].get_value()
    assert breakfast_table.root_link not in dishtowel.states[ContactBodies].get_value()

    with pytest.raises(NotImplementedError):
        bowl.states[ContactBodies].set_value(None)


@og_test
def test_next_to():
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    bottom_cabinet.set_position([0., 0., 0.38])
    bowl.set_position([0.4, 0., 0.04])
    dishtowel.set_position([0., 0.4, 0.02])

    for _ in range(5):
        og.sim.step()

    assert bowl.states[NextTo].get_value(bottom_cabinet)
    assert bottom_cabinet.states[NextTo].get_value(bowl)
    assert dishtowel.states[NextTo].get_value(bottom_cabinet)
    assert bottom_cabinet.states[NextTo].get_value(dishtowel)

    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[NextTo].get_value(bottom_cabinet)
    assert not bottom_cabinet.states[NextTo].get_value(bowl)
    assert not dishtowel.states[NextTo].get_value(bottom_cabinet)
    assert not bottom_cabinet.states[NextTo].get_value(dishtowel)

    with pytest.raises(NotImplementedError):
        bowl.states[NextTo].set_value(bottom_cabinet, None)


@og_test
def test_overlaid():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    carpet = og.sim.scene.object_registry("name", "carpet")

    breakfast_table.set_position([0., 0., 0.53])
    carpet.set_position([0.0, 0., 0.67])

    for _ in range(5):
        og.sim.step()

    assert carpet.states[Overlaid].get_value(breakfast_table)

    carpet.set_position([20., 20., 1.])

    for _ in range(5):
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

    pp = dishtowel.root_link.particle_positions
    assert np.allclose(dishtowel.states[AABB].get_value(), (pp.min(axis=0), pp.max(axis=0)))
    assert np.all((dishtowel.states[AABB].get_value()[0] < pos2) & (pos2 < dishtowel.states[AABB].get_value()[1]))

    with pytest.raises(NotImplementedError):
        breakfast_table.states[AABB].set_value(None)


@og_test
def test_adjacency():
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    bottom_cabinet.set_position([0., 0., 0.38])
    bowl.set_position([0.4, 0., 0.04])
    dishtowel.set_position([0., 0.4, 0.02])

    # Need to take one sim step
    og.sim.step()

    assert bottom_cabinet in set.union(
        *(axis.positive_neighbors | axis.negative_neighbors
          for coordinate in bowl.states[HorizontalAdjacency].get_value() for axis in coordinate)
    )

    assert bottom_cabinet in set.union(
        *(axis.positive_neighbors | axis.negative_neighbors
          for coordinate in dishtowel.states[HorizontalAdjacency].get_value() for axis in coordinate)
    )

    bottom_cabinet.set_position([0., 0., 0.38])
    bowl.set_position([0., -0.08, 1.])
    dishtowel.set_position([0., -0.08, 2.0])

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

    microwave.set_position_orientation([0., 0., 0.15], [0, 0, 0, 1])
    stove.set_position_orientation([1, 0., 0.45], [0, 0, 0, 1])
    fridge.set_position_orientation([2, 0., 0.98], [0, 0, 0, 1])
    plywood.set_position_orientation([3, 0, 0.05], [0, 0, 0, 1])

    # Set the objects to be far away
    bagel.set_position_orientation([-0.5, 0., 0.03], [0, 0, 0, 1])
    dishtowel.set_position_orientation([-1.0, 0.0, 0.02], [0, 0, 0, 1])

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
    bagel.set_position_orientation([-0.5, 0., 0.03], [0, 0, 0, 1])
    dishtowel.set_position_orientation([-1.0, 0.0, 0.02], [0, 0, 0, 1])
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
    bagel.set_position_orientation([2.9, 0, 0.03], [0, 0, 0, 1])
    dishtowel.set_position_orientation([3.1, 0, 0.02], [0, 0, 0, 1])

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

    stove.set_position_orientation([1.5, 0.3, 0.45], T.euler2quat([0, 0, -np.pi / 2.0]))
    robot.set_position_orientation([0.01, 0.38, 0], [0, 0, 0, 1])

    assert not stove.states[ToggledOn].get_value()

    robot.joints["torso_lift_joint"].set_pos(0.0)
    robot.joints["shoulder_pan_joint"].set_pos(np.pi / 2)
    robot.joints["shoulder_lift_joint"].set_pos(np.pi / 36)
    robot.joints["upperarm_roll_joint"].set_pos(0.0)
    robot.joints["elbow_flex_joint"].set_pos(0.0)
    robot.joints["forearm_roll_joint"].set_pos(0.0)
    robot.joints["wrist_flex_joint"].set_pos(0.0)
    robot.joints["wrist_roll_joint"].set_pos(0.0)

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
def test_fluid_source():
    sink = og.sim.scene.object_registry("name", "sink")
    sink.set_position_orientation([0, 0, 0.7], [0, 0, 0, 1])
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


@og_test
def test_fluid_sink():
    sink = og.sim.scene.object_registry("name", "sink")
    sink.set_position_orientation([0, 0, 0.7], [0, 0, 0, 1])
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

    # TODO: current water sink annotation is wrong, so this test is failing.
    # There should be no water particles because the fluid source absorbs them.
    # assert water_system.n_particles == 0


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
def test_draped():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    carpet = og.sim.scene.object_registry("name", "carpet")

    breakfast_table.set_position([0., 0., 0.53])
    carpet.set_position([0.0, 0., 0.67])

    for _ in range(5):
        og.sim.step()

    assert carpet.states[Draped].get_value(breakfast_table)

    carpet.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not carpet.states[Draped].get_value(breakfast_table)

    assert carpet.states[Draped].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        carpet.states[Draped].set_value(breakfast_table, False)


def test_clear_sim():
    og.sim.clear()
