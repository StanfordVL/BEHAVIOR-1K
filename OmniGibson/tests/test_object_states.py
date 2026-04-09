import math

import pytest
import torch as th
from utils import SYSTEM_EXAMPLES, get_random_pose, place_obj_on_floor_plane, place_objA_on_objB_bbox

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import macros as m
from omnigibson.object_states import (
    AABB,
    AttachedTo,
    Burnt,
    ContactParticles,
    Contains,
    Cooked,
    Covered,
    Draped,
    Filled,
    Folded,
    Frozen,
    Heated,
    HeatSourceOrSink,
    HorizontalAdjacency,
    Inside,
    Joint,
    MaxTemperature,
    NextTo,
    OnFire,
    OnTop,
    Open,
    Overlaid,
    ParticleApplier,
    ParticleRemover,
    ParticleSink,
    ParticleSource,
    Pose,
    Saturated,
    Temperature,
    ToggledOn,
    Touching,
    Under,
    Unfolded,
    VerticalAdjacency,
)
from omnigibson.systems import VisualParticleSystem
from omnigibson.utils.physx_utils import apply_force_at_pos
from omnigibson.utils.usd_utils import RigidContactAPI


def test_attached_to(env, bookcase_back, bookcase_shelf, bookcase_baseboard):
    # Lower the mass of the shelf - otherwise, the gravity will create enough torque to break the joint
    bookcase_shelf.root_link.mass = 0.1

    bookcase_back.set_position_orientation(position=[0, 0, 0.01], orientation=[0, 0, 0, 1])
    bookcase_back.keep_still()
    bookcase_shelf.set_position_orientation(
        position=[0, 0.0286, 0.2], orientation=T.euler2quat(th.tensor([0.0, -math.pi / 2.0, math.pi / 2.0]))
    )
    bookcase_shelf.keep_still()

    og.sim.step()

    # The shelf should not be attached to the back panel (no contact yet)
    assert not bookcase_shelf.states[Touching].get_value(bookcase_back)
    assert not bookcase_shelf.states[AttachedTo].get_value(bookcase_back)

    # Let the shelf fall
    for _ in range(10):
        og.sim.step()

    # The shelf should be attached to the back panel
    assert bookcase_shelf.states[Touching].get_value(bookcase_back)
    assert bookcase_shelf.states[AttachedTo].get_value(bookcase_back)

    # Try to attach again (should be no-op)
    assert bookcase_shelf.states[AttachedTo].set_value(bookcase_back, True)
    # The shelf should still be attached to the back panel
    assert bookcase_shelf.states[AttachedTo].get_value(bookcase_back)

    # Detach
    assert bookcase_shelf.states[AttachedTo].set_value(bookcase_back, False)
    # The shelf should not be attached to the back panel
    assert not bookcase_shelf.states[AttachedTo].get_value(bookcase_back)

    # Attach again
    assert bookcase_shelf.states[AttachedTo].set_value(bookcase_back, True)
    # shelf should be attached to the back panel
    assert bookcase_shelf.states[AttachedTo].get_value(bookcase_back)

    force_dir = th.tensor([0, 0, 1])
    # A small force will not break the attachment
    force_mag = 10
    apply_force_at_pos(bookcase_shelf.root_link, force_dir * force_mag, bookcase_shelf.get_position_orientation()[0])
    og.sim.step()
    assert bookcase_shelf.states[AttachedTo].get_value(bookcase_back)

    # A large force will break the attachment
    force_mag = 1000
    apply_force_at_pos(bookcase_shelf.root_link, force_dir * force_mag, bookcase_shelf.get_position_orientation()[0])
    og.sim.step()
    assert not bookcase_shelf.states[AttachedTo].get_value(bookcase_back)

    bookcase_shelf.set_position_orientation(position=[0, 0, 10], orientation=[0, 0, 0, 1])
    assert not bookcase_shelf.states[AttachedTo].set_value(bookcase_back, True)
    # The shelf should not be attached to the back panel because the alignment is wrong
    assert not bookcase_shelf.states[AttachedTo].get_value(bookcase_back)

    assert bookcase_shelf.states[AttachedTo].set_value(bookcase_back, True, bypass_alignment_checking=True)
    # The shelf should be attached to the back panel because the alignment checking is bypassed
    assert bookcase_shelf.states[AttachedTo].get_value(bookcase_back)

    # The shelf baseboard should NOT be attached because the attachment has the wrong type
    bookcase_baseboard.set_position_orientation(position=[0.37, -0.93, 0.03], orientation=[0, 0, 0, 1])
    assert not bookcase_baseboard.states[AttachedTo].set_value(bookcase_back, True, bypass_alignment_checking=True)
    assert not bookcase_baseboard.states[AttachedTo].get_value(bookcase_back)


def test_on_top(env, breakfast_table, bowl, dishtowel):
    place_obj_on_floor_plane(breakfast_table)
    for i, obj in enumerate((bowl, dishtowel)):
        place_objA_on_objB_bbox(obj, breakfast_table)
        for _ in range(5):
            og.sim.step()

        assert obj.states[OnTop].get_value(breakfast_table)

        obj.set_position_orientation(position=th.ones(3) * 10 * (i + 1))
        og.sim.step()

        assert not obj.states[OnTop].get_value(breakfast_table)

    assert bowl.states[OnTop].set_value(breakfast_table, True)
    assert dishtowel.states[OnTop].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        bowl.states[OnTop].set_value(breakfast_table, False)


def test_inside(env, bottom_cabinet, bowl, dishtowel):
    place_obj_on_floor_plane(bottom_cabinet)
    bowl.set_position_orientation(position=[0.0, 0.0, 0.08])
    dishtowel.set_position_orientation(position=[0, 0.0, 0.5])

    for _ in range(5):
        og.sim.step()

    assert bowl.states[Inside].get_value(bottom_cabinet)
    assert dishtowel.states[Inside].get_value(bottom_cabinet)

    bowl.set_position_orientation(position=[10.0, 10.0, 1.0])
    dishtowel.set_position_orientation(position=[20.0, 20.0, 1.0])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[Inside].get_value(bottom_cabinet)
    assert not dishtowel.states[Inside].get_value(bottom_cabinet)

    assert bowl.states[Inside].set_value(bottom_cabinet, True)
    assert dishtowel.states[Inside].set_value(bottom_cabinet, True)

    with pytest.raises(NotImplementedError):
        bowl.states[OnTop].set_value(bottom_cabinet, False)


def test_under(env, breakfast_table, bowl, dishtowel):
    place_obj_on_floor_plane(breakfast_table)
    for i, obj in enumerate((bowl, dishtowel)):
        place_obj_on_floor_plane(obj)
        for _ in range(5):
            og.sim.step()

        assert obj.states[Under].get_value(breakfast_table)

        obj.set_position_orientation(position=th.ones(3) * 10 * (i + 1))
        og.sim.step()

        assert not obj.states[Under].get_value(breakfast_table)

    assert bowl.states[Under].set_value(breakfast_table, True)
    assert dishtowel.states[Under].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        bowl.states[Under].set_value(breakfast_table, False)


def test_touching(env, breakfast_table, bowl, dishtowel):
    place_obj_on_floor_plane(breakfast_table)
    for i, obj in enumerate((bowl, dishtowel)):
        place_objA_on_objB_bbox(obj, breakfast_table)
        for _ in range(5):
            og.sim.step()

        assert obj.states[Touching].get_value(breakfast_table)
        assert breakfast_table.states[Touching].get_value(obj)

        obj.set_position_orientation(position=th.ones(3) * 10 * (i + 1))
        og.sim.step()

        assert not obj.states[Touching].get_value(breakfast_table)
        assert not breakfast_table.states[Touching].get_value(obj)

    with pytest.raises(NotImplementedError):
        bowl.states[Touching].set_value(breakfast_table, None)


def test_rigid_contact_bodies(env, breakfast_table, bowl):
    from omnigibson.utils.usd_utils import RigidContactAPI

    place_obj_on_floor_plane(breakfast_table)
    place_objA_on_objB_bbox(bowl, breakfast_table)
    for _ in range(5):
        og.sim.step()

    # Bowl should be in contact with the table
    assert RigidContactAPI.is_in_contact(
        scene_idx=env.scene.idx, query_set=[bowl], with_set=[breakfast_table], ignore_set=None, current_only=False
    )
    assert RigidContactAPI.is_in_contact(
        scene_idx=env.scene.idx, query_set=[bowl], with_set=[breakfast_table], ignore_set=None, current_only=True
    )

    # Let bodies settle/sleep and verify contacts persist
    for _ in range(300):
        og.sim.step()
    assert breakfast_table.is_asleep, "Table should be asleep"
    assert bowl.root_link.is_asleep, "Bowl should be asleep"
    assert RigidContactAPI.is_in_contact(
        scene_idx=env.scene.idx, query_set=[bowl], with_set=[breakfast_table], ignore_set=None, current_only=False
    )
    assert RigidContactAPI.is_in_contact(
        scene_idx=env.scene.idx, query_set=[bowl], with_set=[breakfast_table], ignore_set=None, current_only=True
    )

    # Move the bowl far away and verify contacts clear
    bowl.set_position_orientation(position=th.tensor([10.0, 10.0, 10.0]))
    og.sim.step()
    assert not RigidContactAPI.is_in_contact(
        scene_idx=env.scene.idx, query_set=[bowl], with_set=[breakfast_table], ignore_set=None, current_only=False
    )
    assert not RigidContactAPI.is_in_contact(
        scene_idx=env.scene.idx, query_set=[bowl], with_set=[breakfast_table], ignore_set=None, current_only=True
    )
    assert not RigidContactAPI.is_in_contact(
        scene_idx=env.scene.idx, query_set=[breakfast_table], with_set=[bowl], ignore_set=None, current_only=False
    )
    assert not RigidContactAPI.is_in_contact(
        scene_idx=env.scene.idx, query_set=[breakfast_table], with_set=[bowl], ignore_set=None, current_only=True
    )


def test_next_to(env, bottom_cabinet, bowl, dishtowel):
    place_obj_on_floor_plane(bottom_cabinet)
    for i, (axis, obj) in enumerate(zip(("x", "y"), (bowl, dishtowel))):
        place_obj_on_floor_plane(obj, **{f"{axis}_offset": 0.3})
        for _ in range(5):
            og.sim.step()

        assert obj.states[NextTo].get_value(bottom_cabinet)
        assert bottom_cabinet.states[NextTo].get_value(obj)

        obj.set_position_orientation(position=th.ones(3) * 10 * (i + 1))
        og.sim.step()

        assert not obj.states[NextTo].get_value(bottom_cabinet)
        assert not bottom_cabinet.states[NextTo].get_value(obj)

    with pytest.raises(NotImplementedError):
        bowl.states[NextTo].set_value(bottom_cabinet, None)


@pytest.mark.skip(reason="disable cloth tests until #2042")
def test_overlaid(env, breakfast_table, carpet):
    place_obj_on_floor_plane(breakfast_table)
    place_objA_on_objB_bbox(carpet, breakfast_table)

    for _ in range(5):
        og.sim.step()

    assert carpet.states[Overlaid].get_value(breakfast_table)

    carpet.set_position_orientation(position=th.ones(3) * 20.0)
    og.sim.step()

    assert not carpet.states[Overlaid].get_value(breakfast_table)

    assert carpet.states[Overlaid].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        carpet.states[Overlaid].set_value(breakfast_table, False)


def test_pose(env, breakfast_table, dishtowel):
    pos1, orn1 = get_random_pose()
    breakfast_table.set_position_orientation(position=pos1, orientation=orn1)

    pos2, orn2 = get_random_pose()
    dishtowel.set_position_orientation(position=pos2, orientation=orn2)

    assert th.allclose(breakfast_table.states[Pose].get_value()[0], pos1)
    assert th.allclose(breakfast_table.states[Pose].get_value()[1], orn1) or th.allclose(
        breakfast_table.states[Pose].get_value()[1], -orn1
    )
    assert th.allclose(dishtowel.states[Pose].get_value()[0], pos2)
    assert th.allclose(dishtowel.states[Pose].get_value()[1], orn2) or th.allclose(
        dishtowel.states[Pose].get_value()[1], -orn2
    )

    with pytest.raises(NotImplementedError):
        breakfast_table.states[Pose].set_value(None)


def test_joint(env, breakfast_table, bottom_cabinet):
    lo = bottom_cabinet.joint_lower_limits
    hi = bottom_cabinet.joint_upper_limits
    q_rand = lo + (hi - lo) * th.rand(bottom_cabinet.n_joints)
    bottom_cabinet.set_joint_positions(q_rand)

    assert th.allclose(bottom_cabinet.states[Joint].get_value(), q_rand)
    assert len(breakfast_table.states[Joint].get_value()) == 0

    with pytest.raises(NotImplementedError):
        bottom_cabinet.states[Joint].set_value(None)


def test_aabb(env, breakfast_table, dishtowel):
    pos1, orn1 = get_random_pose()
    breakfast_table.set_position_orientation(position=pos1, orientation=orn1)

    pos2, orn2 = get_random_pose()
    dishtowel.set_position_orientation(position=pos2, orientation=orn2)

    # Need to take one sim step
    og.sim.step()

    assert th.allclose(breakfast_table.states[AABB].get_value()[0], breakfast_table.aabb[0])
    assert th.allclose(breakfast_table.states[AABB].get_value()[1], breakfast_table.aabb[1])
    assert th.all(
        (breakfast_table.states[AABB].get_value()[0] < pos1) & (pos1 < breakfast_table.states[AABB].get_value()[1])
    )

    # skip until #2042
    # pp = dishtowel.root_link.compute_particle_positions()
    # offset = dishtowel.root_link.cloth_system.particle_contact_offset
    # particle_aabb = (pp.min(dim=0).values - offset, pp.max(dim=0).values + offset)
    # assert th.allclose(dishtowel.states[AABB].get_value()[0], particle_aabb[0])
    # assert th.allclose(dishtowel.states[AABB].get_value()[1], particle_aabb[1])
    # assert th.all(
    #     (dishtowel.states[AABB].get_value()[0] < pos2) & (pos2 < dishtowel.states[AABB].get_value()[1])
    # ).item()

    with pytest.raises(NotImplementedError):
        breakfast_table.states[AABB].set_value(None)


def test_adjacency(env, bottom_cabinet, bowl, dishtowel):
    place_obj_on_floor_plane(bottom_cabinet)
    for i, (axis, obj) in enumerate(zip(("x", "y"), (bowl, dishtowel))):
        place_obj_on_floor_plane(obj, **{f"{axis}_offset": 0.4})
        og.sim.step()

        assert bottom_cabinet in set.union(
            *(
                axis.positive_neighbors | axis.negative_neighbors
                for coordinate in obj.states[HorizontalAdjacency].get_value()
                for axis in coordinate
            )
        )

    bowl.set_position_orientation(position=[0.0, 0.0, 1.0])
    dishtowel.set_position_orientation(position=[0.0, 0.0, 2.0])

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


def test_temperature(env, microwave, stove, fridge, plywood, bagel, cookable_dishtowel):
    dishtowel = cookable_dishtowel

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
    microwave.joints["j_link_0"].set_pos(math.pi / 2)

    # Set the objects to be inside the microwave
    bagel.set_position_orientation(position=[0, 0.03, 0.11], orientation=[0, 0, 0, 1])
    dishtowel.set_position_orientation(position=[0.0, -0.12, 0.11], orientation=[0, 0, 0, 1])

    for _ in range(5):
        og.sim.step()

    # Not affected by any heat source (the microwave is NOT toggled on)
    assert bagel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE

    # The microwave should not turn on when the door is open
    assert not microwave.states[ToggledOn].set_value(True)

    for _ in range(5):
        og.sim.step()

    # Not affected by any heat source (the microwave is open)
    assert bagel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE
    assert dishtowel.states[Temperature].get_value() == m.object_states.temperature.DEFAULT_TEMPERATURE

    microwave.joints["j_link_0"].set_pos(0.0)
    og.sim.step()
    assert microwave.states[ToggledOn].set_value(True)

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
    bagel.set_position_orientation(position=[0.78, -0.2, 0.88], orientation=[0, 0, 0, 1])
    dishtowel.set_position_orientation(position=[0.84, -0.15, 0.89], orientation=[0, 0, 0, 1])

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
    bagel.set_position_orientation(position=[1.9, 0, 0.7], orientation=[0, 0, 0, 1])
    dishtowel.set_position_orientation(position=[2.1, 0, 0.7], orientation=[0, 0, 0, 1])

    assert fridge.states[Open].set_value(False, fully=True)

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


def test_max_temperature(env, bagel, cookable_dishtowel):
    dishtowel = env.scene.object_registry("name", "cookable_dishtowel")

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


def test_heat_source_or_sink(env, microwave, stove, fridge):
    assert microwave.states[HeatSourceOrSink].requires_inside
    assert microwave.states[HeatSourceOrSink].requires_closed
    assert microwave.states[HeatSourceOrSink].requires_toggled_on

    microwave.joints["j_link_0"].set_pos(math.pi / 2)
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

    fridge.joints["j_link_0"].set_pos(math.pi / 2)

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


def test_cooked(env, bagel, cookable_dishtowel):
    dishtowel = env.scene.object_registry("name", "cookable_dishtowel")

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


def test_burnt(env, bagel, cookable_dishtowel):
    dishtowel = env.scene.object_registry("name", "cookable_dishtowel")

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


def test_frozen(env, bagel, cookable_dishtowel):
    dishtowel = env.scene.object_registry("name", "cookable_dishtowel")

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


def test_heated(env, bagel, cookable_dishtowel):
    dishtowel = env.scene.object_registry("name", "cookable_dishtowel")

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


def test_on_fire(env, plywood):
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


@pytest.mark.skip(
    reason="investigate why extra steps are needed for the contact to be registered beyond CAN_TOGGLE_STEPS"
)
def test_toggled_on(env, stove, robot):
    stove.set_position_orientation([1.487, 0.3, 0.443], T.euler2quat(th.tensor([0, 0, math.pi], dtype=th.float32)))
    robot.set_position_orientation(position=[0.0, 0.38, 0.0], orientation=[0, 0, 0, 1])

    assert not stove.states[ToggledOn].get_value()

    # Make the toggle button huge to add tolerance
    stove.states[ToggledOn].link.scale = 3.0

    q = robot.get_joint_positions()
    jnt_idxs = {name: i for i, name in enumerate(robot.joints.keys())}
    q[jnt_idxs["torso_lift_joint"]] = 0.0
    q[jnt_idxs["shoulder_pan_joint"]] = th.deg2rad(th.tensor([90.0])).item()
    q[jnt_idxs["shoulder_lift_joint"]] = th.deg2rad(th.tensor([9.0])).item()
    q[jnt_idxs["upperarm_roll_joint"]] = 0.0
    q[jnt_idxs["elbow_flex_joint"]] = 0.0
    q[jnt_idxs["forearm_roll_joint"]] = 0.0
    q[jnt_idxs["wrist_flex_joint"]] = 0.0
    q[jnt_idxs["wrist_roll_joint"]] = 0.0
    q[jnt_idxs["l_gripper_finger_joint"]] = 0.0
    q[jnt_idxs["r_gripper_finger_joint"]] = 0.0
    robot.set_joint_positions(q, drive=False)

    steps = m.object_states.toggle.CAN_TOGGLE_STEPS
    for _ in range(steps):
        og.sim.step()

    # End-effector not close to the button, stays False
    assert not stove.states[ToggledOn].get_value()

    # Settle robot
    for _ in range(10):
        og.sim.step()

    q[jnt_idxs["shoulder_pan_joint"]] = 0.0
    robot.set_joint_positions(q, drive=False)

    for _ in range(steps - 1):
        og.sim.step()
        robot.set_joint_positions(q, drive=False)
        robot.keep_still()

    # End-effector close to the button, but not enough time has passed, still False
    assert not stove.states[ToggledOn].get_value()

    robot.set_joint_positions(q, drive=False)
    robot.keep_still()
    og.sim.step()

    # Enough time has passed, turns True
    assert stove.states[ToggledOn].get_value()

    # Setter should work
    assert stove.states[ToggledOn].set_value(False)
    assert not stove.states[ToggledOn].get_value()


def test_particle_source(env, furniture_sink):
    sink = env.scene.object_registry("name", "furniture_sink")

    place_obj_on_floor_plane(sink)
    for _ in range(3):
        og.sim.step()

    assert not sink.states[ToggledOn].get_value()
    water_system = env.scene.get_system("water")
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

    water_system.remove_all_particles()


def test_particle_sink(env, furniture_sink):
    sink = env.scene.object_registry("name", "furniture_sink")
    place_obj_on_floor_plane(sink)
    for _ in range(3):
        og.sim.step()

    water_system = env.scene.get_system("water")
    # There should be no water particles.
    assert water_system.n_particles == 0

    sink_pos = sink.states[ParticleSink].link.get_position_orientation()[0]
    water_system.generate_particles(positions=[(sink_pos + th.tensor([0, 0, 0.05])).tolist()])
    # There should be exactly 1 water particle.
    assert water_system.n_particles == 1

    for _ in range(sink.states[ParticleSink].n_steps_per_modification):
        og.sim.step()

    # There should be no water particles because the fluid source absorbs them.
    assert water_system.n_particles == 0

    # Cannot set this state
    with pytest.raises(NotImplementedError):
        sink.states[ParticleSink].set_value(True)

    water_system.remove_all_particles()


@pytest.mark.skip(reason="investigate why particle applier and remover tests are failing, see issue #2066")
def test_particle_applier(env, breakfast_table, acetone_atomizer, applier_dishtowel):
    # Test projection

    place_obj_on_floor_plane(breakfast_table)
    place_objA_on_objB_bbox(acetone_atomizer, breakfast_table, z_offset=0.1)
    acetone_atomizer.set_orientation(th.tensor([0.707, 0, 0, 0.707]))
    for _ in range(3):
        og.sim.step()

    assert not acetone_atomizer.states[ToggledOn].get_value()
    water_system = env.scene.get_system("water")
    # Spray bottle is toggled off, no water should be present
    assert water_system.n_particles == 0

    # Take number of steps for water to be generated, make sure there is still no water
    n_applier_steps = acetone_atomizer.states[ParticleApplier].n_steps_per_modification
    for _ in range(n_applier_steps):
        og.sim.step()

    assert water_system.n_particles == 0

    # Turn particle applier on, and verify particles are generated after the same number of steps are taken
    acetone_atomizer.states[ToggledOn].set_value(True)

    for _ in range(n_applier_steps):
        og.sim.step()

    # Some water should be present
    assert water_system.n_particles > 0

    # Test adjacency

    water_system.remove_all_particles()
    acetone_atomizer.set_position_orientation(position=th.ones(3) * 50.0, orientation=th.tensor([0, 0, 0, 1.0]))

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
        acetone_atomizer.states[ParticleApplier].set_value(True)

    water_system.remove_all_particles()


@pytest.mark.skip(reason="investigate why particle applier and remover tests are failing, see issue #2066")
def test_particle_remover(env, breakfast_table, vacuum, remover_dishtowel):
    # Test projection

    place_obj_on_floor_plane(breakfast_table)
    place_objA_on_objB_bbox(vacuum, breakfast_table, z_offset=0.02)
    for _ in range(3):
        og.sim.step()

    assert not vacuum.states[ToggledOn].get_value()
    water_system = env.scene.get_system("water")
    # Place single particle of water on middle of table
    water_system.generate_particles(
        positions=[[0, 0, breakfast_table.aabb[1][2].item() + water_system.particle_radius]]
    )
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

    vacuum.set_position_orientation(position=th.ones(3) * 50.0)
    place_objA_on_objB_bbox(remover_dishtowel, breakfast_table, z_offset=0.03)
    og.sim.step()
    # Place single particle of water on middle of table
    water_system.generate_particles(
        positions=[[0, 0, breakfast_table.aabb[1][2].item() + water_system.particle_radius]]
    )

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

    water_system.remove_all_particles()


def test_saturated(env, remover_dishtowel):
    place_obj_on_floor_plane(remover_dishtowel)

    for _ in range(5):
        og.sim.step()

    water_system = env.scene.get_system("water")

    # Place single row of water above dishtowel
    n_particles = 5
    remover_dishtowel.states[Saturated].set_limit(water_system, n_particles)
    water_system.generate_particles(
        positions=[
            [0, 0, remover_dishtowel.aabb[1][2].item() + water_system.particle_radius * (1 + 2 * i)]
            for i in range(n_particles)
        ],
    )

    # Take a few steps
    for _ in range(20):
        og.sim.step()

    # Make sure Saturated is True, and no particles exist
    assert water_system.n_particles == 0
    assert remover_dishtowel.states[Saturated].get_value(water_system)

    # Make sure we can toggle saturated to be true and false
    assert remover_dishtowel.states[Saturated].set_value(water_system, False)
    assert remover_dishtowel.states[Saturated].set_value(water_system, True)

    water_system.remove_all_particles()


def test_open(env, microwave, bottom_cabinet):
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


@pytest.mark.skip(reason="disable cloth tests until #2042")
def test_folded_unfolded(env, carpet):
    place_obj_on_floor_plane(carpet)

    for _ in range(10):
        og.sim.step()

    carpet.reset()
    carpet.keep_still()

    for _ in range(10):
        og.sim.step()

    assert not carpet.states[Folded].get_value()
    assert carpet.states[Unfolded].get_value()

    pos = carpet.root_link.compute_particle_positions()
    y_min, y_max = th.min(pos, dim=0).values[1], th.max(pos, dim=0).values[1]
    y_extent = y_max - y_min
    # Get indices for the bottom 10 percent vertices in the y-axis
    indices = th.argsort(pos, dim=0)[:, 1][: (pos.shape[0] // 10)]
    start = th.clone(pos[indices])

    # lift up a bit
    mid = th.clone(start)
    mid[:, 2] += y_extent * 0.2

    # move towards y_max
    end = th.clone(mid)
    end[:, 1] += y_extent * 0.9

    increments = 25
    total_points = increments * 2

    # Create interpolation weights
    t = th.linspace(0, 1, total_points).unsqueeze(1).unsqueeze(2)

    # Interpolate between start, mid, and end
    first_half = t < 0.5
    ctrl_pts = th.where(first_half, start * (1 - 2 * t) + mid * (2 * t), mid * (2 - 2 * t) + end * (2 * t - 1))

    for pt in ctrl_pts:
        carpet.root_link.set_particle_positions(pt, idxs=indices)
        og.sim.step()

    assert carpet.states[Folded].get_value()
    assert not carpet.states[Unfolded].get_value()
    assert carpet.states[Unfolded].set_value(True)

    with pytest.raises(NotImplementedError):
        carpet.states[Unfolded].set_value(False)

    with pytest.raises(NotImplementedError):
        carpet.states[Folded].set_value(True)


@pytest.mark.skip(reason="disable cloth tests until #2042")
def test_draped(env, breakfast_table, carpet):
    place_obj_on_floor_plane(breakfast_table)
    place_objA_on_objB_bbox(carpet, breakfast_table)

    for _ in range(10):
        og.sim.step()

    assert carpet.states[Draped].get_value(breakfast_table)

    carpet.set_position_orientation(position=[20.0, 20.0, 1.0])

    for _ in range(5):
        og.sim.step()

    assert not carpet.states[Draped].get_value(breakfast_table)

    assert carpet.states[Draped].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        carpet.states[Draped].set_value(breakfast_table, False)


def test_filled(env, stockpot):
    systems = [
        env.scene.get_system(system_name)
        for system_name, system_class in SYSTEM_EXAMPLES.items()
        if not issubclass(system_class, VisualParticleSystem)
    ]
    for system in systems:
        stockpot.set_position_orientation(position=th.ones(3) * 50.0, orientation=[0, 0, 0, 1.0])
        place_obj_on_floor_plane(stockpot)
        for _ in range(5):
            og.sim.step()

        assert stockpot.states[Filled].set_value(system, True)
        og.sim.step()
        assert stockpot.states[Filled].get_value(system)

        # Cannot set Filled state False
        with pytest.raises(NotImplementedError):
            stockpot.states[Filled].set_value(system, False)

        system.remove_all_particles()
        og.sim.step()
        assert not stockpot.states[Filled].get_value(system)


def test_contains(env, stockpot):
    systems = [env.scene.get_system(system_name) for system_name, system_class in SYSTEM_EXAMPLES.items()]
    for system in systems:
        print(f"Testing Contains {stockpot.name} with {system.name}")
        stockpot.set_position_orientation(position=th.ones(3) * 50.0, orientation=[0, 0, 0, 1.0])
        place_obj_on_floor_plane(stockpot)
        for _ in range(5):
            og.sim.step()

        # Sample single particle
        if env.scene.is_physical_particle_system(system_name=system.name):
            system.generate_particles(positions=[[0, 0, stockpot.aabb[1][2].item() - 0.1]])
        else:
            if system.get_group_name(stockpot) not in system.groups:
                system.create_attachment_group(stockpot)
            system.generate_group_particles(
                group=system.get_group_name(stockpot),
                positions=[th.tensor([0, 0, stockpot.aabb[1][2] - 0.1])],
                link_prim_paths=[stockpot.root_link.prim_path],
            )

        og.sim.step()
        assert stockpot.states[Contains].get_value(system)

        # Remove all particles and make sure contains returns False
        stockpot.states[Contains].set_value(system, False)
        og.sim.step()
        assert not stockpot.states[Contains].get_value(system)

        # Cannot set Contains state
        with pytest.raises(NotImplementedError):
            stockpot.states[Contains].set_value(system, True)

        system.remove_all_particles()


def test_covered(env, bracelet, bowl, microwave):
    systems = [env.scene.get_system(system_name) for system_name, system_class in SYSTEM_EXAMPLES.items()]
    for obj in (bracelet, bowl, microwave):
        for system in systems:
            # bracelet is too small to sample physical particles on it
            sampleable = env.scene.is_visual_particle_system(system.name) or obj != bracelet
            if sampleable:
                print(f"Testing Covered {obj.name} with {system.name}")
                obj.set_position_orientation(position=th.ones(3) * 50.0, orientation=[0, 0, 0, 1.0])
                place_obj_on_floor_plane(obj)
                for _ in range(5):
                    og.sim.step()

                assert obj.states[Covered].set_value(system, True)
                og.sim.step()
                assert obj.states[Covered].get_value(system)

                assert obj.states[Covered].set_value(system, False)
                # We don't call og.sim.step() here because it's possible for the "second" layer of particles to fall down
                # and make Covered to be True again. Instead, we clear the caches and check that Covered is False.
                obj.states[Covered].clear_cache()
                obj.states[ContactParticles].clear_cache()
                assert not obj.states[Covered].get_value(system)

                system.remove_all_particles()

        obj.set_position_orientation(position=th.ones(3) * 75.0, orientation=[0, 0, 0, 1.0])


def test_kinematic_only_contact_no_error():
    """
    A scene containing only kinematic/fixed-base rigid objects (no dynamic
    bodies) must not crash during RigidContactAPI.initialize_view(), and contact queries must
    return False rather than raise an exception.
    """
    test_clear_sim()

    cfg = {
        "scene": {"type": "Scene"},
        "objects": [
            {
                "type": "DatasetObject",
                "name": "table",
                "category": "breakfast_table",
                "model": "skczfi",
                "kinematic_only": True,
                "fixed_base": True,
                "position": [0, 0, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "bowl",
                "category": "bowl",
                "model": "ajzltc",
                "kinematic_only": True,
                "fixed_base": True,
                "position": [0, 0, 1.0],
            },
        ],
    }
    env = og.Environment(configs=cfg)

    # og.sim.step() internally calls RigidContactAPI.initialize_view(); must not raise.
    og.sim.step()

    # Explicit call must also not raise.
    RigidContactAPI.initialize_view()

    scene_idx = env.scene.idx
    table = env.scene.object_registry("name", "table")
    bowl = env.scene.object_registry("name", "bowl")

    # Contact queries against a kinematic-only scene must return False, not error.
    assert not RigidContactAPI.is_in_contact(
        scene_idx=scene_idx, query_set=[table], with_set=[bowl], ignore_set=None, current_only=False
    )
    assert not RigidContactAPI.is_in_contact(
        scene_idx=scene_idx, query_set=[table], with_set=[bowl], ignore_set=None, current_only=True
    )
    assert not RigidContactAPI.is_in_contact(
        scene_idx=scene_idx, query_set=[bowl], with_set=[table], ignore_set=None, current_only=False
    )
    assert not RigidContactAPI.is_in_contact(
        scene_idx=scene_idx, query_set=[bowl], with_set=[table], ignore_set=None, current_only=True
    )

    test_clear_sim()


def test_clear_sim():
    if og.sim is not None:
        og.clear()
