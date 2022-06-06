import numpy as np
from omni.isaac.core.utils.viewports import set_camera_view

from igibson import ig_dataset_path, object_states
from igibson.objects.dataset_object import DatasetObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator_omni import Simulator

sim = Simulator()


def setup_scene_for_abilities(abilities1, abilities2):
    global sim

    sim.clear_and_remove_scene()
    scene = EmptyScene(floor_plane_visible=True)
    sim.import_scene(scene)

    objs = [None, None]
    abilities_arr = [abilities1, abilities2]
    position_arr = [np.array([0, 0, 0.04]), np.array([2, 0, 0.8])]
    for idx, (obj_category, obj_model) in enumerate((("apple", "00_0"), ("fridge", "12252"))):
        name = obj_category
        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"
        objs[idx] = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            abilities=abilities_arr[idx],
        )
        sim.import_object(objs[idx], auto_initialize=True)
        objs[idx].set_position_orientation(position=position_arr[idx])

    sim._set_viewer_camera("/OmniverseKit_Persp")
    set_camera_view(eye=[-4, 3, 3], target=[2, 0, 0])

    sim.play()
    for _ in range(5):
        sim.step()

    return objs[0], objs[1]


######################################################################################
# StickyAttachment
#   can attach if touching and at least one object has sticky state.
######################################################################################
obj1, obj2 = setup_scene_for_abilities(abilities1={"sticky": {}}, abilities2={})
assert object_states.StickyAttachment in obj1.states

# Obj1 moves towards obj2 and they are attached together.
obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
for i in range(200):
    sim.step()
assert obj1.states[object_states.StickyAttachment].get_value(obj2)

# Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
for i in range(200):
    sim.step()

# Unattach obj1 and obj2.
obj1.states[object_states.StickyAttachment].set_value(obj2, False)
assert not obj1.states[object_states.StickyAttachment].get_value(obj2)

# Obj1 moves away from obj2.
obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
for i in range(200):
    sim.step()

######################################################################################
# MagneticAttachment
#   can attach if touching and both objects have magnetic state.
######################################################################################
obj1, obj2 = setup_scene_for_abilities(abilities1={"magnetic": {}}, abilities2={"magnetic": {}})
assert object_states.MagneticAttachment in obj1.states
assert object_states.MagneticAttachment in obj2.states

# Obj1 moves towards obj2 and they are attached together.
obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
for i in range(200):
    sim.step()
assert obj1.states[object_states.MagneticAttachment].get_value(obj2)
assert obj2.states[object_states.MagneticAttachment].get_value(obj1)

# Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
for i in range(200):
    sim.step()

# Unattach obj1 and obj2.
obj1.states[object_states.MagneticAttachment].set_value(obj2, False)
assert not obj1.states[object_states.MagneticAttachment].get_value(obj2)
assert not obj2.states[object_states.MagneticAttachment].get_value(obj1)

# Obj1 moves away from obj2.
obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
for i in range(200):
    sim.step()


######################################################################################
# MagneticAttachment - FAIL because only 1 object has magnetic state
#   can attach if touching and both objects have magnetic state.
######################################################################################
obj1, obj2 = setup_scene_for_abilities(abilities1={"magnetic": {}}, abilities2={})
assert object_states.MagneticAttachment in obj1.states
assert object_states.MagneticAttachment not in obj2.states

# Obj1 moves towards obj2 and but they are NOT attached together.
obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
for i in range(200):
    sim.step()
assert not obj1.states[object_states.MagneticAttachment].get_value(obj2)

# Obj1 moves away from obj2.
obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
for i in range(200):
    sim.step()


######################################################################################
# MaleAttachment / FemaleAttachment
#   can attach if touching, self is male and the other is female.
######################################################################################
obj1, obj2 = setup_scene_for_abilities(abilities1={"maleAttachable": {}}, abilities2={"femaleAttachable": {}})
assert object_states.MaleAttachment in obj1.states
assert object_states.FemaleAttachment in obj2.states

# Obj1 moves towards obj2 and they are attached together.
obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
for i in range(200):
    sim.step()
assert obj1.states[object_states.MaleAttachment].get_value(obj2)
assert obj2.states[object_states.FemaleAttachment].get_value(obj1)

# Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
for i in range(200):
    sim.step()

# Unattach obj1 and obj2.
obj1.states[object_states.MaleAttachment].set_value(obj2, False)
assert not obj1.states[object_states.MaleAttachment].get_value(obj2)
assert not obj2.states[object_states.FemaleAttachment].get_value(obj1)

# Obj1 moves away from obj2.
obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
for i in range(200):
    sim.step()


######################################################################################
# MaleAttachment - FAIL because the other object is not female
#   can attach if touching, self is male and the other is female.
######################################################################################
obj1, obj2 = setup_scene_for_abilities(abilities1={"maleAttachable": {}}, abilities2={"maleAttachable": {}})
assert object_states.MaleAttachment in obj1.states
assert object_states.FemaleAttachment not in obj2.states

# Obj1 moves towards obj2 and but they are NOT attached together.
obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
for i in range(200):
    sim.step()
assert not obj1.states[object_states.MaleAttachment].get_value(obj2)

# Obj1 moves away from obj2.
obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
for i in range(200):
    sim.step()


######################################################################################
# HungMaleAttachment / HungFemaleAttachment
#   can attach if touching, self is male, the other is female,
#   and the male hanging object is "below" the female mounting object (center of bbox).
######################################################################################
obj1, obj2 = setup_scene_for_abilities(abilities1={"hungMaleAttachable": {}}, abilities2={"hungFemaleAttachable": {}})
assert object_states.HungMaleAttachment in obj1.states
assert object_states.HungFemaleAttachment in obj2.states

# Obj1 moves towards obj2 and they are attached together.
obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
for i in range(200):
    sim.step()
assert obj1.states[object_states.HungMaleAttachment].get_value(obj2)
assert obj2.states[object_states.HungFemaleAttachment].get_value(obj1)

# Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
for i in range(200):
    sim.step()

# Unattach obj1 and obj2.
obj1.states[object_states.HungMaleAttachment].set_value(obj2, False)
assert not obj1.states[object_states.HungMaleAttachment].get_value(obj2)
assert not obj2.states[object_states.HungFemaleAttachment].get_value(obj1)

# Obj1 moves away from obj2.
obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
for i in range(200):
    sim.step()


######################################################################################
# HungMaleAttachment - FAIL because the other object is not female hung
#   can attach if touching, self is male, the other is female,
#   and the male hanging object is "below" the female mounting object (center of bbox).
######################################################################################
obj1, obj2 = setup_scene_for_abilities(abilities1={"hungMaleAttachable": {}}, abilities2={"hungMaleAttachable": {}})
assert object_states.HungMaleAttachment in obj1.states
assert object_states.HungFemaleAttachment not in obj2.states

# Obj1 moves towards obj2 and but they are NOT attached together.
obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
for i in range(200):
    sim.step()
assert not obj1.states[object_states.HungMaleAttachment].get_value(obj2)

# Obj1 moves away from obj2.
obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
for i in range(200):
    sim.step()


######################################################################################
# HungMaleAttachment / FemaleAttachment - FAIL because the male object is above the female object
#   can attach if touching, self is male, the other is female,
#   and the male hanging object is "below" the female mounting object (center of bbox).
######################################################################################
obj1, obj2 = setup_scene_for_abilities(abilities1={"hungMaleAttachable": {}}, abilities2={"hungFemaleAttachable": {}})
assert object_states.HungMaleAttachment in obj1.states
assert object_states.HungFemaleAttachment in obj2.states

# Obj1 moves towards obj2 and but they are NOT attached together.
obj1.set_linear_velocity(velocity=np.array([5.0, 0, 5.0]))
for i in range(200):
    sim.step()
assert not obj1.states[object_states.HungMaleAttachment].get_value(obj2)

# Obj1 moves away from obj2.
obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
for i in range(200):
    sim.step()
