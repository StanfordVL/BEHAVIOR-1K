from omnigibson import app, og_dataset_path, Simulator
from omnigibson.scenes.empty_scene import EmptyScene
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.constants import PrimType
import os

sim = Simulator()
scene = EmptyScene(floor_plane_visible=True)
sim.import_scene(scene=scene)

cube1 = PrimitiveObject(prim_path="/World/RigidCube", primitive_type="Cube", prim_type=PrimType.RIGID)
sim.import_object(cube1)

cube2 = PrimitiveObject(prim_path="/World/ClothCube", primitive_type="Cube", prim_type=PrimType.CLOTH)
sim.import_object(cube2)

carpet1 = DatasetObject(
    prim_path="/World/RigidCarpet",
    usd_path=os.path.join(og_dataset_path, "objects", "carpet", "carpet_0", "usd", "carpet_0.usd"),
    prim_type=PrimType.RIGID
)
sim.import_object(carpet1)

carpet2 = DatasetObject(
    prim_path="/World/ClothCarpet1",
    usd_path=os.path.join(og_dataset_path, "objects", "carpet", "carpet_0", "usd", "carpet_0.usd"),
    prim_type=PrimType.CLOTH
)
sim.import_object(carpet2)

carpet3 = DatasetObject(
    prim_path="/World/ClothCarpet2",
    usd_path=os.path.join(og_dataset_path, "objects", "carpet", "carpet_0", "usd", "carpet_0.usd"),
    prim_type=PrimType.CLOTH
)
sim.import_object(carpet3)

sim.play()
for _ in range(100000):
    sim.step()
