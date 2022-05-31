from igibson import app, ig_dataset_path, Simulator
from igibson.scenes.empty_scene import EmptyScene
from igibson.objects.primitive_object import PrimitiveObject
from igibson.objects.dataset_object import DatasetObject
from igibson.utils.constants import PrimType
import os

sim = Simulator()
scene = EmptyScene(floor_plane_visible=True)
sim.import_scene(scene=scene)
sim.step()

cube1 = PrimitiveObject(prim_path="/World/TestCube1", primitive_type="Cube", prim_type=PrimType.RIGID)
sim.import_object(cube1)
sim.step()

cube2 = PrimitiveObject(prim_path="/World/TestCube2", primitive_type="Cube", prim_type=PrimType.CLOTH)
sim.import_object(cube2)
sim.step()

carpet1 = DatasetObject(
    prim_path="/World/TestCarpet1",
    usd_path=os.path.join(ig_dataset_path, "objects", "carpet", "carpet_0", "usd", "carpet_0.usd"),
    prim_type=PrimType.RIGID
)
sim.import_object(carpet1)
sim.step()

carpet2 = DatasetObject(
    prim_path="/World/TestCarpet2",
    usd_path=os.path.join(ig_dataset_path, "objects", "carpet", "carpet_0", "usd", "carpet_0.usd"),
    prim_type=PrimType.CLOTH
)
sim.import_object(carpet2)

carpet3 = DatasetObject(
    prim_path="/World/TestCarpet3",
    usd_path=os.path.join(ig_dataset_path, "objects", "carpet", "carpet_0", "usd", "carpet_0.usd"),
    prim_type=PrimType.CLOTH
)
sim.import_object(carpet3)

sim.step()
