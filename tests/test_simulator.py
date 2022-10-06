from igibson.objects.ycb_object import YCBObject
from igibson.scenes.stadium_scene import StadiumScene
from igibson.simulator import Simulator
from igibson.utils.asset_utils import download_assets


def test_simulator():
    download_assets()
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)

    for i in range(10):
        obj = YCBObject("006_mustard_bottle")
        s.import_object(obj)

    for i in range(10):
        obj = YCBObject("002_master_chef_can")
        s.import_object(obj)

    for i in range(1000):
        s.step()
    s.disconnect()
