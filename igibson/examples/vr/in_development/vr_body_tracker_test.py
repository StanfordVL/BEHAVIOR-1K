"""
Demo for testing VR body based on torso tracker
"""
from omnigibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from omnigibson.robots.behavior_robot import BehaviorRobot
from omnigibson.scenes.empty_scene import EmptyScene
from omnigibson.simulator import Simulator


def main():
    s = Simulator(mode="vr", rendering_settings=MeshRendererSettings(enable_shadow=True, optimized=True))
    scene = EmptyScene()
    s.import_scene(scene)
    vr_agent = BehaviorRobot()

    # Main simulation loop
    while True:
        s.step()
        vr_agent.apply_action()


if __name__ == "__main__":
    main()
