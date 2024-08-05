import torch as th

import omnigibson as og
from omnigibson.renderer_settings.renderer_settings import RendererSettings


def main(random_selection=False, headless=False, short_exec=False):
    """
    Shows how to use RendererSettings class
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Specify objects to load
    banana_cfg = dict(
        type="DatasetObject",
        name="banana",
        category="banana",
        model="vvyyyv",
        scale=[3.0, 5.0, 2.0],
        position=[-0.906661, -0.545106, 0.136824],
        orientation=[0, 0, 0.76040583, -0.6494482],
    )

    door_cfg = dict(
        type="DatasetObject",
        name="door",
        category="door",
        model="ohagsq",
        position=[-2.0, 0, 0.70000001],
        orientation=[0, 0, -0.38268343, 0.92387953],
    )

    # Create the scene config to load -- empty scene with a few objects
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [banana_cfg, door_cfg],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Set camera to appropriate viewing pose
    cam = og.sim.viewer_camera
    cam.set_position_orientation(
        position=th.tensor([-4.62785, -0.418575, 0.933943]),
        orientation=th.tensor([0.52196595, -0.4231939, -0.46640436, 0.5752612]),
    )

    def steps(n):
        for _ in range(n):
            env.step(th.empty(0))

    # Take a few steps to let objects settle
    steps(25)

    # Create renderer settings object.
    renderer_setting = RendererSettings()

    # RendererSettings is a singleton.
    renderer_setting2 = RendererSettings()
    assert renderer_setting == renderer_setting2

    # Set current renderer.
    if not short_exec:
        input("Setting renderer to Real-Time. Press [ENTER] to continue.")
    renderer_setting.set_current_renderer("Real-Time")
    assert renderer_setting.get_current_renderer() == "Real-Time"
    steps(5)

    if not short_exec:
        input("Setting renderer to Interactive (Path Tracing). Press [ENTER] to continue.")
    renderer_setting.set_current_renderer("Interactive (Path Tracing)")
    assert renderer_setting.get_current_renderer() == "Interactive (Path Tracing)"
    steps(5)

    # Get all available settings.
    print(renderer_setting.settings.keys())

    if not short_exec:
        input(
            "Showcasing how to use RendererSetting APIs. Please see example script for more information. "
            "Press [ENTER] to continue."
        )

    # Set setting (2 lines below are equivalent).
    renderer_setting.set_setting(path="/app/renderer/skipMaterialLoading", value=True)
    renderer_setting.common_settings.materials_settings.skip_material_loading.set(True)

    # Get setting (3 lines below are equivalent).
    assert renderer_setting.get_setting_from_path(path="/app/renderer/skipMaterialLoading") == True
    assert renderer_setting.common_settings.materials_settings.skip_material_loading.value == True
    assert renderer_setting.common_settings.materials_settings.skip_material_loading.get() == True

    # Reset setting (2 lines below are equivalent).
    renderer_setting.reset_setting(path="/app/renderer/skipMaterialLoading")
    renderer_setting.common_settings.materials_settings.skip_material_loading.reset()
    assert renderer_setting.get_setting_from_path(path="/app/renderer/skipMaterialLoading") == False

    # Set setting to an unallowed value using top-level method.
    # Examples below will use the "top-level" setting method.
    try:
        renderer_setting.set_setting(path="/app/renderer/skipMaterialLoading", value="foo")
    except AssertionError as e:
        print(e)  # All good. We got an AssertionError.

    # Set setting to a value out-of-range.
    try:
        renderer_setting.set_setting(path="/rtx/fog/fogColorIntensity", value=0.0)
    except AssertionError as e:
        print(e)  # All good. We got an AssertionError.

    # Set unallowed setting.
    try:
        renderer_setting.set_setting(path="foo", value="bar")
    except NotImplementedError as e:
        print(e)  # All good. We got a NotImplementedError.

    # Set setting but the setting group is not enabled.
    # Setting is successful but there will be a warning message printed.
    renderer_setting.set_setting(path="/rtx/fog/fogColorIntensity", value=1.0)

    # Shutdown sim
    if not short_exec:
        input("Completed demo. Press [ENTER] to shutdown simulation.")
    og.clear()


if __name__ == "__main__":
    main()
