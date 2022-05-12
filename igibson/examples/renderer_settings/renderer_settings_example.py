from igibson.renderer_settings.renderer_settings import RendererSettings
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator_omni import Simulator

# Create simulator and import scene.
sim = Simulator()
scene = EmptyScene(floor_plane_visible=False)
sim.import_scene(scene)

# Create renderer settings object.
renderer_setting = RendererSettings()

# Set current renderer.
renderer_setting.set_current_renderer("Real-Time")
assert renderer_setting.get_current_renderer() == "Real-Time"
renderer_setting.set_current_renderer("Path-Traced")
assert renderer_setting.get_current_renderer() == "Path-Traced"

# Get all available settings.
print(renderer_setting.settings.keys())

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
