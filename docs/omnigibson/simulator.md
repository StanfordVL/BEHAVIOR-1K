# :material-repeat: **Simulator**

## Description

**`OmniGibson`**'s [Simulator](../reference/simulator.md) class is the global singleton that serves as the interface with omniverse's low-level physx (physics) backend. It provides utility functions for modulating the ongoing simulation as well as the low-level interface for importing scenes and objects. For standard use-cases, interfacing with the simulation exclusively through a created [environment](./environments.md) or [vector environment](./vector_environments.md) should be sufficient, though for more advanced or prototyping use-cases it may be common to interface via this simulator class.

## Usage

### Creating

Because this `Simulator` is a global singleton, it is instantiated every time the stage is cleared and always occurs at the _very beginning_ of **`OmniGibson`**'s launching. This either occurs when an environment instance is created (`env = Environment(...)` or `vec_env = VectorEnvironment(...)`), when `og.launch()` is explicitly called, or whenever the stage is reset.

### Runtime

After **`OmniGibson`** is launched, the simulator interface can be accessed globally via `og.sim`. Below, we briefly describe multiple common usages of the simulation interface:

#### Importing and Removing Scenes / Objects
The simulator can directly import a scene via `sim.import_scene(scene)`. The imported scene and its corresponding objects can be directly accessed through `sim.scenes`. To remove a desired object, call `scene.remove_object(object)` or, for a batched call, `og.sim.batch_remove_objects(objects)`. The simulator can also clear all the scenes via `og.clear()`.

#### Propagating Physics
The simulator can be manually stepped, with or without physics / rendering (`sim.step()`, `sim.step_physics()`, `sim.render()`), and can be stopped (`sim.stop()`), paused (`sim.pause()`), or played (`sim.play()`). Note that physics only runs when the simulator is playing! The current sim mode can be checked via `sim.is_stopped()`, `sim.is_paused()`, and `sim.is_playing()`.

??? warning annotate "Cycling the sim can cause unexpected behavior"

    If the simulator is playing and then suddenly stopped, all objects are immediately teleported back to their "canonical poses" -- i.e.: the corresponding global poses that were set _before_ the sim started playing. Thus, if `og.sim.play()` is immediately called after, the objects will _not_ teleport back to their respective pre-stopped poses. You can think of the canonical poses as a sort of "initial states", and we recommend explicitly setting a desired initial sim scene file configuration via `scene.update_initial_file()` and then calling `scene.reset()` after `og.sim.play()`.

#### Modifying Physics
If necessary, low-level physics behavior can also be set as well, via the physics interface (`sim.pi`), physics simulation interface (`sim.psi`), physics scene query interface (`sim.psqi`), and physics context (`sim.get_physics_context()`). The simulation timesteps can also be directly set via `sim.set_simulation_dt(...)`.

#### Callbacks
It may be useful to have callbacks trigger when certain simulation events occur. We provide utility functions to add / remove callbacks when `sim.play()`, `sim.stop()`, `sim._post_import_object()`, and `sim._pre_remove_object()` is called.

#### Viewport Camera
The simulator owns a global [`VisionSensor`](./sensors.md) camera which can be controlled by the user and accessed via `sim.viewer_camera`. To enable keyboard teleoperation of the camera, call `sim.enable_viewer_camera_teleoperation()`.

#### Saving / Loading State
To record the current global simulation state (which includes the state of all tracked objects within the simulator), call `state = sim.dump_state(serialized)`, where `serialized` can be `True` or `False`, defining whether the outputted state should be in nested dictionary (more interpretable) format or a more compressed, 1D numpy array format. Both representations are equivalent, with the former being more useful for prototyping and the latter being useful when recording large amounts of data (e.g.: when collecting demonstrations). To load a sim state, call `sim.load_state(state, serialized)`.

Note, however, that the above paradigm generally assumes a consistent configuration of objects -- `sim.load_state(...)` will not import / remove objects if there is a mismatch between the desired state and current simulator object set. Instead, we provide `sim.save(fpath)` and `sim.restore(fpath)` functionality for arbitrarily saving and restoring a scene / object configuration from scratch, where `fpath` defines the absolute filepath to the simulator state + metadata.
