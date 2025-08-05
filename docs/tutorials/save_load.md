# :material-zip-disk: **Saving and Loading Simulation State**

## Memory

You can dump the current simulation state to memory by calling `dump_state`:

```{.python .annotate}
state_dict = og.sim.dump_state(serialized=False)
```
This will return a dictionary containing all the information about the current state of the simulator.

If you want to save the state to a flat array, you can call `dump_state` with `serialized=True`:
```{.python .annotate}
state_flat_array = og.sim.dump_state(serialized=True)
```

You can then load the state back into the simulator by calling `load_state`:
```{.python .annotate}
og.sim.load_state(state_dict, serialized=False)
```
Or
```{.python .annotate}
og.sim.load_state(state_flat_array, serialized=True)
```

??? warning annotate "`load_state` assumes object permanence!"
    `load_state` assumes that the objects in the state match the objects in the current simulator. Only the state of the objects will be restored, not the objects themselves, i.e. no objects will be added or removed.
    If there is an object in the state that is not in the simulator, it will be ignored. If there is an object in the simulator that is not in the state, it will be left unchanged.

## Disk
Alternatively, you can save the state to disk by calling `save`:

```{.python .annotate}
og.sim.save(["path/to/scene_0.json"])
```

The number of json files should match the number of scenes in the simulator (by default, 1).

You can then load the state back into the simulator by calling `og.clear()` first and then `restore`:

```{.python .annotate}
og.clear()
og.sim.restore(["path/to/scene_0.json"])
```

??? warning annotate "`restore` assumes an empty simulator!"
    Always remember to call `og.clear()`, which clears the entire simualtor, before calling `restore`.
    Otherwise, the saved scenes will be appended to the existing scenes of the current simulator, which may lead to unexpected behavior.