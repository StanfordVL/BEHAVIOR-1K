# :octicons-gear-24: **Setting Macros**

Macros are a global set of hard-coded, "magic" numbers that are used as default values across **OmniGibson**. These values can have significant implications that broadly impact **OmniGibson**'s runtime (such as setting `HEADLESS` or `DEFAULT_PHYSICS_FREQ`), or can have a much more narrow scope that impacts only a specific module within **OmniGibson** (such as `FIRE_EMITTER_HEIGHT_RATIO`).

All macros within **OmniGibson** can be directly accessed and set via the [`omnigibson/macros.py`](../reference/macros.md) module. There are two sets of macros:

1. **Global Macros**: Accessed via the `gm` module variable, these are fundamental settings that generally impact all parts of **OmniGibson** runtime, and include values such as `gm.HEADLESS`, `gm.DEFAULT_PHYSICS_FREQ`, and `gm.ENABLE_HQ_RENDERING`. Descriptions of each global macro can be seen directly in the `omnigibson/macros.py` file.
2. **Module Macros** Accessed via the `macros` module variable, these are module-level settings used by individual modules throughout **OmniGibson**. These tend to only impact the module they are defined in, though they can be referenced by other modules as well. Examples include values such as `macros.objects.stateful_object.FIRE_EMITTER_HEIGHT_RATIO` and `macros.robots.manipulation_robot.ASSIST_GRASP_MASS_THRESHOLD`. Descriptions of each module-level macro can be seen directly at the top of the module that it is defined in.

## Reading Values
Macros can be read like any other python value. Both `gm` and `macros` are [Addict](https://github.com/mewwts/addict) dictionary-like objects, and whose keyword-indexed values can be queried via dot notation. Note that for `macros`, this is a nested addict object and represents values as keyword-entries indexed by the relative path of the module it is specified in, expressed in dot notation. For example, to read a macro from `omnigibson/prims/entity_prim.py`, you would query `macros.prims.entity_prim.<MACRO_NAME>`. We provide an example snippet of code below:

??? code "read_macros.py"
    ``` python linenums="1"

    from omnigibson.macros import gm, macros

    # Read some global macros
    print(f"Is headless: {gm.HEADLESS}")
    print(f"Physics freq: {gm.DEFAULT_PHYSICS_FREQ}")

    # Read some module-level macros
    print(f"Object sleep threshold: {macros.prims.entity_prim.DEFAULT_SLEEP_THRESHOLD}")
    print(f"Visual particle removal limit: {macros.object_states.particle_modifier.VISUAL_PARTICLES_REMOVAL_LIMIT}")

    # Print out all global macros
    print(f"All global macros:\n\n{gm}")

    # Print out all module-level macros
    print(f"All module-level macros:\n\n{macros}")
    ```

## Setting Values
Macros can be set like any other python value, and utilize the same convention for accessing as stated above. However, a key caveat is that once a macro is read, it **cannot** be set. This is to guarantee consistent runtime performance and avoid silent bugs due to macros being updated but not read by relevant downstream use cases. For example, `gm.HEADLESS` is read when `og.launch()` or `og.Environment()` is called to start up the **OmniGibson** application, and cannot be modified afterward. If there was no guard in place, setting this macro after **OmniGibson** is launched would result in an inconsistency between the observed state (i.e.: the mode requested when **OmniGibson** was originally launched) and the read state (modified value of `gm.HEADLESS`). In general, we recommend setting all macros *before* `og.launch()` or `og.Environment()` is called. We provide an example snippet of code below:

??? code "set_macros.py"
    ``` python linenums="1"

    import omnigibson as og
    from omnigibson.macros import gm, macros

    # Set some global macros
    gm.HEADLESS = True
    gm.DEFAULT_PHYSICS_FREQ = 60

    # Set some module-level macros
    macros.prims.entity_prim.DEFAULT_SLEEP_THRESHOLD = 0.002
    macros.object_states.particle_modifier.VISUAL_PARTICLES_REMOVAL_LIMIT = 80

    # Now launch omnigibson
    env = og.Environment({"scene": {"type": "Scene"}})
    ```

