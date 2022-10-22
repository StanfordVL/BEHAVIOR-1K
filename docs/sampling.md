# Sampling Scene Instances

The [logic states](extended_states.md) implemented in OmniGibson since the version 2.0 provide a mechanism that facilitates the generation of simulated scenes to study and develop robotic solutions.
Users of OmniGibson can specify the desired configuration of the environment in the logic language [BDDL](https://github.com/StanfordVL/bddl#readme), part of the [BEHAVIOR benchmark](behavior.stanford.edu).
This language has similarities to the Planning Domain Definition Language (PDDL), allowing researchers to define scenes in logic-semantic manner (e.g., objects on top, next, inside of others) instead of the time consuming and tedious work of specifying manually their positions.
Given a scene definition in BDDL, OmniGibson provides the functionalities to sample compliant instances of the logic description to be used in simulation.
The image below shows an example of different instances sampled from the same logic description (three books on a table).

![sampling2.gif](images/sampling2.gif)

The first step to generate a new activity in OmniGibson is to create its BDDL description.
Please, follow the instructions [here](https://behavior.stanford.edu/activity-annotation) to create your own BDDL description using our online web interface, or modify some of the existing descriptions included as part of BEHAVIOR (see [here](https://github.com/StanfordVL/bddl/tree/master/bddl/activity_definitions)).

The next step is to download and install [BDDL](https://github.com/StanfordVL/bddl). Place your own BDDL description under `bddl/activity_definitions/<new_activity>/problem0.bddl`.

Then you can select a *vanilla scene* to instantiate the BDDL description on.
We provide 15 scenes as part of the OmniGibson dataset, furnished and with clutter objects to be specialized for multiple activities. See the available scenes in `og_dataset/scenes` or [here](http://svl.stanford.edu/omnigibson/docs/dataset.html).

With the BDDL description and a list of scene names, OmniGibson can generate valid scene instances of the description with the following script:
```
python -m omnigibson.utils.data_utils.sampling_task.sampling_saver --task <new_activity> --task_id 0 --scenes <scene_name_1> <scene_name_2> ...
```

The script will sample possible poses of object models of the indicated classes until all conditions are fulfilled.
The result will be stored as `og_dataset/scenes/<scene_name_1>/urdf/<scene_name_1>_task_<new_activity>_0_0.urdf`, a description of the scene with additional objects that fulfill the initial conditions in the BDDL description.
The user should ensure that the definition is sampleable in the given scene. Otherwise, after a certain number of sampling attempts, the script will fail and return.

We recommend to use the BEHAVIOR Dataset of 3D objects to get access to hundreds of object models to create new activities.

