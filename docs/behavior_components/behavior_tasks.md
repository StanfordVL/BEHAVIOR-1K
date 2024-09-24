---
icon: material/silverware-fork-knife
---

# 🍴 **BEHAVIOR Tasks**

## Overview

BEHAVIOR is short for Benchmark for Everyday Household Activities in Virtual, Interactive, and ecOlogical enviRonments.

[**`BehaviorTask`**](../reference/tasks/behavior_task.md) represents a family of 1000 long-horizon household activities that humans benefit the most from robots' help based on our survey results.

To browse and modify the definition of BEHAVIOR tasks, you might find it helpful to download a local editable copy of our `bddl` repo.
```{.python .annotate}
git clone https://github.com/StanfordVL/bddl.git
```

Then you can install it in the same conda environment as OmniGibson.
```{.python .annotate}
conda activate omnigibson
cd bddl
pip install -e .
```

You can verify the installation by running the following command. This should now point to the local `bddl` repo, instead of the PyPI one.
```{.python .annotate}
>>> import bddl; print(bddl)
<module 'bddl' from '/path/to/BDDL/bddl/__init__.py'>
```

## Browse 1000 BEHAVIOR Tasks
All 1000 activities are defined in BDDL, a domain-specific language designed for BEHAVIOR.

You can find them in [`bddl/activity_definitions`](https://github.com/StanfordVL/bddl/tree/master/bddl/activity_definitions) folder.

Alternatively, you can browse them on the [BEHAVIOR Knowledgebase](https://behavior.stanford.edu/knowledgebase/tasks).

Here is an example of a BEHAVIOR task definition, which consists of several components:

- **:objects**: task-relevant objects, where each line represents a [**WordNet**](https://wordnet.princeton.edu/) synset of the object. For example, `candle.n.01_1 candle.n.01_2 candle.n.01_3 candle.n.01_4 - candle.n.01` indicates that four objects that belong to the `candoe.n.01` synset are needed for this task.
- **:init**: initial conditions of the task, where each line represents a ground predicate that holds at the beginning of the task. For example, `(ontop candle.n.01_1 table.n.02_1)` indicates that the first candle is on top of the first table when the task begins.
- **:goal**: goal conditions of the task, where each line represents a ground predicate and each block represents a non-ground predicate (e.g. `forall`, `forpairs`, `and`, `or`, etc) that should hold for the task to be considered solved. For example, `(inside ?candle.n.01 ?wicker_basket.n.01)` indicates that the candle should be inside the wicker basket at the end of the task.

??? code "assembling_gift_baskets.bddl"
    ``` yaml linenums="1"
    (define (problem assembling_gift_baskets-0)
        (:domain omnigibson)
    
        (:objects
            wicker_basket.n.01_1 wicker_basket.n.01_2 wicker_basket.n.01_3 wicker_basket.n.01_4 - wicker_basket.n.01
            floor.n.01_1 - floor.n.01
            candle.n.01_1 candle.n.01_2 candle.n.01_3 candle.n.01_4 - candle.n.01
            butter_cookie.n.01_1 butter_cookie.n.01_2 butter_cookie.n.01_3 butter_cookie.n.01_4 - butter_cookie.n.01
            swiss_cheese.n.01_1 swiss_cheese.n.01_2 swiss_cheese.n.01_3 swiss_cheese.n.01_4 - swiss_cheese.n.01
            bow.n.08_1 bow.n.08_2 bow.n.08_3 bow.n.08_4 - bow.n.08
            table.n.02_1 table.n.02_2 - table.n.02
            agent.n.01_1 - agent.n.01
        )
        
        (:init 
            (ontop wicker_basket.n.01_1 floor.n.01_1) 
            (ontop wicker_basket.n.01_2 floor.n.01_1) 
            (ontop wicker_basket.n.01_3 floor.n.01_1) 
            (ontop wicker_basket.n.01_4 floor.n.01_1) 
            (ontop candle.n.01_1 table.n.02_1) 
            (ontop candle.n.01_2 table.n.02_1) 
            (ontop candle.n.01_3 table.n.02_1) 
            (ontop candle.n.01_4 table.n.02_1) 
            (ontop butter_cookie.n.01_1 table.n.02_1) 
            (ontop butter_cookie.n.01_2 table.n.02_1) 
            (ontop butter_cookie.n.01_3 table.n.02_1) 
            (ontop butter_cookie.n.01_4 table.n.02_1) 
            (ontop swiss_cheese.n.01_1 table.n.02_2) 
            (ontop swiss_cheese.n.01_2 table.n.02_2) 
            (ontop swiss_cheese.n.01_3 table.n.02_2) 
            (ontop swiss_cheese.n.01_4 table.n.02_2) 
            (ontop bow.n.08_1 table.n.02_2) 
            (ontop bow.n.08_2 table.n.02_2) 
            (ontop bow.n.08_3 table.n.02_2) 
            (ontop bow.n.08_4 table.n.02_2) 
            (inroom floor.n.01_1 living_room) 
            (inroom table.n.02_1 living_room) 
            (inroom table.n.02_2 living_room) 
            (ontop agent.n.01_1 floor.n.01_1)
        )
        
        (:goal 
            (and 
                (forpairs 
                    (?wicker_basket.n.01 - wicker_basket.n.01) 
                    (?candle.n.01 - candle.n.01) 
                    (inside ?candle.n.01 ?wicker_basket.n.01)
                ) 
                (forpairs 
                    (?wicker_basket.n.01 - wicker_basket.n.01) 
                    (?swiss_cheese.n.01 - swiss_cheese.n.01) 
                    (inside ?swiss_cheese.n.01 ?wicker_basket.n.01)
                ) 
                (forpairs 
                    (?wicker_basket.n.01 - wicker_basket.n.01) 
                    (?butter_cookie.n.01 - butter_cookie.n.01) 
                    (inside ?butter_cookie.n.01 ?wicker_basket.n.01)
                ) 
                (forpairs 
                    (?wicker_basket.n.01 - wicker_basket.n.01) 
                    (?bow.n.08 - bow.n.08) 
                    (inside ?bow.n.08 ?wicker_basket.n.01)
                )
            )
        )
    )
    ```

## Sample BEHAVIOR Tasks

Given a BEHAVIOR task definition, you can sample an instance of the task in OmniGibson by specifying the `activity_name` and `activity_definition_id` in the task configuration, which correspounds to `bddl/activity_definitions/<activity_name>/problem<activity_definition_id>.bddl`.

Here is an example of sample a BEHAVIOR task in OmniGibson for [laying_wood_floors](https://github.com/StanfordVL/bddl/blob/master/bddl/activity_definitions/laying_wood_floors/problem0.bddl).
```{.python .annotate}
import omnigibson as og
cfg = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
    },
    "robots": [
        {
            "type": "Fetch",
            "obs_modalities": ["rgb"],
            "default_arm_pose": "diagonal30",
            "default_reset_mode": "tuck",
        },
    ],
    "task": {
        "type": "BehaviorTask",
        "activity_name": "laying_wood_floors",
        "activity_definition_id": 0,
        "activity_instance_id": 0,
        "online_object_sampling": True,
    },
}
env = og.Environment(configs=cfg)
```

Each time you run the code above, a different instance of the task will be generated:

- A different object category might be sampled. For example, for a high-level synset like `fruit.n.01`, different types of fruits like apple, banana, and orange might be sampled.
- A different object model might be sampled. For example, different models of the same category (e.g. apple) might be sampled
- A different object pose might be sampled. For example, the apple might be placed at a different location in the scene.

Sampling can also fail for a wide variety of reasons:

- Missing room types: a required room type doesn't exist in the current scene
- No valid scene objects: cannot find appropriate scene objects (objects with the `inroom` predicate in the task definition), e.g. category mismatch.
- Cannot sample initial conditions: cannot find an appropraite physical configuration that satisfies all the initial conditions in the task definition, e.g. size mismatch.
- Many more...

Once a task is successfully sampled, you can save it to disk.
```{.python .annotate}
env.task.save_task()
```
The default path for saving the task is:
```
<gm.DATASET_PATH>/scenes/<SCENE_MODEL>/json/<scene_model>_task_{activity_name}_{activity_definition_id}_{activity_instance_id}_template.json
```

## Load Pre-sampled BEHAVIOR Tasks

Here is an example of loading a pre-sampled BEHAVIOR task instance in OmniGibson that you just saved.

```{.python .annotate}
import omnigibson as og
cfg = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
    },
    "robots": [
        {
            "type": "Fetch",
            "obs_modalities": ["rgb"],
            "default_arm_pose": "diagonal30",
            "default_reset_mode": "tuck",
        },
    ],
    "task": {
        "type": "BehaviorTask",
        "activity_name": "laying_wood_floors",
        "activity_definition_id": 0,
        "activity_instance_id": 0,
        "online_object_sampling": False,
    },
}
env = og.Environment(configs=cfg)
```

Curently, in our publicly available dataset, we have pre-sampled exactly **1** instance of all 1000 BEHAVIOR tasks.
We recommend you to set `online_object_sampling` to `False` to load the pre-sampled task instances in the dataset.
You can run the following command to find out the path to the pre-sampled task instances.
```bash
ls -l <gm.DATASET_PATH>/scenes/*/json/*task*
```

## (Advanced) Customize BEHAVIOR Tasks

The easiest way to create custom BEHAVIOR tasks is to add new task definitions to the `bddl` repo.

For instance, you can emulate the existing task definitions and create a new task definition at `bddl/activity_definitions/<my_new_task>/problem0.bddl`.

Then you can run the following tests to ensure that your new task is compatible with the rest of BEHAVIOR knowledgebase (e.g. you are using valid synsets with valid states).
```bash
cd bddl
python tests/bddl_tests.py batch_verify
python tests/tm_tests.py
```

Finally, you can sample and load your custom BEHAVIOR tasks in OmniGibson as shown above.