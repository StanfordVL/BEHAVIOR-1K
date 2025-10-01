# :material-silverware-fork-knife: **BEHAVIOR Tasks**

[`BehaviorTask`](../reference/tasks/behavior_task.md) represents a family of **1000 long-horizon household activities** that humans benefit the most from robots' help based on our survey results.

## Getting Started

BDDL (BEHAVIOR Domain Definition Language) is automatically installed when you follow the [standard BEHAVIOR installation guide](../getting_started/installation.md). This section is only needed if you want to customize or edit task definitions.

### Installing BDDL for Customization (Optional)

If you want to modify or create new task definitions, you'll need to install the BDDL repository in development mode:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/StanfordVL/bddl.git
   ```

2. **Install the package:**
   ```bash
   conda activate behavior
   cd bddl
   pip install -e .
   ```

3. **Verify installation:**
   ```python
   >>> import bddl; print(bddl)
   <module 'bddl' from '/path/to/BDDL/bddl/__init__.py'>
   ```

This should now point to your local `bddl` repo, instead of the PyPI version, giving you full editing capabilities.

## Task Library

### Where to Find Tasks

All 1000 activities are defined using **BDDL** (BEHAVIOR Domain Definition Language), specifically designed for household robotics tasks.

- **Local Files:** [`bddl/activity_definitions`](https://github.com/StanfordVL/bddl/tree/master/bddl/activity_definitions) folder
- **Online Browser:** [BEHAVIOR Knowledgebase](https://behavior.stanford.edu/knowledgebase/tasks) for interactive exploration

### Task Structure

Every BEHAVIOR task consists of three main components:

#### 1. Objects
Each line represents a [**WordNet**](https://wordnet.princeton.edu/) synset of required objects.

**Example:**
```yaml
candle.n.01_1 candle.n.01_2 candle.n.01_3 candle.n.01_4 - candle.n.01
```
This means four objects belonging to the `candle.n.01` synset are needed.

#### 2. Initial Conditions
Ground predicates that define the world state when the task begins.

**Example:**
```yaml
(ontop candle.n.01_1 table.n.02_1)
```
The first candle starts on top of the first table.

#### 3. Goal Conditions
Predicates and logical blocks that must be satisfied for task completion.

**Example:**
```yaml
(inside ?candle.n.01 ?wicker_basket.n.01)
```
All candles must end up inside wicker baskets.

### Complete Task Example

Here's a full task definition for assembling gift baskets:

```yaml
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

## Working with Tasks

### Sampling New Task Instances

Generate fresh instances of existing tasks with randomized elements for variety and robustness testing.

**Example: Sampling a wood floor laying task**

```python
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
        "activity_name": "laying_wood_floors",  # Task name
        "activity_definition_id": 0,           # Problem variant
        "activity_instance_id": 0,             # Instance number
        "online_object_sampling": True,        # Enable sampling
    },
}

env = og.Environment(configs=cfg)
```

Each sampling run produces different variations:

- **Object Categories:** High-level synsets sample different specific types (e.g., `fruit.n.01` → apple, banana, orange)
- **Object Models:** Same categories use different 3D models with varying shapes and colors
- **Object Poses:** Spatial arrangements vary while satisfying constraints

### Sampling Challenges

Sampling can fail for various reasons - this is normal behavior:

| Failure Type | Description |
|--------------|-------------|
| **Missing Rooms** | Required room type doesn't exist in the scene |
| **No Valid Objects** | Cannot find appropriate scene objects for the task |
| **Physical Constraints** | Cannot satisfy initial conditions (e.g., objects too large) |

### Saving Task Instances

Once successfully sampled, preserve the configuration for reuse:

```python
# Save the current task instance
env.task.save_task()
```

Default save location:
```
<gm.DATA_PATH>/behavior-1k-assets/scenes/<SCENE_MODEL>/json/<scene_model>_task_{activity_name}_{activity_definition_id}_{activity_instance_id}_template.json
```

## Loading Pre-sampled Tasks

### Using Existing Instances

For consistent, reproducible experiments, load pre-sampled task instances from the dataset. Our publicly available dataset includes **exactly 1 pre-sampled instance** of all 1000 BEHAVIOR tasks.

**Example: Loading a pre-sampled task**

```python
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
        "online_object_sampling": False,  # Load pre-sampled
    },
}

env = og.Environment(configs=cfg)
```

### Finding Pre-sampled Tasks

Discover available pre-sampled task instances in your dataset:

```bash
ls -l <gm.DATA_PATH>/*/scenes/*/json/*task*
```

**Recommendation:** Set `online_object_sampling=False` to load the stable, pre-sampled task instances for consistent evaluation and comparison.

## Creating Custom Tasks

### Step-by-Step Process

1. **Create task directory:**
   ```bash
   mkdir bddl/activity_definitions/my_new_task
   ```

2. **Create task definition file:**
   ```bash
   touch bddl/activity_definitions/my_new_task/problem0.bddl
   ```

3. **Define task components:**
   - Specify required objects (`:objects`)
   - Set initial conditions (`:init`) 
   - Define goal conditions (`:goal`)

4. **Validate compatibility:**
   ```bash
   cd bddl
   python tests/bddl_tests.py batch_verify
   python tests/tm_tests.py
   ```

5. **Test your custom task:**
   ```python
   cfg = {
       # ... standard configuration ...
       "task": {
           "type": "BehaviorTask",
           "activity_name": "my_new_task",  # Your custom task
           "activity_definition_id": 0,
           "activity_instance_id": 0,
           "online_object_sampling": True,
       },
   }
   
   env = og.Environment(configs=cfg)
   ```

### Activity Identification

Tasks are uniquely identified by three parameters:

- **`activity_name`** - The task type (e.g., "laying_wood_floors")
- **`activity_definition_id`** - Problem variant within the task (usually 0)
- **`activity_instance_id`** - Specific instance number for the sampled configuration

You now have the complete toolkit for working with BEHAVIOR tasks! Start by exploring the 1000 pre-defined activities, then create your own custom household robotics challenges.