---
icon: material/silverware-fork-knife
---

# 🍴 **BEHAVIOR Tasks**
 
[`BehaviorTask`](../reference/tasks/behavior_task.md) represents a family of **1000 long-horizon household activities** that humans benefit the most from robots' help based on our survey results.

---

## :octicons-goal-16: **Getting Started**

Before working with BEHAVIOR tasks, you'll need to set up the BDDL (BEHAVIOR Domain Definition Language) repository for task definition editing and browsing.

### Installation Steps

=== ":material-download: Clone Repository"

    ```bash
    git clone https://github.com/StanfordVL/bddl.git
    ```

=== ":material-package: Install Package"

    ```bash
    conda activate omnigibson
    cd bddl
    pip install -e .
    ```

=== ":material-check-circle: Verify Installation"

    ```python
    >>> import bddl; print(bddl)
    <module 'bddl' from '/path/to/BDDL/bddl/__init__.py'>
    ```

!!! success "Installation Complete"
    This should now point to your local `bddl` repo, instead of the PyPI version, giving you full editing capabilities.

---

## :material-library: **Exploring the Task Library**

### 📂 Task Locations

All 1000 activities are defined using **BDDL** (BEHAVIOR Domain Definition Language), specifically designed for household robotics tasks.

!!! info "Where to Find Tasks"
    - **📁 Local Files:** [`bddl/activity_definitions`](https://github.com/StanfordVL/bddl/tree/master/bddl/activity_definitions) folder
    - **🌐 Online Browser:** [BEHAVIOR Knowledgebase](https://behavior.stanford.edu/knowledgebase/tasks) - Interactive exploration

### 🏗️ Task Structure

Every BEHAVIOR task consists of three main components that define the complete activity specification:

=== ":material-cube-outline: Objects"

    !!! info "Task-Relevant Objects"
        Each line represents a [**WordNet**](https://wordnet.princeton.edu/) synset of required objects.

    **Example:**
    ```yaml
    candle.n.01_1 candle.n.01_2 candle.n.01_3 candle.n.01_4 - candle.n.01
    ```
    
    ↳ *Four objects belonging to the `candle.n.01` synset are needed*

=== ":material-play-circle: Initial Conditions"

    !!! info "Starting State"
        Ground predicates that define the world state when the task begins.

    **Example:**
    ```yaml
    (ontop candle.n.01_1 table.n.02_1)
    ```
    
    ↳ *The first candle starts on top of the first table*

=== ":material-flag-checkered: Goal Conditions"

    !!! info "Success Criteria"
        Predicates and logical blocks that must be satisfied for task completion.

    **Example:**
    ```yaml
    (inside ?candle.n.01 ?wicker_basket.n.01)
    ```
    
    ↳ *All candles must end up inside wicker baskets*

### 📖 Complete Task Example

Here's a full task definition showing all components working together:

??? example "assembling_gift_baskets.bddl - Complete Task Definition"
    ```yaml linenums="1"
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

---

## :material-cogs: **Working with Tasks**

### 🎲 Sampling New Task Instances

Generate fresh instances of existing tasks with randomized elements for variety and robustness testing.

!!! tip "Dynamic Sampling Benefits"
    Each sample creates unique variations in object types, models, poses, and configurations while maintaining task semantics.

**Example: Sampling a Wood Floor Laying Task**

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

### 🎯 Sampling Variations

Each sampling run produces different variations:

=== ":material-apple: Object Categories"

    **High-level synsets** sample different specific types:
    
    - `fruit.n.01` → apple, banana, orange, etc.
    - `chair.n.01` → office chair, dining chair, recliner, etc.

=== ":material-shape: Object Models"

    **Same categories** use different 3D models:
    
    - Apple category → different apple models with varying shapes, colors
    - Table category → various table designs and sizes

=== ":material-map-marker: Object Poses"

    **Spatial arrangements** vary while satisfying constraints:
    
    - Objects placed at different valid locations
    - Orientations adjusted within acceptable ranges

### ⚠️ Sampling Challenges

!!! warning "Common Sampling Failures"
    Sampling can fail for various reasons - this is normal behavior:

| Failure Type | Description | Example |
|--------------|-------------|---------|
| 🏠 **Missing Rooms** | Required room type doesn't exist | Task needs kitchen, scene has none |
| 🎯 **No Valid Objects** | Cannot find appropriate scene objects | Category mismatch with scene contents |
| 📐 **Physical Constraints** | Cannot satisfy initial conditions | Objects too large for designated spaces |
| 🔗 **Dependency Issues** | Complex constraint satisfaction fails | Multiple interconnected placement rules |

### 💾 Saving Task Instances

Once successfully sampled, preserve the configuration for reuse:

```python
# Save the current task instance
env.task.save_task()
```

!!! info "Default Save Location"
    ```
    <gm.DATASET_PATH>/scenes/<SCENE_MODEL>/json/<scene_model>_task_{activity_name}_{activity_definition_id}_{activity_instance_id}_template.json
    ```

---

## :material-folder-open: **Loading Pre-sampled Tasks**

### 📦 Using Existing Instances

For consistent, reproducible experiments, load pre-sampled task instances from the dataset.

!!! success "Dataset Availability"
    Our publicly available dataset includes **exactly 1 pre-sampled instance** of all 1000 BEHAVIOR tasks.

**Example: Loading a Pre-sampled Task**

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

### 🔍 Finding Pre-sampled Tasks

Discover available pre-sampled task instances in your dataset:

```bash
ls -l <gm.DATASET_PATH>/scenes/*/json/*task*
```

!!! tip "Recommended Usage"
    Set `online_object_sampling=False` to load the stable, pre-sampled task instances for consistent evaluation and comparison.

---

## :material-wrench: **Advanced Customization**

### 🎨 Creating Custom Tasks

The most straightforward approach to creating custom BEHAVIOR tasks is extending the existing BDDL repository structure.

#### Step-by-Step Process

=== ":material-file-plus: Create Task Definition"

    **1. Add New Task Directory**
    ```bash
    mkdir bddl/activity_definitions/my_new_task
    ```

    **2. Create Task Definition File**
    ```bash
    touch bddl/activity_definitions/my_new_task/problem0.bddl
    ```

    **3. Define Task Components**
    - Specify required objects (`:objects`)
    - Set initial conditions (`:init`) 
    - Define goal conditions (`:goal`)

=== ":material-test-tube: Validate Compatibility"

    **Run Compatibility Tests**
    ```bash
    cd bddl
    python tests/bddl_tests.py batch_verify
    python tests/tm_tests.py
    ```

    !!! warning "Validation Importance"
        These tests ensure your task uses:
        
        - ✅ Valid synsets from the knowledgebase
        - ✅ Compatible object states and properties
        - ✅ Proper BDDL syntax and semantics

=== ":material-rocket-launch: Deploy & Test"

    **Sample Your Custom Task**
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

### 🧩 Task Design Guidelines

!!! tip "Best Practices for Custom Tasks"
    
    **Synset Selection:**
    - Use existing synsets from the [BEHAVIOR Knowledgebase](behavior_knowledgebase.md)
    - Ensure objects have required abilities for your task goals
    
    **Spatial Constraints:**
    - Consider object sizes and room layouts
    - Design achievable initial and goal configurations
    
    **Logical Structure:**
    - Use clear, unambiguous goal conditions
    - Test with multiple object instances when applicable

---

## :material-lightbulb: **Key Concepts**

### Task Sampling vs. Loading

| Aspect | **Sampling** (`online_object_sampling=True`) | **Loading** (`online_object_sampling=False`) |
|--------|---------------------------------------------|---------------------------------------------|
| 🎲 **Variability** | High - new instance each time | Low - same instance always |
| 🔧 **Use Case** | Research, robustness testing | Evaluation, comparison |

### Activity Identification

Tasks are uniquely identified by three parameters:

- **`activity_name`** - The task type (e.g., "laying_wood_floors")
- **`activity_definition_id`** - Problem variant within the task (usually 0)
- **`activity_instance_id`** - Specific instance number for the sampled configuration

---

!!! success "Ready to Explore?"
    You now have the complete toolkit for working with BEHAVIOR tasks! Start by exploring the 1000 pre-defined activities, then create your own custom household robotics challenges. 🚀