---
icon: material/bookshelf
---

# :material-bookshelf: **BEHAVIOR Knowledgebase**

The [**BEHAVIOR Knowledgebase**](https://behavior.stanford.edu/knowledgebase/) is a comprehensive resource containing information about valid synsets, their relationships, abilities, hyperparameters, and hand-specified [transition rules](../omnigibson/transition_rules.md).

## Core Components

The BEHAVIOR Knowledgebase provides an interactive web interface to visualize and look up the following components:

### Tasks

A comprehensive family of [**1000 long-horizon household activities**](https://behavior.stanford.edu/knowledgebase/tasks) that cover the full spectrum of everyday tasks.

**What's included:**

- Task-relevant objects and their specifications
- Initial conditions - setup requirements
- Goal conditions - completion criteria
- Scene compatibility information
- (Experimental) Transition paths for goal achievement

Each task definition contains a structured representation of objects needed and conditions that must be satisfied for successful completion.

### Synsets

[**Synsets**](https://behavior.stanford.edu/knowledgebase/synsets) are the basic building blocks of the knowledgebase.

**Structure & Organization:**

- **WordNet Hierarchy** - follows established semantic relationships
- **Custom Extensions** - additional synsets for robotics needs
- **Parent-Child Relationships** - hierarchical organization
- **Leaf Synsets** - terminal nodes without children

**Synset Abilities:**

| Type | Examples | Purpose |
|------|----------|---------|
| **Physical** | `liquid`, `cloth`, `visualSubstance` | OmniGibson simulation behavior |
| **Semantic** | `fillable`, `openable`, `cookable` | Object affordances |
| **Parameterized** | `heatSource` with temperature settings | Detailed behavior control |

The knowledgebase provides extensive cross-references including predicates used in task definitions, tasks involving each synset, associated object categories and models, relevant transition rules, and WordNet hierarchy position.

### Categories

[**Categories**](https://behavior.stanford.edu/knowledgebase/categories) connect abstract synsets to concrete OmniGibson objects and substances.

**Key principles:**

- **One-to-One Mapping** - each category maps to exactly one leaf synset
- **Many-to-One Allowed** - multiple categories can share the same synset
- **Mass & Size Consistency** - objects in same category are interchangeable
- **Randomization Ready** - supports object randomization

**Example:**
```
Synset: sink.n.01
├── drop_in_sink (category)
└── pedestal_sink (category)

Both share identical properties but differ physically
```

**What you'll find:**

- Object images - visual representation of category members
- Object collections - all objects belonging to the category
- Hierarchy navigation - synset relationships and position

### Objects

[**Objects**](https://behavior.stanford.edu/knowledgebase/objects) represent specific 3D models in the dataset with one-to-one mapping.

**Object properties:**

- **Unique Identity** - belongs to exactly one category
- **File System** - corresponds to `<gm.DATASET_PATH>/objects/category/id`
- **Meta Links** - annotations for object states in OmniGibson

**Meta Link System example:**
For `coffee_maker-fwlabx`:

- `connectedpart` → `AttachedTo` state
- `heatsource` → `HeatSourceOrSink` state  
- `toggleButton` → `ToggledOn` state

**Additional information:**

- Object imagery and visual representation
- Scene/room appearances - where objects are found
- Synset hierarchy - semantic relationships

### Scenes

[**Scenes**](https://behavior.stanford.edu/knowledgebase/scenes) define specific 3D environment configurations in the dataset.

**Scene structure:**

- **Room Organization** - multiple rooms per scene
- **Naming Convention** - `<room_type>_<room_id>` (e.g., `living_room_0`, `kitchen_1`)
- **Object Inventory** - detailed object counts per room

**Example:**
In [`Beechwood_0_int`](https://behavior.stanford.edu/knowledgebase/scenes/Beechwood_0_int/index.html):

`countertop-tpuwys: 6` means the `kitchen_0` room contains 6 instances of the `countertop-tpuwys` object.

### Transition Rules

[**Transition Rules**](https://behavior.stanford.edu/knowledgebase/transitions/index.html) define physical and chemical interactions not natively supported by Omniverse.

**Rule components:**

- **Input Synsets** - required starting materials
- **Output Synsets** - resulting products
- **Conditions** - requirements for transition to occur

**Example:**
[`beef_stew`](https://behavior.stanford.edu/knowledgebase/transitions/beef_stew) rule:

**Inputs:** `ground_beef.n.01` + `beef_broth.n.01` + `pea.n.01` + `diced__carrot.n.01` + `diced__vidalia_onion.n.01`

**Output:** `beef_stew.n.01`

**Conditions:** heated in `stove.n.01` or `stockpot.n.01`

## Usage & Integration

### Python API Access

OmniGibson interfaces with the BEHAVIOR Knowledgebase through the [`ObjectTaxonomy`](https://github.com/StanfordVL/bddl/blob/master/bddl/object_taxonomy.py) class.

```python
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
```

### Common Operations

#### Hierarchy Navigation

```python
# Get family relationships
parents = OBJECT_TAXONOMY.get_parents("fruit.n.01")
children = OBJECT_TAXONOMY.get_children("fruit.n.01")
ancestors = OBJECT_TAXONOMY.get_ancestors("fruit.n.01")
descendants = OBJECT_TAXONOMY.get_descendants("fruit.n.01")
leaf_descendants = OBJECT_TAXONOMY.get_leaf_descendants("fruit.n.01")
```

#### Validation & Queries

```python
# Checker functions for synsets
is_leaf = OBJECT_TAXONOMY.is_leaf("fruit.n.01")
is_ancestor = OBJECT_TAXONOMY.is_ancestor("fruit.n.01", "apple.n.01")
is_descendant = OBJECT_TAXONOMY.is_descendant("apple.n.01", "fruit.n.01")
is_valid = OBJECT_TAXONOMY.is_valid_synset("fruit.n.01")
```

#### Abilities & Properties

```python
# Get synset abilities
# Returns: {'rigidBody': {...}, 'heatSource': {...}, 'toggleable': {...}, ...}
abilities = OBJECT_TAXONOMY.get_abilities("coffee_maker.n.01")

# Check specific ability
has_ability = OBJECT_TAXONOMY.has_ability("coffee_maker.n.01", "heatSource")
```

#### Category Mappings

```python
# Synset ↔ Category conversion
object_synset = OBJECT_TAXONOMY.get_synset_from_category("apple")  # → "apple.n.01"
object_categories = OBJECT_TAXONOMY.get_categories("apple.n.01")   # → ["apple"]

# Get all categories in subtree
leaf_descendant_categories = OBJECT_TAXONOMY.get_subtree_categories("fruit.n.01")
# → ["apple", "banana", "orange", ...]
```

#### Substance Handling

```python
# Substance synset mappings
substance_synset = OBJECT_TAXONOMY.get_synset_from_substance("water")  # → "water.n.06"
substance_categories = OBJECT_TAXONOMY.get_substances("water.n.06")    # → ["water"]

# Substance subtrees
leaf_descendant_substances = OBJECT_TAXONOMY.get_subtree_substances("liquid.n.01")
# → ["water", "milk", "juice", ...]
```

## Transition Rule Types

The BEHAVIOR system supports six distinct types of transition rules, each handling different aspects of object and substance interactions.

### Core Transition Rules

#### Washer Rule
Removes "dirty" substances when proper solvent is present and applies water effects.

**Process:**
- Remove dirty substances from washer
- Apply water saturation or coverage
- Clean objects inside washer

#### Dryer Rule
Removes water saturation and moisture from objects.

**Process:**
- Remove water saturation from objects
- Clear all water from dryer
- Dry objects completely

#### Slicing Rule
Cutting objects into halves based on contact with a `slicer` object.

**Requirements:**
- Object with `slicer` ability
- Target with `sliceable` ability
- **Result:** Two object halves

#### Dicing Rule
Transformation into diced substances.

**Requirements:**
- Object with `slicer` ability 
- Target with `diceable` ability
- **Result:** Diced substance

#### Melting Rule
Heat-induced state changes to melted substances.

**Requirements:**
- Object with `meltable` ability
- Reaching critical temperature
- **Result:** Melted substance

#### Recipe Rule
General framework for recipe-based transitions with multiple inputs and custom conditions.

**Components:**

| Element | Description |
|---------|-------------|
| **input_objects** | Required objects and counts |
| **input_systems** | Required particle systems |
| **output_objects** | Produced objects and counts |
| **output_systems** | Produced systems (volume-based) |
| **input_states** | Required input conditions |
| **output_states** | Resulting output conditions |
| **fillable_categories** | Required containers (pots, pans, etc.) |

### Recipe Rule Subtypes

#### Cooking Rules

**CookingPhysicalParticleRule**

Transforms physical particles through cooking, with optional water requirements.

**Examples:**
- **With water:** `rice` + `cooked__water` → `cooked__rice`
- **Without water:** `diced__chicken` → `cooked__diced__chicken`

#### ToggleableMachine Rules

**ToggleableMachineRule**

Uses toggleable appliances that must be in `ToggledOn` state.

**Examples:**
- **Object output:** `flour` + `butter` + `sugar` → `dough` (electric_mixer)
- **System output:** `strawberry` + `milk` → `strawberry_smoothie` (blender)

#### Mixing Rules

**MixingToolRule**

Leverages mixing tools in contact with fillable containers.

**Example:**
`water` + `lemon_juice` + `sugar` → `lemonade` (spoon + container)

#### Advanced Cooking

**CookingRule Variants**

| Type | Input | Output | Heat Source | Container |
|------|-------|---------|-------------|-----------|
| **CookingObjectRule** | `bagel_dough` + `egg` + `sesame_seed` | `bagel` | `oven` | `baking_sheet` |
| **CookingSystemRule** | `beef` + `tomato` + `chicken_stock` | `stew` | `stove` | `stockpot` |

## Advanced Customization

For advanced users who need to modify or extend the BEHAVIOR Knowledgebase, you can customize the source data and rebuild the system.

**Warning:** Knowledgebase customization requires understanding of the BEHAVIOR ecosystem and careful attention to data consistency.

### Modification Workflow

1. **Edit Source CSV Files** - Modify the underlying data definitions
2. **Rebuild Knowledgebase** - Generate updated system files  
3. **Validate Consistency** - Ensure task compatibility

### Key Source Files

#### Core Mappings

**[category_mapping.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/category_mapping.csv)**

- **Purpose:** Map object categories to synsets
- **When to modify:** Adding new object categories
- **Dependency:** Requires updating `avg_category_specs.json` for canonical density

**[substance_hyperparams.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/substance_hyperparams.csv)**

- **Purpose:** Map substance categories to synsets with physical/visual properties
- **When to modify:** Adding new substance categories
- **Dependencies:** Requires metadata and particle prototypes in dataset

**[synsets.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/synsets.csv)**

- **Purpose:** Define synset hierarchy and abilities
- **When to modify:** Adding new synsets
- **Dependencies:** Update corresponding property parameter annotations

#### Property Parameters

**Heat & Temperature**
- [`heatSource.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/heatSource.csv) - heating behavior parameters
- [`coldSource.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/coldSource.csv) - cooling behavior parameters
- [`cookable.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/cookable.csv) - cooking thresholds
- [`flammable.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/flammable.csv) - ignition properties

**Particle Interactions**
- [`particleApplier.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/particleApplier.csv) - substance application
- [`particleSource.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/particleSource.csv) - substance generation
- [`particleRemover.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/particleRemover.csv) - substance removal

**Physical Transformations**
- [`diceable.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/diceable.csv) - dicing parameters
- [`sliceable.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/sliceable.csv) - slicing parameters  
- [`meltable.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/meltable.csv) - melting parameters

#### Transition Rules

**Recipe Definitions**
- [`heat_cook.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_raw_data/heat_cook.csv) - cooking transformations
- [`mixing_stick.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_raw_data/mixing_stick.csv) - mixing operations
- [`single_toggleable_machine.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_raw_data/single_toggleable_machine.csv) - appliance usage
- [`washer.csv`](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_raw_data/washer.csv) - cleaning rules

### Rebuild Process

```bash
# Rebuild the knowledgebase
cd bddl
python data_generation/generate_datafiles.py

# Validate consistency
python tests/bddl_tests.py batch_verify
python tests/tm_tests.py
```

If errors occur during rebuilding, carefully read error messages and address issues systematically. Common problems include missing dependencies, inconsistent mappings, or malformed CSV entries.

Visit the [**BEHAVIOR Knowledgebase Website**](https://behavior.stanford.edu/knowledgebase) to explore the full dataset interactively!