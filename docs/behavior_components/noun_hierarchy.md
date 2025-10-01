# :material-file-tree-outline: **BEHAVIOR Noun Hierarchy**

The BEHAVIOR ecosystem organizes objects and substances using a hierarchical structure that extends WordNet. This hierarchy includes synsets (concepts), categories (object types), individual objects, scenes, tasks, and transition rules that define how objects interact and transform.

Here are the important conceptual components of the BEHAVIOR hierarchy:

### **Tasks**
A family of 1000 long-horizon household activities.

- As illustrated in the [**BEHAVIOR Tasks tutorial**](behavior_tasks.md), each task definition contains a list of task-relevant objects, and their initial and goal conditions.

### **Synsets**
The basic building block of the BEHAVIOR hierarchy.

- We follow the [**WordNet**](https://wordnet.princeton.edu/) hierarchy while expanding it with additional ("custom") synsets to suit the need of BEHAVIOR.
- Each synset has at least one parent synset, and can have many children synsets (no children means it's a leaf synset).
- Each synset can have many abilities (or properties).
    - Some properties define the physical attributes of the object and how OmniGibson simulates them, e.g. `liquid`, `cloth`, `visualSubstance`, etc.
    - Some properties define the semantic attributes (or affordances) of the object, e.g. `fillable`, `openable`, `cookable`, etc.
    - Each property might contain additional hyperparameters that define the exact behavior of the property, e.g. `heatSource` has hyperparameters `requires_toggled_on` (bool), `requires_closed` (bool), `requires_inside` (bool), `temperature` (float), and `heating_rate` (float).
- Synsets are used in task definitions through predicates, appear in various tasks, and are involved in transition rules.

### **Categories**
The bridge between the WordNet(-like) synsets and OmniGibson's object and substance categories.

- Each category is mapped to **exactly one leaf synset**, e.g. `apple` is mapped to `apple.n.01`.
- Multiple categories can be mapped to the same synset, e.g. `drop_in_sink` and `pedestal_sink` both map to `sink.n.01`, and share the exact same properties (because properties are annotated at the synset level, not the category level).
- All objects belonging to the same category should share similar mass and size, i.e. should be interchangeable if object randomization is performed.

### **Objects**
One-to-one mapping to a specific 3D object model in our dataset.

- Each object belongs to **exactly one category**, e.g. `coffee_maker-fwlabx` belongs to `coffee_maker`, correspounding to the object model residing at `<gm.DATA_PATH>/behavior-1k-assets/objects/coffee_maker/fwlabx`.
- Each object can have multiple meta links that serve the relevant object states in OmniGibson. For example, for the `coffee_maker-fwlabx` object, it is annotated with `connectedpart` for the `AttachedTo` state, `heatsource` for the `HeatSourceOrSink` state, and `toggleButton` for the `ToggledOn` state.
- Objects appear in specific scenes and rooms within the BEHAVIOR environment.

### **Scenes**
One-to-one mapping to a specific 3D scene model in our dataset.

- Each scene consists of multiple rooms with the following naming convention: `<room_type>_<room_id>`, e.g. `living_room_0`, `kitchen_1`, etc.
- Each room contains a list of objects, e.g. in the `Beechwood_0_int` scene, `countertop-tpuwys: 6` means the `kitchen_0` room has 6 copies of the `countertop-tpuwys` object.

### **Transition Rules**
Hand-specified rules that define complex physical or chemical interactions between objects and substances that are not natively supported by Omniverse.

- Each transition rule specifies a list of input synsets and a list of output synsets, as well as the conditions that need to be satisfied for the transition to occur.
- For instance, in the `beef_stew` rule, the input synsets are `ground_beef.n.01`, `beef_broth.n.01`, `pea.n.01`, `diced__carrot.n.01` and `diced__vidalia_onion.n.01` and the output synset is `beef_stew.n.01`.
- The conditions can be inspected in the [JSON files](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_jsons).

We have 6 different types of transition rules:

- `WasherRule`: remove "dirty" substance from the washer if the necessary solvent is present, and wet the objects inside by making them either `Saturated` with or `Covered` by `water`.
- `DryerRule`: dry the objects inside by making them not `Saturated` with `water`, and remove all `water` from the dryer.
- `SlicingRule`: when an object with the `slicer` ability exerts a sufficient force on another object with the `sliceable` ability, it slices the latter object into two halves.
- `DicingRule`: when an object with the `slicer` ability exerts a sufficient force on another object with the `diceable` ability, it dices the latter object into the corresounding diced substance.
- `MeltingRule`: when an object with the `meltable` ability reaches a certain temperature, it melts into the corresounding melted substance.
- `RecipeRule`: a general framework of recipe-based transitions that involve multiple objects and substances, and custom defined conditions.
    - `input_objects`: input objects and their counts that are required
    - `input_systems`: input systems that are required
    - `output_objects`: output objects and their counts that are produced
    - `output_systems`: output systems that are produced (the quantity depends on the collective volume of the input objects and systems)
    - `input_states`: the states that the input objects and systems should satisfy, e.g. an ingredient should not be `cooked` already.
    - `output_states`: the states that the output objects and systems should satisfy, e.g. the dish should be `cooked` after the recipe is done.
    - `fillable_categories`: `fillable` object categories needed for the recipe, e.g. pots and pans for cooking, and coffee makers for brewing coffee.

We have 5 different types of `RecipeRule`s:

- `CookingPhysicalParticleRule`: "cook" physical particles. It might or might not require water, depending on the synset's property `waterCook`.
    - Requires water: `rice` + `cooked__water` -> `cooked__rice`.
    - Doesn't require water: `diced__chicken` -> `cooked__diced__chicken`.
- `ToggleableMachineRule`: leverages a `toggleable` ability machine (e.g. electric mixer, coffee machine, blender) that needs to be `ToggledOn`.
    - Output is a single object: `flour` + `butter` + `sugar` -> `dough`; the machine is `electric_mixer`.
    - Output is a single system: `strawberry` + `milk` -> `strawberry_smoothie`; the machine is `blender`.
- `MixingToolRule`: leverages a `mixingTool` ability object that gets into contact with a `fillable` ability object.
    - Output is a single system: `water` + `lemon_juice` + `sugar` -> `lemonade`; the mixing tool is `spoon`.
- `CookingRule`: leverages a `heatsource` ability object and a `fillable` ability object for general cooking.
    - `CookingObjectRule`: Output is one or more objects: `bagel_dough` + `egg` + `sesame_seed` -> `bagel`; the heat source is `oven`; the container is `baking_sheet`.
    - `CookingSystemRule`: Output is a single system: `beef` + `tomato` + `chicken_stock` -> `stew`; the heat source is `stove`; the container is `stockpot`.

## Usage

OmniGibson interfaces with the BEHAVIOR hierarchy via a single interface: the [`ObjectTaxonomy`](https://github.com/StanfordVL/bddl/blob/master/bddl/object_taxonomy.py) class.

Here is an example of how to use the `ObjectTaxonomy` class to query the BEHAVIOR hierarchy.

```{.python .annotate}
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY

# Get parents / children / ancestors / descendants / leaf descendants of a synset
parents = OBJECT_TAXONOMY.get_parents("fruit.n.01")
children = OBJECT_TAXONOMY.get_children("fruit.n.01")
ancestors = OBJECT_TAXONOMY.get_ancestors("fruit.n.01")
descendants = OBJECT_TAXONOMY.get_descendants("fruit.n.01")
leaf_descendants = OBJECT_TAXONOMY.get_leaf_descendants("fruit.n.01")

# Checker functions for synsets
is_leaf = OBJECT_TAXONOMY.is_leaf("fruit.n.01")
is_ancestor = OBJECT_TAXONOMY.is_ancestor("fruit.n.01", "apple.n.01")
is_descendant = OBJECT_TAXONOMY.is_descendant("apple.n.01", "fruit.n.01")
is_valid = OBJECT_TAXONOMY.is_valid_synset("fruit.n.01")

# Get the abilities of a synset, e.g. "coffee_maker.n.01" -> {'rigidBody': {...}, 'heatSource': {...}, 'toggleable': {...}, ...}
abilities = OBJECT_TAXONOMY.get_abilities("coffee_maker.n.01")

# Check if a synset has a specific ability, e.g. "coffee_maker.n.01" has "heatSource"
has_ability = OBJECT_TAXONOMY.has_ability("coffee_maker.n.01", "heatSource")

# Get the synset of a object category, e.g. "apple" -> "apple.n.01"
object_synset = OBJECT_TAXONOMY.get_synset_from_category("apple")

# Get the object categories of a synset, e.g. "apple.n.01" -> ["apple"]
object_categories = OBJECT_TAXONOMY.get_categories("apple.n.01")

# Get the object categories of all the leaf descendants of a synset, e.g. "fruit.n.01" -> ["apple", "banana", "orange", ...]
leaf_descendant_categories = OBJECT_TAXONOMY.get_subtree_categories("fruit.n.01")

# Get the synset of a substance category , e.g. "water" -> "water.n.06"
substance_synset = OBJECT_TAXONOMY.get_synset_from_substance("water")

# Get the substance categories of a synset, e.g. "water.n.06" -> ["water"]
substance_categories = OBJECT_TAXONOMY.get_substances("water.n.06")

# Get the substance categories of all the leaf descendants of a synset, e.g. "liquid.n.01" -> ["water", "milk", "juice", ...]
leaf_descendant_substances = OBJECT_TAXONOMY.get_subtree_substances("liquid.n.01")
```

## (Advanced) Customize BEHAVIOR Hierarchy

To customize the BEHAVIOR hierarchy, you can modify the source CSV files in the [bddl](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data) repository, and then rebuild the system.

### Modify Source CSV Files

You can use Excel, Google Sheets or any other spreadsheet software to modify the source CSV files below.

[category_mapping.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/category_mapping.csv)

- **Information**: map an object category to a synset.
- **When modify**: add a new object category.
- **Caveat**: you also need to add the canonical density of the object category to `<gm.DATA_PATH>/behavior-1k-assets/metadata/avg_category_specs.json`.

[substance_hyperparams.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/substance_hyperparams.csv)

- **Information**: map a substance category to a synset, and also specify the substance's type (e.g. `fluid`, `macro_physical_particle`), physical attributes (e.g. `is_viscous`, `particle_density`) and visual appearance (e.g. `material_mtl_name`, `diffuse_reflection_color`).
- **When modify**: add a new substance category.
- **Caveat**: you also need to add the metadata (in a JSON file) and (optionally) particle prototypes to the `<gm.DATA_PATH>/behavior-1k-assets/systems/<substance_category>`.
    - `fluid`: only metadata is needed, e.g. `<gm.DATA_PATH>/behavior-1k-assets/systems/water/metadata.json`.
    - `granular`: both metadata and particle prototypes are needed, e.g. `<gm.DATA_PATH>/behavior-1k-assets/systems/salt/metadata.json` and `<gm.DATA_PATH>/behavior-1k-assets/systems/sugar/iheusv`.
    - `macro_physical_particle`: both hyperparams and particle prototypes are needed, e.g. `<gm.DATA_PATH>/behavior-1k-assets/systems/cashew/metadata.json` and `<gm.DATA_PATH>/behavior-1k-assets/systems/cashew/qyglnm`.
    - `macro_visual_particle`: both hyperparams and particle prototypes are needed, e.g. `<gm.DATA_PATH>/behavior-1k-assets/systems/stain/metadata.json` and `<gm.DATA_PATH>/behavior-1k-assets/systems/stain/ahkjul`.

[synsets.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/synsets.csv)

- **Information**: specify the parent and abilities of a synset.
- **When modify**: add a new synset.
- **Caveat**: feel free to create custom synsets if you can't find existing ones from WordNet; you also need to update the property parameter annotations in the `prop_param_annots` folder accordingly (see below).

[prop_param_annots/*](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots)

- **Information**: specify the hyperparameters of the abilities (or properties) of a synset.
- **When modify**: add a new synset that has the ability, or modify the hyperparameters of the ability.
- **Caveat**: if a new object or substance synset is involved, you also need to modify `synsets.csv`, `category_mapping` and `substance_hyperparams.csv` accordingly (see above).

[prop_param_annots/heatSource.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/heatSource.csv)

- **Information**: specify the hyperparameters of the `heatSource` ability, e.g. whether the object needs to be toggled on or have its doors closed, whether it requires other objects to be inside it, and the heating temperature and rate.
- **When modify**: add a new synset that has the `heatSource` ability.

[prop_param_annots/coldSource.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/coldSource.csv)

- **Information**: specify the hyperparameters of the `coldSource` ability, e.g. whether the object needs to be toggled on or have its doors closed, whether it requires other objects to be inside it, and the heating temperature and rate.
- **When modify**: add a new synset that has the `coldSource` ability.

[prop_param_annots/cookable.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/cookable.csv)

- **Information**: specify the hyperparameters of the `cookable` ability, e.g. the temperature threshold, and the cooked version of the substance synset (if applicable).
- **When modify**: add a new synset that has the `cookable` ability.

[prop_param_annots/flammable.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/flammable.csv)

- **Information**: specify the hyperparameters of the `flammable` ability, e.g. the ignition and fire temperature, the heating rate and distance threshold.
- **When modify**: add a new synset that has the `flammable` ability.

[prop_param_annots/particleApplier.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/particleApplier.csv)

- **Information**: specify the hyperparameters of the `particleApplier` ability, e.g. modification method, conditions, and substance synset to be applied.
- **When modify**: add a new synset that has the `particleApplier` ability.

[prop_param_annots/particleSource.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/particleSource.csv)

- **Information**: specify the hyperparameters of the `particleSource` ability, e.g. conditions, and substance synset to be applied.
- **When modify**: add a new synset that has the `particleSource` ability.

[prop_param_annots/particleRemover.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/particleRemover.csv)

- **Information**: specify the hyperparameters of the `particleRemover` ability, e.g. conditions to remove white-listed substance synsets, and conditions to remove everything else.
- **When modify**: add a new synset that has the `particleRemover` ability.

[prop_param_annots/particleSink.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/particleSink.csv)

- **Information**: specify the hyperparameters of the `particleSink` ability (deprecated).
- **When modify**: add a new synset that has the `particleSink` ability.

[prop_param_annots/diceable.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/diceable.csv)

- **Information**: specify the hyperparameters of the `diceable` ability, e.g. the uncooked and cooked diced substance synsets.
- **When modify**: add a new synset that has the `diceable` ability.

[prop_param_annots/sliceable.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/sliceable.csv)

- **Information**: specify the hyperparameters of the `sliceable` ability, e.g. the sliced halves' synset.
- **When modify**: add a new synset that has the `sliceable` ability.

[prop_param_annots/meltable.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/prop_param_annots/meltable.csv)

- **Information**: specify the hyperparameters of the `meltable` ability, e.g. the melted substance synset.
- **When modify**: add a new synset that has the `meltable` ability.

[transition_map/tm_raw_data/*](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_raw_data)

- **Information**: specify the transition rules for different types of transitions.
- **Caveat**: if a new object or substance synset is involved, you also need to modify `synsets.csv`, `category_mapping` and `substance_hyperparams.csv` accordingly (see above).

[transition_map/tm_raw_data/heat_cook.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_raw_data/heat_cook.csv)

- **Information**: specify the transition rules for `CookingObjectRule` and `CookingSystemRule`, i.e. the input synsets / states, the output synsets / states, the heat source, the container, and the timesteps to cook.
- **When modify**: add a new transition rule for cooking objects or systems.

[transition_map/tm_raw_data/mixing_stick.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_raw_data/mixing_stick.csv)

- **Information**: specify the transition rules for `MixingToolRule`, i.e. the input synsets, and the output synsets.
- **When modify**: add a new transition rule for mixing systems.

[transition_map/tm_raw_data/single_toggleable_machine.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_raw_data/single_toggleable_machine.csv)

- **Information**: specify the transition rules for `ToggleableMachineRule`, i.e. the input synsets / states, the output synsets / states, and the machine.
- **When modify**: add a new transition rule for toggleable machines.

[transition_map/tm_raw_data/washer.csv](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_raw_data/washer.csv)

- **Information**: specify the transition rules for `WasherRule`, similar to `prop_param_annots/particleRemover.csv` , i.e. solvents required to remove white-listed substance synsets, and conditions to remove everything else.
- **When modify**: add a new transition rule for washing machines.

### Rebuild the Hierarchy

To rebuild the hierarchy, you need to run the following command:

```bash
cd bddl
python data_generation/generate_datafiles.py
```

To make sure the new hierarchy is consistent with the task definitions, you should also run the following command:

```bash
python tests/bddl_tests.py batch_verify
python tests/tm_tests.py
```

If you encounter any errors during the rebuilding process, please read the error messages carefully and try to fix the issues accordingly.