# :material-bookshelf: **BEHAVIOR Knowledgebase Website**

The [**BEHAVIOR Knowledgebase**](https://behavior.stanford.edu/knowledgebase/) is a web-based visualization tool that provides an interactive interface for exploring the BEHAVIOR hierarchy. It allows users to browse and understand the relationships between synsets, categories, objects, scenes, tasks, and transition rules.

The website serves as a comprehensive reference for researchers and developers working with the BEHAVIOR dataset and OmniGibson simulation environment.

## Website Features

### [**Tasks**](https://behavior.stanford.edu/knowledgebase/tasks)
The website displays information about the 1000 long-horizon household activities.

- For each task, the website shows:
    - Which scenes this task is compatible with
    - (Experimental) The transition paths that help achieve the goal conditions

### [**Synsets**](https://behavior.stanford.edu/knowledgebase/synsets)
The website provides detailed information about each synset in the hierarchy.

- For each synset, the website displays:
    - The predicates that are used for the synset in the task definitions
    - The tasks that involve the synset
    - The object categories and models that belong to the synset
    - The transition rules that involve the synset
    - The synset's position in the WordNet hierarchy (e.g. ancestors, descendants, etc)

### [**Categories**](https://behavior.stanford.edu/knowledgebase/categories)
The website shows the mapping between categories and synsets.

- For each category, the website displays:
    - The objects that belong to the category, as well as their images
    - The corresponding synset's position in the WordNet hierarchy (e.g. ancestors, descendants, etc)

### [**Objects**](https://behavior.stanford.edu/knowledgebase/objects)
The website provides detailed information about individual 3D object models.

- For each object (e.g. [`coffee_maker-fwlabx`](https://behavior.stanford.edu/knowledgebase/objects/coffee_maker-fwlabx/index.html)), the website shows:
    - The object's image
    - The scenes / rooms the object appears in
    - The corresponding synset's position in the WordNet hierarchy (e.g. ancestors, descendants, etc)

### [**Scenes**](https://behavior.stanford.edu/knowledgebase/scenes)
The website displays information about the 3D scene models in the dataset.

- For each scene (e.g. [`Beechwood_0_int`](https://behavior.stanford.edu/knowledgebase/scenes/Beechwood_0_int/index.html)), the website shows:
    - The rooms in the scene and their object inventories
    - For example, `countertop-tpuwys: 6` indicates that the `kitchen_0` room has 6 copies of the `countertop-tpuwys` object

### [**Transition Rules**](https://behavior.stanford.edu/knowledgebase/transitions/index.html)
The website visualizes the transition rules that define object interactions.

- For each transition rule (e.g. [`beef_stew`](https://behavior.stanford.edu/knowledgebase/transitions/beef_stew)), the website displays:
    - The input synsets required for the transition
    - The output synsets produced by the transition
    - Note: The conditions are not yet visualized on the website, but can be manually inspected in the [JSON files](https://github.com/StanfordVL/bddl/tree/master/bddl/generated_data/transition_map/tm_jsons)
