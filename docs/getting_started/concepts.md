# Concepts
In this document, we discuss and disambiguate a number of concepts that are central to working with OmniGibson and BEHAVIOR-1K.

## Components of the BEHAVIOR ecosystem
The BEHAVIOR ecosystem consists of four components: BDDL (the symbolic knowledgebase), OmniGibson (the simulator), the BEHAVIOR dataset (the scene and object assets) and the OmniGibson assets (robots etc).

### OmniGibson
OmniGibson is the main software component of the BEHAVIOR ecosystem. It is a robotics simulator built on NVIDIA Isaac Sim and is the successor of the BEHAVIOR team's previous well known simulator, iGibson. OmniGibson is designed to meet the needs of the BEHAVIOR project, including realistic rendering, high-fidelity physics, and the ability to simulate soft bodies and fluids.

OmniGibson is a Python package, and it requires Isaac Sim to be available locally to function. It can also be used independently from the BEHAVIOR ecosystem to perform robot learning on different robots, assets, and tasks. The OmniGibson stack is discussed further in the "OmniGibson, Omniverse and Isaac Sim" section.

### OmniGibson Assets
The OmniGibson assets are a collection of robots and other simple graphical assets that are downloaded into the omnigibson/data directory. These assets are necessary to be able to OmniGibson (e.g. no robot simulation without robots!) for any purpose, and as such are shipped separately from the BEHAVIOR dataset which contains the items needed to simulate BEHAVIOR tasks. These assets are not encrypted.

### The BEHAVIOR dataset
The BEHAVIOR dataset consists of the scene, object and particle system assets that are used to simulate the BEHAVIOR-1K tasks. Most of the assets were procured through ShapeNet and TurboSquid and the dataset is encrypted to comply with their license.

* Objects are represented as USD files that contain the geometry, materials, and physics properties of the objects. Materials are separately provided.
* Scene assets are represented as JSON files containing OmniGibson state dumps that describe a particular configuration of the USD objects in a scene. Scene directories also include additional information such as traversability maps of the scene with various subsets of objects included. *In the currently shipped versions of OmniGibson scenes, "clutter" objects that are not task-relevant are not included (e.g. the products for sale at the supermarket), to reduce the complexity of the scenes and improve simulation performance.*
* The particle system assets are represented as JSON files describing the parameters of the particle system. Some particle systems also contain USD assets that are used as particles of that system. Other systems are rendered directly using isosurfaces, etc.

### BDDL
The BEHAVIOR Domain Definition Language (BDDL) library contains the symbolic knowledgebase for the BEHAVIOR ecosystem and the tools for interacting with it. The BDDL library contains the below main components:

* The BEHAVIOR Object Taxonomy, which contains a tree of nouns ("synsets") derived from WordNet and enriched with annotations and relationships that are useful for robotics and AI. The Object Taxonomy also includes mapping of BEHAVIOR dataset categories and systems to synsets. The Object Taxonomy can be accessed using the `bddl.object_taxonomy` module.
* The BEHAVIOR Domain Definition Language (BDDL) standard, parsers, and implementations of all of the first-order logic predicates and functions defined in the standard.
* The definitions of the 1,000 tasks that are part of the BEHAVIOR-1K dataset. These are defined with initial and goal conditions as first-order logic predicates in BDDL.
* The backend abstract base class, which needs to be implemented by a simulator (e.g. OmniGibson) to provide the necessary functionality to sample the initial conditions and check the predicates in goal conditions of tasks.
* Transition rule definitions, which define recipes, like cooking, that result in the removal and addition of nouns into the environment state at a given time. Some of these transitions are critical to completion of a task, e.g. blending lemons and water in a blender need to produce the blender substance for a `making_lemonade` task to be feasible. These need to be implemented by the simulator.
* The knowledgebase module (`bddl.knowledge_base`) that contains an ORM representation of all of the BDDL + BEHAVIOR dataset concepts. This can be used to investigate the relationships between objects, synsets, categories, substances, systems, and tasks. The [BEHAVIOR knowledgebase website](https://behavior.stanford.edu/knowledgebase) is a web interface to this module.

### BDDL

## BEHAVIOR concepts
### Tasks

### Synsets

### Categories

### Objects

### Substances / Systems


## OmniGibson, Omniverse and Isaac Sim
TODO

## USD, Physx and Fabric
TODO

## Play and Stop
TODO

## Prims and Objects
TODO

## Dataset vs Assets
TODO

## Lazy Imports