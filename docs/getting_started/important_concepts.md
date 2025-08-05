# :material-lightbulb: **Important Concepts**

In this document, we discuss and disambiguate a number of concepts that are central to working with OmniGibson and BEHAVIOR-1K.

## **BEHAVIOR concepts**

At a high level, the BEHAVIOR dataset consists of tasks, synsets, categories, objects and substances. These are all interconnected and are used to define and simulate household robotics.

### Tasks

Tasks in the BEHAVIOR are first order logic formalizations of 1000+ long-horizon household activities that survey participants indicated they would benefit from robot help with. Each task is defined in a single BDDL file that includes the list of objects needed for the task (the *object scope*), and their *initial conditions* (e.g. what a scene should look like when the task begins) and *goal conditions* (e.g. what needs to be true for the task to be considered completed). Task definitions are symbolic - they can be grounded in a particular scene with particular object, which is called a *task instance*. Task instances are created through a process called *sampling* that finds scenes and rooms that match the task's requirements, and configures non-scene objects into configurations that satisfy the task's initial conditions.

### Synsets

Synsets are the nouns used in the BDDL object scopes, expanded from the WordNet hierarchy with additional synsets to suit BEHAVIOR needs. Synsets are laid out in the form of a directed acyclic graph, so each synset can have parents/ancestors and children/descendants. When a task object scope requires a synset (e.g. "grocery.n.01"), instantiations of the task might use objects belonging to any descendant of that synset (e.g. an apple, assigned to "apple.n.01"), allowing a high degree of flexibility in task definitions. Each synset is annotated with abilities and parameters that define the kind of behaviors expected from objects of that synset (e.g. a faucet is a source of water, a door is openable, a stove is a heat source, etc.)

### Categories

Categories act as a bridge between synsets and OmniGibson's objects. Each category is mapped to one leaf synset and can contain multiple objects. The purpose of a category is to disambiguate between objects that are semantically the same but functionally & physically not; e.g. both a wall-mounted sink and a standing sink are `sink.n.01` semantically (e.g. they have the same functions and can be used for the same purposes), but they should not be swapped for one another during object randomization for the sake of physical and visual realism. As a result, wall_mounted_sink and standing_sink are different categories, but they are mapped to the same synset and thus can be used for the same task-relevant purposes.

### Objects

Objects denote specific 3D object models in the dataset. Each object belongs to one category and has a unique 6-character ID that identifies that object in the dataset. Objects can have articulations and metadata annotated, used in OmniGibson to simulate the abilities expected by the object's assigned synset. For example, a faucet is a fluid source, so it needs to have an annotation for the position the water will come out of.

### Scenes

Scenes are specific configurations of objects. A scene file by default will contain the information needed to lay out all the objects to form the scene. BEHAVIOR-1K ships with 50 base scenes that show a variety of different environments like houses, offices, restaurants, etc. and these scenes can be randomized by performing object randomization by replacing objects with other objects from the same category within the existing objects' bounding boxes. During task sampling, additional objects as requested in the object scope can be added, and these scene/task combinations (*task instances*) can be saved separately. BEHAVIOR-1K ships with at least one instantiation of each task.

### Substances / Systems

Some synsets, such as water, are marked as substances. For substance synsets, categories and objects are not provided, instead, these synsets are mapped to *particle systems* inside OmniGibson. Particle systems can act in a variety of ways: some like water act and are rendered as fluids, others like stains are simply visual particles with custom meshes. Substances are implemented singletons at the scene level, e.g. there is only one *water* particle system in a scene, and its particles may be arbitrarily placed in the scene. At a symbolic level, other objects can be filled with, covered in, or simply containing, particles of a particle system.

### Transition Rules

Transition rules define complex physical or chemical interactions between objects and substances not natively supported by Omniverse. They specify input and output synsets, conditions for transitions, and involve rules for washing, drying, slicing, dicing, melting, and recipe-based transformations. Each rule type has specific input and output requirements and conditions. When the input requirements are satisfied, the rule will be applied, causing the removal of some objects/substances and the addition of others into the scene.


## **Components of the BEHAVIOR ecosystem**

The BEHAVIOR ecosystem consists of four components: BDDL (the symbolic knowledgebase), OmniGibson (the simulator), OmniGibson assets (robots etc), and the BEHAVIOR dataset (the scene and object assets).

### BDDL

The BEHAVIOR Domain Definition Language (BDDL) library contains the symbolic knowledgebase for the BEHAVIOR ecosystem and the tools for interacting with it. The BDDL library contains the below main components:

* The BEHAVIOR Object Taxonomy, which contains a tree of nouns ("synsets") derived from WordNet and enriched with annotations and relationships that are useful for robotics and AI. The Object Taxonomy also includes mapping of BEHAVIOR dataset categories and systems to synsets. The Object Taxonomy can be accessed using the `bddl.object_taxonomy` module.
* The BEHAVIOR Domain Definition Language (BDDL) standard, parsers, and implementations of all of the first-order logic predicates and functions defined in the standard.
* The definitions of the 1,000 tasks that are part of the BEHAVIOR-1K dataset. These are defined with initial and goal conditions as first-order logic predicates in BDDL.
* The backend abstract base class, which needs to be implemented by a simulator (e.g. OmniGibson) to provide the necessary functionality to sample the initial conditions and check the predicates in goal conditions of tasks.
* Transition rule definitions, which define recipes, like cooking, that result in the removal and addition of nouns into the environment state at a given time. Some of these transitions are critical to completion of a task, e.g. blending lemons and water in a blender need to produce the blender substance for a `making_lemonade` task to be feasible. These need to be implemented by the simulator.
* The knowledgebase module (`bddl.knowledge_base`) that contains an ORM representation of all of the BDDL + BEHAVIOR dataset concepts. This can be used to investigate the relationships between objects, synsets, categories, substances, systems, and tasks. The [BEHAVIOR knowledgebase website](https://behavior.stanford.edu/knowledgebase) is a web interface to this module.

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


## **OmniGibson, Omniverse, Isaac Sim and PhysX**

OmniGibson is an open-source project that is built on top of NVIDIA's Isaac Sim and Omniverse. Here we discuss the relationship between these components.

### Omniverse

Omniverse is a platform developed by NVIDIA that provides a set of tools and services for creating, sharing, and rendering 3D content.

Omniverse on its own is a SDK containing a UI, a photorealistic renderer (RTX/Hydra), a scene representation (USD), a Physics engine (PhysX) and a number of other features. Its components, and other custom code, can be used in different combinations to create "Omniverse apps".

An Omniverse app usually involves rendering, but does not have to involve physics simulation. NVIDIA develops a number of such apps in-house, e.g. Omniverse Create which can be used as a CAD design tool, and Isaac Sim, which is an application for robotics simulation.

### PhysX

PhysX is a physics engine owned and developed by NVIDIA and used in a variety of games and platforms like Unity. It is integrated into Omniverse and thus can be used to apply physics updates to the state of the scene in an Omniverse app.

PhysX supports important features that are necessary for robotics simulation, such as articulated bodies, joints, motors, controllers, etc.

### Isaac Sim

Isaac Sim is an Omniverse app developed by NVIDIA that is designed for robotics simulation. It is built on top of Omniverse and uses PhysX for physics simulation. As an Omniverse app, it's defined as a list of Omniverse components that need to be enabled to comprise the application, as well as providing a thin layer of custom logic to support launching the application as a library and programmatically stepping the simulation rather than launching it as an asychronous, standalone desktop application.

It's important to note that the Omniverse SDK is generally meant as a CAD / collaboration / rendering platform and is monetized as such. Isaac Sim is a bit of a special case in that its main purpose is robotics simulation, which usually involves starting with a fixed state and simulating through physics, rather than manually making changes to a CAD file manually, or by making animations using keyframes. The application also runs as a MDP where the viewport updates on step rather than asynchronously like a typical interactive desktop app. As a result, a lot of Omniverse features are not used in Isaac Sim, and some features (e.g. timestamps, live windows, etc.) do not quite work as expected.

### OmniGibson

OmniGibson is a Python package that is built by the BEHAVIOR team at the Stanford Vision and Learning Group on top of Isaac Sim and provides a number of features that are necessary for simulating BEHAVIOR tasks. OmniGibson:

* completely abstracts away the Isaac Sim interface (e.g. users do not interact with NVIDIA code / interfaces / abstractions at all), instead providing a familiar scene/object/robot/task interface similar to those introduced in iGibson
* provides a number of fast high-level APIs for interacting with the simulator, such as loading scenes, setting up tasks, and controlling robots
* implements samplers and checkers for all of the predicates and functions defined in the BDDL standard to allow instantiation and simulation of BEHAVIOR-1K tasks
* includes utilities for working with the BEHAVIOR dataset including decryption, saving / loading scene states, etc.
* supports very simple vectorization across multiple copies of the scene to aid with training reinforcement learning agents
* provides easily configurable controllers (direct joint control, inverse kinematics, operational space, differential drive, etc.) that can be used to control robots in the simulator

OmniGibson is shipped as a Python package through pip or GitHub, however, it requires Isaac Sim to be installed locally to function. It can also be used independently from the BEHAVIOR ecosystem to perform robot learning on different robots, assets, and tasks.