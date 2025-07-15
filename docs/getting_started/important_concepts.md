---
icon: material/lightbulb
---

# **Important Concepts**

In this document, we discuss and disambiguate a number of concepts that are central to working with OmniGibson and BEHAVIOR-1K.

---

## :material-brain: **BEHAVIOR Concepts**

At a high level, the BEHAVIOR dataset consists of **tasks**, **synsets**, **categories**, **objects**, and **substances**. These are all interconnected and are used to define and simulate household robotics.

### Core Concepts

=== ":material-format-list-checks: Tasks"

    !!! info "What are Tasks?"
        Tasks in BEHAVIOR are **first-order logic formalizations** of 1000+ long-horizon household activities that survey participants indicated they would benefit from robot help with.

    **Key Components:**
    
    - **Object Scope**: List of objects needed for the task
    - **Initial Conditions**: What a scene should look like when the task begins
    - **Goal Conditions**: What needs to be true for the task to be considered completed
    
    !!! example "Task Definition Process"
        - Task definitions are **symbolic** - they can be grounded in a particular scene with particular objects
        - This creates a **task instance** through a process called **sampling**
        - Sampling finds scenes and rooms that match requirements and configures objects to satisfy initial conditions

=== ":material-graph: Synsets"

    !!! info "What are Synsets?"
        Synsets are the **nouns** used in BDDL object scopes, expanded from the WordNet hierarchy with additional synsets to suit BEHAVIOR needs.

    **Structure:**
    
    - Organized as a **directed acyclic graph** 
    - Each synset can have parents/ancestors and children/descendants
    - High flexibility: when a task requires `grocery.n.01`, it can use any descendant like `apple.n.01`
    
    !!! tip "Synset Annotations"
        Each synset is annotated with **abilities and parameters** that define expected behaviors:
        
        - 🚰 Faucet → source of water
        - 🚪 Door → openable
        - 🔥 Stove → heat source

=== ":material-tag: Categories"

    !!! info "What are Categories?"
        Categories act as a **bridge** between synsets and OmniGibson's objects. Each category maps to one leaf synset and can contain multiple objects.

    **Purpose: Disambiguation**
    
    Both wall-mounted and standing sinks are semantically `sink.n.01` (same functions), but they shouldn't be swapped during randomization for physical/visual realism.
    
    !!! example "Category Example"
        ```
        Synset: sink.n.01
        ├── Category: wall_mounted_sink
        └── Category: standing_sink
        ```

=== ":material-cube: Objects"

    !!! info "What are Objects?"
        Objects denote **specific 3D object models** in the dataset. Each object belongs to one category and has a unique 6-character ID.

    **Features:**
    
    - **Articulations**: Movable parts and joints
    - **Metadata**: Annotations for synset abilities
    - **Physics Properties**: Materials, collision, etc.
    
    !!! example "Object Annotation"
        A faucet object needs annotation for **water output position** because its synset defines it as a fluid source.

=== ":material-home: Scenes"

    !!! info "What are Scenes?"
        Scenes are **specific configurations of objects** that form complete environments.

    **BEHAVIOR-1K Ships With:**
    
    - 🏠 **50 base scenes** showing variety of environments
    - 🏢 Houses, offices, restaurants, etc.
    - 🎲 **Object randomization** capability
    - 📋 **Task instances** with pre-configured setups
    
    !!! tip "Scene Flexibility"
        Objects can be replaced with others from the same category within existing bounding boxes during randomization.

=== ":material-water: Substances & Systems"

    !!! info "What are Substances?"
        Some synsets (like water) are marked as **substances** and implemented as **particle systems** in OmniGibson.

    **Types of Particle Systems:**
    
    | Type | Behavior | Rendering |
    |------|----------|-----------|
    | 💧 Fluids | Water-like physics | Fluid rendering |
    | 🎨 Stains | Visual particles | Custom meshes |
    | 🌫️ Smoke | Atmospheric effects | Volumetric rendering |
    
    !!! note "Singleton Implementation"
        There's only **one** particle system per substance type per scene (e.g., one water system), but particles can be placed arbitrarily.

=== ":material-arrow-decision: Transition Rules"

    !!! info "What are Transition Rules?"
        Transition rules define **complex physical or chemical interactions** between objects and substances not natively supported by Omniverse.

    **Rule Types:**
    
    - 🧼 **Washing**: Cleaning objects
    - 🌬️ **Drying**: Removing moisture
    - 🔪 **Slicing/Dicing**: Cutting objects
    - 🔥 **Melting**: State changes
    - 👨‍🍳 **Recipe-based**: Cooking transformations
    
    !!! example "Rule Application"
        When input requirements are satisfied → remove some objects/substances → add new ones to the scene

---

## :material-puzzle: **Components of the BEHAVIOR Ecosystem**

The BEHAVIOR ecosystem consists of four main components that work together to enable household robotics simulation.

=== ":material-code-braces: BDDL"

    !!! abstract "BEHAVIOR Domain Definition Language"
        The symbolic knowledgebase and tools for interacting with the BEHAVIOR ecosystem.

    **Main Components:**

    ### :material-tree: Object Taxonomy
    - Tree of nouns ("synsets") derived from WordNet
    - Enriched with robotics-specific annotations
    - Mapping of dataset categories to synsets
    - Accessible via `bddl.object_taxonomy` module

    ### :material-script: BDDL Standard
    - Parsers and implementations of first-order logic predicates
    - Functions defined in the BDDL standard
    - 1,000 task definitions with initial/goal conditions

    ### :material-api: Backend Interface
    - Abstract base class for simulator implementation
    - Provides functionality to sample initial conditions
    - Checks predicates in goal conditions

    ### :material-arrow-decision: Transition Rules
    - Recipe definitions (cooking, blending, etc.)
    - Critical for task completion
    - Must be implemented by simulator

    ### :material-database: Knowledge Base
    - ORM representation of all BDDL concepts
    - Investigate relationships between objects, synsets, categories
    - [**BEHAVIOR Knowledgebase Website**](https://behavior.stanford.edu/knowledgebase) 🌐

=== ":material-robot-excited: OmniGibson"

    !!! abstract "Main Simulator"
        The primary software component built on NVIDIA Isaac Sim, successor to iGibson.

    **Key Features:**

    ### :material-eye: Rendering & Physics
    - 🎨 **Realistic rendering**
    - ⚡ **High-fidelity physics**
    - 🌊 **Soft bodies and fluids**

    ### :material-api: Python Integration
    - 🐍 **Pure Python package**
    - 🔧 **Requires Isaac Sim locally**
    - 🎯 **Independent usage possible**

    ### :material-feature-search: Capabilities
    - Fast high-level APIs
    - Scene/object/robot/task interface
    - BDDL predicate samplers and checkers
    - Vectorization support for RL training
    - Configurable controllers (IK, operational space, etc.)

=== ":material-download: OmniGibson Assets"

    !!! abstract "Essential Assets"
        Collection of robots and graphical assets downloaded to `omnigibson/data`.

    **Contents:**
    
    - 🤖 **Robot models**
    - 🎨 **Graphical assets**
    - 🔓 **Unencrypted**
    - ✅ **Required for any OmniGibson usage**

=== ":material-database-outline: BEHAVIOR Dataset"

    !!! abstract "Scene & Object Assets"
        Encrypted dataset containing assets for simulating BEHAVIOR-1K tasks.

    **Asset Types:**

    ### :material-cube-outline: Objects
    - **USD files** with geometry, materials, physics
    - Separately provided materials
    - Procured from ShapeNet and TurboSquid

    ### :material-home-outline: Scenes
    - **JSON files** with OmniGibson state dumps
    - Traversability maps included
    - Clutter objects excluded for performance

    ### :material-water-outline: Particle Systems
    - **JSON parameter files**
    - Some include USD particle assets
    - Others use isosurfaces for rendering

    !!! warning "Encryption"
        Dataset is **encrypted** to comply with ShapeNet and TurboSquid licenses.

---

## :material-layers: **Technical Stack**

Understanding the relationship between OmniGibson and NVIDIA's technology stack.

=== ":material-earth: Omniverse"

    !!! abstract "NVIDIA's 3D Platform"
        Platform providing tools and services for creating, sharing, and rendering 3D content.

    **Core Components:**
    
    | Component | Purpose |
    |-----------|---------|
    | 🎨 **RTX/Hydra** | Photorealistic renderer |
    | 🎬 **USD** | Scene representation |
    | ⚡ **PhysX** | Physics engine |
    | 🖥️ **UI Framework** | User interface |
    
    !!! info "Omniverse Apps"
        Different combinations of components create "Omniverse apps":
        
        - **Omniverse Create** → CAD design tool
        - **Isaac Sim** → Robotics simulation

=== ":material-lightning-bolt: PhysX"

    !!! abstract "Physics Engine"
        NVIDIA's physics engine used in games, Unity, and integrated into Omniverse.

    **Robotics Features:**
    
    - 🦾 **Articulated bodies**
    - 🔗 **Joints and motors**
    - 🎮 **Controllers**
    - ⚙️ **Mechanical constraints**

=== ":material-robot-excited-outline: Isaac Sim"

    !!! abstract "Robotics Simulation App"
        Specialized Omniverse app designed for robotics simulation.

    **Key Characteristics:**
    
    - 🏗️ **Built on Omniverse + PhysX**
    - 📚 **Library-based usage**
    - 🎯 **Programmatic stepping**
    - 🎮 **MDP-style operation**
    
    !!! note "Special Case"
        Unlike typical CAD apps, Isaac Sim focuses on **physics simulation** rather than manual editing or keyframe animation.

=== ":material-code-tags: OmniGibson"

    !!! abstract "High-Level Robotics Interface"
        Python package by Stanford Vision and Learning Group, built on Isaac Sim.

    **What OmniGibson Provides:**

    ### :material-shield: Complete Abstraction
    - No direct interaction with NVIDIA code
    - Familiar scene/object/robot/task interface
    - Similar to iGibson but more powerful

    ### :material-api: High-Level APIs
    - 🏠 **Scene loading**
    - 📋 **Task setup**
    - 🤖 **Robot control**
    - 🎯 **Fast operations**

    ### :material-check-circle: BDDL Integration
    - Samplers for all BDDL predicates
    - Checkers for goal conditions
    - Full BEHAVIOR-1K task support

    ### :material-tools: Utilities
    - 🔐 **Dataset decryption**
    - 💾 **State saving/loading**
    - 🔢 **Simple vectorization**
    - 🎮 **Configurable controllers**

    !!! tip "Flexible Usage"
        Can be used **independently** from BEHAVIOR ecosystem for custom robots, assets, and tasks.

---

!!! success "Ready to Get Started?"
    Now that you understand the key concepts, check out our [**Examples**](examples.md) to see OmniGibson in action! 🚀
