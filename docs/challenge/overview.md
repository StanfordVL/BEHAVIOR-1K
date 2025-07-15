---
icon: material/trophy
---

# 🏆 **BEHAVIOR Challenge**

!!! info "Challenge Overview"
    **BEHAVIOR** is a robotics challenge for everyday household tasks - the first of its kind that requires robots to demonstrate high-level reasoning, long-range locomotion, and dexterous bimanual manipulation in house-scale environments.

!!! success "🎯 Challenge Focus"
    This year's challenge features **50 carefully selected tasks** from our comprehensive benchmark of 1,000 everyday household activities, designed to test the full spectrum of household robotics capabilities.

---

## :material-puzzle: **Challenge Components**

BEHAVIOR consists of three interconnected components that create the most comprehensive household robotics benchmark available:

=== ":material-format-list-checks: Task Definitions"

    !!! abstract "1,000 Everyday Household Activities"
        Comprehensive task definitions covering the full spectrum of household activities that humans need help with most.

    **📋 Task Categories Include:**
    
    - **🔄 Rearrangement** - Organizing and repositioning objects
    - **🧽 Cleaning & Wiping** - Surface maintenance and hygiene
    - **🍳 Cooking & Freezing** - Meal preparation and preservation
    - **🎨 Painting & Spraying** - Surface treatment and decoration
    - **🔧 Hanging & Installing** - Setup and mounting tasks
    - **🔪 Slicing & Dicing** - Food preparation
    - **🥖 Baking** - Complex cooking processes
    - **👕 Laundry** - Cleaning and drying clothes

    !!! tip "Human-Grounded Design"
        All tasks are based on extensive surveys asking real people: **"What do you want robots to do for you?"**

=== ":material-home-city: Interactive Environments"

    !!! abstract "50 Fully Interactive Scenes + 10,000 Objects"
        Richly detailed household environments with thousands of interactive objects for realistic simulation.

    **🏠 Environment Features:**
    
    - **50 Complete Scenes** - Full house layouts with up to 26 rooms
    - **10,000+ Objects** - Richly annotated with physical properties
    - **Realistic Layouts** - Based on real-world household configurations
    - **Interactive Elements** - Every object and furniture can be manipulated and interacted with

    !!! success "Scale & Diversity"
        House-scale environments requiring navigation, manipulation, and complex reasoning across multiple rooms and object types.

=== ":material-robot: OmniGibson Simulator"

    !!! abstract "Feature-CompletePhysics & Interaction Simulation"
        State-of-the-art simulation environment capable of modeling complex real-world interactions.

    **⚙️ Simulation Capabilities:**
    
    - **🔧 Rigid Bodies** - Precise physics simulation
    - **🧶 Deformable Objects** - Cloth, fabric, and flexible materials
    - **💧 Fluids** - Water, oils, and liquid interactions
    - **🌡️ Thermal Effects** - Heat transfer and temperature changes
    - **🔥 Visual Effects** - Fire, smoke, and environmental effects

    !!! info "Realism Focus"
        OmniGibson enables unprecedented realism in household robotics simulation, bridging the gap between simulation and reality.

---

## :material-database: **Data & Baselines**

### 📊 Human Demonstration Dataset

!!! success "Dataset Scale"
    - **10,000 Trajectories** - Comprehensive coverage across task categories
    - **1,500+ Hours** - Total human teleoperation data
    - **Diverse Behaviors** - Spanning all major household activity types

### 📈 Rich Data Annotations

Each demonstration trajectory contains comprehensive multi-modal data:

=== ":material-camera: Visual Data"

    - **RGBD Observations** - Synchronized RGB and depth information
    - **Object & Part Segmentation** - Object and part-level segmentation masks and bounding boxes
    - **Point Clouds** - Ready-to-use point clouds
    - **Multi-Camera Views** - Egocentric and wrist-mounted views

=== ":material-robot-outline: Robot Data"

    - **Proprioceptive Information** - Joint angles, forces, and torques
    - **Action Sequences** - Complete robot action trajectories

=== ":material-cube-scan: State Information"

    - **Ground-Truth Object States** - Exact object positions and orientations
    - **Scene Graph** - Coming soon!

=== ":material-text-box-check: Dense Annotation"

    - **Subtask and Skill Segmentation** - Coming soon!
    - **Natural Language Annotation** - Coming soon!

### 🤖 Baseline Methods

We provide training and evaluation pipelines for popular state-of-the-art methods:

!!! example "Available Baseline Methods"

    | Method | Type | Description |
    |--------|------|-------------|
    | **BC-RNN** | Imitation Learning | Behavioral cloning with recurrent networks |
    | **ACT** | Imitation Learning | Action Chunking with Transformers |
    | **Diffusion Policy** | Imitation Learning | Diffusion-based policy learning |
    | **WB-VIMA** | Imitation Learning | Kinematic-aware hierarchical policy learning |
    | **OpenVLA** | Vision-Language-Action | Open-source VLA pretrained on Open X-Embodiment dataset|
    | **π0** | Vision-Language-Action | General-purpose robot foundation model |

!!! tip "Getting Started"
    All baseline implementations are provided with complete training scripts, evaluation protocols, and documentation to help participants get started quickly.

---

## :material-chart-line: **Evaluation Framework**

### 📊 Evaluation Metrics

We evaluate agents across three critical dimensions of household robotics performance:

=== ":material-flag-checkered: Task Completion Rate"

    !!! info "Primary Evaluation Metric"
        **Definition:** Fraction of satisfied predicates in the goal condition of BDDL (BEHAVIOR Domain Definition Language) task definitions within the given time limit.

=== ":material-speedometer: Agent Efficiency"

    !!! info "Performance Optimization"
        **Metrics:** Total distance traveled and energy expended during task execution.

=== ":material-school: Data Efficiency"

    !!! info "Learning Efficiency"
        **Metrics:** Total number of frames used during training (demonstrations for IL, simulator steps for RL).

### 📈 Reporting & Confidence

!!! success "Statistical Rigor"
    All results are reported with **95% confidence intervals** to ensure statistical significance and reproducibility.

**Leaderboard Features:**

- 🏆 **Primary Ranking** - Based on task completion rate
- 📊 **Comprehensive Metrics** - All evaluation dimensions displayed
- 📈 **Confidence Intervals** - Statistical significance reporting
- 🔄 **Regular Updates** - Real-time submission processing

---

## :material-web: **Resources & Participation**

### 🌐 Open Source Ecosystem

Everything you need to participate is freely available:

=== ":material-code-tags: Code & Documentation"

    **📁 Complete Codebase:**
    
    - 🔧 **Simulator Installation** - Step-by-step setup guides
    - 📦 **3D Asset Downloads** - Complete object and scene libraries
    - 👀 **Data Visualization** - Tools for exploring demonstrations
    - 🚀 **Baseline Implementations** - Ready-to-use training code

=== ":material-book-open: Learning Resources"

    **📚 Comprehensive Tutorials:**
    
    - 🎯 **Getting Started** - From installation to first experiments
    - 📊 **Data Handling** - Working with demonstration trajectories
    - 🤖 **Baseline Training** - How to train and evaluate models
    - 🏆 **Challenge Protocols** - Rules, submission guidelines, and best practices

=== ":material-account-group: Community & Support"

    **🤝 Participant Support:**
    
    - 💬 **Discord Community** - Real-time discussion and help
    - 📧 **Technical Support** - Direct assistance for technical issues
    - 📖 **Documentation** - Comprehensive guides and API references
    - 🔄 **Regular Updates** - Challenge updates and announcements

### 🎯 How to Participate

!!! success "Ready to Join?"
    
    **1. 📝 Register Your Team**
    
    - Visit our **EvalAI** platform for team registration
    - Complete participant information and agreements
    - Receive access to submission systems

    **2. 🔧 Set Up Your Environment**
    
    - Follow installation guides at [behavior.stanford.edu](https://behavior.stanford.edu/)
    - Download datasets and 3D assets
    - Verify setup with provided test scripts

    **3. 🚀 Start Development**
    
    - Explore baseline implementations
    - Develop your approach using our training pipelines
    - Test on validation sets before submission

    **4. 📊 Submit & Compete**
    
    - Submit results through EvalAI platform
    - Track your progress on the leaderboard
    - Iterate and improve based on feedback

---

## :material-trophy: **What Makes BEHAVIOR Special**

!!! quote "First-of-its-Kind Challenge"
    BEHAVIOR is the **first robotics challenge** that requires the full spectrum of household robotics capabilities:
    
    - 🧠 **High-Level Reasoning** - Complex task planning and execution
    - 🏃 **Long-Range Locomotion** - Navigation across house-scale environments  
    - 🤲 **Dexterous Bimanual Manipulation** - Coordinated two-handed object handling
    - 🏠 **House-Scale Complexity** - Real-world environmental challenges
