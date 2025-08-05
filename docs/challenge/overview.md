# 🏆 **2025 BEHAVIOR Challenge**

**Join us and solve 50 full-length household tasks in the realistic BEHAVIOR-1K environment, with 10,000 teleoperated expert demonstrations (1000+ hours) available!** 🤖

---

## :material-graph-outline: **Overview**

**BEHAVIOR** is a robotics challenge for everyday household tasks. It's a large-scale, human-grounded benchmark that consists of three main components:

1. **1,000 everyday household activities** task definitions
2. **50 fully interactive scenes** and around 10,000 richly annotated objects  
3. **OmniGibson**, a simulation environment capable of modeling complex interactions with rigid bodies, deformable objects, and fluids

BEHAVIOR is the first challenge of its kind that requires a robot's capability in high-level reasoning, long-range locomotion, and dexterous bimanual manipulation in house-scale scenes. This year's challenge includes **50 tasks**.

## :material-format-list-checks: **Challenge Components**

### Task Definitions

The benchmark includes 1,000 everyday household activities covering diverse behaviors across: **rearrangement**, **cleaning/wiping**, **cooking/freezing**, **painting/spraying**, **hanging/installing**, **slicing/dicing**, **baking**, and **doing laundry**.

### Interactive Environments

- 50 fully interactive scenes with house-scale layouts
- 10,000+ richly annotated objects

### OmniGibson Simulator

The simulation environment supports:

- Rigid body physics
- Deformable objects (cloth, fabric)
- Fluid interactions (water, oils)
- Object semantic states (e.g., open, filled, on-top, inside, etc.)

## :material-database: **Data and Baselines**

### Dataset

The benchmark includes **10,000 human-demonstrated trajectories** with diverse behaviors across all task categories. Each demonstration contains:

- Synchronized RGBD observations
- Object and part-level segmentation masks
- Ground-truth object states  
- Robot proprioception
- Robot actions
- Skill and subtask annotations

### Available Baseline Methods

Participants have access to training and evaluation pipelines for these baseline methods: **ACT**, **Diffusion Policy**, **BC-RNN**, **WB-VIMA**, **OpenVLA**, and **π0**.

## :material-chart-box: **Evaluation**

### Metrics

Agents are evaluated across three areas:

1. **Task completion rate** (primary metric): Fraction of satisfied predicates in the goal condition of BDDL (BEHAVIOR Domain Definition Language) task definition
2. **Agent efficiency**: Total distance traveled and energy expended during task execution
3. **Data efficiency**: Total number of frames from demonstrations (IL) or simulator (RL) used during training

### Reporting

- Results are reported with 95% confidence intervals
- Primary ranking based on task completion rate
- All metrics displayed on the leaderboard
- EvalAI platform used for team registration, submission and leaderboard management

## :octicons-person-add-16: **Resources and Participation**

### Available Resources

All code, data, and documentation are open-source and available at [behavior.stanford.edu](https://behavior.stanford.edu/), including:

- Tutorial on simulator installation
- 3D asset downloads
- Demonstration data download and visualization tools
- Starter code for baseline methods
- Challenge rules and protocols

### How to Participate

1. **Register** your team on the EvalAI platform
2. **Install** the simulator and download the required data
3. **Develop** your approach using the provided baselines and training pipelines
4. **Submit** your results through EvalAI
5. **Track** your progress on the leaderboard

The challenge provides comprehensive documentation, tutorials, and baseline implementations to help participants get started with developing household robotics solutions.