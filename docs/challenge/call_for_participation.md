
# **Announcing the 1st BEHAVIOR Challenge @ NeurIPS 2025**

Robots in the BEHAVIOR simulator perform everyday activities (like cooking or cleaning) in virtual home environments. *BEHAVIOR* (Benchmark for Everyday Household Activities in Virtual, Interactive, and Realistic environments) is a large-scale embodied AI benchmark with 1,000 defined household tasks grounded in real human needs. These tasks introduce long-horizon mobile manipulation challenges in realistic settings, bridging the gap between current research and real-world, human-centric applications. Even the state-of-the-art robot learning solutions struggle with the complexity and extended duration of BEHAVIOR’s activities, which is why we are thrilled to announce the 1st BEHAVIOR Challenge at NeurIPS 2025. This competition invites the community to tackle 50 of these full-length tasks in a realistic simulator - pushing the frontiers of both high-level planning and low-level control in house-scale environments.

<iframe width="560" height="315" src="https://www.youtube.com/embed/iSFpinMiT0s?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Challenge 2025 Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## **What is the BEHAVIOR Benchmark?**

**BEHAVIOR-1K** is a comprehensive simulation benchmark for embodied AI and robotics, focusing on everyday household tasks that people most want robots to assist with. It consists of three core components:

1. **Task Definitions:** 1,000 everyday activities (e.g., cooking, cleaning, organizing) formally defined in *BEHAVIOR Domain Definition Language* (BDDL). These span diverse categories like **rearrangement**, **cleaning/wiping**, **cooking/freezing**, **painting/spraying**, **hanging/installing**, **slicing/dicing**, **baking**, and **laundry**. Each task has a set of initial conditions and goal conditions specifying what constitutes success (for example, *“all candles must end up inside the baskets”* for an *assembling gift baskets* task).  
2. **Interactive Environments:** 50 high-fidelity, fully interactive scenes populated with around **10,000** objects. These are realistic, house-scale layouts (kitchens, living rooms, offices, etc.) with annotated objects that robots can manipulate. Every object has rich semantic properties and can exist in different states (for instance, a fridge can be *open* or *closed*, a pan can be *empty* or *filled* with water, a tomato can be *sliced*).  
3. **OmniGibson Simulator:** Built upon NVIDIA’s Omniverse, a physics simulation platform that makes these tasks possible with realistic interaction modeling. OmniGibson supports rigid-body physics as well as advanced phenomena like **deformable objects** (e.g. cloth and fabric), **fluid interactions** (pouring liquids), and complex **object state changes** (heating, cooling, cutting, etc.). This means a robot can do things like fold clothes, pour water, cook food, or clean up spills in simulation - bringing us closer to real-world robot capabilities.

Importantly, BEHAVIOR is the first benchmark of its kind that requires a broad range of robot abilities *at the same time*. A successful robot must perform **high-level reasoning**, **long-range navigation** through rooms, and **agile bimanual manipulation** of objects. In short, it’s not just a pick-and-place test or a navigation task - it’s both and more. The 50 tasks chosen for this year’s challenge are *full-length activities* that integrate dozens of sub-goals and skill primitives into one continuous sequence.

## **The BEHAVIOR Challenge @ NeurIPS 2025**

**The 1st BEHAVIOR Challenge** will officially take place as part of the NeurIPS 2025 competition. Participants will develop solutions (in simulation) to solve 50 household tasks drawn from the BEHAVIOR benchmark. Each task is a realistic scenario - for example, *“tidy bedroom”, “wash dog toys”, “collect children's toys”, “spray for bugs”,* or *“make a pizza”* \- that may require **several minutes** of autonomous execution, multiple rooms of exploration, and manipulation of many different objects in sequence. The goal is to complete the task as defined by the BDDL goal conditions (e.g., all the target objects in desired states and locations).

To empower learning these complex behaviors, we provide an extensive **dataset of expert demonstrations**. In total, the challenge offers **10,000 teleoperated trajectories** (over **1,200 hours** of data) collected from human experts solving the tasks in simulation. These demonstrations are *richly annotated* and are designed to be a valuable training resource for participants. Key features of the challenge include:

### **Large-Scale Demonstration Data** 

For each of the 50 tasks, we provide 200 human demonstrations, totaling 10k trajectories and 1,200+ hours of data. Each demonstration is recorded with multiple observation modalities (RGB-D camera streams, robot proprioception), along with the ground-truth state of objects and fine-grained annotations. Not only do we segment each demo into **subtasks/skills**, we also annotate spatial relations and provide natural-language descriptions at multiple granularities (e.g., step-by-step commentary and high-level summaries). This large, detailed dataset is a unique asset of BEHAVIOR.

For the 1st challenge, we use Galaxea’s R1 Pro robot, a wheeled humanoid, as the default embodiment. The choice is justified in our previous work, [BEHAVIOR Robot Suite](https://behavior-robot-suite.github.io/), for its capabilities of extensive end-effector reachability, bimanual coordination, and stable and accurate navigation that are essential for household activities.

<iframe width="560" height="315" src="https://www.youtube.com/embed/oVr3IYnQiys?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Annotation Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### **High-Quality Teleoperated Trajectories** 

All demonstrations were collected via **JoyLo**, our custom whole-body teleoperation interface. Using two handheld controllers (inspired by game consoles) and two toy robot arms, human operators can smoothly drive the mobile manipulator’s base, torso, and dual arms. We enforced strict quality control to ensure the data is *near-optimal and clean*. That means no failed grasps, no accidental collisions with the environment, and no jittery, unnatural motions - only smooth and purposeful manipulation behavior. Teleoperators moved at a moderate, consistent speed, providing expert examples. In short, this dataset reflects successful task executions, with minimal noise or trial-and-error, which is ideal for methods like imitation learning or offline reinforcement learning. All of our data and annotations are purchased from Simovation.

*[JoyLo](../behavior_components/joylo.md)* is a low-cost teleoperation system that lets a human operator control a complex robot. Our vendor, Simovation, used JoyLo to collect over 1,200 hours of expert demonstrations for BEHAVIOR tasks. JoyLo originally demonstrated impressive results on real robots and now works in simulation too, enabling operators to command the robot’s base, torso, arms, and grippers fluidly. This interface significantly speeds up data collection and improves data quality. 

<iframe width="560" height="315" src="https://www.youtube.com/embed/fFAtUzEETe4?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Data Quality Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

We thank [Simovation](https://www.linkedin.com/company/simovationinc/) for providing high-quality JoyLo teleoperation data in simulation. Their long-horizon, near-optimal, and clean data, together with their strict QA standards, provide a solid foundation for training robots in the BEHAVIOR challenge. Such scale and quality are rarely matched in the field, making Simovation a trusted source for high-quality data collection.


### **Long-Horizon Mobile Manipulation** 

BEHAVIOR tasks are truly **long-horizon**; they often involve dozens of skills and can span **several minutes** of continuous execution. In the simulator, tasks don’t “time out” until a fairly long duration, giving robots ample opportunity to try. Crucially, these activities play out in **household-scale environments**: the robot might have to navigate from the kitchen to the living room, remember where it left an item, and then return to a cupboard, all in one task. This poses a stern test of **memory, planning, and reasoning** over extended timescales. Solving such tasks means a robot must integrate navigation with manipulation in a coordinated plan. The challenge scenarios require robots to plan ahead (you can’t cook a meal without gathering ingredients first), adapt to intermediate state changes, and recover from mistakes, much like a human would when doing chores at home.  

<iframe width="560" height="315" src="https://www.youtube.com/embed/3XKhbg9_MS4?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Long-Horizon Task Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>  

### **Diverse State Changes & Low-level Skills** 

A striking aspect of BEHAVIOR is the variety of object **state transitions and low-level skills** involved in these tasks. This goes far beyond classic pick-and-place. For instance, objects can be *opened* or *closed* (cabinets, doors, appliances), *attached* or *detached* (attaching a camera to a tripod, hanging a picture), *heated* or *frozen* (food can become *cooked or frozen*), or even set *on fire*! Also, surfaces or objects can be *covered by particles* (e.g., with dust, stain, paint, etc), and these particles can be cleaned. The robot may need to *slice* or *dice* food items or *toggle on/off* electronic devices. In total, the tasks call for at least **30 primitive skills** and an understanding of complex state transitions. To succeed, a policy must know how to perform **low-level manipulation skills like** **pour**, **wipe**, **cut**, **attach**, **spray**, **cook**, **clean**, and more. This rich diversity of required skills is what makes BEHAVIOR especially challenging and exciting. It encourages research into more general, flexible robots that can handle whatever the household throws at them, rather than just a single specialized trick.  

<iframe width="560" height="315" src="https://www.youtube.com/embed/FeD8_KgVOag?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Skills Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## **Baselines and Getting Started**

To lower the entry barrier and facilitate rapid progress, we provide a suite of **baseline models** and tools that participants can use out of the box. Specifically, the challenge release includes training and evaluation pipelines for a number of state-of-the-art methods:

*Classic Behavioral Cloning baselines:* **ACT**, **Diffusion Policy**, **BC-RNN**, **WB-VIMA** - these are standard imitation learning approaches that learn from the provided demonstrations.

*Pre-trained Visuo-Language Action models:* **OpenVLA** and **π0**.  These models are pretrained by a large amount of demonstration data, giving an alternative to models that need to be trained from scratch.

Participants are encouraged to test and build upon these baselines. All the necessary code, simulation assets, and data are available. In fact, **everything is open-source** on our website: you’ll find instructions to install the OmniGibson simulator, download the 3D scene assets and object models, and load the demonstration dataset. We also include a **starter kit** with example training scripts and evaluation routines, so you can reproduce baseline results and then go beyond them. The **documentation** covers how to set up the environment, use our APIs for robot control, and visualize the task executions - making it as easy as possible to get started with developing your solution.

If you’re new to embodied AI, don’t worry: the BEHAVIOR challenge provides tutorials and a step-by-step guide (from simulator setup to submitting results). We’re excited to see what creative ideas teams will bring to tackle these tasks!

## **Evaluation, Timeline, and Prizes**

**Please see the detailed challenge rules [on this page](./evaluation.md). Here is a high-level overview.**

### **How will entries be evaluated?** 

We focus on **task success rate** as the primary metric. We count partial success - the number of the goal conditions a robot satisfies by the end of the episode. For each task, the simulator checks the BDDL predicates (e.g., *“candle is inside basket”* or *“floor is clean”*) to see if the robot achieved the desired outcome. A robot that fully completes the task will get 100%, but partial credit is possible if only some goals are achieved. This encourages solutions that make progress on the task, even if they don’t perfectly finish every detail.

In addition, we track a couple of secondary metrics to diagnose *how* the solution performed:

### **Efficiency** 

We measure the time taken, the distance the robot traveled, and the total joint movement. A clever robot that takes a shorter path or minimizes unnecessary movements is rated as more efficient. This can distinguish “messy” solutions from elegant ones - for example, two robots might both complete the task, but one might wander around, waste time, and cause unnecessary changes in the environment.

### **Data Utilization** 

We also consider how much training data each submission used. Teams can use as much or as little of the 10k demonstrations (for imitation learning) or the simulator (for RL) as they want, but we record the total frames of experience. This gives a sense of **data efficiency** - e.g., did a team achieve high performance with only a few demonstrations? Such insights could help us understand the “scaling law” of embodied AI.

Submissions will be made through Google Form. Teams will evaluate their solutions on a **validation set** of tasks initially (with an open leaderboard during the development phase), and final scoring will be done on a hidden **test set** of task instances to determine the winners. The **timeline** is as follows:

**September 2nd, 2025:** Challenge **launch** - participants can register on our platform, download all data, and begin developing their solutions. During this phase, you can submit to the public leaderboard to see where you stand and iterate on your models.

**November 16th, 2025:** **Final evaluation** - the organizers will run all qualified submissions on the hidden test set in a controlled environment. We’ll then aggregate the results and identify the top-performing solutions.

**December 6-7, 2025:** **NeurIPS 2025 Competition Event** - final results are revealed and the **winners announced** at the NeurIPS conference in San Diego. The top teams will be invited to present their approaches at the NeurIPS competition track workshop. This is a great opportunity to discuss what worked, what didn’t, and share insights with the broader research community.

To encourage wide participation, we’ve secured **prizes** for the winners: **$1,000** for 1st place, **$500** for 2nd, and **$300** for 3rd place (along with, of course, the opportunity to publish your methods in the proceedings!). The real reward is contributing to the advance of embodied AI — and perhaps claiming the title of the robot that best “behaves” in a virtual home.

## **Looking Ahead: A New Frontier in Embodied AI**

This challenge marks the beginning of a new frontier in embodied AI and robotics. By taking on BEHAVIOR’s challenging tasks, we as a community can start answering some **important open questions** in the field:

**How close are we** to solving truly *human-centric* household tasks with current AI techniques? Can today’s algorithms tidy a house or cook a meal without supervision, or do we still have a long way to go?

**What are the generalization limits** of models in these environments? If a robot learns to *make a bed* once, will it reliably work again in a slightly different instance? How well do our methods handle new task instances (i.e., different initial configurations of the environments, including the objects and the robot) that they weren’t explicitly trained on?

**Are there scaling laws** for embodied AI systems, similar to those in language models or vision? For instance, does doubling the amount of demonstration data significantly improve performance, or are we bottlenecked by other factors?

By participating in the BEHAVIOR Challenge, you’ll help shed light on these questions. Perhaps your approach will demonstrate a breakthrough in low-level manipulation skill learning or long-horizon planning, or maybe it will reveal that current methods plateau and we need new ideas. Either way, the findings will be invaluable to guide future research.

We also warmly invite everyone to **join our BEHAVIOR [Discord community](https://discord.gg/bccR5vGFEx)** and our public office hours every Monday & Thursday 4:30-5:30pm PST via **[Zoom](https://stanford.zoom.us/j/92909660940?pwd=RgFrdC8XeB3nVxABqb1gxrK96BCRBa.1)**. This is the central hub for participants to ask questions, share progress, and get help from the organizers. Whether you’re a robotics expert or just starting out in embodied AI, you’ll find peers and mentors on the Discord ready to collaborate. Our team will be active there to provide technical support, clarify rules, and hear your feedback — we’re here to help you succeed.

In conclusion, the 1st BEHAVIOR Challenge is not just a competition for a leaderboard spot — it's a collective exploration of what it takes to make AI agents that truly **understand and interact with the human world**. The tasks are hard, yes, but they are also deeply meaningful: they represent the kind of assistance that could one day improve people's daily lives. By tackling them in simulation now, we take important steps toward that future. We encourage you to bring your creativity, your best algorithms, and maybe a bit of courage (to face the wild world of household chores!) and join us in this challenge.

The tasks are ready, the data is waiting, and the robots are ready to learn. How far can we push the state of embodied intelligence? - **Let's find out together!**

---

## **BibTeX**

To cite BEHAVIOR-1K, please use:
```bibtex
@article{li2024behavior,
  title={Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation},
  author={Li, Chengshu and Zhang, Ruohan and Wong, Josiah and Gokmen, Cem and Srivastava, Sanjana and Mart{\'i}n-Mart{\'i}n, Roberto and Wang, Chen and Levine, Gabrael and Ai, Wensi and Martinez, Benjamin and Yin, Hang and Lingelbach, Michael and Hwang, Minjune and Hiranaka, Ayano and Garlanka, Sujay and Aydin, Arman and Lee, Sharon and Sun, Jiankai and Anvari, Mona and Sharma, Manasi and Bansal, Dhruva and Hunter, Samuel and Kim, Kyu-Young and Lou, Alan and Matthews, Caleb R. and Villa-Renteria, Ivan and Tang, Jerry Huayang and Tang, Claire and Xia, Fei and Li, Yunzhu and Savarese, Silvio and Gweon, Hyowon and Liu, C. Karen and Wu, Jiajun and Fei-Fei, Li},
  journal={arXiv preprint arXiv:2403.09227},
  year={2024}
}
```
