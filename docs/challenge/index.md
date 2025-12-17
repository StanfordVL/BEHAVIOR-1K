# 🏆 **2025 BEHAVIOR Challenge**

**Join us and solve 50 full-length household tasks in the realistic BEHAVIOR-1K environment, with 10,000 teleoperated expert demonstrations (1200+ hours) available!** 🤖

- Event: Foundation Models Meet Embodied Agents Challenge @ NeurIPS 2025
- Time: Dec. 7th, 11:00 AM – 1:45 PM PST
- Location: Mezzanine Room 15AB, Convention Center, or [over zoom](https://us06web.zoom.us/j/84419017872?pwd=EeE71l6XbsjdU6kqy8jHJF45MSR9vM.1)


---

## 📣 **Announcements**

!!! info "🗓️ 12/01/2025 — Leaderboard Open"

    [Leaderboard](./leaderboard.md) for the 2025 BEHAVIOR Challenge is now officially online! Congradulations to the winners, and huge thanks to the participants for all the impressive submissions!


!!! info "🗓️ 11/17/2025 — Submissions Closed"

    Submissions for the 2025 BEHAVIOR Challenge have officially closed! We are thrilled to announce that we have received submissions from 18 teams across academia, industry, and individuals; participating from 4 countries (US, China, Canada, South Korea) and including some multinational teams. Stay tuned as we verify results for the final leaderboard!


!!! info "🗓️ 11/13/2025 — Deadline Extension"

    To allow any final evaluations to complete, we're extending the submission deadline by 24 hours.
    The new deadline will now be November 16th 11:59PM AoE. Please plan accordingly as we will not be able to accept any late submissions!


!!! info "🗓️ 11/07/2025 — Rule Clarifications"
    Key updates this week:
    
    - Documentation updates
    - Bug fixes & Hidden test logic addition. 

    [Read full details →](./updates.md#11072025)


!!! info "🗓️ 10/30/2025 — Rule Clarifications & Features Update"
    Key updates this week:
    
    - We have released language annotations for all 50 tasks.
    - We have included task id as part of the observation dict that the policy will receive. 
    - Added more utilities (HeavyRobotWrapper, score_utils, etc.)

    [Read full details →](./updates.md#10302025)

!!! info "🗓️ 10/08/2025 — Rule Clarifications, Bug Fixes & NVIDIA Sponsorship"
    Key updates this week:
    
    - Clarified evaluation setup: only task-relevant object poses and the robot’s initial pose will be randomized.  
    - Privileged info allowed during training for both tracks.  
    - Multiple bug fixes (`eval_utils.py`, USD asset format, partial credit).  
    - Updated [submission guidelines](./submission.md) + sample Dockerfile.  
    - **New sponsor:** [NVIDIA](https://www.nvidia.com/en-us/)!  
    - 💰 Prize pool updated:
        - 1st: $1,000 + GeForce 5080  
        - 2nd: $500 + (Jetson Orin Nano Super or $1,000 Brev Credits)   
        - 3rd: $300 + $500 Brev Credits  

    [Read full details →](./updates.md#10082025)

!!! info "🗓️ 09/28/2025 — Dataset Fixes & CLI Improvements"
    Highlights:
    
    - No formal registration required — submit directly!  
    - Fixed dataset sharding, robot start poses, and improved baseline checkpoints.  
    - Added new CLI args for evaluation (`testing_on_train_instances`, `max_steps`, `partial_scene_load`).  

    [Read full details →](./updates.md#09282025)

!!! info "🗓️ 09/19/2025 — Rule Clarifications, Evaluation Protocol & Tutorial"
    Highlights:
    
    - BDDL task definitions can be used for both tracks and are identical during evaluation.  
    - Additional self-collected data allowed for both tracks.
    - Defined evaluation timeout and success score metrics.  
    - Various bug fixes (Windows setup, dataset timestamp, evaluation scripts).  
    - Added new tutorial: [Configure robot action space](./evaluation.md#configure-robot-action-space).  

    [Read full details →](./updates.md#09192025)

---

## :material-graph-outline: **Overview**

**BEHAVIOR** is a robotics challenge for everyday household tasks. It's a large-scale, human-grounded benchmark that tests a robot's capability in high-level reasoning, long-range locomotion, and dexterous bimanual manipulation in house-scale scenes.

This year's challenge features:

- **50 full-length household tasks** from our 1,000 activity collection, covering diverse activities like rearrangement, cooking, cleaning, and installation
- **10,000 teleoperated demonstrations** (1200+ hours) for training

BEHAVIOR challenge is co-hosted with the [Embodied Agent Interface Competition](https://foundation-models-meet-embodied-agents.github.io/eai_challenge/) at NeurIPS 2025.

---

## :material-database: **Dataset & Baselines**

### Teleoperated Demonstrations

**10,000 expert demonstrations** (1200+ hours) collected via teleoperation:

- Synchronized RGBD observations
- Object and part-level segmentation
- Ground-truth object states
- Robot proprioception and actions
- Skill and subtask annotations

[Dataset details →](./dataset.md)

### Baseline Methods

Pre-implemented training & evaluation pipelines for:

- **Behavioral Cloning baselines**: ACT, Diffusion Policy, BC-RNN, WB-VIMA - these are diverse imitation learning approaches that learn from the provided demonstrations.
- **Pre-trained Visuo-Language Action models**: OpenVLA and π0. These models are pretrained by a large amount of demonstration data, giving an alternative to models that need to be trained from scratch.

[Baselines details →](./baselines.md)

## :material-chart-box: **Evaluation & Rules**

The organizers reserve the right of final interpretation of the challenge rules. 

### Challenge Tracks

**Standard track:** Limited to provided robot onboard observations (RGB + depth + instance segmentation + proprioception).

**Privileged information track:** May query simulator for any information (object poses, scene point clouds, etc.).

🏆 **Prizes per track:**

1. 🥇 $1,000 + GeForce 5080
2. 🥈 $500 + (Jetson Orin Nano Super or $1,000 Brev Credits)
3. 🥉 $300 + $500 Brev Credits

Top 3 teams from each track will be invited to present at the workshop!

### Evaluation Metrics

**Primary metric (for ranking):** Task success rate averaged across 50 tasks. Partial credit given as fraction of satisfied BDDL goal predicates.

**Secondary metrics (efficiency):**

- **Simulated time** - Total simulation steps × time per step
- **Distance navigated** - Total base movement distance
- **Hand displacement** - Cumulative hand movement

[Evaluation details & Full challenge rules →](./evaluation.md)


## :octicons-person-add-16: **Participating**

### Resources

Join our community to ask questions and discuss the challenge:

- **Discord**: [Join our Discord Server](https://discord.gg/bccR5vGFEx)
- **Office Hours**: Monday and Thursday, 4:30-6pm PST via [Zoom](https://stanford.zoom.us/j/92909660940?pwd=RgFrdC8XeB3nVxABqb1gxrK96BCRBa.1)

Whether you're a robotics veteran or just entering the field, we're here to support you.

### Important Dates

- **Challenge Launch**: September 2, 2025
- **Submission Deadline**: November 16th 11:59PM AoE, 2025
- **Winners Announcement**: December 6-7, 2025 @ NeurIPS conference in San Diego

## :material-book-edit: **BibTeX**

To cite BEHAVIOR-1K, please use:
```bibtex
@article{li2024behavior,
  title={Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation},
  author={Li, Chengshu and Zhang, Ruohan and Wong, Josiah and Gokmen, Cem and Srivastava, Sanjana and Mart{\'i}n-Mart{\'i}n, Roberto and Wang, Chen and Levine, Gabrael and Ai, Wensi and Martinez, Benjamin and Yin, Hang and Lingelbach, Michael and Hwang, Minjune and Hiranaka, Ayano and Garlanka, Sujay and Aydin, Arman and Lee, Sharon and Sun, Jiankai and Anvari, Mona and Sharma, Manasi and Bansal, Dhruva and Hunter, Samuel and Kim, Kyu-Young and Lou, Alan and Matthews, Caleb R. and Villa-Renteria, Ivan and Tang, Jerry Huayang and Tang, Claire and Xia, Fei and Li, Yunzhu and Savarese, Silvio and Gweon, Hyowon and Liu, C. Karen and Wu, Jiajun and Fei-Fei, Li},
  journal={arXiv preprint arXiv:2403.09227},
  year={2024}
}
```

## :material-handshake: **Sponsors**

High-quality simulation data provided by Simovation. 

We gratefully acknowledge the support of our sponsors who make this challenge possible:

<div style="display: flex; gap: 2rem; justify-content: center; align-items: center; margin: 1rem 0;">
  <a href="https://www.linkedin.com/company/simovationinc/" title="Simovation" style="display: flex; align-items: center; justify-content: center; width: 200px; height: 100px;">
    <img src="../assets/challenge_2025/simovation_logo.png" alt="Simovation" style="max-height: 100%; max-width: 100%; width: auto; height: auto; object-fit: contain;" />
  </a>
  <a href="https://www.imda.gov.sg/" title="IMDA" style="display: flex; align-items: center; justify-content: center; width: 200px; height: 100px;">
    <img src="../assets/challenge_2025/imda_logo.png" alt="IMDA" style="max-height: 100%; max-width: 100%; width: auto; height: auto; object-fit: contain;" />
  </a>
  <a href="https://hai.stanford.edu/" title="Stanford HAI" style="display: flex; align-items: center; justify-content: center; width: 200px; height: 100px;">
    <img src="../assets/challenge_2025/hai_logo.png" alt="Stanford HAI" style="max-height: 100%; max-width: 100%; width: auto; height: auto; object-fit: contain;" />
  </a>
  <a href="https://tsffoundation.org/" title="Schmidt Family Foundation" style="display: flex; align-items: center; justify-content: center; width: 200px; height: 100px;">
    <img src="../assets/challenge_2025/schmidt_family_foundation_logo.png" alt="Schmidt Family Foundation" style="max-height: 100%; max-width: 100%; width: auto; height: auto; object-fit: contain;" />
  </a>
  <a href="https://www.nvidia.com/" title="NVIDIA" style="display: flex; align-items: center; justify-content: center; width: 200px; height: 100px;">
    <img src="../assets/challenge_2025/nvidia_logo.png" alt="NVIDIA" style="max-height: 100%; max-width: 100%; width: auto; height: auto; object-fit: contain;" />
  </a>
</div>
