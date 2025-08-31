# 🏆 **2025 BEHAVIOR Challenge**

**Join us and solve 50 full-length household tasks in the realistic BEHAVIOR-1K environment, with 10,000 teleoperated expert demonstrations (1200+ hours) available!** 🤖

---

## :material-video: **Challenge Walkthrough**

[Challenge Video Placeholder - Main walkthrough video for participants]

## :material-graph-outline: **Overview**

**BEHAVIOR** is a robotics challenge for everyday household tasks. It's a large-scale, human-grounded benchmark that tests a robot's capability in high-level reasoning, long-range locomotion, and dexterous bimanual manipulation in house-scale scenes.

This year's challenge features:

- **50 full-length household tasks** from our 1,000 activity collection, covering diverse activities like rearrangement, cooking, cleaning, and installation
- **10,000 teleoperated demonstrations** (1200+ hours) for training

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
- **Pre-trained Visuo-Language Action models**: OpenVLA and π0.  These models are pretrained by a large amount of demonstration data, giving an alternative to models that need to be trained from scratch.

[Baseline details →](./baselines.md)

## :material-chart-box: **Evaluation & Rules**

### Challenge Tracks

**Standard track:** Limited to provided robot onboard observations (RGB + depth + instance segmentation + proprioception).

**Privileged information track:** May query simulator for any information (object poses, scene point clouds, etc.).

🏆 **Prizes per track:** 🥇 $1,000 | 🥈 $500 | 🥉 $300

Top 3 teams from each track will be invited to present at the workshop!

[Full challenge rules →](./rules.md)

### Evaluation Metrics

**Primary metric (for ranking):** Task success rate averaged across 50 tasks. Partial credit given as fraction of satisfied BDDL goal predicates.

**Secondary metrics (efficiency):**

- **Simulated time** - Total simulation steps × time per step
- **Distance navigated** - Total base movement distance
- **Hand displacement** - Cumulative hand movement

[Evaluation details →](./evaluation.md)


## :octicons-person-add-16: **Participating**

### Resources

- [Discord community](https://discord.gg/bccR5vGFEx)
- [Challenge office hours](#) (TBD)

### Important Dates

- **Challenge Launch**: September 2, 2025
- **Submission Deadline**: November 15, 2025
- **Winners Announcement**: December 6-7, 2025 @ NeurIPS conference in San Diego

To encourage participation, we’ve secured prizes for the winners: $1,000 for 1st place, $500 for 2nd, and $300 for 3rd place (along with, of course, the opportunity to publish your methods in the proceedings!). The real reward is contributing to the advance of embodied AI - and perhaps claiming the title of the robot that best “behaves” in a virtual home.

## :material-handshake: **Sponsors**

We gratefully acknowledge the support of our sponsors who make this challenge possible:

<div style="display: flex; gap: 2rem; justify-content: center; align-items: center; margin: 1rem 0;">
  <a href="https://lightwheel.ai/home" title="Lightwheel" style="display: flex; align-items: center; justify-content: center; width: 200px; height: 100px;">
    <img src="../assets/challenge_2025/lightwheel_logo.png" alt="Lightwheel" style="max-height: 100%; max-width: 100%; width: auto; height: auto; object-fit: contain;" />
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
</div>