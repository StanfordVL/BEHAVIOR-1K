# 🏆 **2026 BEHAVIOR Challenge**

**Join us for the second year of the BEHAVIOR challenge and solve 100 full-length household tasks in the realistic BEHAVIOR-1K environment!** 🤖

- Event: *To be announced*
- Time: *To be announced*
- Location: *To be announced*

!!! warning "🚧 Page under construction"

    The 2026 BEHAVIOR Challenge is being prepared. Dates, prizes, sponsors, and dataset
    details below are placeholders (**TBD**) and will be finalized before launch. Looking for
    last year's challenge? See [Past Challenges](./archive/index.md).

---

## 📣 **Announcements**

See the [Announcements / Updates](./updates.md) page for the latest news.

---

## :material-graph-outline: **Overview**

**BEHAVIOR** is a robotics challenge for everyday household tasks. It's a large-scale, human-grounded benchmark that tests a robot's capability in high-level reasoning, long-range locomotion, and dexterous bimanual manipulation in house-scale scenes.

This year's challenge features:

- **100 full-length household tasks** from our 1,000 activity collection — the 50 tasks from the 2025 challenge **plus 50 new tasks** — covering diverse activities like rearrangement, cooking, cleaning, and installation
- **Four new scenes** including office, restaurant, and hotel environments

Browse the full task list in the [Demo Gallery](./tasks/).

---

## :material-database: **Dataset & Baselines**

### Teleoperated Demonstrations

Expert demonstrations collected via teleoperation, with:

- Synchronized RGBD observations
- Object and part-level segmentation
- Ground-truth object states
- Robot proprioception and actions
- Skill and subtask annotations

The dataset for the 50 new 2026 tasks is **TBD** and will be released before the challenge launch. See [Dataset details →](./dataset.md).

### Baseline Methods

Pre-implemented training & evaluation pipelines (carried over from 2025, updates **TBD**):

- **Behavioral Cloning baselines**: ACT, Diffusion Policy, BC-RNN, WB-VIMA
- **Pre-trained Visuo-Language Action models**: OpenVLA and π0

[Baselines details →](./baselines.md)

## :material-chart-box: **Evaluation & Rules**

The organizers reserve the right of final interpretation of the challenge rules.

### Challenge Tracks

**Standard track:** Limited to provided robot onboard observations (RGB + depth + instance segmentation + proprioception).

**Privileged information track:** May query simulator for any information (object poses, scene point clouds, etc.).

🏆 **Prizes per track:** *To be announced*

### Evaluation Metrics

**Primary metric (for ranking):** Task success rate averaged across all tasks. Partial credit given as fraction of satisfied BDDL goal predicates.

**Secondary metrics (efficiency):**

- **Simulated time** - Total simulation steps × time per step
- **Distance navigated** - Total base movement distance
- **Hand displacement** - Cumulative hand movement

[Evaluation details & Full challenge rules →](./evaluation.md)


## :octicons-person-add-16: **Participating**

### Resources

Join our community to ask questions and discuss the challenge:

- **Discord**: [Join our Discord Server](https://discord.gg/bccR5vGFEx)

Whether you're a robotics veteran or just entering the field, we're here to support you.

### Important Dates

- **Challenge Launch**: *To be announced*
- **Submission Deadline**: *To be announced*
- **Winners Announcement**: *To be announced*

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

*To be announced.*
