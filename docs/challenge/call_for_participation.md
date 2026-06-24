# **Call for Participation: 2026 BEHAVIOR Challenge**

We invite researchers, students, and robotics practitioners to participate in the **2nd BEHAVIOR Challenge**. Teams will build agents for **100 full-length household tasks** in the realistic BEHAVIOR-1K environment, using robot onboard observations: **RGB + depth + proprioception**.

BEHAVIOR-1K is a large-scale embodied AI benchmark for everyday household activities. The challenge tests whether embodied agents can combine high-level reasoning, long-horizon navigation, and dexterous bimanual manipulation in interactive house-scale scenes.

<iframe width="560" height="315" src="https://www.youtube.com/embed/iSFpinMiT0s?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Challenge Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## **Important Dates**

| Milestone | Date |
| --- | --- |
| Challenge launch | June 30, 2026 |
| Submission deadline | October 16, 2026 |
| Winners announcement | November 4, 2026 |

Event details, including venue and presentation format, will be announced as they are finalized. The [leaderboard](https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard) will track public submissions during the challenge.

## **Challenge at a Glance**

| Topic | 2026 Challenge |
| --- | --- |
| Tasks | 100 full-length household tasks |
| Environments | 7 scenes, including 4 new scenes |
| Evaluation track | One track using RGB + depth + proprioception |
| Demonstrations | 20,000 human teleoperation demos |
| Baselines | π0.5 (pi0.5) and GR00T N1.7 |
| Ranking metric | Average task success score with BDDL partial credit |
| Prizes | To be announced, with special prizes for outstanding open-source solutions |

Detailed specifications live on the canonical challenge pages: [Dataset](./dataset.md), [Baselines](./baselines.md), [Evaluation and Rules](./evaluation.md), and [Submission Guidelines](./submission.md).

## **Demonstration Data**

The challenge provides large-scale human teleoperation demonstrations for learning long-horizon household behaviors. The release includes RGB and depth observations, robot proprioception and actions, and skill/subtask annotations; the full dataset format and statistics are documented on the [Dataset](./dataset.md) page.

<iframe width="560" height="315" src="https://www.youtube.com/embed/oVr3IYnQiys?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Annotation Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Demonstrations were collected with **JoyLo**, a whole-body teleoperation interface for controlling the robot base, torso, arms, and grippers. We thank [Simovation](https://www.linkedin.com/company/simovationinc/) for providing high-quality JoyLo teleoperation data in simulation.

<iframe width="560" height="315" src="https://www.youtube.com/embed/fFAtUzEETe4?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Data Quality Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## **Why Participate**

BEHAVIOR tasks go beyond short pick-and-place or navigation benchmarks. Agents must search across rooms, manipulate many objects, handle object state changes, and satisfy symbolic BDDL goal conditions after several minutes of autonomous execution.

The 2026 challenge is intended as a shared benchmark for testing robot foundation models, imitation learning, reinforcement learning, task and motion planning, memory systems, SLAM, and LLM-assisted policies under the same realistic evaluation protocol.

<iframe width="560" height="315" src="https://www.youtube.com/embed/3XKhbg9_MS4?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Long-Horizon Task Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

The tasks also exercise diverse object state changes and low-level skills, including opening, closing, pouring, wiping, spraying, attaching, toggling, cooking, and slicing.

<iframe width="560" height="315" src="https://www.youtube.com/embed/FeD8_KgVOag?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Skills Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## **How to Participate**

1. Join the [Discord community](https://discord.gg/bccR5vGFEx) for announcements and participant discussion.
2. Attend office hours every Monday, 5-6pm PST, over [Zoom](https://stanford.zoom.us/j/92909660940?pwd=RgFrdC8XeB3nVxABqb1gxrK96BCRBa.1).
3. Review the [Dataset](./dataset.md), [Baselines](./baselines.md), and [Evaluation and Rules](./evaluation.md) pages.
4. Develop and evaluate your method using the official challenge-track observation restrictions.
5. Prepare your submission using the [Submission Guidelines](./submission.md). The submission portal will be announced.

We look forward to seeing what the community builds.
