# 🏆 **2026 BEHAVIOR Challenge**

Join us for the second year of the BEHAVIOR Challenge: solve **100 full-length household tasks** in the realistic BEHAVIOR-1K environment. BEHAVIOR tests whether embodied agents can combine high-level reasoning, long-horizon navigation, and dexterous bimanual manipulation in house-scale scenes.

<div class="challenge-teaser-player" data-youtube-src="https://www.youtube.com/embed/ihihRCf5NI4?autoplay=1&mute=1&playsinline=1&enablejsapi=1&modestbranding=1&showinfo=0&rel=0&controls=1">
  <button class="challenge-teaser-button" type="button" aria-label="Play BEHAVIOR Challenge teaser video">
    <img src="../assets/challenge_teaser_frame_240.png" alt="BEHAVIOR Challenge teaser">
    <span class="challenge-teaser-play" aria-hidden="true"></span>
  </button>
</div>

<script>
document.querySelectorAll(".challenge-teaser-player").forEach((player) => {
  const button = player.querySelector(".challenge-teaser-button");
  if (!button) {
    return;
  }

  button.addEventListener("click", () => {
    const iframe = document.createElement("iframe");
    const videoUrl = new URL(player.dataset.youtubeSrc);
    videoUrl.searchParams.set("origin", window.location.origin);
    videoUrl.searchParams.set("widget_referrer", window.location.href);
    iframe.src = videoUrl.toString();
    iframe.title = "BEHAVIOR Challenge Video";
    iframe.setAttribute("frameborder", "0");
    iframe.setAttribute("allow", "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share");
    iframe.setAttribute("allowfullscreen", "");
    iframe.setAttribute("referrerpolicy", "strict-origin-when-cross-origin");
    player.replaceChildren(iframe);
  }, { once: true });
});
</script>

<div class="challenge-action-grid">
  <section class="challenge-action-card challenge-action-card--deadlines">
    <h2>Important Dates</h2>
    <ul class="challenge-action-list">
      <li><strong>Challenge Launch:</strong> <span>07/02/2026</span></li>
      <li><strong>Submission Deadline:</strong> <span>10/16/2026</span></li>
      <li><strong>Winners Announcement:</strong> <span>11/04/2026</span></li>
    </ul>
    <div class="challenge-action-links">
      <a href="https://behavior-1k-2026-challenge-leaderboard.hf.space/submit">Submission Portal</a>
      <a href="./submission.html">Submission Guidelines</a>
    </div>
  </section>

  <section class="challenge-action-card challenge-action-card--venue">
    <h2>Event Details</h2>
    <ul class="challenge-action-list">
      <li><strong>Event:</strong> <span>To be announced</span></li>
      <li><strong>Time:</strong> <span>To be announced</span></li>
      <li><strong>Location:</strong> <span>To be announced</span></li>
    </ul>
    <div class="challenge-action-links">
      <a href="https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard">Leaderboard</a>
      <a href="./evaluation.html">Evaluation & Rules</a>
    </div>
  </section>
</div>

## :material-trophy: **Prize Pool**

<div class="challenge-prize-panel">
  <div class="challenge-prize-total">
    <span><b class="challenge-prize-icon">🏆</b> Total Prize Pool</span>
    <strong>$11,000</strong>
  </div>
  <div class="challenge-prize-grid">
    <div class="challenge-prize-card challenge-prize-card--first">
      <b class="challenge-prize-icon">🥇</b>
      <span>1st Place</span>
      <strong>$5,000</strong>
    </div>
    <div class="challenge-prize-card challenge-prize-card--second">
      <b class="challenge-prize-icon">🥈</b>
      <span>2nd Place</span>
      <strong>$3,000</strong>
    </div>
    <div class="challenge-prize-card challenge-prize-card--third">
      <b class="challenge-prize-icon">🥉</b>
      <span>3rd Place</span>
      <strong>$2,000</strong>
    </div>
    <div class="challenge-prize-card challenge-prize-card--open">
      <b class="challenge-prize-icon">🌐</b>
      <span>Outstanding Open Source</span>
      <strong>$1,000</strong>
    </div>
  </div>
</div>

## :material-view-dashboard: **Challenge at a Glance**

<table class="challenge-data-table">
  <tbody>
    <tr>
      <td>Tasks</td>
      <td>100 full-length household tasks</td>
    </tr>
    <tr>
      <td>Environments</td>
      <td>7 scenes, including 4 new scenes</td>
    </tr>
    <tr>
      <td>Evaluation track</td>
      <td>One track using RGB + depth + proprioception</td>
    </tr>
    <tr>
      <td>Demonstrations</td>
      <td>20,000 human teleoperation demos, 1,950 hours in total</td>
    </tr>
    <tr>
      <td>Baselines</td>
      <td>π0.5 (pi0.5) and GR00T N1.7</td>
    </tr>
    <tr>
      <td>Ranking metric</td>
      <td>Average task success score with BDDL partial credit</td>
    </tr>
  </tbody>
</table>

Detailed specifications live on the canonical challenge pages: [Dataset](./dataset.md), [Baselines](./baselines.md), [Evaluation and Rules](./evaluation.md), and [Submission Guidelines](./submission.md). Browse the full task list in the [Demo Gallery](./tasks/index.md).

## :material-database: **Demonstration Data**

The challenge provides large-scale human teleoperation demonstrations for learning long-horizon household behaviors. The release includes RGB and depth observations, robot proprioception and actions, and skill/subtask annotations; the full dataset format and statistics are documented on the [Dataset](./dataset.md) page.

Demonstrations were collected with **JoyLo**, a whole-body teleoperation interface for controlling the robot base, torso, arms, and grippers. We thank [Simovation](https://www.linkedin.com/company/simovationinc/) for providing high-quality JoyLo teleoperation data in simulation.

<div class="challenge-video-grid">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/oVr3IYnQiys?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Annotation Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <iframe width="560" height="315" src="https://www.youtube.com/embed/fFAtUzEETe4?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Data Quality Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

## :material-lightbulb-on-outline: **Why Participate**

BEHAVIOR tasks go beyond short pick-and-place or navigation benchmarks. Agents must search across rooms, manipulate many objects, handle object state changes, and satisfy symbolic BDDL goal conditions after several minutes of autonomous execution.

The 2026 challenge is intended as a shared benchmark for testing robot foundation models, imitation learning, reinforcement learning, task and motion planning, memory systems, SLAM, and LLM-assisted policies under the same realistic evaluation protocol.

The tasks also exercise diverse object state changes and low-level skills, including opening, closing, pouring, wiping, spraying, attaching, toggling, cooking, and slicing.

<div class="challenge-video-grid">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/3XKhbg9_MS4?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Long-Horizon Task Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <iframe width="560" height="315" src="https://www.youtube.com/embed/FeD8_KgVOag?modestbranding=1&showinfo=0&rel=0&controls=1" title="BEHAVIOR Skills Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

## :octicons-person-add-16: **Getting Started**

1. Join the [Discord community](https://discord.gg/bccR5vGFEx) for announcements and participant discussion.
2. Attend office hours every Monday, 5-6pm Pacific Time, over [Zoom](https://stanford.zoom.us/j/98056621630?pwd=G3JqHl6lWB0qnMAV3BZMJaWgoiXeqT.1).
3. Download the dataset and review the [dataset documentation](./dataset.md).
4. Start from the [π0.5 and GR00T N1.7 baseline pipelines](./baselines.md).
5. Run evaluation and prepare your submission using the [submission guidelines](./submission.md).

Whether you're a robotics veteran or just entering the field, we're here to support you.

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

<div style="display: flex; flex-wrap: nowrap; gap: 1.5rem; justify-content: center; align-items: center; margin: 1rem 0;">
  <a href="https://www.linkedin.com/company/simovationinc/" title="Simovation" style="display: flex; align-items: center; justify-content: center; width: 170px; height: 100px;">
    <img src="../assets/challenge_2025/simovation_logo.png" alt="Simovation" style="max-height: 100%; max-width: 100%; width: auto; height: auto; object-fit: contain;" />
  </a>
  <a href="https://www.imda.gov.sg/" title="IMDA" style="display: flex; align-items: center; justify-content: center; width: 170px; height: 100px;">
    <img src="../assets/challenge_2025/imda_logo.png" alt="IMDA" style="max-height: 100%; max-width: 100%; width: auto; height: auto; object-fit: contain;" />
  </a>
  <a href="https://hai.stanford.edu/" title="Stanford HAI" style="display: flex; align-items: center; justify-content: center; width: 170px; height: 100px;">
    <img src="../assets/challenge_2025/hai_logo.png" alt="Stanford HAI" style="max-height: 100%; max-width: 100%; width: auto; height: auto; object-fit: contain;" />
  </a>
  <a href="https://tsffoundation.org/" title="Schmidt Family Foundation" style="display: flex; align-items: center; justify-content: center; width: 170px; height: 100px;">
    <img src="../assets/challenge_2025/schmidt_family_foundation_logo.png" alt="Schmidt Family Foundation" style="max-height: 100%; max-width: 100%; width: auto; height: auto; object-fit: contain;" />
  </a>
</div>
