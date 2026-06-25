# Dataset

## Data Format

For the 2026 challenge, we provide the following datasets hosted on HuggingFace:

**Raw HDF5 replay data.** [2026-challenge-rawdata](https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata) contains the original raw HDF5 data for the 20k teleoperation demos. These files contain everything needed to replay exact trajectories in OmniGibson. Use them with `OmniGibson/scripts/learning/replay_obs.py` to replay trajectories and collect additional visual observations.

**LeRobot demo dataset.** [2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos) contains 20,000 human-collected teleoperation demos across 100 tasks. It follows the [LeRobot](https://github.com/huggingface/lerobot) V3 format with customizations for better data handling.

The demo dataset has the following structure:

<table class="challenge-data-table">
  <thead>
    <tr>
      <th>Folder</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>annotations</code></td>
      <td>Language annotations for each episode.</td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>Low-dimensional data, including proprioceptions, actions, camera poses, and related episode data.</td>
    </tr>
    <tr>
      <td><code>meta</code></td>
      <td>Metadata folder containing episode-level information.</td>
    </tr>
    <tr>
      <td><code>videos</code></td>
      <td>Visual observations, including RGB and depth.</td>
    </tr>
  </tbody>
</table>

The dataset includes 2 visual modalities: RGB (`rgb`) and Depth (`depth_linear`):

<div class="challenge-modality-grid">
  <section>
    <div>
      <h3>RGB</h3>
      <p>RGB image of the scene from the camera perspective.</p>
      <dl>
        <dt>Shape</dt>
        <dd><code>(height, width, 4)</code></dd>
        <dt>Type</dt>
        <dd><code>numpy.uint8</code></dd>
        <dt>Resolution</dt>
        <dd>720 x 720 head camera; 480 x 480 wrist cameras</dd>
        <dt>Range</dt>
        <dd>[0, 255]</dd>
      </dl>
    </div>
    <img src="../assets/challenge_2025/dataset_rgb.png" alt="RGB observation example">
  </section>
  <section>
    <div>
      <h3>Depth Linear</h3>
      <p>Distance between the camera and scene geometry, with measurement linearly proportional to actual distance.</p>
      <dl>
        <dt>Shape</dt>
        <dd><code>(height, width)</code></dd>
        <dt>Type</dt>
        <dd><code>numpy.float32</code></dd>
        <dt>Encoding</dt>
        <dd>Depth videos are log-quantized during replay and dequantized back to metric depth values.</dd>
        <dt>Range</dt>
        <dd>[0, 10] meters</dd>
      </dl>
    </div>
    <img src="../assets/challenge_2025/dataset_depth.png" alt="Depth observation example">
  </section>
</div>


## Dataset Statistics

<table class="challenge-data-table challenge-data-table--stats">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Total Trajectories</td>
      <td>20,000</td>
    </tr>
    <tr>
      <td>Total Tasks</td>
      <td>100</td>
    </tr>
    <tr>
      <td>Total Skills</td>
      <td>270,600</td>
    </tr>
    <tr>
      <td>Unique Skills</td>
      <td>31</td>
    </tr>
    <tr>
      <td>Avg. Skills per Trajectory</td>
      <td>27.06</td>
    </tr>
    <tr>
      <td>Avg. Trajectory Duration</td>
      <td>351.54 seconds / 5.9 minutes</td>
    </tr>
  </tbody>
</table>

<details>
<summary><b>Show unique skills breakdown</b></summary>

<ul>
<li>attach</li>
<li>chop</li>
<li>close door</li>
<li>close drawer</li>
<li>close lid</li>
<li>hand over</li>
<li>hang</li>
<li>hold</li>
<li>ignite</li>
<li>insert</li>
<li>move to</li>
<li>open door</li>
<li>open drawer</li>
<li>open lid</li>
<li>pick up from</li>
<li>place in</li>
<li>place in next to</li>
<li>place on</li>
<li>place on next to</li>
<li>place under</li>
<li>pour</li>
<li>press</li>
<li>push to</li>
<li>release</li>
<li>spray</li>
<li>sweep surface</li>
<li>tip over</li>
<li>turn off switch</li>
<li>turn on switch</li>
<li>turn to</li>
<li>wipe hard</li>
</ul>

</details>

<p class="challenge-plot-title">Overall Demo Duration</p>

![Overall Demo Duration](../assets/challenge_2026/overall_demo_duration.png)

<p class="challenge-plot-title">Per Task Demo Duration</p>

![Per Task Demo Duration](../assets/challenge_2026/per_task_demo_duration.png)
