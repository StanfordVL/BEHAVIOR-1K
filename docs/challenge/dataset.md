# Dataset

## Dataset Access

We host our dataset on Hugging Face:

**Dataset URL**: [https://huggingface.co/datasets/behavior-1k/2025-challenge-demos](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos)

**Rawdata URL**: [https://huggingface.co/datasets/behavior-1k/2025-challenge-rawdata](https://huggingface.co/datasets/behavior-1k/2025-challenge-rawdata)

## Data Format

Our demonstration data is provided in **LeRobot format**, a widely-adopted format for robot learning datasets. LeRobot provides a unified interface for robot demonstration data, making it easy to load, process, and use the data for training policies.

To learn more about the LeRobot format, visit the official [LeRobot repository](https://github.com/huggingface/lerobot). We also provide tutorial notebooks about [loading the dataset](https://github.com/StanfordVL/b1k-baselines/blob/main/tutorials/dataset.ipynb) and [generating custom data](https://github.com/StanfordVL/b1k-baselines/blob/main/tutorials/generate_custom_data.ipynb)


## Dataset Statistics

| Metric | Value |
| ------ | ----- |
| Total Trajectories | 10,000 |
| Total Tasks | 50 |
| Total Skills | 270,600 |
| Unique Skills | 31 |
| Avg. Skills per Trajectory | 27.06 |
| Avg. Trajectory Duration | 397.04 seconds / 6.6 minutes |

### Overall Demo Duration

![Overall Demo Duration](../assets/challenge_2025/overall_demo_duration.png)

### Per Task Demo Duration

![Per Task Demo Duration](../assets/challenge_2025/per_task_demo_duration.png)

Note: Language annotations will be released gradually and will be available soon. Currently, 15 tasks are fully annotated across all demonstrations, with the remaining annotations in the final stages of QA before release. 
