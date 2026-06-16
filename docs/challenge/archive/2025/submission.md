# Submission

- **Submission Portal**: [Google Form](https://forms.gle/54tVqi5zs3ANGutn7)

- **Leaderboard**: [here](./leaderboard.md)


## **Submission details**

After running the [evaluation script](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/learning/eval.py) (see [evaluation](./evaluation.md) for more details), there will be two output files: an json file containing the metric results, and a mp4 video recording of the rollout trajectory. Here is a sample output json file for one episode of evaluation:

```
{
    "agent_distance": {
        "base": 9.703554042062024e-06, 
        "left": 0.019627160858362913, 
        "right": 0.015415858360938728
    }, 
    "normalized_agent_distance": {
        "base": 4.93031697036899e-06, 
        "left": 0.006022007241065448, 
        "right": 0.0037894888066205374
    }, 
    "q_score": {
        "final": 0.0
    }, 
    "time": {
        "simulator_steps": 6, 
        "simulator_time": 0.2, 
        "normalized_time": 0.002791165032284476
    }
}
```

- Submit your results and models at [Google Form](https://forms.gle/54tVqi5zs3ANGutn7). No formal registration is required to participate in the challenge. We encourage you to submit intermediate results and models to be showcased on our leaderboard. The same model with different checkpoints from the same team will be considered as a single entry.

- **Partial submission is allowed**: Since each tasks will be evaluated on 10 instances and 1 rollout each, there should be 500 json files after the full evaluation. However, you are allowed to evaluate your policy on a subset of the tasks (or instances). Any rollout instances not submitted will be counted as zero when calculating the final score of the submission. 


## **Final model submission and evaluation**

There are two ways to submit your model for evaluation:

1. (**Recommended**) Docker-based evaluation
    
    We have provided a sample docker image [here](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/docker/submission.Dockerfile), which will start up a dummy local policy that always outputs zero action. Below is a tutorial on how to test the evaluation pipeline with the provided dockerfile:
    
    1. Start up an evaluation instance in another terminal: 
    ```
    python OmniGibson/omnigibson/learning/eval.py log_path=$LOG_PATH policy=websocket task.name=turning_on_radio
    ```
    2. build the dockerfile: `docker build -f OmniGibson/docker/submission.Dockerfile -t b1k-challenge-example .`
    3. run the docker container: `docker run -p 8000:8000 b1k-challenge-example`

    **NOTE: While this Docker image contains a copy of OmniGibson for use in your policy code as a utility library if you so desire, the OmniGibson simulation should NOT be launched inside the container.** Isaac Sim is not installed in the container and as such cannot be used to run simulation. For evaluation, we will run OmniGibson **outside** the container and connect to your policy inside the container using the WebSocket policy client. You should perform your testing under this setup, too.

    We will use this similar pipeline for our evaluation, except for the second step the submitted docker image will be pulled. 
    
    The model should run on a single 24GB VRAM GPU. We will use the following GPUs to perform the final evaluation: RTX 3090, A5000, TitanRTX

2. IP address-based evaluation: You can serve your models and provide us with corresponding IP addresses that allow us to query your models for evaluation. We recommend common model serving libraries, such as [TorchServe](https://docs.pytorch.org/serve/), [LitServe](https://lightning.ai/docs/litserve/home), [vLLM](https://docs.vllm.ai/en/latest/index.html), [NVIDIA Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html), etc.


**YOU ARE NOT ALLOWED TO MODIFY THE OUTPUT JSON FILES AND VIDEOS IN ANY WAY**. Your final submission will be a zip file containing the following:

1. All the json files, one for each rollout you performed (up to 500);
3. Wrapper code (.py) used during evaluation;
4. Robot (R1Pro) config file (.yaml) used during evaluation; 
5. A readme file (.md) that specifies the details to perform evaluation with your policy:
    - For docker image-based submission, please include the link to your docker image, as well as the image digest hash
    - For IP address-based evaluation, please provide the corresponding IP address that you used to serve the policy.
    - Please also include any other information you deem necessary to help us evaluation your policy


In addition, we require you to submit a link to all the mp4 videos, one for each rollout you performed (up to 500). See our submission [Google Form](https://forms.gle/54tVqi5zs3ANGutn7) for more details. 
