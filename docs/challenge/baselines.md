# Baselines

For the 2026 BEHAVIOR Challenge, we provide starter training and evaluation pipelines for two baseline methods:

- **[π0.5](https://www.physicalintelligence.company/blog/pi05)** — fine-tuned with a challenge fork of [OpenPI](https://github.com/wensi-ai/openpi)
- **[GR00T N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B)** — fine-tuned with a challenge fork of [Isaac-GR00T](https://github.com/wensi-ai/Isaac-GR00T)

These baselines are meant to help participants verify the full challenge workflow — loading the demonstration dataset, fine-tuning a policy, running evaluation in OmniGibson, and preparing outputs for submission — and serve as reference implementations for the expected observation and action interfaces of the challenge track. Participants are encouraged to build on these pipelines, compare against them, and open-source improvements when possible.

Both walkthroughs share the same first steps ([Common setup](#common-setup)) and the same last step ([Evaluation](#evaluation)); only the training and serving steps differ between the two methods.

## Provided checkpoints

We provide fine-tuned checkpoints so participants can run evaluation without training:

| Task               | π0.5                                     | GR00T N1.7                               |
| ------------------ | ---------------------------------------- | ---------------------------------------- |
| `turning_on_radio` | [checkpoint](https://drive.google.com/file/d/1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2/view?usp=sharing) | [checkpoint](https://drive.google.com/file/d/1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789/view?usp=sharing) |

!!! tip "Evaluation only?"
    Complete the [Common setup](#common-setup), then jump straight to the corresponding *Serve the policy* step and [Evaluation](#evaluation).

## Common setup

Both baselines use [uv](https://docs.astral.sh/uv/) to manage Python dependencies for training, and a conda environment for the OmniGibson evaluator. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up.

### Placeholders

The commands on this page use a few shell variables. Set them once per terminal, replacing the example values with local paths and the task to train or evaluate:

```bash
export PATH_TO_BEHAVIOR_1K=~/BEHAVIOR-1K       # path to the BEHAVIOR-1K checkout
export TASK_NAME=turning_on_radio              # any challenge task
export DATA_ROOT=~/2026-challenge-demos/b1k    # holds one demo folder per task
export DATASET_PATH=$DATA_ROOT/$TASK_NAME      # the LeRobot dataset for $TASK_NAME
```

See the [dataset page](./dataset.md) for the full task list.

### Install BEHAVIOR-1K

Clone the challenge release of BEHAVIOR-1K and run the setup script. It creates a `behavior` conda environment with OmniGibson, the simulation assets, and the evaluation dependencies:

```bash
git clone -b v3.9.0 https://github.com/StanfordVL/BEHAVIOR-1K.git $PATH_TO_BEHAVIOR_1K
cd $PATH_TO_BEHAVIOR_1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
```

See the [installation guide](../getting_started/installation.md) for prerequisites and troubleshooting.

### Download the demonstrations

The demos ship as a [LeRobot](https://github.com/huggingface/lerobot) v3.0 dataset on HuggingFace: [behavior-1k/2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos). Download only the relevant task folder — the full dataset is 3.27 TB:

```bash
huggingface-cli download behavior-1k/2026-challenge-demos \
    --repo-type dataset \
    --local-dir "$(dirname $DATA_ROOT)" \
    --include "b1k/$TASK_NAME/**"
```

This places the demos at `$DATASET_PATH`. Drop the `--include` filter to download all 100 tasks. See the [dataset page](./dataset.md) for details on the data format.

## π0.5

This walkthrough fine-tunes [π0.5](https://www.physicalintelligence.company/blog/pi05) on the challenge dataset using a fork of OpenPI.

### Setup

Clone the `behavior` branch of the OpenPI fork and set up its environment:

```bash
export OPENPI_DIR=~/openpi     # where to put the OpenPI checkout

git clone -b behavior https://github.com/wensi-ai/openpi.git $OPENPI_DIR
cd $OPENPI_DIR
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Prepare the dataset

Every command below passes the dataset location explicitly via `--data.repo_id=$TASK_NAME --data.base_config.dataset_root=$DATASET_PATH`. Alternatively, set `repo_id` and `dataset_root` once in the `pi05_b1k` block of `src/openpi/training/config.py` and drop those flags.

Before training, compute normalization statistics for the dataset:

```bash
cd $OPENPI_DIR
uv run scripts/compute_norm_stats.py pi05_b1k \
    --data.repo_id=$TASK_NAME \
    --data.base_config.dataset_root=$DATASET_PATH
```

This writes `norm_stats.json` under `outputs/assets/pi05_b1k/$TASK_NAME`.

### Train

The `pi05_b1k` config fine-tunes the π0.5 base model (`gs://openpi-assets/checkpoints/pi05_base/params`) on the R1Pro robot with a 32-step action horizon. Robot and task definitions live in `src/openpi/configs/robots/b1k.py` and `src/openpi/configs/tasks/b1k.py`.

Pick a unique experiment name `$EXP_NAME` for each run and start training from `$OPENPI_DIR`:

=== "Single GPU"

```
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/b1k/train_b1k.py pi05_b1k \
    --exp_name=$EXP_NAME \
    --overwrite \
    --batch_size=64 \
    --data.repo_id=$TASK_NAME \
    --data.base_config.dataset_root=$DATASET_PATH
```
```

=== "Multi-GPU (single node)"

```
Pass the GPU count and device list as positional arguments (e.g. `... pi05_b1k 8 0,1,2,3,4,5,6,7`):

```bash
./scripts/b1k/train_b1k.sh pi05_b1k $NUM_GPUS $CUDA_VISIBLE_DEVICES \
    --data.repo_id=$TASK_NAME \
    --data.base_config.dataset_root=$DATASET_PATH
```
```

=== "SLURM"

```
Use `scripts/b1k/train_b1k.sbatch.sh`. Before submitting, edit the `#SBATCH` directives at the top of the script (account, partition, GPU type/count, memory, time limit) and update the virtual-environment activation line to point to the OpenPI install.

Submit a new run:

```bash
EXP_NAME=my_run sbatch scripts/b1k/train_b1k.sbatch.sh pi05_b1k \
    --overwrite \
    --data.repo_id=$TASK_NAME \
    --data.base_config.dataset_root=$DATASET_PATH
```

Resume an existing run by omitting `--overwrite` (by default the script resumes from `outputs/checkpoints/pi05_b1k/$EXP_NAME/`):

```bash
EXP_NAME=my_run sbatch scripts/b1k/train_b1k.sbatch.sh pi05_b1k \
    --data.repo_id=$TASK_NAME \
    --data.base_config.dataset_root=$DATASET_PATH
```

Any extra arguments after the config name are passed through to `train_b1k.py`. Job logs are written to the path set by `#SBATCH --output`.
```

Checkpoints are saved under `outputs/checkpoints/pi05_b1k/$EXP_NAME/`.

### Serve the policy

Set `$PATH_TO_CKPT` to a checkpoint step directory (for example `outputs/checkpoints/pi05_b1k/$EXP_NAME/<step>`, or the provided checkpoint from the [table above](#provided-checkpoints)) and start the policy server:

```bash
cd $OPENPI_DIR
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro \
    --task b1k/$TASK_NAME \
    --repo-id $TASK_NAME \
    --policy.config pi05_b1k \
    --policy.dir $PATH_TO_CKPT \
    --control_mode receding_horizon \
    --action_horizon 16 \
    --port 8000
```

This starts a websocket policy server on port 8000. Leave it running and continue to [Evaluation](#evaluation).

## GR00T N1.7

This walkthrough fine-tunes [GR00T N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B) on the challenge dataset using a fork of Isaac-GR00T.

### Setup

Clone the Isaac-GR00T fork and set up its environment:

```bash
export GROOT_DIR=~/Isaac-GR00T     # where to put the Isaac-GR00T checkout

git clone https://github.com/wensi-ai/Isaac-GR00T $GROOT_DIR
cd $GROOT_DIR
uv sync --frozen --python 3.10
uv pip install --python .venv/bin/python websockets
source .venv/bin/activate
```

!!! warning "Gated backbone"
    The N1.7 backbone `[nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)` is gated. Accept the gate before training.

### Prepare the dataset

Deploy the GR00T modality configuration (`meta/modality.json`) into every task dataset under `$DATA_ROOT`:

```bash
cd $GROOT_DIR
python scripts/b1k/deploy_modality.py $DATA_ROOT
```

Normalization statistics (`meta/stats.json`) are generated automatically on the first training run.

??? note "Optional: convert the dataset to LeRobot v2.1"
    The GR00T loader reads both the released v3.0 demos and v2.1 natively — no conversion is needed to train. Convert only if downstream tooling requires v2.1.

```
The conversion runs **in place**: `$DATA_ROOT/$TASK_NAME` becomes v2.1 and the original v3.0 copy is backed up to `$DATA_ROOT/${TASK_NAME}_v3.0`.

```bash
cd $GROOT_DIR/scripts/lerobot_conversion
uv venv --python 3.11 .venv && source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475" \
  huggingface_hub jsonlines numpy pyarrow tqdm
python convert_v3_to_v2.py --root $DATA_ROOT --repo-id $TASK_NAME
cd $GROOT_DIR
source .venv/bin/activate      # re-activate the GR00T venv (conversion used its own)
```

Re-run `deploy_modality.py` afterwards — the conversion does not carry `modality.json` over.
```

??? tip "Optional: pre-cache the base models"
    Training downloads both models automatically on the first run; pre-cache them to fail fast on access or network issues:

```
```bash
export HF_TOKEN=hf_xxx         # the account that accepted the Cosmos-Reason2-2B gate
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN")
snapshot_download("nvidia/GR00T-N1.7-3B", token=tok)
snapshot_download("nvidia/Cosmos-Reason2-2B", token=tok)  # gated backbone
PY
```
```

### Train

`scripts/b1k/train_b1k.py` fine-tunes the GR00T N1.7 base model (`nvidia/GR00T-N1.7-3B`) on the R1Pro robot with a 16-step action horizon, tuning the projector and diffusion action head while keeping the vision encoder and LLM frozen. The R1Pro embodiment — cameras, proprioception, and action groups — is defined in `examples/b1k/r1pro.py` and registered under the `NEW_EMBODIMENT` tag.

Run the following from `$GROOT_DIR` to fine-tune on 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=online OMP_NUM_THREADS=4 \
torchrun --nproc_per_node=8 --master_port=29500 scripts/b1k/train_b1k.py \
    --experiment-name b1k-$TASK_NAME \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path $DATASET_PATH \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/b1k/r1pro.py \
    --num-gpus 8 \
    --global-batch-size 2048 \
    --output-dir outputs \
    --save-steps 1500 --save-total-limit 5 --max-steps 150000 \
    --dataloader-num-workers 8 --decode-only-used-frames
```

Checkpoints land in `outputs/b1k-$TASK_NAME/checkpoint-<step>/`, each one standalone and directly servable. Metrics are logged to the `B1K` Weights & Biases project (`WANDB_MODE=offline` to log locally). Reruns start fresh by default; pass `--resume-from-checkpoint` to continue from the latest checkpoint.

!!! tip
    Match `OMP_NUM_THREADS` and `--dataloader-num-workers` to the CPU, and `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, and `--num-gpus` to the GPU count. `--decode-only-used-frames` skips decoding unused video frames to speed up data loading.

### Serve the policy

Set `$PATH_TO_CKPT` to a checkpoint directory (for example `outputs/b1k-$TASK_NAME/checkpoint-<step>`, or the provided checkpoint from the [table above](#provided-checkpoints)) and start the policy server:

```bash
cd $GROOT_DIR
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/b1k/serve_b1k.py \
    --model-path $PATH_TO_CKPT \
    --modality-config-path examples/b1k/r1pro.py \
    --embodiment-tag NEW_EMBODIMENT \
    --host 127.0.0.1 --port 8000
```

This starts a websocket policy server on port 8000, applying temporal ensembling by default; health-check it with `curl -s http://127.0.0.1:8000/healthz`. Leave it running and continue to [Evaluation](#evaluation).

## Evaluation

Evaluation runs as two processes: the policy server started in *Serve the policy*, and the OmniGibson evaluator from BEHAVIOR-1K. With the server running, open a second terminal and run the evaluator inside the `behavior` conda environment created during [Common setup](#common-setup):

```bash
conda activate behavior
cd $PATH_TO_BEHAVIOR_1K
export LOG_PATH=./eval_logs/$TASK_NAME     # where metrics and videos are written

python -m omnigibson.eval.eval \
    --task-name $TASK_NAME \
    --host 127.0.0.1 \
    --port 8000 \
    --output-dir $LOG_PATH \
    --write-video
```

The evaluator writes per-rollout metrics and videos to `$LOG_PATH`. For the full evaluator flag reference, observation wrappers, and custom robot configurations, see [Evaluation and Rules](./evaluation.md); for preparing a submission, see the [Submission Guidelines](./submission.md).