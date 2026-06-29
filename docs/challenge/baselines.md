# Baselines

For the 2026 BEHAVIOR Challenge, we provide starter training and evaluation pipelines for two baseline methods:

- **π0.5 (pi0.5)**
- **GR00T N1.7**

These baselines are meant to help participants verify the full workflow: loading the demonstration dataset, training or adapting a policy, running evaluation in OmniGibson, and preparing outputs for submission. They are also useful reference implementations for the expected observation and action interfaces in this year's single challenge track.

Participants are encouraged to build on these pipelines, compare against them, and open-source improvements when possible. Additional setup instructions, checkpoints, and runnable examples will be linked here as they are finalized for the 2026 release.

## π0.5 (π₀.₅)

This tutorial provides a minimal walkthrough for fine-tuning [π₀.₅](https://www.physicalintelligence.company/blog/pi05) on the 2026 BEHAVIOR-1K Challenge dataset and running evaluation in OmniGibson. 

We provide a Pi0.5 checkpoint for:

- turning_on_radio task [here](TODO: add checkpoint link).

If you would like to run eval only feel free to skip to the evaluation section.

Throughout this tutorial, replace the placeholders below with your local paths and the task you want to train or evaluate:

- `$OPENPI_DIR` — path to your OpenPi checkout
- `$PATH_TO_BEHAVIOR_1K` — path to your BEHAVIOR-1K checkout
- `$TASK_NAME` — BDDL task name (see the [dataset page](./dataset.md) for the full task list)
- `$DATASET_ROOT` — local path to the LeRobot dataset for that task (typically `$OPENPI_DIR/2026-challenge-demos/b1k/$TASK_NAME`)
- `$EXP_NAME` — experiment name for a training run
- `$PATH_TO_CKPT` — checkpoint step directory (for example, `outputs/checkpoints/pi05_b1k/$EXP_NAME/$STEP`)
- `$LOG_PATH` — directory where the evaluator writes metrics and videos

### Repo Clone

```bash
git clone https://github.com/wensi-ai/openpi.git -b behavior
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

### Installation

OpenPi uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the training environment:

```bash
cd $OPENPI_DIR
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Install BEHAVIOR-1K packages needed for evaluation (a separate `behavior` conda environment is recommended):

```bash
cd $PATH_TO_BEHAVIOR_1K
pip install -e bddl3
pip install -e OmniGibson[eval]
```

### Dataset

Download the [2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos) LeRobot dataset from HuggingFace. For a single task, you can download only the relevant folder:

```bash
huggingface-cli download behavior-1k/2026-challenge-demos \
    --repo-type dataset \
    --local-dir $OPENPI_DIR/2026-challenge-demos \
    --include "b1k/$TASK_NAME/**"
```

Update `dataset_root` and `repo_id` in the `pi05_b1k` block of `src/openpi/training/config.py`, or override them from the command line when computing stats or training:

```bash
--data.repo_id=$TASK_NAME \
--data.base_config.dataset_root=$DATASET_ROOT
```

### Finetune Pi0.5

The `pi05_b1k` config fine-tunes the π₀.₅ base model (`gs://openpi-assets/checkpoints/pi05_base/params`) on the R1Pro robot with a 32-step action horizon. Robot and task definitions live in `src/openpi/configs/robots/b1k.py` and `src/openpi/configs/tasks/b1k.py`.

Before training, compute normalization statistics for your dataset:

```bash
cd $OPENPI_DIR
uv run scripts/compute_norm_stats.py pi05_b1k \
    --data.repo_id=$TASK_NAME \
    --data.base_config.dataset_root=$DATASET_ROOT
```

This writes `norm_stats.json` under `outputs/assets/pi05_b1k/$TASK_NAME`.

Then start fine-tuning. On a single GPU:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/b1k/train_b1k.py pi05_b1k \
    --exp_name=$EXP_NAME \
    --overwrite \
    --batch_size=64 \
    --data.repo_id=$TASK_NAME \
    --data.base_config.dataset_root=$DATASET_ROOT
```

For multi-GPU training on a single node, use the helper script:

```bash
./scripts/b1k/train_b1k.sh pi05_b1k $NUM_GPUS $CUDA_VISIBLE_DEVICES \
    --data.repo_id=$TASK_NAME \
    --data.base_config.dataset_root=$DATASET_ROOT
```

For SLURM clusters, use `scripts/b1k/train_b1k.sbatch.sh`. Before submitting, edit the `#SBATCH` directives at the top of the script for your account, partition, GPU type/count, memory, and time limit, and update the virtual-environment activation line to point to your OpenPi install.

Submit a new training run from the OpenPi repo root (set `EXP_NAME` to a unique run name):

```bash
cd $OPENPI_DIR
EXP_NAME=my_run sbatch scripts/b1k/train_b1k.sbatch.sh pi05_b1k \
    --overwrite \
    --data.repo_id=$TASK_NAME \
    --data.base_config.dataset_root=$DATASET_ROOT
```

Resume an existing run:

```bash
EXP_NAME=my_run sbatch scripts/b1k/train_b1k.sbatch.sh pi05_b1k \
    --data.repo_id=$TASK_NAME \
    --data.base_config.dataset_root=$DATASET_ROOT
```

The sbatch script passes any extra arguments after the config name through to `train_b1k.py`. By default it resumes from `outputs/checkpoints/pi05_b1k/$EXP_NAME/`; pass `--overwrite` to start a fresh run with a new `$EXP_NAME`. Job logs are written to the path set by `#SBATCH --output` in the script.

Checkpoints are saved under `outputs/checkpoints/pi05_b1k/$EXP_NAME/`.

### Evaluation

Evaluation runs as two processes: a policy server (OpenPi) and the OmniGibson evaluator (BEHAVIOR-1K).

#### 1. Deploy the fine-tuned checkpoint

From the OpenPi repo:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro \
    --task b1k/$TASK_NAME \
    --repo-id $TASK_NAME \
    policy:checkpoint \
    --policy.config pi05_b1k \
    --policy.dir $PATH_TO_CKPT \
    --control_mode receding_horizon \
    --action_horizon 16 \
    --port 8000
```

This starts a websocket policy server on `0.0.0.0:8000`.

#### 2. Run evaluation in OmniGibson

In a separate terminal, activate the `behavior` conda environment and run the evaluator from the BEHAVIOR-1K repo:

```bash
conda activate behavior
cd $PATH_TO_BEHAVIOR_1K
OMNIGIBSON_HEADLESS=1 python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=$TASK_NAME \
    log_path=$LOG_PATH
```

## GR00T N1.7

This tutorial provides a simplest version instruction to finetune GR00T N1.7 on the 2026 BEHAVIOR-1K Challenge dataset.

### Repo Clone

```
git clone https://github.com/wensi-ai/Isaac-GR00T
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

This finetuning instruction is adapted from the original Isaac-GR00T repo. 

### Installation

GR00T uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```
cd Isaac-GR00T
uv sync --frozen --python 3.10
uv pip install --python .venv/bin/python websockets

source .venv/bin/activate

# Install behavior for eval (creates a separate `behavior` conda env)
cd $PATH_TO_BEHAVIOR_1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
```

The N1.7 backbone `nvidia/Cosmos-Reason2-2B` is gated. Accept the gate at [https://huggingface.co/nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) before training. 

### Finetune GR00T

We provide a GR00T N1.7 checkpoint for:

- turning_on_radio task [here](TODO: add checkpoint link).

If you would like to run eval only feel free to skip to the last section.

```
export TASK=turning_on_radio                                            # any challenge task
export DATA_ROOT=$PATH_TO_BEHAVIOR_1K/datasets/2026-challenge-demos/b1k # holds one folder per task
export DATASET_PATH=$DATA_ROOT/$TASK                                    # e.g. .../2026-challenge-demos/b1k/turning_on_radio
export OUTPUT_DIR=outputs/b1k-$TASK
```

#### Dataset version: LeRobot v3.0 (default) or v2.1

The challenge demos ship as **LeRobot v3.0**. The GR00T loader reads both **v3.0** and **v2.1** natively (it auto-detects the version from `meta/info.json`); it only additionally needs the GR00T-specific `meta/modality.json` deployed below. Choose one:

- **v3.0 — default, no conversion.** Train directly on the demos as released; `$DATASET_PATH` already points at them.
- **v2.1 — optional, convert first.** Only if your tooling specifically needs v2.1. The converter builds its own environment and runs **in place**: `$DATA_ROOT/$TASK` becomes v2.1 and the original v3.0 is backed up to `$DATA_ROOT/${TASK}_v3.0`.

To convert to v2.1:

```
cd scripts/lerobot_conversion
uv venv --python 3.11 .venv && source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475" \
  huggingface_hub jsonlines numpy pyarrow tqdm
python convert_v3_to_v2.py --root $DATA_ROOT --repo-id $TASK
cd ../..                       # back to the repo root
source .venv/bin/activate      # re-activate the GR00T venv (conversion used its own)
```

#### Deploy modality.json

Before we can run training, we need GR00T-specific `meta/modality.json`. Deploy it into each task dataset (point it at the root that holds your task folders — run this **after** any v2.1 conversion, since conversion does not carry it over):

```
python scripts/b1k/deploy_modality.py $DATA_ROOT
```

Normalization statistics (`meta/stats.json`) are generated automatically on the first training run.

#### (Optional) Pre-cache base models

Training auto-downloads the base model and its gated backbone on the first run (with `HF_TOKEN` set), but you can pre-cache them first to fail fast on access/network issues:

```
export HF_TOKEN=hf_xxx         # the account that accepted the Cosmos-Reason2-2B gate
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN")
snapshot_download("nvidia/GR00T-N1.7-3B", token=tok)
snapshot_download("nvidia/Cosmos-Reason2-2B", token=tok)  # gated backbone
PY
```

#### Train

Run the following command to finetune GR00T:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=online OMP_NUM_THREADS=4 \
torchrun --nproc_per_node=8 --master_port=29500 scripts/b1k/train_b1k.py \
    --experiment-name b1k-$TASK \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path $DATASET_PATH \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/b1k/r1pro.py \
    --num-gpus 8 \
    --global-batch-size 2048 \
    --output-dir $OUTPUT_DIR \
    --save-steps 1500 --save-total-limit 5 --max-steps 150000 \
    --dataloader-num-workers 8 --decode-only-used-frames
```

Checkpoints land in `$OUTPUT_DIR/b1k-$TASK/checkpoint-<step>/`, each one standalone and directly servable.

**Tune** `OMP_NUM_THREADS` **and** `--dataloader-num-workers` **to your CPU.**

### Evaluation

After finetuning, you can run evaluation by following the steps below:

1. Deploy finetuned checkpoint:
  ```
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python scripts/b1k/serve_b1k.py \
        --model-path $PATH_TO_CKPT \
        --modality-config-path examples/b1k/r1pro.py \
        --embodiment-tag NEW_EMBODIMENT \
        --host 127.0.0.1 --port 8000
  ```
    This opens a connection listening on 127.0.0.1:8000. Health-check it with `curl -s http://127.0.0.1:8000/healthz` (returns `OK`).
2. Run the evaluation on BEHAVIOR:
  Assume you have behavior env installed (check [https://github.com/StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) for more details), run the following command within the BEHAVIOR-1K directory:

