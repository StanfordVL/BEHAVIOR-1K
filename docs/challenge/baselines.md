# Baselines

For the 2026 BEHAVIOR Challenge, we provide starter training and evaluation pipelines for two baseline methods:

- **π0.5 (pi0.5)**
- **GR00T N1.7**

These baselines are meant to help participants verify the full workflow: loading the demonstration dataset, training or adapting a policy, running evaluation in OmniGibson, and preparing outputs for submission. They are also useful reference implementations for the expected observation and action interfaces in this year's single challenge track.

Participants are encouraged to build on these pipelines, compare against them, and open-source improvements when possible. Additional setup instructions, checkpoints, and runnable examples will be linked here as they are finalized for the 2026 release.

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

