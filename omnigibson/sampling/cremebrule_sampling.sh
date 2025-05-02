#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities cook_eggplant
