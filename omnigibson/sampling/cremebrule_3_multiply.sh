#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#python ${DIR}/multiply_b1k_tasks.py --seed 0 --start_idx 1 --end_idx 150 --partial_save --scene_model house_double_floor_lower --activity rearranging_kitchen_furniture
#python ${DIR}/multiply_b1k_tasks.py --seed 0 --start_idx 1 --end_idx 150 --partial_save --scene_model house_double_floor_lower --activity picking_up_trash
#python ${DIR}/multiply_b1k_tasks.py --seed 0 --start_idx 1 --end_idx 150 --partial_save --scene_model house_double_floor_lower --activity turning_on_radio
python ${DIR}/multiply_b1k_tasks.py --seed 0 --start_idx 1 --end_idx 150 --partial_save --scene_model house_double_floor_lower --activity picking_up_trash


