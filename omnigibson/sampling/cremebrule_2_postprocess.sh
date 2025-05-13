#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity picking_up_trash
#python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity rearranging_kitchen_furniture
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity picking_up_toys
#python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity turning_on_radio

