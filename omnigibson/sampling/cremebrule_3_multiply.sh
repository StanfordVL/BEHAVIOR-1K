#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python ${DIR}/multiply_b1k_tasks.py --seed 0 --start_idx 1 --end_idx 10 --partial_save --scene_model house_double_floor_lower --activity picking_up_trash
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities cook_eggplant
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_double_floor_lower --activities chopping_wood
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities slicing_vegetables
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_double_floor_lower --activities chop_an_onion
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities installing_alarms
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_double_floor_lower --activities turning_on_radio
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities thawing_frozen_food
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities freeze_meat
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities freeze_vegetables
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities freeze_fruit
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities freeze_pies
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_double_floor_lower --activities cool_cakes
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities cook_a_brisket
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities cook_bacon
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_double_floor_lower --activities can_beans
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_double_floor_lower --activities spraying_for_bugs
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities canning_food
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities cook_cabbage

