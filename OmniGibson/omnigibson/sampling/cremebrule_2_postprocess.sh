#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity assembling_gift_baskets
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity can_beans
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity can_meat
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity canning_food
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity carrying_in_groceries
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity chop_an_onion
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity chopping_wood
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity clean_up_broken_glass
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity cleaning_up_plates_and_food
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity cook_a_brisket
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity cook_bacon
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity cook_cabbage
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity cook_eggplant
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity cool_cakes
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity freeze_fruit
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity freeze_meat
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity freeze_pies
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity freeze_vegetables
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity hanging_pictures
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity hiding_Easter_eggs
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity loading_the_dishwasher
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity picking_up_toys
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity picking_up_trash
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity putting_away_Halloween_decorations
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity putting_away_toys
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity putting_up_Christmas_decorations_inside
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity rearranging_kitchen_furniture
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_upper --activity setting_mousetraps
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity slicing_vegetables
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_upper --activity sorting_mail
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity spraying_for_bugs
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity thawing_frozen_food
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity turning_on_radio
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_single_floor --activity put_togethera_basic_pruning_kit
python ${DIR}/postprocess_sampled_task.py --overwrite --scene_model house_double_floor_lower --activity spraying_fruit_trees
