#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities cook_eggplant
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_double_floor_lower --activities chopping_wood
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities slicing_vegetables
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_double_floor_lower --activities chop_an_onion
#python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_single_floor --activities installing_alarms
python ${DIR}/sample_b1k_tasks.py --offline --scene_model house_double_floor_lower --activities turning_on_radio

