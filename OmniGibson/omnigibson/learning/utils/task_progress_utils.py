import torch as th
from omnigibson.object_states import ToggledOn, OnTop, Inside, Open, NextTo, Under

ROBOT_OBJECT_DISTANCE_THRESHOLD = 0.5  # meters


def check_progress(env, check_specs):
    """Generic progress checker using declarative specs."""
    objs = env.task.object_scope
    results = {}

    for name, spec in check_specs.items():
        check_type = spec[0]

        if check_type == "near":
            # ("near", obj1_key, obj2_key)
            obj1, obj2 = objs[spec[1]].unwrapped, objs[spec[2]].unwrapped
            pos1, pos2 = obj1.get_position_orientation()[0], obj2.get_position_orientation()[0]
            # Only consider x and y coordinates (horizontal distance)
            dist = th.linalg.norm(pos1[:2] - pos2[:2])
            results[name] = bool(dist < ROBOT_OBJECT_DISTANCE_THRESHOLD)

        elif check_type == "state":
            # Check if it's a relational or non-relational state based on argument pattern
            if len(spec) == 4 and isinstance(spec[3], bool):
                # ("state", obj_key, state_name, expected_bool) - non-relational state, e.g. ToggledOn, Open
                obj = objs[spec[1]].unwrapped
                results[name] = obj.states[spec[2]].get_value() == spec[3]
            elif len(spec) == 5:
                # ("state", obj1_key, state_name, obj2_key, expected_bool) - relational state with explicit bool, e.g. Inside, OnTop
                obj1, obj2 = objs[spec[1]].unwrapped, objs[spec[3]].unwrapped
                results[name] = obj1.states[spec[2]].get_value(obj2) == spec[4]
            else:
                raise ValueError(f"Invalid state spec: {spec}")

    return results


# Task specifications
CHALLENGE_TASKS_PROGRESS_APPROXIMATION = {
    "turning_on_radio": lambda env: check_progress(
        env,
        {
            "robot_near_radio": ("near", "agent.n.01_1", "radio_receiver.n.01_1"),
            "radio_picked_up": ("state", "radio_receiver.n.01_1", OnTop, "table.n.02_1", False),
            "radio_on": ("state", "radio_receiver.n.01_1", ToggledOn, True),
        },
    ),
    "picking_up_trash": lambda env: check_progress(
        env,
        {
            "robot_near_trash_can": ("near", "agent.n.01_1", "ashcan.n.01_1"),
            "robot_near_can_of_soda_1": ("near", "agent.n.01_1", "can__of__soda.n.01_1"),
            "robot_near_can_of_soda_2": ("near", "agent.n.01_1", "can__of__soda.n.01_2"),
            "robot_near_can_of_soda_3": ("near", "agent.n.01_1", "can__of__soda.n.01_3"),
            "can_of_soda_1_picked_up": ("state", "can__of__soda.n.01_1", OnTop, "floor.n.01_1", False),
            "can_of_soda_2_picked_up": ("state", "can__of__soda.n.01_2", OnTop, "floor.n.01_1", False),
            "can_of_soda_3_picked_up": ("state", "can__of__soda.n.01_3", OnTop, "floor.n.01_1", False),
            "can_of_soda_1_in_trash": ("state", "can__of__soda.n.01_1", Inside, "ashcan.n.01_1", True),
            "can_of_soda_2_in_trash": ("state", "can__of__soda.n.01_2", Inside, "ashcan.n.01_1", True),
            "can_of_soda_3_in_trash": ("state", "can__of__soda.n.01_3", Inside, "ashcan.n.01_1", True),
        },
    ),
    "putting_away_Halloween_decorations": lambda env: check_progress(
        env,
        {
            "cabinet_open": ("state", "cabinet.n.01_1", Open, True),
            "robot_near_candle_1": ("near", "agent.n.01_1", "candle.n.01_1"),
            "robot_near_candle_2": ("near", "agent.n.01_1", "candle.n.01_2"),
            "robot_near_candle_3": ("near", "agent.n.01_1", "candle.n.01_3"),
            "candle_1_picked_up": ("state", "candle.n.01_1", OnTop, "floor.n.01_1", False),
            "candle_2_picked_up": ("state", "candle.n.01_2", OnTop, "floor.n.01_1", False),
            "candle_3_picked_up": ("state", "candle.n.01_3", OnTop, "floor.n.01_1", False),
            "candle_1_in_cabinet": ("state", "candle.n.01_1", Inside, "cabinet.n.01_1", True),
            "candle_2_in_cabinet": ("state", "candle.n.01_2", Inside, "cabinet.n.01_1", True),
            "candle_3_in_cabinet": ("state", "candle.n.01_3", Inside, "cabinet.n.01_1", True),
            "robot_near_pumpkin_1": ("near", "agent.n.01_1", "pumpkin.n.02_1"),
            "robot_near_pumpkin_2": ("near", "agent.n.01_1", "pumpkin.n.02_2"),
            "pumpkin_1_picked_up": ("state", "pumpkin.n.02_1", OnTop, "floor.n.01_1", False),
            "pumpkin_2_picked_up": ("state", "pumpkin.n.02_2", OnTop, "floor.n.01_1", False),
            "pumpkin_1_in_cabinet": ("state", "pumpkin.n.02_1", Inside, "cabinet.n.01_1", True),
            "pumpkin_2_in_cabinet": ("state", "pumpkin.n.02_2", Inside, "cabinet.n.01_1", True),
            "robot_near_caldron": ("near", "agent.n.01_1", "caldron.n.01_1"),
            "caldron_picked_up": ("state", "caldron.n.01_1", OnTop, "floor.n.01_1", False),
            "caldron_next_to_table": ("state", "caldron.n.01_1", NextTo, "table.n.02_1", True),
        },
    ),
    "cleaning_up_plates_and_food": lambda env: check_progress(
        env,
        {
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "robot_near_table": ("near", "agent.n.01_1", "breakfast_table.n.01_1"),
            "robot_near_sink": ("near", "agent.n.01_1", "sink.n.01_1"),
            "fridge_opened": ("state", "electric_refrigerator.n.01_1", Open, None, True),
            "plate_1_picked_up": ("state", "plate.n.04_1", OnTop, "breakfast_table.n.01_1", False),
            "plate_2_picked_up": ("state", "plate.n.04_2", OnTop, "breakfast_table.n.01_1", False),
            "pizza_1_in_fridge": ("state", "pizza.n.01_1", Inside, "electric_refrigerator.n.01_1", True),
            "pizza_2_in_fridge": ("state", "pizza.n.01_2", Inside, "electric_refrigerator.n.01_1", True),
            "plate_1_in_fridge": ("state", "plate.n.04_1", Inside, "electric_refrigerator.n.01_1", True),
            "plate_2_in_fridge": ("state", "plate.n.04_2", Inside, "electric_refrigerator.n.01_1", True),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, None, False),
            "bowl_1_picked_up": ("state", "bowl.n.01_1", OnTop, "breakfast_table.n.01_1", False),
            "bowl_2_picked_up": ("state", "bowl.n.01_2", OnTop, "breakfast_table.n.01_1", False),
            "bowl_1_in_sink": ("state", "bowl.n.01_1", Inside, "sink.n.01_1", True),
            "bowl_2_in_sink": ("state", "bowl.n.01_2", Inside, "sink.n.01_1", True),
        },
    ),
    "can_meat": lambda env: check_progress(
        env,
        {
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "robot_near_chopping_board": ("near", "agent.n.01_1", "chopping_board.n.01_1"),
            "cabinet_opened": ("state", "cabinet.n.01_1", Open, None, True),
            "jar_1_picked_up": ("state", "hinged_jar.n.01_1", Inside, "cabinet.n.01_1", False),
            "jar_2_picked_up": ("state", "hinged_jar.n.01_2", Inside, "cabinet.n.01_1", False),
            "jar_1_opened": ("state", "hinged_jar.n.01_1", Open, None, True),
            "jar_2_opened": ("state", "hinged_jar.n.01_2", Open, None, True),
            "bratwurst_1_picked_up": ("state", "bratwurst.n.01_1", OnTop, "chopping_board.n.01_1", False),
            "bratwurst_2_picked_up": ("state", "bratwurst.n.01_2", OnTop, "chopping_board.n.01_1", False),
            "bratwurst_3_picked_up": ("state", "bratwurst.n.01_3", OnTop, "chopping_board.n.01_1", False),
            "bratwurst_4_picked_up": ("state", "bratwurst.n.01_4", OnTop, "chopping_board.n.01_1", False),
            "bratwurst_1_in_jar": ("state", "bratwurst.n.01_1", Inside, "hinged_jar.n.01_1", True),
            "bratwurst_2_in_jar": ("state", "bratwurst.n.01_2", Inside, "hinged_jar.n.01_1", True),
            "bratwurst_3_in_jar": ("state", "bratwurst.n.01_3", Inside, "hinged_jar.n.01_2", True),
            "bratwurst_4_in_jar": ("state", "bratwurst.n.01_4", Inside, "hinged_jar.n.01_2", True),
            "jar_1_closed": ("state", "hinged_jar.n.01_1", Open, None, False),
            "jar_2_closed": ("state", "hinged_jar.n.01_2", Open, None, False),
            "jar_1_back_in_cabinet": ("state", "hinged_jar.n.01_1", Inside, "cabinet.n.01_1", True),
            "jar_2_back_in_cabinet": ("state", "hinged_jar.n.01_2", Inside, "cabinet.n.01_1", True),
            "cabinet_closed": ("state", "cabinet.n.01_1", Open, None, False),
        },
    ),
    "setting_mousetraps": lambda env: check_progress(
        env,
        {
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "robot_near_sink": ("near", "agent.n.01_1", "sink.n.01_1"),
            "mousetrap_1_picked_up": ("state", "mousetrap.n.01_1", OnTop, "cabinet.n.01_1", False),
            "mousetrap_2_picked_up": ("state", "mousetrap.n.01_2", OnTop, "cabinet.n.01_1", False),
            "mousetrap_3_picked_up": ("state", "mousetrap.n.01_3", OnTop, "cabinet.n.01_1", False),
            "mousetrap_4_picked_up": ("state", "mousetrap.n.01_4", OnTop, "cabinet.n.01_1", False),
            "mousetrap_1_on_floor": ("state", "mousetrap.n.01_1", OnTop, "floor.n.01_1", True),
            "mousetrap_2_on_floor": ("state", "mousetrap.n.01_2", OnTop, "floor.n.01_1", True),
            "mousetrap_3_on_floor": ("state", "mousetrap.n.01_3", OnTop, "floor.n.01_1", True),
            "mousetrap_4_on_floor": ("state", "mousetrap.n.01_4", OnTop, "floor.n.01_1", True),
            "mousetrap_1_near_sink": ("state", "mousetrap.n.01_1", Under, "sink.n.01_1", True),
            "mousetrap_2_near_sink": ("state", "mousetrap.n.01_2", Under, "sink.n.01_1", True),
        },
    ),
    "hiding_Easter_eggs": lambda env: check_progress(
        env,
        {
            "robot_near_basket": ("near", "agent.n.01_1", "wicker_basket.n.01_1"),
            "robot_near_tree": ("near", "agent.n.01_1", "tree.n.01_1"),
            "egg_1_out_of_basket": ("state", "easter_egg.n.01_1", Inside, "wicker_basket.n.01_1", False),
            "egg_2_out_of_basket": ("state", "easter_egg.n.01_2", Inside, "wicker_basket.n.01_1", False),
            "egg_3_out_of_basket": ("state", "easter_egg.n.01_3", Inside, "wicker_basket.n.01_1", False),
            "egg_1_on_lawn": ("state", "easter_egg.n.01_1", OnTop, "lawn.n.01_1", True),
            "egg_2_on_lawn": ("state", "easter_egg.n.01_2", OnTop, "lawn.n.01_1", True),
            "egg_3_on_lawn": ("state", "easter_egg.n.01_3", OnTop, "lawn.n.01_1", True),
            # TODO: what do we do with wildcards?
            # "egg_1_next_to_tree": ("state", "easter_egg.n.01_1", NextTo, "tree.n.01_1", True),
            # "egg_2_next_to_tree": ("state", "easter_egg.n.01_2", NextTo, "tree.n.01_1", True),
            # "egg_3_next_to_tree": ("state", "easter_egg.n.01_3", NextTo, "tree.n.01_1", True),
        },
    ),
    "picking_up_toys": lambda env: check_progress(
        env,
        {
            "robot_near_bed": ("near", "agent.n.01_1", "bed.n.01_1"),
            "robot_near_table": ("near", "agent.n.01_1", "table.n.02_1"),
            "robot_near_toy_box": ("near", "agent.n.01_1", "toy_box.n.01_1"),
            "board_game_1_picked_up": ("state", "board_game.n.01_1", OnTop, "bed.n.01_1", False),
            "board_game_2_picked_up": ("state", "board_game.n.01_2", OnTop, "bed.n.01_1", False),
            "board_game_3_picked_up": ("state", "board_game.n.01_3", OnTop, "table.n.02_1", False),
            "jigsaw_1_picked_up": ("state", "jigsaw_puzzle.n.01_1", OnTop, "table.n.02_1", False),
            "jigsaw_2_picked_up": ("state", "jigsaw_puzzle.n.01_2", OnTop, "table.n.02_1", False),
            "tennis_ball_picked_up": ("state", "tennis_ball.n.01_1", OnTop, "table.n.02_1", False),
            "board_game_1_in_box": ("state", "board_game.n.01_1", Inside, "toy_box.n.01_1", True),
            "board_game_2_in_box": ("state", "board_game.n.01_2", Inside, "toy_box.n.01_1", True),
            "board_game_3_in_box": ("state", "board_game.n.01_3", Inside, "toy_box.n.01_1", True),
            "jigsaw_1_in_box": ("state", "jigsaw_puzzle.n.01_1", Inside, "toy_box.n.01_1", True),
            "jigsaw_2_in_box": ("state", "jigsaw_puzzle.n.01_2", Inside, "toy_box.n.01_1", True),
            "tennis_ball_in_box": ("state", "tennis_ball.n.01_1", Inside, "toy_box.n.01_1", True),
        },
    ),
    "rearranging_kitchen_furniture": lambda env: check_progress(
        env,
        {
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "cabinet_opened": ("state", "cabinet.n.01_1", Open, None, True),
            "toaster_picked_up": ("state", "toaster.n.02_1", OnTop, "countertop.n.01_1", False),
            "food_processor_picked_up": ("state", "food_processor.n.01_1", OnTop, "countertop.n.01_1", False),
            "french_press_picked_up": ("state", "french_press.n.01_1", OnTop, "countertop.n.01_1", False),
            "toaster_in_cabinet": ("state", "toaster.n.02_1", Inside, "cabinet.n.01_1", True),
            "food_processor_in_cabinet": ("state", "food_processor.n.01_1", Inside, "cabinet.n.01_1", True),
            "french_press_in_cabinet": ("state", "french_press.n.01_1", Inside, "cabinet.n.01_1", True),
            "cabinet_closed": ("state", "cabinet.n.01_1", Open, None, False),
        },
    ),
    "putting_up_Christmas_decorations_inside": lambda env: check_progress(
        env,
        {
            "robot_near_basket": ("near", "agent.n.01_1", "wicker_basket.n.01_1"),
            "robot_near_tree": ("near", "agent.n.01_1", "christmas_tree.n.05_1"),
            "robot_near_table": ("near", "agent.n.01_1", "table.n.02_1"),
            "robot_near_sofa": ("near", "agent.n.01_1", "sofa.n.01_1"),
            "wreath_out_of_basket": ("state", "wreath.n.01_1", Inside, "wicker_basket.n.01_1", False),
            "candy_cane_1_out_of_basket": ("state", "candy_cane.n.01_1", Inside, "wicker_basket.n.01_1", False),
            "candy_cane_2_out_of_basket": ("state", "candy_cane.n.01_2", Inside, "wicker_basket.n.01_1", False),
            "candy_cane_3_out_of_basket": ("state", "candy_cane.n.01_3", Inside, "wicker_basket.n.01_1", False),
            "candle_1_out_of_basket": ("state", "pillar_candle.n.01_1", Inside, "wicker_basket.n.01_1", False),
            "candle_2_out_of_basket": ("state", "pillar_candle.n.01_2", Inside, "wicker_basket.n.01_1", False),
            "gift_1_near_tree": ("state", "gift_box.n.01_1", NextTo, "christmas_tree.n.05_1", True),
            "gift_2_near_tree": ("state", "gift_box.n.01_2", NextTo, "christmas_tree.n.05_1", True),
            "gift_3_near_tree": ("state", "gift_box.n.01_3", NextTo, "christmas_tree.n.05_1", True),
            "candle_1_on_table": ("state", "pillar_candle.n.01_1", OnTop, "table.n.02_1", True),
            "candle_2_on_table": ("state", "pillar_candle.n.01_2", OnTop, "table.n.02_1", True),
            "candy_cane_on_table": ("state", "candy_cane.n.01_1", OnTop, "table.n.02_1", True),
            "wreath_on_sofa": ("state", "wreath.n.01_1", OnTop, "sofa.n.01_1", True),
            "candy_cane_2_on_sofa": ("state", "candy_cane.n.01_2", OnTop, "sofa.n.01_1", True),
            "candy_cane_3_on_sofa": ("state", "candy_cane.n.01_3", OnTop, "sofa.n.01_1", True),
        },
    ),
    "set_up_a_coffee_station_in_your_kitchen": lambda env: check_progress(
        env,
        {
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_shelf": ("near", "agent.n.01_1", "shelf.n.01_1"),
            "robot_near_coffee_maker": ("near", "agent.n.01_1", "coffee_maker.n.01_1"),
            "filter_picked_up": ("state", "paper_coffee_filter.n.01_1", OnTop, "countertop.n.01_1", False),
            "filter_in_coffee_maker": ("state", "paper_coffee_filter.n.01_1", OnTop, "coffee_maker.n.01_1", True),
            "coffee_bottle_picked_up": ("state", "bottle__of__coffee.n.01_1", OnTop, "shelf.n.01_1", False),
            "coffee_bottle_near_maker": ("state", "bottle__of__coffee.n.01_1", NextTo, "coffee_maker.n.01_1", True),
            "kettle_repositioned": ("state", "electric_kettle.n.01_1", NextTo, "coffee_maker.n.01_1", True),
            "saucer_near_maker": ("state", "saucer.n.02_1", NextTo, "coffee_maker.n.01_1", True),
            "cup_picked_up": ("state", "coffee_cup.n.01_1", OnTop, "countertop.n.01_1", False),
            "cup_on_saucer": ("state", "coffee_cup.n.01_1", OnTop, "saucer.n.02_1", True),
        },
    ),
    "putting_dishes_away_after_cleaning": lambda env: check_progress(
        env,
        {
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "robot_near_countertop_1": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_countertop_2": ("near", "agent.n.01_1", "countertop.n.01_2"),
            "cabinet_opened": ("state", "cabinet.n.01_1", Open, None, True),
            "plate_1_picked_up": ("state", "plate.n.04_1", OnTop, "countertop.n.01_1", False),
            "plate_2_picked_up": ("state", "plate.n.04_2", OnTop, "countertop.n.01_1", False),
            "plate_3_picked_up": ("state", "plate.n.04_3", OnTop, "countertop.n.01_1", False),
            "plate_4_picked_up": ("state", "plate.n.04_4", OnTop, "countertop.n.01_1", False),
            "plate_5_picked_up": ("state", "plate.n.04_5", OnTop, "countertop.n.01_2", False),
            "plate_6_picked_up": ("state", "plate.n.04_6", OnTop, "countertop.n.01_2", False),
            "plate_7_picked_up": ("state", "plate.n.04_7", OnTop, "countertop.n.01_2", False),
            "plate_8_picked_up": ("state", "plate.n.04_8", OnTop, "countertop.n.01_2", False),
            "plate_1_in_cabinet": ("state", "plate.n.04_1", Inside, "cabinet.n.01_1", True),
            "plate_2_in_cabinet": ("state", "plate.n.04_2", Inside, "cabinet.n.01_1", True),
            "plate_3_in_cabinet": ("state", "plate.n.04_3", Inside, "cabinet.n.01_1", True),
            "plate_4_in_cabinet": ("state", "plate.n.04_4", Inside, "cabinet.n.01_1", True),
            "plate_5_in_cabinet": ("state", "plate.n.04_5", Inside, "cabinet.n.01_1", True),
            "plate_6_in_cabinet": ("state", "plate.n.04_6", Inside, "cabinet.n.01_1", True),
            "plate_7_in_cabinet": ("state", "plate.n.04_7", Inside, "cabinet.n.01_1", True),
            "plate_8_in_cabinet": ("state", "plate.n.04_8", Inside, "cabinet.n.01_1", True),
            "cabinet_closed": ("state", "cabinet.n.01_1", Open, None, False),
        },
    ),
    "preparing_lunch_box": lambda env: check_progress(
        env,
        {
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_chopping_board": ("near", "agent.n.01_1", "chopping_board.n.01_1"),
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "lunch_box_picked_up": ("state", "packing_box.n.02_1", OnTop, "countertop.n.01_1", False),
            "sandwich_picked_up": ("state", "club_sandwich.n.01_1", OnTop, "chopping_board.n.01_1", False),
            "apple_1_picked_up": ("state", "half__apple.n.01_1", OnTop, "chopping_board.n.01_1", False),
            "apple_2_picked_up": ("state", "half__apple.n.01_2", OnTop, "chopping_board.n.01_1", False),
            "cookie_picked_up": ("state", "chocolate_chip_cookie.n.01_1", OnTop, "chopping_board.n.01_1", False),
            "fridge_opened": ("state", "electric_refrigerator.n.01_1", Open, None, True),
            "tea_picked_up": ("state", "bottle__of__tea.n.01_1", Inside, "electric_refrigerator.n.01_1", False),
            "sandwich_in_box": ("state", "club_sandwich.n.01_1", Inside, "packing_box.n.02_1", True),
            "apple_1_in_box": ("state", "half__apple.n.01_1", Inside, "packing_box.n.02_1", True),
            "apple_2_in_box": ("state", "half__apple.n.01_2", Inside, "packing_box.n.02_1", True),
            "cookie_in_box": ("state", "chocolate_chip_cookie.n.01_1", Inside, "packing_box.n.02_1", True),
            "tea_in_box": ("state", "bottle__of__tea.n.01_1", Inside, "packing_box.n.02_1", True),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, None, False),
        },
    ),
    "loading_the_car": lambda env: check_progress(
        env,
        {
            "robot_near_car": ("near", "agent.n.01_1", "car.n.01_1"),
            "robot_near_table": ("near", "agent.n.01_1", "table.n.02_1"),
            "robot_near_container": ("near", "agent.n.01_1", "container.n.01_1"),
            "car_opened": ("state", "car.n.01_1", Open, None, True),
            "container_picked_up": ("state", "container.n.01_1", OnTop, "floor.n.01_2", False),
            "camera_picked_up": ("state", "digital_camera.n.01_1", OnTop, "table.n.02_1", False),
            "racket_picked_up": ("state", "tennis_racket.n.01_1", OnTop, "table.n.02_1", False),
            "container_in_car": ("state", "container.n.01_1", Inside, "car.n.01_1", True),
            "camera_in_container": ("state", "digital_camera.n.01_1", Inside, "container.n.01_1", True),
            "racket_in_car": ("state", "tennis_racket.n.01_1", Inside, "car.n.01_1", True),
            "car_closed": ("state", "car.n.01_1", Open, None, False),
        },
    ),
    "carrying_in_groceries": lambda env: check_progress(
        env,
        {
            "robot_near_car": ("near", "agent.n.01_1", "car.n.01_1"),
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "bag_picked_up": ("state", "sack.n.01_1", Inside, "car.n.01_1", False),
            "car_closed": ("state", "car.n.01_1", Open, None, False),
            "tomato_out_of_bag": ("state", "beefsteak_tomato.n.01_1", Inside, "sack.n.01_1", False),
            "milk_out_of_bag": ("state", "carton__of__milk.n.01_1", Inside, "sack.n.01_1", False),
            "fridge_opened": ("state", "electric_refrigerator.n.01_1", Open, None, True),
            "tomato_in_fridge": ("state", "beefsteak_tomato.n.01_1", Inside, "electric_refrigerator.n.01_1", True),
            "milk_in_fridge": ("state", "carton__of__milk.n.01_1", Inside, "electric_refrigerator.n.01_1", True),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, None, False),
        },
    ),
    "make_microwave_popcorn": lambda env: check_progress(
        env,
        {
            "robot_near_microwave": ("near", "agent.n.01_1", "microwave.n.02_1"),
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "microwave_opened": ("state", "microwave.n.02_1", Open, None, True),
            "popcorn_bag_picked_up": ("state", "popcorn__bag.n.01_1", OnTop, "countertop.n.01_1", False),
            "popcorn_bag_in_microwave": ("state", "popcorn__bag.n.01_1", Inside, "microwave.n.02_1", True),
            "microwave_closed": ("state", "microwave.n.02_1", Open, None, False),
            "microwave_turned_on": ("state", "microwave.n.02_1", ToggledOn, None, True),
            # TODO: figure out what to do for "Real/exist"
        },
    ),
    # TODO: add more tasks here
}
