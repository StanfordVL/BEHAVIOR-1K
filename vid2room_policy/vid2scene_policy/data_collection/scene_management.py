import logging
import random

import torch as th
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.object_states import OnTop
from omnigibson.utils.asset_utils import get_all_object_category_models

logger = logging.getLogger(__name__)


def get_scene_objects_by_category(scene, whitelist: list[str]) -> dict[str, list]:
    objects_by_category = scene.object_registry.get_dict("category")
    filtered = {}
    for cat in whitelist:
        if cat in objects_by_category:
            filtered[cat] = list(objects_by_category[cat])
    return filtered


def spawn_and_place_object(scene, category: str, support, robot_pos: th.Tensor = None) -> DatasetObject | None:
    """Spawn object and place it on support surface, biased toward robot."""
    EDGE_MARGIN = 0.15

    models = get_all_object_category_models(category)
    if not models:
        return None

    model = random.choice(models)
    try:
        obj = DatasetObject(
            name=f"spawned_{category}_{random.randint(0, 9999)}",
            category=category,
            model=model,
        )
        scene.add_object(obj)

        for _ in range(5):
            og.sim.step()

        if OnTop in obj.states:
            success = obj.states[OnTop].set_value(support, True)
            if success:
                for _ in range(30):
                    og.sim.step()

                obj_pos, obj_ori = obj.get_position_orientation()
                aabb = support.aabb
                min_x, min_y = aabb[0][0].item(), aabb[0][1].item()
                max_x, max_y = aabb[1][0].item(), aabb[1][1].item()
                center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

                if robot_pos is not None:
                    rx, ry = robot_pos[0].item(), robot_pos[1].item()
                    # Place closer to the edge near robot
                    CENTER_MARGIN = 0.30  # Stay away from center by this margin
                    if rx < center_x:
                        edge_x = min_x + EDGE_MARGIN
                        inner_x = center_x - (center_x - min_x) * CENTER_MARGIN
                        new_x = random.uniform(edge_x, inner_x)
                    else:
                        edge_x = max_x - EDGE_MARGIN
                        inner_x = center_x + (max_x - center_x) * CENTER_MARGIN
                        new_x = random.uniform(inner_x, edge_x)
                    if ry < center_y:
                        edge_y = min_y + EDGE_MARGIN
                        inner_y = center_y - (center_y - min_y) * CENTER_MARGIN
                        new_y = random.uniform(edge_y, inner_y)
                    else:
                        edge_y = max_y - EDGE_MARGIN
                        inner_y = center_y + (max_y - center_y) * CENTER_MARGIN
                        new_y = random.uniform(inner_y, edge_y)
                else:
                    new_x = random.uniform(min_x + EDGE_MARGIN, max_x - EDGE_MARGIN)
                    new_y = random.uniform(min_y + EDGE_MARGIN, max_y - EDGE_MARGIN)

                new_pos = th.tensor([new_x, new_y, obj_pos[2].item()])
                obj.set_position_orientation(position=new_pos, orientation=obj_ori)
                for _ in range(10):
                    og.sim.step()

                return obj
            else:
                scene.remove_object(obj)
                return None
        else:
            scene.remove_object(obj)
            return None

    except Exception as e:
        logger.debug("Failed to spawn %s: %s", category, e)
        return None


def safe_remove_object(scene, obj, robot=None):
    try:
        if robot is not None:
            for arm in robot.arm_names:
                obj_in_hand = robot._ag_obj_in_hand.get(arm)
                if obj_in_hand is not None and obj_in_hand.name == obj.name:
                    try:
                        robot.release_grasp_immediately(arm=arm)
                    except Exception:
                        pass
                    for _ in range(10):
                        og.sim.step()
                    break
        scene.remove_object(obj)
    except Exception:
        pass
