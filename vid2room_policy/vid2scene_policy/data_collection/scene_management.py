import json
import logging
import os
import random
from pathlib import Path

import torch as th
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.object_states import OnTop

logger = logging.getLogger(__name__)

_GRASPABLE_MODELS = None
EDGE_MARGIN_M = 0.15
CENTER_MARGIN_RATIO = 0.30
SPAWN_SETTLE_STEPS = 5
POST_PLACEMENT_SETTLE_STEPS = 30
POST_REPOSITION_STEPS = 10
POST_RELEASE_STEPS = 10

def _load_graspable_models():
    global _GRASPABLE_MODELS
    if _GRASPABLE_MODELS is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "graspable_models.json"
        with open(config_path, "r", encoding="utf-8") as f:
            raw_models = [tuple(x) for x in json.load(f)]

        valid_models = []
        missing = 0
        for dataset_name, category, model in raw_models:
            usd_path = DatasetObject.get_usd_path(
                category=category,
                model=model,
                dataset_name=dataset_name,
            )
            encrypted_path = usd_path.replace(".usd", ".encrypted.usd")
            if os.path.exists(usd_path) or os.path.exists(encrypted_path):
                valid_models.append((dataset_name, category, model))
            else:
                missing += 1

        _GRASPABLE_MODELS = valid_models
        logger.info(
            "Loaded %d spawnable models (%d missing assets filtered)",
            len(valid_models),
            missing,
        )
    return _GRASPABLE_MODELS


def get_scene_objects_by_category(scene, whitelist: list[str]) -> dict[str, list]:
    objects_by_category = scene.object_registry.get_dict("category")
    filtered = {}
    for cat in whitelist:
        if cat in objects_by_category:
            filtered[cat] = list(objects_by_category[cat])
    return filtered


def spawn_and_place_object(scene, support, robot_pos: th.Tensor = None) -> DatasetObject | None:
    """Spawn a random graspable object and place it on support surface."""
    graspables = _load_graspable_models()
    dataset_name, category, model = random.choice(graspables)
    print(f"[Episode] Spawning {category}/{model}...", flush=True)

    try:
        obj = DatasetObject(
            name=f"spawned_{category}_{random.randint(0, 9999)}",
            category=category,
            model=model,
            dataset_name=dataset_name,
        )
        scene.add_object(obj)

        for _ in range(SPAWN_SETTLE_STEPS):
            og.sim.step()

        if OnTop in obj.states:
            success = obj.states[OnTop].set_value(support, True, use_trav_map=True)
            if success:
                for _ in range(POST_PLACEMENT_SETTLE_STEPS):
                    og.sim.step()

                obj_pos, obj_ori = obj.get_position_orientation()
                aabb = support.aabb
                min_x, min_y = aabb[0][0].item(), aabb[0][1].item()
                max_x, max_y = aabb[1][0].item(), aabb[1][1].item()
                center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

                if robot_pos is not None:
                    rx, ry = robot_pos[0].item(), robot_pos[1].item()
                    # Bias spawn toward the robot-facing side.
                    if rx < center_x:
                        edge_x = min_x + EDGE_MARGIN_M
                        inner_x = center_x - (center_x - min_x) * CENTER_MARGIN_RATIO
                        new_x = random.uniform(edge_x, inner_x)
                    else:
                        edge_x = max_x - EDGE_MARGIN_M
                        inner_x = center_x + (max_x - center_x) * CENTER_MARGIN_RATIO
                        new_x = random.uniform(inner_x, edge_x)
                    if ry < center_y:
                        edge_y = min_y + EDGE_MARGIN_M
                        inner_y = center_y - (center_y - min_y) * CENTER_MARGIN_RATIO
                        new_y = random.uniform(edge_y, inner_y)
                    else:
                        edge_y = max_y - EDGE_MARGIN_M
                        inner_y = center_y + (max_y - center_y) * CENTER_MARGIN_RATIO
                        new_y = random.uniform(inner_y, edge_y)
                else:
                    new_x = random.uniform(min_x + EDGE_MARGIN_M, max_x - EDGE_MARGIN_M)
                    new_y = random.uniform(min_y + EDGE_MARGIN_M, max_y - EDGE_MARGIN_M)

                new_pos = th.tensor([new_x, new_y, obj_pos[2].item()])
                obj.set_position_orientation(position=new_pos, orientation=obj_ori)
                for _ in range(POST_REPOSITION_STEPS):
                    og.sim.step()

                return obj
            else:
                scene.remove_object(obj)
                return None
        else:
            scene.remove_object(obj)
            return None

    except Exception:
        logger.exception("Failed to spawn object %s/%s", category, model)
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
                        logger.debug("Failed immediate grasp release for arm %s", arm, exc_info=True)
                    for _ in range(POST_RELEASE_STEPS):
                        og.sim.step()
                    break
        scene.remove_object(obj)
    except Exception:
        logger.exception("Failed to safely remove object %s", getattr(obj, "name", "<unknown>"))
