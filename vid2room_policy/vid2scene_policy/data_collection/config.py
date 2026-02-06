import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class DataCollectionConfig:
    scene_model: str = "Rs_int"
    dataset_name: str = "behavior-1k-assets"
    num_episodes: int = 100
    max_steps_per_episode: int = 500
    max_navigation_steps: int = 2000
    max_rotation_steps: int = 300
    max_grasp_steps: int = 100
    max_place_steps: int = 100
    fps: int = 30
    output_dir: str = "./lerobot_datasets"
    repo_id: str = "pick_place_dataset"
    # Whitelist paths (used when object_filter_method="whitelist")
    whitelist_graspable_path: str = "configs/whitelist_graspable_objects.json"
    whitelist_support_path: str = "configs/whitelist_support_objects.json"
    approach_dist: float = 0.5
    angle_threshold: float = 0.1
    # Object filter method: "whitelist" or "classifier"
    object_filter_method: str = "classifier"
    # Classifier paths (used when object_filter_method="classifier", None uses defaults)
    classifier_embeddings_path: str | None = None
    classifier_models_dir: str | None = None
    classifier_threshold: float = 0.5


def load_whitelists(config: DataCollectionConfig) -> tuple[list[str], list[str]]:
    with open(config.whitelist_graspable_path) as f:
        graspable = json.load(f)["graspable_objects"]
    with open(config.whitelist_support_path) as f:
        support = json.load(f)["support_objects"]
    logger.info("Loaded %d graspable, %d support categories", len(graspable), len(support))
    return graspable, support


def get_object_filters(config: DataCollectionConfig) -> tuple[Callable[[str], bool], Callable[[str], bool]]:
    """Get object filter functions based on config method.

    Returns:
        Tuple of (is_support_fn, is_graspable_fn) functions
    """
    from .object_classifier import get_object_filter

    if config.object_filter_method == "whitelist":
        graspable_whitelist, support_whitelist = load_whitelists(config)
        return get_object_filter(
            method="whitelist",
            support_whitelist=support_whitelist,
            graspable_whitelist=graspable_whitelist,
        )
    elif config.object_filter_method == "classifier":
        # Combine whitelist + classifier: use both for more coverage
        graspable_whitelist, support_whitelist = load_whitelists(config)
        classifier_support_fn, classifier_graspable_fn = get_object_filter(
            method="classifier",
            embeddings_path=config.classifier_embeddings_path,
            models_dir=config.classifier_models_dir,
            threshold=config.classifier_threshold,
        )
        # Return combined filter: whitelist OR classifier
        def combined_support_fn(category: str) -> bool:
            return (category in support_whitelist or classifier_support_fn(category)) and category not in ("bottom_cabinet_no_top", "top_cabinet")
        def combined_graspable_fn(category: str) -> bool:
            return category in graspable_whitelist or classifier_graspable_fn(category)
        return combined_support_fn, combined_graspable_fn
    else:
        raise ValueError(f"Unknown object_filter_method: {config.object_filter_method}")
