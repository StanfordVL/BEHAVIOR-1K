import json
import logging
from dataclasses import dataclass
from typing import Callable, Literal

logger = logging.getLogger(__name__)
ObjectFilterMethod = Literal["whitelist", "classifier"]
EXCLUDED_SUPPORT_CATEGORIES = {"bottom_cabinet_no_top", "top_cabinet"}


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
    success_file: str | None = None
    # Whitelist mode inputs.
    whitelist_graspable_path: str = "configs/whitelist_graspable_objects.json"
    whitelist_support_path: str = "configs/whitelist_support_objects.json"
    approach_dist: float = 0.5
    angle_threshold: float = 0.1
    object_filter_method: ObjectFilterMethod = "classifier"
    # Classifier mode inputs.
    classifier_embeddings_path: str | None = None
    classifier_models_dir: str | None = None
    classifier_threshold: float = 0.5
    # Base-start sampling constraints.
    max_arm_reach_m: float = 1.0
    support_search_radius_m: float = 2.5
    support_erosion_extra_margin_m: float = 0.25
    ignored_nav_obstacle_categories: tuple[str, ...] = (
        "armchair",
        "straight_chair",
        "folding_chair",
        "rocking_chair",
        "swivel_chair",
        "highchair",
        "eames_chair",
        "garden_chair",
        "ottoman",
    )
    # Temporarily move unrelated movable objects away per attempt.
    remove_other_movable_objects: bool = False


def load_whitelists(config: DataCollectionConfig) -> tuple[list[str], list[str]]:
    with open(config.whitelist_graspable_path, "r", encoding="utf-8") as f:
        graspable = json.load(f)["graspable_objects"]
    with open(config.whitelist_support_path, "r", encoding="utf-8") as f:
        support = json.load(f)["support_objects"]
    logger.info("Loaded %d graspable, %d support categories", len(graspable), len(support))
    return graspable, support


def _build_combined_filter_fns(
    support_whitelist: list[str],
    graspable_whitelist: list[str],
    classifier_support_fn: Callable[[str], bool],
    classifier_graspable_fn: Callable[[str], bool],
) -> tuple[Callable[[str], bool], Callable[[str], bool]]:
    """Combine whitelist and classifier filters with support exclusions."""

    def combined_support_fn(category: str) -> bool:
        return (
            category in support_whitelist or classifier_support_fn(category)
        ) and category not in EXCLUDED_SUPPORT_CATEGORIES

    def combined_graspable_fn(category: str) -> bool:
        return category in graspable_whitelist or classifier_graspable_fn(category)

    return combined_support_fn, combined_graspable_fn


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
    if config.object_filter_method == "classifier":
        graspable_whitelist, support_whitelist = load_whitelists(config)
        classifier_support_fn, classifier_graspable_fn = get_object_filter(
            method="classifier",
            embeddings_path=config.classifier_embeddings_path,
            models_dir=config.classifier_models_dir,
            threshold=config.classifier_threshold,
        )
        return _build_combined_filter_fns(
            support_whitelist=support_whitelist,
            graspable_whitelist=graspable_whitelist,
            classifier_support_fn=classifier_support_fn,
            classifier_graspable_fn=classifier_graspable_fn,
        )

    raise ValueError(f"Unknown object_filter_method: {config.object_filter_method}")
