import argparse
from pathlib import Path

from vid2scene_policy.data_collection.config import DataCollectionConfig
from vid2scene_policy.data_collection.episode import run_data_collection


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for data collection."""
    base_dir = Path(__file__).resolve().parent
    whitelist_graspable = base_dir / "configs" / "whitelist_graspable_objects.json"
    whitelist_support = base_dir / "configs" / "whitelist_support_objects.json"

    parser = argparse.ArgumentParser(description="Collect R1Pro pick-only data")
    parser.add_argument("--scene_name", "--scene", dest="scene_name", default="Rs_int", help="Scene model name")
    parser.add_argument(
        "--scene_dataset",
        "--dataset",
        dest="scene_dataset",
        default="behavior-1k-assets",
        help="Scene dataset name (e.g. behavior-1k-assets, spoc, proctor)",
    )
    parser.add_argument(
        "--num_episodes", "--episodes", dest="num_episodes", type=int, default=100, help="Number of episodes to collect"
    )
    parser.add_argument("--output", default="./lerobot_datasets", help="Output directory")
    parser.add_argument("--repo-id", default=None, help="Repository ID (default: {scene}_r1pro_pick_only)")
    parser.add_argument("--max-nav-steps", type=int, default=1500, help="Max navigation steps per episode")
    parser.add_argument("--max-rotation-steps", type=int, default=300, help="Max steps for rotation")
    parser.add_argument(
        "--max-arm-reach-m",
        type=float,
        default=0.9,
        help="Max robot-base to support distance (meters) when selecting start poses",
    )
    parser.add_argument(
        "--support-search-radius-m",
        type=float,
        default=2.5,
        help="Search radius (meters) around support for deterministic base-pose sampling",
    )
    parser.add_argument(
        "--support-erosion-extra-margin-m",
        type=float,
        default=0.25,
        help="Extra safety margin (meters) added to base radius during support-sampling erosion",
    )
    parser.add_argument("--object-filter", default="classifier", choices=["whitelist", "classifier"],
                        help="Object filter method (default: classifier)")
    parser.add_argument("--classifier-embeddings", default=None, help="Path to classifier embeddings")
    parser.add_argument("--classifier-models", default=None, help="Path to classifier models directory")
    parser.add_argument("--classifier-threshold", type=float, default=0.5, help="Classifier threshold")
    parser.add_argument(
        "--ignore-nav-obstacle-categories",
        type=str,
        default=None,
        help="Comma-separated categories to clear from traversability before erosion (e.g. straight_chair,armchair)",
    )
    parser.add_argument(
        "--remove-other-movable-objects",
        action="store_true",
        help="If set, temporarily move unrelated movable objects away during each attempt (default: disabled)",
    )
    parser.set_defaults(
        _whitelist_graspable_path=str(whitelist_graspable),
        _whitelist_support_path=str(whitelist_support),
    )
    return parser


def _parse_ignored_categories(raw_categories: str | None) -> tuple[str, ...] | None:
    if raw_categories is None:
        return None
    return tuple(category.strip() for category in raw_categories.split(",") if category.strip())


def _build_config(args: argparse.Namespace) -> DataCollectionConfig:
    repo_id = args.repo_id or f"{args.scene_name}_r1pro_pick_only"
    ignored_nav_obstacle_categories = _parse_ignored_categories(args.ignore_nav_obstacle_categories)

    default_ignored = DataCollectionConfig.__dataclass_fields__["ignored_nav_obstacle_categories"].default

    return DataCollectionConfig(
        scene_model=args.scene_name,
        dataset_name=args.scene_dataset,
        num_episodes=args.num_episodes,
        output_dir=args.output,
        repo_id=repo_id,
        whitelist_graspable_path=args._whitelist_graspable_path,
        whitelist_support_path=args._whitelist_support_path,
        max_navigation_steps=args.max_nav_steps,
        max_rotation_steps=args.max_rotation_steps,
        max_grasp_steps=100,
        max_place_steps=100,
        max_arm_reach_m=args.max_arm_reach_m,
        support_search_radius_m=args.support_search_radius_m,
        support_erosion_extra_margin_m=args.support_erosion_extra_margin_m,
        object_filter_method=args.object_filter,
        classifier_embeddings_path=args.classifier_embeddings,
        classifier_models_dir=args.classifier_models,
        classifier_threshold=args.classifier_threshold,
        ignored_nav_obstacle_categories=(
            ignored_nav_obstacle_categories if ignored_nav_obstacle_categories is not None else default_ignored
        ),
        remove_other_movable_objects=args.remove_other_movable_objects,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _build_config(args)
    run_data_collection(config)


if __name__ == "__main__":
    main()
