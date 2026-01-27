"""Collect pick-and-place episodes for policy training.

Usage:
    python collect_data.py --scene Rs_int --episodes 10
    python collect_data.py --dataset spoc --scene train_505 --episodes 5
    python collect_data.py --object-filter whitelist --scene Rs_int

Options:
    --scene NAME          Scene model (default: Rs_int)
    --dataset NAME        Dataset: behavior-1k-assets, spoc (default: behavior-1k-assets)
    --episodes N          Number of episodes (default: 100)
    --output DIR          Output directory (default: ./lerobot_datasets)
    --repo-id ID          Repository ID (default: {scene}_stretch_pick_place)
    --max-nav-steps N     Max navigation steps (default: 1500)
    --object-filter MODE  whitelist or classifier (default: classifier)
    --classifier-threshold  Classification threshold (default: 0.5)
"""
import argparse

from vid2scene_policy.data_collection.config import DataCollectionConfig
from vid2scene_policy.data_collection.episode import run_data_collection


def main():
    parser = argparse.ArgumentParser(description="Collect pick-and-place data")
    parser.add_argument("--scene", default="Rs_int", help="Scene model name")
    parser.add_argument("--dataset", default="behavior-1k-assets", help="Dataset name (behavior-1k-assets or spoc)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--output", default="./lerobot_datasets", help="Output directory")
    parser.add_argument("--repo-id", default=None, help="Repository ID (default: {scene}_stretch_pick_place)")
    parser.add_argument("--max-nav-steps", type=int, default=1500, help="Max navigation steps per episode")
    parser.add_argument("--object-filter", default="classifier", choices=["whitelist", "classifier"],
                        help="Object filter method (default: classifier)")
    parser.add_argument("--classifier-embeddings", default=None, help="Path to classifier embeddings")
    parser.add_argument("--classifier-models", default=None, help="Path to classifier models directory")
    parser.add_argument("--classifier-threshold", type=float, default=0.5, help="Classifier threshold")
    args = parser.parse_args()

    repo_id = args.repo_id or f"{args.scene}_stretch_pick_place"

    config = DataCollectionConfig(
        scene_model=args.scene,
        dataset_name=args.dataset,
        num_episodes=args.episodes,
        output_dir=args.output,
        repo_id=repo_id,
        whitelist_graspable_path="configs/whitelist_graspable_objects.json",
        whitelist_support_path="configs/whitelist_support_objects.json",
        max_navigation_steps=args.max_nav_steps,
        max_grasp_steps=100,
        max_place_steps=100,
        object_filter_method=args.object_filter,
        classifier_embeddings_path=args.classifier_embeddings,
        classifier_models_dir=args.classifier_models,
        classifier_threshold=args.classifier_threshold,
    )
    run_data_collection(config)


if __name__ == "__main__":
    main()
