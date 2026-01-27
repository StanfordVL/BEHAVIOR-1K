#!/usr/bin/env python3
"""Train support and graspable object classifiers using SigLIP embeddings.

This script trains two logistic regression classifiers:
1. Support classifier: predicts if an object is a support surface (table, counter, etc.)
2. Graspable classifier: predicts if an object is graspable (mug, apple, etc.)

The classifiers use SigLIP text embeddings of object category names as input.

Usage:
    # Retrain with default settings
    python scripts/train_object_classifiers.py

    # Custom training examples
    python scripts/train_object_classifiers.py \
        --support-positives table counter desk shelf \
        --graspable-positives apple mug bottle cup
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


# Default positive examples for support objects
DEFAULT_SUPPORT_POSITIVES = [
    "table", "desk", "counter", "countertop", "shelf", "cabinet", "nightstand",
    "dining_table", "coffee_table", "kitchen_table", "breakfast_table", "side_table",
    "end_table", "console_table", "work_table", "writing_desk", "office_desk",
    "kitchen_counter", "bathroom_counter", "vanity", "dresser", "sideboard",
    "buffet", "credenza", "bookshelf", "rack", "stand", "bench", "ottoman",
    "storage_bench", "kitchen_island", "bar_counter", "workbench", "display_table",
]

# Default negative examples for support objects (things that are NOT supports)
DEFAULT_SUPPORT_NEGATIVES = [
    "chair", "sofa", "couch", "bed", "lamp", "plant", "tv", "computer", "phone",
    "book", "mug", "cup", "plate", "bowl", "bottle", "vase", "picture", "mirror",
    "clock", "fan", "heater", "door", "window", "curtain", "rug", "carpet",
    "pillow", "blanket", "towel", "clothes", "shoe", "bag", "toy", "ball",
    "wall", "floor", "ceiling", "stairs", "bathtub", "toilet", "sink",
]

# Default positive examples for graspable objects
DEFAULT_GRASPABLE_POSITIVES = [
    "apple", "orange", "banana", "mug", "cup", "bottle", "can", "jar", "box",
    "book", "phone", "remote", "toy", "ball", "pen", "pencil", "scissors",
    "plate", "bowl", "spoon", "fork", "knife", "glass", "tissue_box",
    "soap", "shampoo", "toothbrush", "hairbrush", "comb", "towel",
    "shoe", "bag", "wallet", "key", "flashlight", "camera", "calculator",
    "spray_bottle", "soap_bottle", "salt_shaker", "pepper_shaker",
    "avocado", "tomato", "lemon", "lime", "pear", "peach", "plum",
]

# Default negative examples for graspable objects (things that are NOT graspable)
DEFAULT_GRASPABLE_NEGATIVES = [
    "table", "desk", "chair", "sofa", "bed", "cabinet", "shelf", "counter",
    "wall", "floor", "ceiling", "door", "window", "stairs", "bathtub",
    "toilet", "sink", "oven", "refrigerator", "dishwasher", "washing_machine",
    "tv", "computer_monitor", "fireplace", "radiator", "air_conditioner",
    "car", "bicycle", "motorcycle", "tree", "bush", "fence", "pool",
    "couch", "armchair", "recliner", "wardrobe", "closet", "dresser",
]


def find_similar_categories(
    target: str,
    object_names: list[str],
    object_embeds: np.ndarray,
    name_to_idx: dict[str, int],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Find categories most similar to target using cosine similarity."""
    if target not in name_to_idx:
        # Try variations
        for variant in [target.lower(), target.replace(" ", "_"), target.replace("_", " ")]:
            if variant in name_to_idx:
                target = variant
                break
        else:
            return []

    target_embed = object_embeds[name_to_idx[target]]

    # Compute cosine similarities
    norms = np.linalg.norm(object_embeds, axis=1) * np.linalg.norm(target_embed)
    similarities = np.dot(object_embeds, target_embed) / (norms + 1e-8)

    # Get top-k (excluding self)
    top_indices = np.argsort(similarities)[::-1][:top_k + 1]

    results = []
    for idx in top_indices:
        if object_names[idx] != target:
            results.append((object_names[idx], similarities[idx]))
        if len(results) >= top_k:
            break

    return results


def expand_categories(
    seed_categories: list[str],
    object_names: list[str],
    object_embeds: np.ndarray,
    name_to_idx: dict[str, int],
    similarity_threshold: float = 0.85,
    max_per_seed: int = 10,
) -> set[str]:
    """Expand seed categories to include similar ones."""
    expanded = set()

    for seed in seed_categories:
        if seed in name_to_idx:
            expanded.add(seed)
        # Also try variations
        for variant in [seed.lower(), seed.replace(" ", "_"), seed.replace("_", " ")]:
            if variant in name_to_idx:
                expanded.add(variant)

        # Find similar categories
        similar = find_similar_categories(seed, object_names, object_embeds, name_to_idx, max_per_seed)
        for name, sim in similar:
            if sim >= similarity_threshold:
                expanded.add(name)

    return expanded


def prepare_training_data(
    positive_categories: list[str],
    negative_categories: list[str],
    object_names: list[str],
    object_embeds: np.ndarray,
    name_to_idx: dict[str, int],
    expand_positives: bool = True,
    expand_negatives: bool = True,
    similarity_threshold: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare training data from category lists."""

    # Expand categories if requested
    if expand_positives:
        positives = expand_categories(
            positive_categories, object_names, object_embeds, name_to_idx, similarity_threshold
        )
    else:
        positives = set()
        for cat in positive_categories:
            if cat in name_to_idx:
                positives.add(cat)
            for variant in [cat.lower(), cat.replace(" ", "_"), cat.replace("_", " ")]:
                if variant in name_to_idx:
                    positives.add(variant)

    if expand_negatives:
        negatives = expand_categories(
            negative_categories, object_names, object_embeds, name_to_idx, similarity_threshold
        )
    else:
        negatives = set()
        for cat in negative_categories:
            if cat in name_to_idx:
                negatives.add(cat)
            for variant in [cat.lower(), cat.replace(" ", "_"), cat.replace("_", " ")]:
                if variant in name_to_idx:
                    negatives.add(variant)

    # Remove overlap
    negatives = negatives - positives

    print(f"  Positives: {len(positives)}, Negatives: {len(negatives)}")

    # Build training data
    X = []
    y = []

    for cat in positives:
        if cat in name_to_idx:
            X.append(object_embeds[name_to_idx[cat]])
            y.append(1)

    for cat in negatives:
        if cat in name_to_idx:
            X.append(object_embeds[name_to_idx[cat]])
            y.append(0)

    return np.array(X), np.array(y)


def train_classifier(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> LogisticRegression:
    """Train a logistic regression classifier."""
    model = LogisticRegression(max_iter=1000, C=C, class_weight="balanced")
    model.fit(X, y)
    return model


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "vid2scene_policy/data_collection/classifiers/object_embeddings.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "vid2scene_policy/data_collection/classifiers"


def main():
    parser = argparse.ArgumentParser(description="Train object classifiers")
    parser.add_argument("--embeddings", type=str, default=str(DEFAULT_EMBEDDINGS),
                        help="Path to object_embeddings.npz")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for trained models")
    parser.add_argument("--support-positives", type=str, nargs="+",
                        help="Positive examples for support classifier")
    parser.add_argument("--support-negatives", type=str, nargs="+",
                        help="Negative examples for support classifier")
    parser.add_argument("--graspable-positives", type=str, nargs="+",
                        help="Positive examples for graspable classifier")
    parser.add_argument("--graspable-negatives", type=str, nargs="+",
                        help="Negative examples for graspable classifier")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                        help="Threshold for expanding categories")
    parser.add_argument("--no-expand", action="store_true",
                        help="Don't expand categories with similar ones")
    parser.add_argument("--regularization", type=float, default=1.0,
                        help="Regularization strength (C parameter)")
    args = parser.parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    data = np.load(args.embeddings)
    object_names = list(data["object_names"])
    object_embeds = data["object_embeds"]
    name_to_idx = {name: i for i, name in enumerate(object_names)}
    print(f"Loaded {len(object_names)} categories with {object_embeds.shape[1]}-dim embeddings")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train support classifier
    print("\nTraining support classifier...")
    support_positives = args.support_positives or DEFAULT_SUPPORT_POSITIVES
    support_negatives = args.support_negatives or DEFAULT_SUPPORT_NEGATIVES

    X_support, y_support = prepare_training_data(
        support_positives, support_negatives,
        object_names, object_embeds, name_to_idx,
        expand_positives=not args.no_expand,
        expand_negatives=not args.no_expand,
        similarity_threshold=args.similarity_threshold,
    )

    support_model = train_classifier(X_support, y_support, args.regularization)

    # Cross-validation
    cv_scores = cross_val_score(support_model, X_support, y_support, cv=5)
    print(f"  CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Save support classifier
    with open(output_dir / "support_classifier.pkl", "wb") as f:
        pickle.dump(support_model, f)
    print(f"  Saved to {output_dir / 'support_classifier.pkl'}")

    # Train graspable classifier
    print("\nTraining graspable classifier...")
    graspable_positives = args.graspable_positives or DEFAULT_GRASPABLE_POSITIVES
    graspable_negatives = args.graspable_negatives or DEFAULT_GRASPABLE_NEGATIVES

    X_graspable, y_graspable = prepare_training_data(
        graspable_positives, graspable_negatives,
        object_names, object_embeds, name_to_idx,
        expand_positives=not args.no_expand,
        expand_negatives=not args.no_expand,
        similarity_threshold=args.similarity_threshold,
    )

    graspable_model = train_classifier(X_graspable, y_graspable, args.regularization)

    # Cross-validation
    cv_scores = cross_val_score(graspable_model, X_graspable, y_graspable, cv=5)
    print(f"  CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Save graspable classifier
    with open(output_dir / "graspable_classifier.pkl", "wb") as f:
        pickle.dump(graspable_model, f)
    print(f"  Saved to {output_dir / 'graspable_classifier.pkl'}")

    # Save training config
    config = {
        "support_positives": support_positives,
        "support_negatives": support_negatives,
        "graspable_positives": graspable_positives,
        "graspable_negatives": graspable_negatives,
        "similarity_threshold": args.similarity_threshold,
        "expanded": not args.no_expand,
        "regularization": args.regularization,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
