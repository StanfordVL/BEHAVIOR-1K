import logging
import pickle
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Default classifier resource directory.
DEFAULT_CLASSIFIERS_DIR = Path(__file__).parent / "classifiers"


class ObjectClassifier:
    """Classifier wrapper over precomputed SigLIP embeddings."""

    def __init__(self, embeddings_path: str | Path, models_dir: str | Path | None = None):
        data = np.load(embeddings_path)
        self.object_names = list(data["object_names"])
        self.object_embeds = data["object_embeds"]
        self.name_to_idx = {name: i for i, name in enumerate(self.object_names)}

        self.support_classifier = None
        self.graspable_classifier = None

        if models_dir is not None:
            self.load_models(models_dir)

    def load_models(self, models_dir: str | Path):
        models_dir = Path(models_dir)

        support_path = models_dir / "support_classifier.pkl"
        if support_path.exists():
            with open(support_path, "rb") as f:
                self.support_classifier = pickle.load(f)

        graspable_path = models_dir / "graspable_classifier.pkl"
        if graspable_path.exists():
            with open(graspable_path, "rb") as f:
                self.graspable_classifier = pickle.load(f)

    def get_embedding(self, category: str) -> np.ndarray | None:
        if category in self.name_to_idx:
            return self.object_embeds[self.name_to_idx[category]]

        category_lower = category.lower()
        if category_lower in self.name_to_idx:
            return self.object_embeds[self.name_to_idx[category_lower]]

        for variant in [category.replace("_", " "), category.replace(" ", "_")]:
            if variant in self.name_to_idx:
                return self.object_embeds[self.name_to_idx[variant]]
            if variant.lower() in self.name_to_idx:
                return self.object_embeds[self.name_to_idx[variant.lower()]]

        return None

    def is_support(self, category: str, threshold: float = 0.5) -> bool:
        if self.support_classifier is None:
            raise RuntimeError("Support classifier not loaded")

        embedding = self.get_embedding(category)
        if embedding is None:
            return False

        prob = self.support_classifier.predict_proba(embedding.reshape(1, -1))[0, 1]
        return prob >= threshold

    def is_graspable(self, category: str, threshold: float = 0.5) -> bool:
        if self.graspable_classifier is None:
            raise RuntimeError("Graspable classifier not loaded")

        embedding = self.get_embedding(category)
        if embedding is None:
            return False

        prob = self.graspable_classifier.predict_proba(embedding.reshape(1, -1))[0, 1]
        return prob >= threshold

    def get_support_prob(self, category: str) -> float:
        if self.support_classifier is None:
            raise RuntimeError("Support classifier not loaded")

        embedding = self.get_embedding(category)
        if embedding is None:
            return 0.0

        return self.support_classifier.predict_proba(embedding.reshape(1, -1))[0, 1]

    def get_graspable_prob(self, category: str) -> float:
        if self.graspable_classifier is None:
            raise RuntimeError("Graspable classifier not loaded")

        embedding = self.get_embedding(category)
        if embedding is None:
            return 0.0

        return self.graspable_classifier.predict_proba(embedding.reshape(1, -1))[0, 1]


def get_object_filter(
    method: str = "whitelist",
    embeddings_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    support_whitelist: list[str] | None = None,
    graspable_whitelist: list[str] | None = None,
    threshold: float = 0.5,
) -> tuple[Callable[[str], bool], Callable[[str], bool]]:
    """Create support and graspable category filter callables."""
    if method == "whitelist":
        if support_whitelist is None or graspable_whitelist is None:
            raise ValueError("Whitelists required for whitelist method")

        support_set = set(s.lower() for s in support_whitelist)
        graspable_set = set(g.lower() for g in graspable_whitelist)

        def is_support(category: str) -> bool:
            return category.lower() in support_set

        def is_graspable(category: str) -> bool:
            return category.lower() in graspable_set

        return is_support, is_graspable

    elif method == "classifier":
        if embeddings_path is None:
            embeddings_path = DEFAULT_CLASSIFIERS_DIR / "object_embeddings.npz"
        if models_dir is None:
            models_dir = DEFAULT_CLASSIFIERS_DIR

        classifier = ObjectClassifier(embeddings_path, models_dir)
        logger.info("Loaded object classifiers from %s", models_dir)

        def is_support(category: str) -> bool:
            return classifier.is_support(category, threshold)

        def is_graspable(category: str) -> bool:
            return classifier.is_graspable(category, threshold)

        return is_support, is_graspable

    else:
        raise ValueError(f"Unknown method: {method}")
