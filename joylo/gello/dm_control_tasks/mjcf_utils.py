from pathlib import Path
from typing import List

from dm_control import mjcf

# Path to the root of the project.
_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent

# Path to the Menagerie submodule.
MENAGERIE_ROOT: Path = _PROJECT_ROOT / "third_party" / "mujoco_menagerie"


def safe_find_all(
    root: mjcf.RootElement,
    feature_name: str,
    immediate_children_only: bool = False,
    exclude_attachments: bool = False,
) -> List[mjcf.Element]:
    """Find all given elements or throw an error if none are found."""
    features = root.find_all(feature_name, immediate_children_only, exclude_attachments)
    if not features:
        raise ValueError(f"No {feature_name} found in the MJCF model.")
    return features
