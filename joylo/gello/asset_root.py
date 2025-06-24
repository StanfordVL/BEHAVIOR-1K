import os
from pathlib import Path


ASSET_ROOT = str(Path(__file__).parent.parent.absolute() / "assets")
assert os.path.exists(ASSET_ROOT), f"Asset root {ASSET_ROOT} does not exist"
