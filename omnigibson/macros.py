"""
Set of macros to use globally for OmniGibson. These are generally magic numbers that were tuned heuristically.

NOTE: This is generally decentralized -- the monolithic @settings variable is created here with some global values,
but submodules within OmniGibson may import this dictionary and add to it dynamically
"""
import os
import pathlib

from addict import Dict

# Initialize settings
macros = Dict()
gm = macros.globals

# Path (either relative to OmniGibson root directory or global absolute path) for data
gm.DATA_PATH = "data"

# Assets correspond to non-objects / scenes (e.g.: robots), and dataset incliudes objects + scene
gm.ASSET_PATH = f"{gm.DATA_PATH}/assets"
gm.DATASET_PATH = f"{gm.DATA_PATH}/og_dataset"
gm.KEY_PATH = f"{gm.DATA_PATH}/omnigibson.key"

# Whether to generate a headless or non-headless application upon OmniGibson startup
gm.HEADLESS = (os.getenv("OMNIGIBSON_HEADLESS", 'False').lower() in ('true', '1', 't'))

# Whether only the viewport should be shown in the GUI or not (if not, other peripherals are additionally shown)
# CANNOT be set at runtime
gm.GUI_VIEWPORT_ONLY = False

# Do not suppress known omni warnings / errors, and also put omnigibson in a debug state
# This includes extra information for things such as object sampling, and also any debug
# logging messages
gm.DEBUG = False

# Whether to print out disclaimers (i.e.: known failure cases resulting from Omniverse's current bugs / limitations)
gm.SHOW_DISCLAIMERS = False

# Whether to use omni's GPU dynamics
# This is necessary for certain features; e.g. particles (fluids / cloth)
gm.USE_GPU_DYNAMICS = False

# Whether to use high-fidelity rendering (this includes, e.g., isosurfaces)
gm.ENABLE_HQ_RENDERING = False

# Whether to use omni's flatcache feature or not (can speed up simulation)
gm.ENABLE_FLATCACHE = False

# Whether to use continuous collision detection or not (slower simulation, but can prevent
# objects from tunneling through each other)
gm.ENABLE_CCD = False

# Pairs setting -- USD default is 256 * 1024, physx default apparently is 32 * 1024.
gm.GPU_PAIRS_CAPACITY = 256 * 1024
# Aggregate pairs setting -- default is 1024, but is often insufficient for large scenes
gm.GPU_AGGR_PAIRS_CAPACITY = (2 ** 14) * 1024

# Maximum particle contacts allowed
gm.GPU_MAX_PARTICLE_CONTACTS = 1024 * 1024

# Whether to enable object state logic or not
gm.ENABLE_OBJECT_STATES = True

# Whether to enable transition rules or not
gm.ENABLE_TRANSITION_RULES = True

# Default settings for the omni UI viewer
gm.DEFAULT_VIEWER_WIDTH = 1280
gm.DEFAULT_VIEWER_HEIGHT = 720

# (Demo-purpose) Whether to activate Assistive Grasping mode for Cloth (it's handled differently from RigidBody)
gm.AG_CLOTH = False

# Forced light intensity for all DatasetObjects. None if the USD-provided intensities should be respected.
gm.FORCE_LIGHT_INTENSITY = 150000

# Forced roughness for all DatasetObjects. None if the USD-provided roughness maps should be respected.
gm.FORCE_ROUGHNESS = 0.7


# Create helper function for generating sub-dictionaries
def create_module_macros(module_path):
    """
    Creates a dictionary that can be populated with module macros based on the module's @module_path

    Args:
        module_path (str): Relative path from the package root directory pointing to the module. This will be parsed
            to generate the appropriate sub-macros dictionary, e.g., for module "dirty" in
            omnigibson/object_states_dirty.py, this would generate a dictionary existing at macros.object_states.dirty

    Returns:
        Dict: addict dictionary which can be populated with values
    """
    # Sanity check module path, make sure omnigibson/ is in the path
    module_path = pathlib.Path(module_path)
    omnigibson_path = pathlib.Path(__file__).parent

    # Trim the .py, and anything before and including omnigibson/, and split into its appropriate parts
    try:
        subsections = module_path.with_suffix("").relative_to(omnigibson_path).parts
    except ValueError:
        raise ValueError("module_path is expected to be a filepath including the omnigibson root directory, got: {module_path}!")

    # Create and return the generated sub-dictionary
    def _recursively_get_or_create_dict(dic, keys):
        # If no entry is in @keys, it returns @dic
        # Otherwise, checks whether the dictionary contains the first entry in @keys, if so, it grabs the
        # corresponding nested dictionary, otherwise, generates a new Dict() as the value
        # It then recurisvely calls this function with the new dic and the remaining keys
        if len(keys) == 0:
            return dic
        else:
            key = keys[0]
            if key not in dic:
                dic[key] = Dict()
            return _recursively_get_or_create_dict(dic=dic[key], keys=keys[1:])

    return _recursively_get_or_create_dict(dic=macros, keys=subsections)

