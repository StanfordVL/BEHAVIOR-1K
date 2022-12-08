from omnigibson import og_dataset_path
from omnigibson.scenes.empty_scene import EmptyScene
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Folded
from omnigibson.macros import gm

import os
import numpy as np
import logging

import omnigibson as og


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of cloth objects that can potentially be folded.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    assert gm.ENABLE_OBJECT_STATES, f"Object states must be enabled in macros.py in order to use this demo!"
    assert gm.USE_GPU_DYNAMICS, f"GPU dynamics must be enabled in macros.py in order to use this demo!"

    # Create the scene config to load -- empty scene + custom cloth object
    cfg = {
        "scene": {
            "type": "EmptyScene",
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "carpet",
                "category": "carpet",
                "model": "carpet_0",
                "prim_type": PrimType.CLOTH,
                "abilities": {"foldable": {}},
                "position": [0, 0, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "dishtowel",
                "category": "dishtowel",
                "model": "Tag_Dishtowel_Basket_Weave_Red",
                "prim_type": PrimType.CLOTH,
                "scale": 5.0,
                "abilities": {"foldable": {}},
                "position": [1, 1, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "shirt",
                "category": "t-shirt",
                "model": "t-shirt_000",
                "prim_type": PrimType.CLOTH,
                "scale": 0.05,
                "abilities": {"foldable": {}},
                "position": [-1, 1, 0.5],
                "orientation": [0.7071, 0., 0.7071, 0.],
            },
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)

    # Grab object references
    carpet = env.scene.object_registry("name", "carpet")
    dishtowel = env.scene.object_registry("name", "dishtowel")
    shirt = env.scene.object_registry("name", "shirt")

    max_steps = 100 if short_exec else -1
    steps = 0

    # Criterion #1: the area of the convex hull of the projection of points onto the x-y plane should be reduced
    # Criterion #2: the diagonal of the convex hull of the projection of points onto the x-y plane should be reduced
    # Criterion #3: the face normals of the cloth should mostly point along the z-axis
    while steps != max_steps:
        og.sim.step()

        flag_area_reduction, flag_diagonal_reduction = carpet.states[Folded].check_projection_area_and_diagonal()
        flag_smoothness = carpet.states[Folded].check_smoothness()
        folded = flag_area_reduction and flag_diagonal_reduction and flag_smoothness
        info = 'carpet: [folded] %d [A] %d [D] %d [S] %d' % (folded, flag_area_reduction, flag_diagonal_reduction, flag_smoothness)

        flag_area_reduction, flag_diagonal_reduction = dishtowel.states[Folded].check_projection_area_and_diagonal()
        flag_smoothness = dishtowel.states[Folded].check_smoothness()
        folded = flag_area_reduction and flag_diagonal_reduction and flag_smoothness
        info += " || dishtowel: [folded] %d [A] %d [D] %d [S] %d" % (folded, flag_area_reduction, flag_diagonal_reduction, flag_smoothness)

        flag_area_reduction, flag_diagonal_reduction = shirt.states[Folded].check_projection_area_and_diagonal()
        flag_smoothness = shirt.states[Folded].check_smoothness()
        folded = flag_area_reduction and flag_diagonal_reduction and flag_smoothness
        info += " || tshirt: [folded] %d [A] %d [D] %d [S] %d" % (folded, flag_area_reduction, flag_diagonal_reduction, flag_smoothness)

        print(info)
        steps += 1

    # Shut down env at the end
    env.close()


if __name__ == "__main__":
    main()
