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
    assert gm.ENABLE_OMNI_PARTICLES, f"Particles must be enabled in macros.py in order to use this demo!"

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "EmptyScene",
        }
    }

    env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)
    og.sim.stop()

    scale_carpet = np.ones(3) * 1.0
    pos_carpet = np.array([0., 0., 0.5])
    quat_carpet = np.array([0., 0., 0., 1.])

    carpet = DatasetObject(
        prim_path="/World/ClothCarpet",
        usd_path=os.path.join(og_dataset_path, "objects", "carpet", "carpet_0", "usd", "carpet_0.usd"),
        scale=scale_carpet,
        prim_type=PrimType.CLOTH,
        abilities={"foldable": {}},
    )
    og.sim.import_object(carpet)
    carpet.set_position_orientation(position=pos_carpet, orientation=quat_carpet)

    scale_dishtowel = np.ones(3) * 5.0
    pos_dishtowel = np.array([1.0, 1.0, 0.5])
    quat_dishtowel = np.array([0., 0., 0., 1.])

    dishtowel= DatasetObject(
        prim_path="/World/DishTowel_Red",
        usd_path=os.path.join(og_dataset_path, "objects", "dishtowel", "Tag_Dishtowel_Basket_Weave_Red", "usd", "Tag_Dishtowel_Basket_Weave_Red.usd"),
        scale=scale_dishtowel,
        prim_type=PrimType.CLOTH,
        abilities={"foldable": {}},
    )
    og.sim.import_object(dishtowel)
    dishtowel.set_position_orientation(position=pos_dishtowel, orientation=quat_dishtowel)

    scale_tshirt = np.ones(3) * 0.05
    pos_tshirt = np.array([-1., 1., 0.5])
    quat_tshirt = np.array([0.7071, 0., 0.7071, 0.])

    tshirt = DatasetObject(
        prim_path="/World/ClothTShirt",
        usd_path=os.path.join(og_dataset_path, "objects", "t-shirt", "t-shirt_000", "usd", "t-shirt_000.usd"),
        scale=scale_tshirt,
        prim_type=PrimType.CLOTH,
        abilities={"foldable": {}},
    )
    og.sim.import_object(tshirt)
    tshirt.set_position_orientation(position=pos_tshirt, orientation=quat_tshirt)

    og.sim.play()

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

        flag_area_reduction, flag_diagonal_reduction = tshirt.states[Folded].check_projection_area_and_diagonal()
        flag_smoothness = tshirt.states[Folded].check_smoothness()
        folded = flag_area_reduction and flag_diagonal_reduction and flag_smoothness
        info += " || tshirt: [folded] %d [A] %d [D] %d [S] %d" % (folded, flag_area_reduction, flag_diagonal_reduction, flag_smoothness)

        print(info)
        steps += 1

    env.close()


if __name__ == "__main__":
    main()
