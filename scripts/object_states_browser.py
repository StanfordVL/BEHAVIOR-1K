import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import choose_from_options, KeyboardEventHandler
from PIL import Image, ImageDraw
from omnigibson.utils.constants import semantic_class_name_to_id, semantic_class_id_to_name
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import *
from omnigibson.utils.constants import PrimType, ParticleModifyCondition, ParticleModifyMethod
from omnigibson.systems import *
import numpy as np

from utils import og_test, get_random_pose, place_objA_on_objB_bbox, place_obj_on_floor_plane, SYSTEM_EXAMPLES

# Make sure object states, GPU dynamics, and transition rules are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = True

TEMP_RELATED_ABILITIES = {"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}}

SYSTEM_EXAMPLES = {
    "water": FluidSystem,
    "white_rice": GranularSystem,
    "diced__apple": MacroPhysicalParticleSystem,
    "stain": MacroVisualParticleSystem,
}

num_objs = 0


def get_obj_cfg(
    name, category, model, prim_type=PrimType.RIGID, scale=None, bounding_box=None, abilities=None, visual_only=False
):
    global num_objs
    num_objs += 1
    return {
        "type": "DatasetObject",
        "fit_avg_dim_volume": scale is None and bounding_box is None,
        "name": name,
        "category": category,
        "model": model,
        "prim_type": prim_type,
        "position": [150, 150, 150 + num_objs * 5],
        "scale": scale,
        "bounding_box": bounding_box,
        "abilities": abilities,
        "visual_only": visual_only,
    }


def main():
    background_id = semantic_class_name_to_id()["background"]
    floor_id = semantic_class_name_to_id()["floors"]

    if og.sim is not None:
        og.sim.stop()

    config = {
        "scene": {
            "type": "Scene",
        },
        "objects": [
            get_obj_cfg("breakfast_table", "breakfast_table", "skczfi"),
            # get_obj_cfg("bottom_cabinet", "bottom_cabinet", "immwzb"),
            # get_obj_cfg("dishtowel", "dishtowel", "dtfspn", prim_type=PrimType.CLOTH, abilities={"cloth": {}}),
            # get_obj_cfg("carpet", "carpet", "ctclvd", prim_type=PrimType.CLOTH, abilities={"cloth": {}}),
            # get_obj_cfg("bowl", "bowl", "ajzltc"),
            # get_obj_cfg("bagel", "bagel", "zlxkry", abilities=TEMP_RELATED_ABILITIES),
            # get_obj_cfg("cookable_dishtowel", "dishtowel", "dtfspn", prim_type=PrimType.CLOTH, abilities={**TEMP_RELATED_ABILITIES, **{"cloth": {}}}),
            # get_obj_cfg("microwave", "microwave", "hjjxmi"),
            # get_obj_cfg("stove", "stove", "yhjzwg"),
            # get_obj_cfg("fridge", "fridge", "dszchb"),
            # get_obj_cfg("plywood", "plywood", "fkmkqa", abilities={"flammable": {}}),
            # get_obj_cfg("shelf_back_panel", "shelf_back_panel", "gjsnrt", abilities={"attachable": {}}),
            # get_obj_cfg("shelf_shelf", "shelf_shelf", "ymtnqa", abilities={"attachable": {}}),
            # get_obj_cfg("shelf_baseboard", "shelf_baseboard", "hlhneo", abilities={"attachable": {}}),
            # get_obj_cfg("bracelet", "bracelet", "thqqmo"),
            # get_obj_cfg("oyster", "oyster", "enzocs"),
            # get_obj_cfg("sink", "sink", "egwapq", scale=np.ones(3)),
            # get_obj_cfg("stockpot", "stockpot", "dcleem", abilities={"fillable": {}, "heatable": {}}),
            # get_obj_cfg("applier_dishtowel", "dishtowel", "dtfspn", abilities={"particleApplier": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}}),
            # get_obj_cfg("remover_dishtowel", "dishtowel", "dtfspn", abilities={"particleRemover": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}}),
            # get_obj_cfg("spray_bottle", "spray_bottle", "asztxi", visual_only=True, abilities={"toggleable": {}, "particleApplier": {"method": ParticleModifyMethod.PROJECTION, "conditions": {"water": [(ParticleModifyCondition.TOGGLEDON, True)]}}}),
            # get_obj_cfg("vacuum", "vacuum", "bdmsbr", visual_only=True, abilities={"toggleable": {}, "particleRemover": {"method": ParticleModifyMethod.PROJECTION, "conditions": {"water": [(ParticleModifyCondition.TOGGLEDON, True)]}}}),
            # get_obj_cfg("blender", "blender", "cwkvib", bounding_box=[0.316, 0.318, 0.649], abilities={"fillable": {}, "toggleable": {}, "heatable": {}}),
            # get_obj_cfg("oven", "oven", "cgtaer", bounding_box=[0.943, 0.837, 1.297]),
            # get_obj_cfg("baking_sheet", "baking_sheet", "yhurut", bounding_box=[0.41607812, 0.43617093, 0.02281223]),
            # get_obj_cfg("bagel_dough", "bagel_dough", "iuembm", scale=np.ones(3) * 0.8),
            # get_obj_cfg("raw_egg", "raw_egg", "ydgivr"),
            # get_obj_cfg("scoop_of_ice_cream", "scoop_of_ice_cream", "dodndj", bounding_box=[0.076, 0.077, 0.065]),
            # get_obj_cfg("food_processor", "food_processor", "gamkbo"),
            # get_obj_cfg("electric_mixer", "electric_mixer", "qornxa"),
            # get_obj_cfg("another_raw_egg", "raw_egg", "ydgivr"),
            # get_obj_cfg("chicken", "chicken", "nppsmz", scale=np.ones(3) * 0.7),
            # get_obj_cfg("tablespoon", "tablespoon", "huudhe"),
            # get_obj_cfg("swiss_cheese", "swiss_cheese", "hwxeto"),
            # get_obj_cfg("apple", "apple", "agveuv"),
            # get_obj_cfg("table_knife", "table_knife", "jxdfyy"),
            # get_obj_cfg("half_apple", "half_apple", "sguztn"),
            # get_obj_cfg("washer", "washer", "dobgmu"),
            # get_obj_cfg("carpet_sweeper", "carpet_sweeper", "xboreo"),
        ],
        # "robots": [
        #         {
        #             "type": "Fetch",
        #             "obs_modalities": ["seg_semantic", "seg_instance", "seg_instance_id"],
        #             "position": [150, 150, 100],
        #             "orientation": [0, 0, 0, 1],
        #         }
        # ]
    }

    states_to_objects = {
        # 'OnTop': ['breakfast_table', 'bowl', 'dishtowel'],
        # 'Inside': ['bottom_cabinet', 'bowl', 'dishtowel'],
        # 'Under': ['breakfast_table', 'bowl', 'dishtowel'],
        # 'Touching': ['breakfast_table', 'bowl', 'dishtowel'],
        # 'ContactBodies': ['breakfast_table', 'bowl', 'dishtowel'],
        # 'NextTo': ['bottom_cabinet', 'bowl', 'dishtowel'],
        # 'Overlaid': ['breakfast_table', 'carpet'],
        # 'Pose': ['breakfast_table', 'dishtowel'],
        # 'AABB': ['breakfast_table', 'dishtowel'],
        # 'Adjacency': ['bottom_cabinet', 'bowl', 'dishtowel'],
        # 'Temperature': ['microwave', 'stove', 'fridge', 'plywood', 'bagel', 'dishtowel'],
        # 'MaxTemperature': ['bagel', 'dishtowel'],
        # 'HeatSourceOrSink': ['microwave', 'stove', 'fridge'],
        # 'Cooked': ['bagel', 'dishtowel'],
        # 'Burnt': ['bagel', 'dishtowel'],
        # 'Frozen': ['bagel', 'dishtowel'],
        # 'Heated': ['bagel', 'dishtowel'],
        # 'OnFire': ['plywood'],
        "ToggledOn": ["stove"],
        # 'AttachedTo': ['shelf_back_panel', 'shelf_shelf', 'shelf_baseboard'],
        "ParticleSource": ["sink"],
        # 'ParticleSink': ['sink'],
        # 'ParticleApplier': ['breakfast_table', 'spray_bottle', 'applier_dishtowel'],
        # 'ParticleRemover': ['breakfast_table', 'vacuum', 'remover_dishtowel'],
        "Saturated": ["remover_dishtowel"],
        # 'Open': ['microwave', 'bottom_cabinet'],
        # 'Folded': ['carpet'],
        # 'Unfolded': ['carpet'],
        "Draped": ["breakfast_table", "carpet"],
        "Filled": ["stockpot"],
        "Contains": ["stockpot"],
        "Covered": ["bracelet", "bowl", "microwave"],
        "IsGrasping": [],
        "ObjectsInFOVOfRobot": [],
    }

    env = og.Environment(configs=config)

    cam_mover = og.sim.enable_viewer_camera_teleoperation()
    og.sim.viewer_camera.add_modality("seg_semantic")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=lambda: terminate.__setitem__(0, True),
    )

    for state, objects in states_to_objects.items():

        print("Load objects and set states")

        print(f"We are now loading state {state}, with objects {objects}")

        breakpoint()

        screenshot = og.sim.viewer_camera.get_obs()[0]["rgb"]
        semantic_image = og.sim.viewer_camera.get_obs()[0]["seg_semantic"]

        # remove background
        screenshot[semantic_image == background_id] = 255
        screenshot[semantic_image == floor_id] = 255

        img = Image.fromarray(screenshot)
        img.save(f"/scr/home/yinhang/object_states_asset/{state}.png")

        og.sim.scene.reset()

    # env.close()
    og.sim.clear()

    env.close()


if __name__ == "__main__":
    main()
