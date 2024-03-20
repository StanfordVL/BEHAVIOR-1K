import argparse
import logging
import os
import numpy as np

from pathlib import Path
import omnigibson as og
from omnigibson.objects import USDObject, LightObject, DatasetObject
from omnigibson.utils.ui_utils import choose_from_options
import omnigibson.utils.transform_utils as T
from omnigibson.utils.render_utils import make_glass
import omnigibson.lazy as lazy
import imageio.v2 as imageio
from omnigibson.utils.python_utils import clear as clear_pu


def set_pt():
    og.app.config["renderer"] = "PathTracing"
    og.app._set_render_settings()
    for i in range(10):
        og.sim.step()


def set_rt():
    og.app.config["renderer"] = "RayTracingLighting"
    og.app.config["anti_aliasing"] = 0
    og.app._set_render_settings()
    for i in range(10):
        og.sim.step()


def make_opaque(obj):
    for link in obj.links.values():
        for vm in link.visual_meshes.values():
            if "material:binding" in vm.prim.GetPropertyNames():
                for target in vm.prim.GetProperty("material:binding").GetTargets():
                    from omni.isaac.core.utils.prims import get_prim_at_path

                    mat = get_prim_at_path(target)
                    if mat:
                        shader = lazy.omni.usd.get_shader_from_material(prim=get_prim_at_path(target))
                        if shader:
                            # shader.GetInput("enable_opacity_texture").Set(False)
                            shader.GetInput("enable_opacity").Set(False)
                            # shader.GetInput("reflection_roughness_constant").Set(0.5)
                            shader.GetInput("reflection_roughness_texture_influence").Set(0.0)
                            # shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(0.3)


SAVE_ROOT_PATH = f"/scr/home/yinhang/obj_videos"


def main(random_selection=False, headless=False, short_exec=False):
    """
    Visualizes object as specified by its USD path, @usd_path. If None if specified, will instead
    result in an object selection from omnigibson's object dataset
    """
    # log.info("*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # # Assuming that if random_selection=True, headless=True, short_exec=True, we are calling it from tests and we
    # # do not want to parse args (it would fail because the calling function is pytest "testfile.py")
    # usd_path = None
    # if not (random_selection and headless and short_exec):
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument(
    #         "--usd_path",
    #         default=None,
    #         help="USD Model to load",
    #     )
    #     args = parser.parse_args()
    #     usd_path = args.usd_path

    if og.sim is not None:
        og.sim.stop()

    config = {
        "scene": {
            "type": "Scene",
            "floor_plane_visible": False,
            "use_skybox": False,
        },
    }

    env = og.Environment(configs=config)

    og.sim.play()
    set_rt()

    og.sim.viewer_width = 1280
    og.sim.viewer_height = 1280

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.00913503, -1.95750906, 1.36407314]),
        orientation=np.array([0.63500641, 0.0, 0.0, 0.77250687]),
    )
    og.sim.viewer_camera.add_modality("seg_semantic")

    # Create a light object
    light0 = LightObject(
        prim_path="/World/sphere_light0",
        light_type="Sphere",
        name="sphere_light0",
        radius=0.01,
        intensity=3000,
    )
    og.sim.import_object(light0)
    light0.set_position(np.array([-2.0, -2.0, 2.0]))

    light1 = LightObject(
        prim_path="/World/sphere_light1",
        light_type="Sphere",
        name="sphere_light1",
        radius=0.01,
        intensity=3000,
    )
    og.sim.import_object(light1)
    light1.set_position(np.array([-2.0, 2.0, 2.0]))

    og.sim.stop()
    og.sim.step()
    og.sim.play()
    for i in range(5):
        og.sim.step()

    # 'espresso_machine': 'tysouc',
    # 'toaster_oven': 'ctrngh',
    # 'fridge': 'petcxr',

    obj_dict = {
        # 'air_conditioner': 'pekain',
        # 'bicycle': 'xjumcf',
        # 'electric_mixer': 'ceaeqf',
        # 'electric_fan': 'lclkju',
        # 'exercise_bike': 'eponae',
        # 'lawn_mower': 'bterwo',
        # 'microwave': 'bfbeeb',
        # 'printer': 'icxhbx',
        # 'fridge': 'petcxr',
        # 'saxophone': 'kladff',
        # 'vacuum': 'bdmsbr',
        # 'power_drill': 'hdvnxd',
        # 'motorcycle': 'aocuum',
        # 'hoodie': 'agftpm',
        # 'swivel_chair': 'iiihwn',
        # 'chicken': 'nppsmz',
        # 'pot_plant': 'udqjui',
        # 'coffee_table': 'aoojzy',
        # 'car': 'xxsgpq',
        # 'dress': 'jmujjo',
        # 'floor_lamp': 'ogolip',
        # 'french_press': 'zidmyo',
        # 'gaming_table': 'ipdvzo',
        # 'goggles': 'wszdxi',
    }

    dataset_path = "/scr/OmniGibson/omnigibson/data/og_dataset"
    for obj_category, obj_model in obj_dict.items():
        category_path = os.path.join(dataset_path, "objects", obj_category)
        if os.path.isdir(category_path) and obj_category not in ["walls", "floors", "ceilings"]:
            print(f"RENDERING CATEGORY/MODEL {obj_category} / {obj_model}...")

            usd_path = DatasetObject.get_usd_path(category=obj_category, model=obj_model)
            usd_path = usd_path.replace(".usd", ".encrypted.usd")
            from omnigibson.utils.asset_utils import decrypted

            with decrypted(usd_path) as fpath:
                stage = lazy.pxr.Usd.Stage.Open(fpath)
                prim = stage.GetDefaultPrim()
                bounding_box = np.array(prim.GetAttribute("ig:nativeBB").Get())

            MAX_BBOX = 1.0
            # Then get the appropriate bounding box such that the object fits in a
            # MAX_BBOX cube.
            scale = MAX_BBOX / np.max(bounding_box)

            # Import the desired object
            obj = USDObject(
                prim_path=f"/World/{obj_category}_{obj_model}",
                name=f"{obj_category}_{obj_model}",
                usd_path=f"{dataset_path}/objects/{obj_category}/{obj_model}/usd/{obj_model}.usd",
                encrypted=True,
                category=obj_category,
                # model=obj_model,
                visual_only=True,
                scale=scale,
            )
            og.sim.import_object(obj)

            for _ in range(10):
                og.sim.step()

            if "glass" in obj_category or "glass" in obj_model:
                make_glass(obj)
            elif "window" in obj_category or "window" in obj_model:
                pass
            else:
                make_opaque(obj)

            # # Standardize the scale of the object so it fits in a [1,1,1] box
            # extents = obj.aabb_extent
            # for i in range(20):
            #     og.sim.step()
            # og.sim.stop()
            # obj.scale = (np.ones(3) / extents).min()
            # og.sim.play()
            # # obj.scale = (np.ones(3) / extents).min()
            # # breakpoint()
            # for i in range(20):
            #     og.sim.step()

            # Move the object so that its center is at [0, 0, 1]
            center_offset = obj.get_position() - obj.aabb_center + np.array([0, 0, 1.0])
            # center_offset = -obj.aabb_center + np.array([0, 0, 1.0])
            obj.set_position(center_offset)

            # # Allow the user to easily move the camera around
            # og.sim.enable_viewer_camera_teleoperation()

            # Rotate the object in place
            steps_per_rotate = 360
            steps_per_joint = steps_per_rotate / 10
            max_steps = steps_per_rotate  # 100 if short_exec else 10000

            for i in range(5):
                og.sim.step()

            save_dir = f"{SAVE_ROOT_PATH}/{obj_category}"
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            video_writer = imageio.get_writer(f"{save_dir}/{obj_category}_{obj_model}.mp4", fps=60)
            for i in range(max_steps):
                z_angle = 2 * np.pi * (i % steps_per_rotate) / steps_per_rotate
                quat = T.euler2quat(np.array([0, 0, z_angle]))
                pos = T.quat2mat(quat) @ center_offset
                if obj.n_dof > 0:
                    frac = (i % steps_per_joint) / steps_per_joint
                    j_frac = -1.0 + 2.0 * frac if (i // steps_per_joint) % 2 == 0 else 1.0 - 2.0 * frac
                    obj.set_joint_positions(positions=j_frac * np.ones(obj.n_dof), normalized=True, drive=False)
                    obj.keep_still()
                obj.set_position_orientation(position=pos, orientation=quat)
                og.sim.step()
                obs = og.sim.viewer_camera.get_obs()[0]
                rgb = obs["rgb"][:, :, :-1]
                ss = obs["seg_semantic"]
                h, w = ss.shape
                black_mask = np.tile(np.where(np.linalg.norm(rgb, axis=-1, keepdims=True) == 0, 1.0, 0.0), (1, 1, 3))
                from omnigibson.utils.constants import semantic_class_name_to_id

                seg_mask = np.tile(
                    np.where(ss == semantic_class_name_to_id()[obj_category], 1.0, 0.0).reshape((h, w, 1)), (1, 1, 3)
                )
                img = np.where(seg_mask == 0, 255, rgb)
                # img = np.where(black_mask != 0, 255, rgb)
                video_writer.append_data(img)

            video_writer.close()
            og.sim.remove_object(obj)
            clear_pu()
            og.sim.stop()
            og.sim.step()
            og.sim.play()
            for i in range(5):
                og.sim.step()


if __name__ == "__main__":
    main()
