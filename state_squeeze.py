from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Folded, Unfolded
from omnigibson.macros import gm
import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
import json
import torch
import time
from omnigibson.prims.xform_prim import XFormPrim
import omnigibson.utils.transform_utils as T

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True

# Set up the 4 walls
def generate_box(box_half_extent=torch.tensor([1, 1, 1], dtype=torch.float32), visualize_wall=False):
    # Temp function to generate the walls for squeezing the cloth
    # The floor plane already exists
    # We just need to generate the side planes
    plane_centers = (
        torch.tensor(
            [
                [1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                [0, -1, 1],
            ]
        )
        * box_half_extent
    )
    plane_prims = []
    plane_motions = []
    for i, pc in enumerate(plane_centers):

        plane = lazy.omni.isaac.core.objects.ground_plane.GroundPlane(
            prim_path=f"/World/plane_{i}",
            name=f"plane_{i}",
            z_position=0,
            size=box_half_extent[2].item(),
            color=None,
            visible=visualize_wall,
        )

        plane_as_prim = XFormPrim(
            relative_prim_path=f"/plane_{i}",
            name=plane.name,
        )
        plane_as_prim.load(None)

        # Build the plane orientation from the plane normal
        horiz_dir = pc - torch.tensor([0, 0, box_half_extent[2]])
        plane_z = -1 * horiz_dir / torch.norm(horiz_dir)
        plane_x = torch.tensor([0, 0, 1], dtype=torch.float32)
        plane_y = torch.cross(plane_z, plane_x)
        plane_mat = torch.stack([plane_x, plane_y, plane_z], dim=1)
        plane_quat = T.mat2quat(plane_mat)
        plane_as_prim.set_position_orientation(pc, plane_quat)

        plane_prims.append(plane_as_prim)
        plane_motions.append(plane_z)
    return plane_prims, plane_motions

def main(visualize_wall=False):
    """
    Demo of cloth objects that can be wall squeezed
    """

    cloth_category_models = [
        ("bandana", "wbhliu"),
    ]
    # cloth_category_models = [
    #     ("hoodie", "agftpm"),
    # ]

    for cloth in cloth_category_models:
        category = cloth[0]
        model = cloth[1]
        print(f"\nCategory: {category}, Model: {model}!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Create the scene config to load -- empty scene + custom cloth object
        cfg = {
            "scene": {
                "type": "Scene",
            },
            "objects": [
                {
                    "type": "DatasetObject",
                    "name": model,
                    "category": category,
                    "model": model,
                    "prim_type": PrimType.CLOTH,
                    "abilities": {"cloth": {}},
                },
            ],
        }

        # Create the environment
        env = og.Environment(configs=cfg)

        plane_prims, plane_motions = generate_box(visualize_wall=visualize_wall)

        # Grab object references
        carpet = env.scene.object_registry("name", model)
        objs = [carpet]

        # Set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=np.array([0.46382895, -2.66703958, 1.22616824]),
            orientation=np.array([0.58779174, -0.00231237, -0.00318273, 0.80900271]),
        )

        for _ in range(100):
            og.sim.step()

        # Calculate end positions for all walls
        end_positions = []
        for i in range(4):
            plane_prim = plane_prims[i]
            position = plane_prim.get_position()
            end_positions.append(np.array(position) + np.array(plane_motions[i]))

        increments = 1000
        for step in range(increments):
            # Move all walls a small amount
            for i in range(4):
                plane_prim = plane_prims[i]
                current_pos = np.linspace(plane_prim.get_position(), end_positions[i], increments)[step]
                plane_prim.set_position_orientation(position=current_pos)

            og.sim.step()

            # Check cloth height
            cloth_positions = objs[0].root_link.compute_particle_positions()
            # import pdb; pdb.set_trace()
            max_height = torch.max(cloth_positions[:, 2])

            # Get distance between facing walls (assuming walls 0-2 and 1-3 are facing pairs)
            wall_dist_1 = torch.linalg.norm(plane_prims[0].get_position() - plane_prims[2].get_position())
            wall_dist_2 = torch.linalg.norm(plane_prims[1].get_position() - plane_prims[3].get_position())
            min_wall_dist = min(wall_dist_1, wall_dist_2)

            # Stop if cloth height exceeds wall distance
            if max_height > min_wall_dist:
                print(f"Stopping: Cloth height ({max_height:.3f}) exceeds wall distance ({min_wall_dist:.3f})")
                break
        

        # TODO: save the cloth in the squeezed state

        # Shut down env at the end
        env.close()


if __name__ == "__main__":
    main()
