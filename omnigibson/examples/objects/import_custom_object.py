"""
Helper script to download OmniGibson dataset and assets.
"""
import math
from typing import Literal

import click
import trimesh

import omnigibson as og
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.asset_conversion_utils import (
    generate_collision_meshes,
    generate_urdf_for_obj,
    import_og_asset_from_urdf,
)


@click.command()
@click.option("--asset-path", required=True, type=click.Path(exists=True, dir_okay=False), help="Absolute path to asset file to import. This can be a raw visual mesh (for single-bodied, static objects), e.g. .obj, .glb, etc., or a more complex (such as articulated) objects defined in .urdf format.")
@click.option("--category", required=True, type=click.STRING, help="Category name to assign to the imported asset")
@click.option("--model", required=True, type=click.STRING, help="Model name to assign to the imported asset. This MUST be a 6-character long string that exclusively contains letters, and must be unique within the given @category")
@click.option(
    "--collision-method",
    type=click.Choice(["coacd", "convex", "none"]),
    default="coacd",
    help="Method to generate the collision mesh. 'coacd' generates a set of convex decompositions, while 'convex' generates a single convex hull. 'none' will not generate any explicit mesh",
)
@click.option("--hull-count", type=int, default=32, help="Maximum number of convex hulls to decompose individual visual meshes into. Only relevant if --collision-method=coacd")
@click.option("--scale", type=float, default=1.0, help="Scale factor to apply to the mesh.")
@click.option("--up-axis", type=click.Choice(["z", "y"]), default="z", help="Up axis for the mesh.")
@click.option("--headless", is_flag=True, help="Run the script in headless mode.")
@click.option("--confirm-bbox", default=True, help="Whether to confirm the scale factor.")
@click.option("--overwrite", is_flag=True, help="Overwrite any pre-existing files")
def import_custom_object(
    asset_path: str,
    category: str,
    model: str,
    collision_method: Literal["coacd", "convex", "none"],
    hull_count: int,
    scale: float,
    up_axis: Literal["z", "y"],
    headless: bool,
    confirm_bbox: bool,
    overwrite: bool,
):
    """
    Imports an externally-defined object asset into an OmniGibson-compatible USD format and saves the imported asset
    files to the external dataset directory (gm.EXTERNAL_DATASET_PATH)
    """
    assert len(model) == 6 and model.isalpha(), "Model name must be 6 characters long and contain only letters."
    collision_method = None if collision_method == "none" else collision_method

    # Sanity check mesh type
    valid_formats = trimesh.available_formats()
    mesh_format = asset_path.split(".")[-1]

    # If we're not a URDF, import the mesh directly first
    if mesh_format != "urdf":
        assert_valid_key(key=mesh_format, valid_keys=valid_formats, name="mesh format")

        # Load the mesh
        visual_mesh: trimesh.Trimesh = trimesh.load(asset_path, force="mesh", process=False)

        # Generate collision meshes if requested
        collision_meshes = generate_collision_meshes(visual_mesh, method=collision_method, hull_count=hull_count) \
            if collision_method is not None else []

        # If the up axis is y, we need to rotate the meshes
        if up_axis == "y":
            rotation_matrix = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
            visual_mesh.apply_transform(rotation_matrix)
            for collision_mesh in collision_meshes:
                collision_mesh.apply_transform(rotation_matrix)

        # If the scale is nonzero, we apply it to the meshes
        if scale != 1.0:
            scale_transform = trimesh.transformations.scale_matrix(scale)
            visual_mesh.apply_transform(scale_transform)
            for collision_mesh in collision_meshes:
                collision_mesh.apply_transform(scale_transform)

        # Check the bounding box size and complain if it's larger than 3 meters
        bbox_size = visual_mesh.bounding_box.extents
        click.echo(f"Visual mesh bounding box size: {bbox_size}")

        if confirm_bbox:
            if any(size > 3.0 for size in bbox_size):
                click.echo(
                    f"Warning: The bounding box sounds a bit large. Are you sure you don't need to scale? "
                    f"We just wanted to confirm this is intentional. You can skip this check by passing --no-confirm-bbox."
                )
                click.confirm("Do you want to continue?", abort=True)

            elif any(size < 0.01 for size in bbox_size):
                click.echo(
                    f"Warning: The bounding box sounds a bit small. Are you sure you don't need to scale? "
                    f"We just wanted to confirm this is intentional. You can skip this check by passing --no-confirm-bbox."
                )
                click.confirm("Do you want to continue?", abort=True)

        # Generate the URDF
        click.echo(f"Generating URDF for {category}/{model}...")
        generate_urdf_for_obj(visual_mesh, collision_meshes, category, model, overwrite=overwrite)
        click.echo("URDF generation complete!")

        urdf_path = None
        collision_method = None
    else:
        urdf_path = asset_path
        collision_method = collision_method

    # Convert to USD
    import_og_asset_from_urdf(
        category=category,
        model=model,
        urdf_path=urdf_path,
        collision_method=collision_method,
        hull_count=hull_count,
        overwrite=overwrite,
        use_usda=False,
    )

    # Visualize if not headless
    if not headless:
        click.echo("The asset has been successfully imported. You can view it and make changes and save if you'd like.")
        while True:
            og.sim.render()


if __name__ == "__main__":
    import_custom_object()
