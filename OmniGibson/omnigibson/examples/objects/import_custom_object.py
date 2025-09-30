"""
Helper script to download BEHAVIOR-1K dataset and assets.
Improved version that can import obj file and articulated file (glb, gltf).
"""

import pathlib
from typing import Literal
import click
import shutil
import tempfile
import omnigibson as og

from omnigibson.utils.asset_conversion_utils import (
    import_og_asset_from_urdf,
    generate_urdf_for_mesh,
)


@click.command()
@click.option(
    "--asset-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Absolute path to asset file to import. This can be a raw visual mesh (for single-bodied, static objects), e.g. .obj, .glb, etc., or a more complex (such as articulated) objects defined in .urdf format.",
)
@click.option("--category", required=True, type=click.STRING, help="Category name to assign to the imported asset")
@click.option(
    "--model",
    required=True,
    type=click.STRING,
    help="Model name to assign to the imported asset. This must be unique within the dataset.",
)
@click.option(
    "--collision-method",
    type=click.Choice(["coacd", "convex", "none"]),
    default="coacd",
    help="Method to generate the collision mesh. 'coacd' generates a set of convex decompositions, while 'convex' generates a single convex hull. 'none' will not generate any explicit mesh",
)
@click.option(
    "--hull-count",
    type=int,
    default=32,
    help="Maximum number of convex hulls to decompose individual visual meshes into. Only relevant if --collision-method=coacd",
)
@click.option("--up-axis", type=click.Choice(["z", "y"]), default="z", help="Up axis for the mesh.")
@click.option("--headless", is_flag=True, help="Run the script in headless mode.")
@click.option("--scale", type=int, default=1, help="User choice scale, will be overwritten if check_scale and rescale")
@click.option("--check_scale", is_flag=True, help="Check meshes scale based on heuristic")
@click.option("--rescale", is_flag=True, help="Rescale meshes based on heuristic if check_scale ")
@click.option("--overwrite", is_flag=True, help="Overwrite any pre-existing files")
def import_custom_object(
    asset_path: str,
    category: str,
    model: str,
    collision_method: Literal["coacd", "convex", "none"],
    hull_count: int,
    up_axis: Literal["z", "y"],
    headless: bool,
    scale: int,
    check_scale: bool,
    rescale: bool,
    overwrite: bool,
):
    """
    Imports a custom-defined object asset into an OmniGibson-compatible USD format and saves the imported asset
    files to the selected dataset directory.
    """

    assert len(model) == 6 and model.isalpha(), "Model name must be 6 characters long and contain only letters."
    collision_method = None if collision_method == "none" else collision_method

    # Resolve the asset path here
    asset_path = pathlib.Path(asset_path).absolute()

    # If we're not a URDF, import the mesh directly first
    temp_dir = tempfile.mkdtemp()

    try:
        if asset_path.suffix != ".urdf":
            # Try to generate URDF, may raise ValueError if too many submeshes
            urdf_path = generate_urdf_for_mesh(
                asset_path,
                temp_dir,
                category,
                model,
                collision_method,
                hull_count,
                up_axis,
                scale=scale,
                check_scale=check_scale,
                rescale=rescale,
                overwrite=True,
            )
            if urdf_path is not None:
                click.echo("URDF generation complete!")
                collision_method = None
            else:
                # Clean up temp directories before exiting
                click.echo("Error during URDF generation")
                raise click.Abort()
        else:
            urdf_path = asset_path
            collision_method = collision_method

        # Convert to USD
        import_og_asset_from_urdf(
            category=category,
            model=model,
            urdf_path=str(urdf_path),
            collision_method=collision_method,
            hull_count=hull_count,
            overwrite=overwrite,
            use_usda=False,
        )

    finally:
        # Clean up temp directories before exiting
        shutil.rmtree(temp_dir)

    # Visualize if not headless
    if not headless:
        click.echo("The asset has been successfully imported. You can view it and make changes and save if you'd like.")
        while True:
            og.sim.render()


if __name__ == "__main__":
    import_custom_object()
