"""
Helper script to download OmniGibson dataset and assets.
Improved version that can import obj file and articulated file (glb, gltf).
"""

from typing import Literal
import click
import trimesh
import sys
import shutil
import select
import tempfile
import time
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
    help="Model name to assign to the imported asset. This MUST be a 6-character long string that exclusively contains letters, and must be unique within the given @category",
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
@click.option("--n_submesh", type=int, help="Maximum of submesh numnber")
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
    n_submesh: int,
):
    """
    Imports a custom-defined object asset into an OmniGibson-compatible USD format and saves the imported asset
    files to the custom dataset directory (gm.CUSTOM_DATASET_PATH)
    """

    assert len(model) == 6 and model.isalpha(), "Model name must be 6 characters long and contain only letters."
    collision_method = None if collision_method == "none" else collision_method

    # Sanity check mesh type
    mesh_format = asset_path.split(".")[-1]

    # If we're not a URDF, import the mesh directly first
    urdf_dep_paths = None
    temp_dirs = []
    if mesh_format != "urdf":
        temp_urdf_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_urdf_dir)

        # Try to generate URDF, may raise ValueError if too many submeshes
        urdf_path = generate_urdf_for_mesh(
            asset_path,
            temp_urdf_dir,
            category,
            model,
            collision_method,
            hull_count,
            up_axis,
            scale=scale,
            check_scale=check_scale,
            rescale=rescale,
            overwrite=overwrite,
            n_submesh=n_submesh
        )
        if urdf_path is not None:
            click.echo("URDF generation complete!")
            urdf_dep_paths = ["material"]
            collision_method = None
        else:
            # Clean up temp directories before exiting
            for tmp_dir in temp_dirs:
                shutil.rmtree(tmp_dir)
            click.echo(f"Error during URDF generation")
            raise click.Abort()
    else:
        urdf_path = asset_path
        collision_method = collision_method

    try:
        # Convert to USD
        import_og_asset_from_urdf(
            category=category,
            model=model,
            urdf_path=urdf_path,
            urdf_dep_paths=urdf_dep_paths,
            collision_method=collision_method,
            hull_count=hull_count,
            overwrite=overwrite,
            use_usda=False,
        )

    except Exception as e:
        click.echo(f"Error during USD conversion: {str(e)}")
        # Clean up temp directories before exiting
        for tmp_dir in temp_dirs:
            shutil.rmtree(tmp_dir)
        raise click.Abort()

    # Clean up temp directories
    for tmp_dir in temp_dirs:
        shutil.rmtree(tmp_dir)

    # Visualize if not headless
    if not headless:
        click.echo("The asset has been successfully imported. You can view it and make changes and save if you'd like.")
        while True:
            og.sim.render()
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()  # Clear the input buffer
                break

if __name__ == "__main__":
    import_custom_object()
