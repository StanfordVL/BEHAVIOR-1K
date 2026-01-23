from collections import defaultdict
from concurrent import futures
import numpy as np
import tqdm
import json

from b1k_pipeline import mesh_tree
from b1k_pipeline.utils import PipelineFS, get_targets, parse_name


def process_target(target):
    # Get the mesh tree
    G = mesh_tree.build_mesh_tree(
        target, load_upper=False, load_bad=False, load_nonzero=False
    )
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]

    # Compute volumes from this target
    volumes = defaultdict(dict)
    errors = {}
    for node in roots:
        try:
            assert (
                "combined_collision_mesh" in G.nodes[node]
            ), f"Node {node} does not have a combined collision mesh."

            obj_volume = G.nodes[node]["combined_collision_mesh"].volume
            assert (
                obj_volume > 0
            ), f"Node {node} has a non-positive volume: {obj_volume}"
            volumes[node[0]][node[1]] = obj_volume
        except Exception as e:
            errors[str(node)] = str(e)

    return volumes, errors


def main():
    target_futures = {}
    targets = get_targets("combined")

    # Collect the volumes from each target
    volumes = defaultdict(dict)
    errors = {}
    with futures.ProcessPoolExecutor(max_workers=16) as executor:
        # Submit all targets for execution
        for target in tqdm.tqdm(targets):
            target_futures[executor.submit(process_target, target)] = target

        # Collect the volumes from each target
        with tqdm.tqdm(total=len(target_futures)) as pbar:
            for future in futures.as_completed(target_futures.keys()):
                target = target_futures[future]
                target_volumes, target_errors = future.result()

                if target_errors:
                    errors[target] = target_errors

                for category, category_volumes in target_volumes.items():
                    volumes[category].update(category_volumes)

                pbar.update(1)

    # Convert the volumes to average volumes
    average_volumes = {
        category: np.median(list(category_volumes.values()))
        for category, category_volumes in volumes.items()
    }

    with PipelineFS() as pipeline_fs:
        with pipeline_fs.pipeline_output().open(
            "collision_average_volumes.json", "w"
        ) as f:
            # TODO: set success
            json.dump(
                {
                    "success": True,
                    "average_volumes": average_volumes,
                    "volumes": volumes,
                    "errors": errors,
                },
                f,
                indent=4,
            )


if __name__ == "__main__":
    main()
