import json

import b1k_pipeline.utils


OUTPUT_PATH = "artifacts/pipeline/check_collisions_challenge.json"


def get_scene_parts(scene_target):
    scene_name = scene_target.replace("scenes/", "")
    object_list_path = (
        b1k_pipeline.utils.PIPELINE_ROOT
        / "cad"
        / scene_target
        / "artifacts"
        / "object_list.json"
    )

    with open(object_list_path, "r") as f:
        room_object_list = json.load(f)

    scene_parts = {scene_name}
    for partial_scene_name in room_object_list.get("outgoing_portals", {}):
        scene_parts.add(partial_scene_name)

    return sorted("scenes/" + scene_part for scene_part in scene_parts)


def load_collision_result(scene_target):
    collision_path = (
        b1k_pipeline.utils.PIPELINE_ROOT
        / "cad"
        / scene_target
        / "artifacts"
        / "check_collisions.json"
    )

    if not collision_path.exists():
        return {
            "success": False,
            "collisions": [],
            "error": f"Missing collision output: {collision_path}",
        }

    with open(collision_path, "r") as f:
        return json.load(f)


def main():
    challenge_scenes = b1k_pipeline.utils.params.get("challenge_scenes", [])
    challenge_scenes_and_deps = b1k_pipeline.utils.params.get(
        "challenge_scenes_and_deps", challenge_scenes
    )

    scene_parts = {}
    errors = {}
    for scene_target in challenge_scenes:
        try:
            parts = get_scene_parts(scene_target)
        except Exception as e:
            parts = [scene_target]
            errors[scene_target] = repr(e)

        scene_parts[scene_target] = parts

    collision_results = {
        scene_target: load_collision_result(scene_target)
        for scene_target in sorted(challenge_scenes_and_deps)
    }
    failed_scenes = sorted(
        scene_target
        for scene_target, result in collision_results.items()
        if not result["success"]
    )

    results = {
        "success": not errors and not failed_scenes,
        "challenge_scenes": challenge_scenes,
        "challenge_scenes_and_deps": challenge_scenes_and_deps,
        "scene_parts": scene_parts,
        "checked_scenes": sorted(challenge_scenes_and_deps),
        "failed_scenes": failed_scenes,
        "errors": errors,
        "collision_results": collision_results,
    }

    output_path = b1k_pipeline.utils.PIPELINE_ROOT / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
