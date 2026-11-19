"""Compare the handoff Warp renderer against Kit path tracing on the same frozen Rs_int scene.

Run in the newton environment, with OMNIGIBSON_HEADLESS=1 and the existing dataset path.
The external benchmark harness is intentionally not a production backend dependency.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.harness))
    from render_bench.common import CAMERAS, gpu_info, load_calibration
    from render_bench.report import psnr, ssim

    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.macros import gm

    gm.PHYSICS_BACKEND = "physx"
    gm.RENDER_BACKEND = "kit"
    width, height = 960, 540
    sensors = []
    for name, camera in CAMERAS.items():
        back = np.subtract(camera["pos"], camera["look_at"])
        back /= np.linalg.norm(back)
        right = np.cross([0.0, 0.0, 1.0], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        quat = Rotation.from_matrix(np.column_stack((right, up, back))).as_quat()
        aperture = 20.995
        focal = aperture * height / width / (2 * np.tan(np.deg2rad(camera["fov"]) / 2))
        sensors.append(
            dict(
                sensor_type="VisionSensor",
                name=name,
                modalities=["rgb"],
                sensor_kwargs=dict(
                    image_width=width, image_height=height, horizontal_aperture=aperture, focal_length=focal
                ),
                position=list(camera["pos"]),
                orientation=quat.tolist(),
                pose_frame="world",
            )
        )
    started = time.perf_counter()
    env = og.Environment(
        configs=dict(
            scene=dict(type="InteractiveTraversableScene", scene_model="Rs_int"), env=dict(external_sensors=sensors)
        )
    )
    summary = dict(
        gpu=gpu_info(),
        cameras=CAMERAS,
        resolution=[width, height],
        load_seconds=time.perf_counter() - started,
        reference="PhysX + Kit 6 path tracing, 256 accumulated samples, same frozen scene",
    )
    try:
        settings = lazy.carb.settings.get_settings()
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", 16)
        settings.set("/rtx/pathtracing/totalSpp", 256)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        # Render only: do not advance physics between reference and candidate captures.
        for _ in range(32):
            og.sim.render()
        reference = {}
        for name, sensor in env.external_sensors[0].items():
            rgb = sensor.get_obs()[0]["rgb"][..., :3].cpu().numpy()
            reference[name] = rgb
            imageio.imwrite(args.output / f"{name}_rtx.png", rgb)

        from render_bench.extract import extract_scene
        from render_bench.warp_renderer import WarpRenderer
        import warp as wp

        # Extract body-local geometry and use live PhysX poses, not stale USD transforms.
        links = [link for obj in env.scene.objects for link in obj.links.values()]
        labels = [link.prim_path for link in links]
        body_q = np.array([th.cat(link.get_position_orientation(frame="world")).cpu().numpy() for link in links])
        started = time.perf_counter()
        scene = extract_scene(og.sim.stage, labels)
        summary["extraction_seconds"] = time.perf_counter() - started
        calibration = load_calibration(args.harness / "output")
        exposure = calibration["warp_exposure"]
        renderer = WarpRenderer(scene, device="cuda:0", ambient=calibration["warp_ambient"])
        renderer.set_body_q(body_q)
        renderer.set_water(None, None)
        summary.update(
            exposure=exposure,
            triangles=scene.triangle_count,
            lights=len(scene.lights.rects),
            dome=scene.lights.dome,
            results={},
        )
        for preset in ("direct", "pt16"):
            renderer.render(list(CAMERAS.values()), width, height, preset=preset, exposure=exposure)
            wp.synchronize()
            started = time.perf_counter()
            images = renderer.render(list(CAMERAS.values()), width, height, preset=preset, exposure=exposure)
            wp.synchronize()
            elapsed = time.perf_counter() - started
            images = images.numpy()
            result = dict(ms_per_camera=elapsed * 1000 / len(CAMERAS), cameras={})
            for (name, ref), candidate in zip(reference.items(), images):
                candidate = candidate[..., :3]
                imageio.imwrite(args.output / f"{name}_{preset}.png", candidate)
                imageio.imwrite(
                    args.output / f"{name}_{preset}_comparison.png", np.concatenate([ref, candidate], axis=1)
                )
                a, b = candidate.astype(np.float32), ref.astype(np.float32)
                result["cameras"][name] = dict(
                    ssim=ssim(a, b), psnr=psnr(a, b), candidate_mean=float(a.mean()), reference_mean=float(b.mean())
                )
            summary["results"][preset] = result
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
