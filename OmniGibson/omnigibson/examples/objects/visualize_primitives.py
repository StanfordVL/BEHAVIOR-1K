"""
Script to visualize custom objects in the simulator.
Supports both:
  - PrimitiveObjects (cubes, spheres, cylinders, etc.) with custom sizes
  - DatasetObjects (pipes, poles, washers, etc.) from BEHAVIOR-1K assets

Usage:
    python omnigibson/examples/objects/visualize_primitives.py
"""

import torch as th
import omnigibson as og


def main():
    """
    Visualize custom objects in the simulator.
    Edit the lists below to customize what you want to see.
    """
    og.log.info(f"Demo {__file__}\n" + "*" * 80)

    # =======================================================================
    # PRIMITIVE OBJECTS - Flattened torus (washer-like)
    # =======================================================================
    primitives = [
        # Flattened torus - washer shape
        # Size controls overall diameter, scale[2] flattens it
        {
            "name": "flat_torus_washer",
            "primitive_type": "Torus",
            "size": 0.03,  # 3cm outer diameter
            "rgba": (0.7, 0.7, 0.7, 1.0),  # Silver/metal color
            "position": [0, 0, 1.0],
            "scale": [1.0, 1.0, 0.3],  # Flatten Z to make washer-like
        },
    ]

    # =======================================================================
    # DATASET OBJECTS - None for now
    # =======================================================================
    dataset_objects = []

    # =======================================================================
    # BUILD SCENE CONFIG
    # =======================================================================
    
    # Lights
    obj_cfgs = [
        {
            "type": "LightObject",
            "light_type": "Sphere",
            "name": "sphere_light0",
            "radius": 0.01,
            "intensity": 1e5,
            "position": [-2.0, -2.0, 2.0],
        },
        {
            "type": "LightObject",
            "light_type": "Sphere", 
            "name": "sphere_light1",
            "radius": 0.01,
            "intensity": 1e5,
            "position": [-2.0, 2.0, 2.0],
        },
    ]

    # Add primitive objects
    for obj_spec in primitives:
        cfg = {
            "type": "PrimitiveObject",
            "name": obj_spec["name"],
            "primitive_type": obj_spec["primitive_type"],
            "rgba": obj_spec["rgba"],
            "position": obj_spec["position"],
            "fixed_base": False,
        }
        if "size" in obj_spec:
            cfg["size"] = obj_spec["size"]
        if "radius" in obj_spec:
            cfg["radius"] = obj_spec["radius"]
        if "height" in obj_spec:
            cfg["height"] = obj_spec["height"]
        if "scale" in obj_spec:
            cfg["scale"] = obj_spec["scale"]
        obj_cfgs.append(cfg)

    # Add dataset objects
    for obj_spec in dataset_objects:
        cfg = {
            "type": "DatasetObject",
            "name": obj_spec["name"],
            "category": obj_spec["category"],
            "model": obj_spec["model"],
            "position": obj_spec["position"],
            "fixed_base": False,
        }
        if "scale" in obj_spec:
            cfg["scale"] = obj_spec["scale"]
        obj_cfgs.append(cfg)

    # Ground plane
    obj_cfgs.append({
        "type": "PrimitiveObject",
        "name": "ground",
        "primitive_type": "Cube",
        "size": 2.0,
        "rgba": (0.3, 0.3, 0.3, 1.0),
        "position": [0, 0, -1.0],
        "fixed_base": True,
    })

    # Create scene
    cfg = {
        "scene": {"type": "Scene"},
        "objects": obj_cfgs,
    }

    # Create environment
    env = og.Environment(configs=cfg)

    # Set camera - closer view for single object
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.15, -0.15, 1.1]),
        orientation=th.tensor([0.56, 0.21, 0.21, 0.77]),
    )
    og.sim.enable_viewer_camera_teleoperation()

    # Print dimensions
    print("\n" + "=" * 80)
    print("OBJECT DIMENSIONS:")
    print("=" * 80)
    
    all_objects = primitives + dataset_objects
    for obj_spec in all_objects:
        obj = env.scene.object_registry("name", obj_spec["name"])
        extent = obj.aabb_extent
        obj_type = "Primitive" if "primitive_type" in obj_spec else "Dataset"
        print(f"{obj_spec['name']:20s} [{obj_type:9s}]: {extent[0]*100:7.2f} x {extent[1]*100:7.2f} x {extent[2]*100:7.2f} cm")
    
    print("=" * 80)
    print("\nUse mouse to move camera. Press Ctrl+C to exit.\n")

    # Run simulation
    try:
        for i in range(10000):
            env.step(th.empty(0))
    except KeyboardInterrupt:
        print("Exiting...")

    og.shutdown()


if __name__ == "__main__":
    main()
