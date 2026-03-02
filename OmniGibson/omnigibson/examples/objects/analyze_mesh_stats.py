"""
Script to analyze mesh statistics (vertex count, face count) for various objects.
This helps understand the mesh complexity of primitives vs dataset objects.

Usage:
    python omnigibson/examples/objects/analyze_mesh_stats.py
"""

import torch as th
import omnigibson as og
import omnigibson.lazy as lazy


def get_mesh_stats(obj):
    """
    Get vertex and face counts for all meshes in an object.
    Returns dict with total vertices, total faces, and per-mesh breakdown.
    """
    total_vertices = 0
    total_faces = 0
    mesh_details = []
    
    try:
        # Get all links
        for link_name, link in obj.links.items():
            # Check visual meshes
            if hasattr(link, 'visual_meshes') and link.visual_meshes:
                for mesh_name, mesh in link.visual_meshes.items():
                    try:
                        prim = mesh.prim
                        points = prim.GetAttribute("points").Get()
                        face_counts = prim.GetAttribute("faceVertexCounts").Get()
                        
                        n_verts = len(points) if points else 0
                        n_faces = len(face_counts) if face_counts else 0
                        
                        total_vertices += n_verts
                        total_faces += n_faces
                        mesh_details.append({
                            "link": link_name,
                            "mesh": mesh_name,
                            "type": "visual",
                            "vertices": n_verts,
                            "faces": n_faces,
                        })
                    except Exception as e:
                        pass
            
            # Check collision meshes
            if hasattr(link, 'collision_meshes') and link.collision_meshes:
                for mesh_name, mesh in link.collision_meshes.items():
                    try:
                        prim = mesh.prim
                        points = prim.GetAttribute("points").Get()
                        face_counts = prim.GetAttribute("faceVertexCounts").Get()
                        
                        n_verts = len(points) if points else 0
                        n_faces = len(face_counts) if face_counts else 0
                        
                        # Don't double count - collision meshes tracked separately
                        mesh_details.append({
                            "link": link_name,
                            "mesh": mesh_name,
                            "type": "collision",
                            "vertices": n_verts,
                            "faces": n_faces,
                        })
                    except Exception as e:
                        pass
                        
    except Exception as e:
        print(f"  Error getting mesh stats: {e}")
    
    return {
        "total_visual_vertices": total_vertices,
        "total_visual_faces": total_faces,
        "details": mesh_details,
    }


def main():
    og.log.info(f"Mesh Statistics Analyzer\n" + "*" * 80)

    # =======================================================================
    # PRIMITIVE OBJECTS - Various sizes
    # =======================================================================
    primitives = [
        # Small sphere
        {"name": "sphere_small", "primitive_type": "Sphere", "radius": 0.01, "rgba": (0.8, 0.2, 0.2, 1.0), "position": [0, 0, 1.0]},
        # Medium sphere
        {"name": "sphere_medium", "primitive_type": "Sphere", "radius": 0.05, "rgba": (0.2, 0.8, 0.2, 1.0), "position": [0.15, 0, 1.0]},
        # Large sphere  
        {"name": "sphere_large", "primitive_type": "Sphere", "radius": 0.15, "rgba": (0.2, 0.2, 0.8, 1.0), "position": [0.4, 0, 1.0]},
        
        # Small cube
        {"name": "cube_small", "primitive_type": "Cube", "size": 0.02, "rgba": (0.8, 0.8, 0.2, 1.0), "position": [0, 0.2, 1.0]},
        # Medium cube
        {"name": "cube_medium", "primitive_type": "Cube", "size": 0.05, "rgba": (0.8, 0.5, 0.2, 1.0), "position": [0.15, 0.2, 1.0]},
        # Large cube
        {"name": "cube_large", "primitive_type": "Cube", "size": 0.15, "rgba": (0.5, 0.2, 0.8, 1.0), "position": [0.4, 0.2, 1.0]},
        
        # Cylinder
        {"name": "cylinder", "primitive_type": "Cylinder", "radius": 0.02, "height": 0.1, "rgba": (0.2, 0.8, 0.8, 1.0), "position": [0, 0.4, 1.0]},
        
        # Torus (ring-like)
        {"name": "torus", "primitive_type": "Torus", "size": 0.03, "rgba": (0.8, 0.2, 0.8, 1.0), "position": [0.15, 0.4, 1.0]},
        
        # Disk
        {"name": "disk", "primitive_type": "Disk", "radius": 0.03, "rgba": (0.5, 0.5, 0.5, 1.0), "position": [0.3, 0.4, 1.0]},
    ]

    # =======================================================================
    # DATASET OBJECTS - Various types
    # =======================================================================
    dataset_objects = [
        # Balls of different sizes
        {"category": "ping_pong_ball", "model": "eutewe", "name": "ping_pong_ball", "position": [-0.3, 0, 1.0]},
        {"category": "tennis_ball", "model": "rgekxe", "name": "tennis_ball", "position": [-0.45, 0, 1.0]},
        {"category": "baseball", "model": "zanmar", "name": "baseball", "position": [-0.6, 0, 1.0]},
        {"category": "soccer_ball", "model": "ikagkc", "name": "soccer_ball", "position": [-0.8, 0, 1.0]},
        
        # Other small objects
        {"category": "dice", "model": "rcgtzz", "name": "dice", "position": [-0.3, 0.2, 1.0]},
        {"category": "ring", "model": "oolbrj", "name": "ring", "position": [-0.45, 0.2, 1.0]},
        {"category": "dowel", "model": "ghnjry", "name": "dowel", "position": [-0.6, 0.2, 1.0]},
        {"category": "pole", "model": "pkzvpk", "name": "pole", "position": [-0.8, 0.2, 1.0]},
    ]

    # =======================================================================
    # BUILD SCENE
    # =======================================================================
    obj_cfgs = [
        {"type": "LightObject", "light_type": "Sphere", "name": "light0", "radius": 0.01, "intensity": 1e5, "position": [-2.0, -2.0, 2.0]},
        {"type": "LightObject", "light_type": "Sphere", "name": "light1", "radius": 0.01, "intensity": 1e5, "position": [-2.0, 2.0, 2.0]},
    ]

    # Add primitives
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
        obj_cfgs.append(cfg)

    # Ground
    obj_cfgs.append({
        "type": "PrimitiveObject",
        "name": "ground",
        "primitive_type": "Cube",
        "size": 3.0,
        "rgba": (0.3, 0.3, 0.3, 1.0),
        "position": [0, 0, -1.5],
        "fixed_base": True,
    })

    # Create environment
    cfg = {"scene": {"type": "Scene"}, "objects": obj_cfgs}
    env = og.Environment(configs=cfg)

    # Step once to ensure everything is loaded
    env.step(th.empty(0))

    # =======================================================================
    # ANALYZE MESH STATS
    # =======================================================================
    print("\n" + "=" * 100)
    print("MESH STATISTICS ANALYSIS")
    print("=" * 100)
    
    all_objects = primitives + dataset_objects
    results = []
    
    for obj_spec in all_objects:
        obj = env.scene.object_registry("name", obj_spec["name"])
        extent = obj.aabb_extent
        stats = get_mesh_stats(obj)
        
        obj_type = "Primitive" if "primitive_type" in obj_spec else "Dataset"
        size_str = f"{extent[0]*100:.1f}x{extent[1]*100:.1f}x{extent[2]*100:.1f}cm"
        
        results.append({
            "name": obj_spec["name"],
            "type": obj_type,
            "size": size_str,
            "vertices": stats["total_visual_vertices"],
            "faces": stats["total_visual_faces"],
        })
    
    # Print results table
    print(f"\n{'Object':<20} {'Type':<10} {'Size (cm)':<25} {'Vertices':>10} {'Faces':>10}")
    print("-" * 85)
    
    for r in results:
        print(f"{r['name']:<20} {r['type']:<10} {r['size']:<25} {r['vertices']:>10} {r['faces']:>10}")
    
    print("-" * 85)
    
    # Summary by type
    print("\n" + "=" * 60)
    print("SUMMARY BY OBJECT TYPE")
    print("=" * 60)
    
    primitive_results = [r for r in results if r['type'] == 'Primitive']
    dataset_results = [r for r in results if r['type'] == 'Dataset']
    
    if primitive_results:
        avg_verts = sum(r['vertices'] for r in primitive_results) / len(primitive_results)
        avg_faces = sum(r['faces'] for r in primitive_results) / len(primitive_results)
        print(f"\nPrimitives:")
        print(f"  Average vertices: {avg_verts:.0f}")
        print(f"  Average faces: {avg_faces:.0f}")
        print(f"  Range: {min(r['vertices'] for r in primitive_results)} - {max(r['vertices'] for r in primitive_results)} vertices")
    
    if dataset_results:
        avg_verts = sum(r['vertices'] for r in dataset_results) / len(dataset_results)
        avg_faces = sum(r['faces'] for r in dataset_results) / len(dataset_results)
        print(f"\nDataset Objects:")
        print(f"  Average vertices: {avg_verts:.0f}")
        print(f"  Average faces: {avg_faces:.0f}")
        print(f"  Range: {min(r['vertices'] for r in dataset_results)} - {max(r['vertices'] for r in dataset_results)} vertices")
    
    print("\n" + "=" * 60)
    print("Note: Primitive objects use the same mesh regardless of size.")
    print("Scaling is applied via transforms, not mesh complexity.")
    print("=" * 60)

    # Shutdown
    og.shutdown()


if __name__ == "__main__":
    main()
