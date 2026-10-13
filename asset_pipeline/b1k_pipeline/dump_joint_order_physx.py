"""Dump the joint order of articulated task models with PhysX alone, without rendering.

Equivalent to dump_joint_order_og.py, for machines where Isaac Sim's RTX
renderer can't start (for example, Isaac Sim 5.1 crashes in librtx.scenedb on
R590+ drivers). It starts a minimal Kit app with only USD and PhysX, references
each decrypted object USD, attaches PhysX, and reads the articulation's joint
names from omni.physics.tensors: the same metadata that OmniGibson's
`obj.joints` (EntityPrim.update_joints) iterates, keeping joints with DOFs.

It needs Isaac Sim, the `cryptography` package and the BEHAVIOR-1K dataset
(with its key), and writes the same metadata/mujoco_joint_order.json:

    python -m b1k_pipeline.dump_joint_order_physx \\
        --instances_zip 2026-challenge-task-instances.zip
"""

import argparse
import json
import pathlib
import tempfile

import b1k_pipeline.utils
from b1k_pipeline.dump_joint_order_og import DEFAULT_INSTANCES_ZIP, OUTPUT, articulated_models, urdf_joint_names

DEFAULT_DATASET = b1k_pipeline.utils.PIPELINE_ROOT.parent / "datasets"

KIT_TEMPLATE = """\
[package]
title = "BEHAVIOR-1K joint order (PhysX only)"
version = "1.0.0"

[dependencies]
"omni.usd" = {{}}
"omni.kit.usd.layers" = {{}}
"omni.kit.loop-isaac" = {{}}
"omni.timeline" = {{}}
"omni.physx" = {{}}
"omni.physics.stageupdate" = {{}}
"omni.physx.tensors" = {{}}
"omni.physics.tensors" = {{}}

[settings.app]
name = "b1k-joint-order"
settings.persistent = false
enableDeveloperWarnings = false

[settings.crashreporter]
enabled = false

[settings.app.exts.folders]
'++' = [{folders}]

[settings.app.extensions]
registryEnabled = false
"""


def write_kit_file(directory):
    """A Kit experience with only USD and PhysX, using the installed Isaac Sim's extensions."""
    import isaacsim

    root = pathlib.Path(isaacsim.__file__).parent
    folders = ", ".join(f'"{root / name}"' for name in ("exts", "extscache", "extsPhysics", "extsDeprecated"))
    path = pathlib.Path(directory) / "physx_only.kit"
    path.write_text(KIT_TEMPLATE.format(folders=folders))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instances_zip", default=str(DEFAULT_INSTANCES_ZIP))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Directory with behavior-1k-assets and omnigibson.key.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    models = articulated_models(args.instances_zip)
    print(f"{len(models)} articulated models with >= 2 joint values")

    from cryptography.fernet import Fernet
    from isaacsim import SimulationApp

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="b1k_joint_order_"))
    app = SimulationApp({"headless": True}, experience=str(write_kit_file(tmp)))

    import omni.kit.app
    import omni.physics.tensors
    import omni.physx
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdPhysics

    dataset = pathlib.Path(args.dataset)
    fernet = Fernet((dataset / "omnigibson.key").read_bytes())
    timeline = omni.timeline.get_timeline_interface()
    simulation = omni.physx.get_physx_simulation_interface()
    order, problems = {}, {}
    for (category, model), count in sorted(models.items()):
        key = f"{category}/{model}"
        encrypted = dataset / "behavior-1k-assets" / "objects" / category / model / "usd" / f"{model}.encrypted.usd"
        usd = tmp / category / model / "usd" / f"{model}.usd"
        usd.parent.mkdir(parents=True, exist_ok=True)
        usd.write_bytes(fernet.decrypt(encrypted.read_bytes()))

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdPhysics.Scene.Define(stage, "/physicsScene")
        prim = stage.DefinePrim("/World/obj", "Xform")
        prim.GetReferences().AddReference(str(usd))
        # As OmniGibson's fixed_base=True: weld the base link to the world.
        UsdPhysics.FixedJoint.Define(stage, "/World/obj/rootJoint").CreateBody1Rel().SetTargets(["/World/obj/base_link"])
        roots = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if not roots:
            problems.setdefault(key, []).append("no ArticulationRootAPI in the USD")
            continue

        timeline.play()
        omni.kit.app.get_app().update()
        stage_id = context.get_stage_id()
        simulation.attach_stage(stage_id)
        simulation.simulate(1.0 / 60.0, 0.0)
        simulation.fetch_results()
        view = omni.physics.tensors.create_simulation_view("numpy", stage_id)
        meta = view.create_articulation_view(str(roots[0].GetPath())).shared_metatype
        names = [meta.joint_names[i] for i in range(meta.joint_count) if meta.joint_dof_counts[i] > 0]
        timeline.stop()
        simulation.detach_stage()
        omni.kit.app.get_app().update()

        order[key] = names
        if len(names) != count:
            problems.setdefault(key, []).append(f"PhysX has {len(names)} joints, instances have {count}")
        urdf_names = urdf_joint_names(category, model)
        if urdf_names is not None and set(names) != urdf_names:
            problems.setdefault(key, []).append(f"joint names differ from the URDF: {sorted(set(names) ^ urdf_names)}")

    with open(args.output, "w") as f:
        json.dump({"format": "b1k_joint_order/1", "models": order, "problems": problems}, f, indent=1, sort_keys=True)
    print(f"Wrote {len(order)} models to {args.output}; {len(problems)} with problems")
    app.close()


if __name__ == "__main__":
    main()
