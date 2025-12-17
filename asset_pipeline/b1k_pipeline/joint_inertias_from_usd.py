from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import pathlib
import tempfile
from cryptography.fernet import Fernet
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import fs.path
from fs.zipfs import ZipFS
import networkx as nx

DATASET_PATH = pathlib.Path("/scr/og-docker-data/datasets/og_dataset")
KEY_PATH = pathlib.Path("/scr/og-docker-data/datasets/omnigibson.key")


def decrypt_file(encrypted_filename, decrypted_filename):
    with open(KEY_PATH, "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def get_joint_inertias_from_usd(input_usd, tempdir):
    encrypted_filename = input_usd
    fd, decrypted_filename = tempfile.mkstemp(suffix=".usd", dir=tempdir)
    os.close(fd)
    decrypt_file(encrypted_filename, decrypted_filename)
    stage = Usd.Stage.Open(str(decrypted_filename))
    prim = stage.GetDefaultPrim()

    # Build a graph of joints and links
    G = nx.DiGraph()
    link_masses = {}

    # Get all the links
    for child in prim.GetChildren():
        if not child.HasAPI(UsdPhysics.MassAPI):
            continue
        path = str(child.GetPath())
        massapi = UsdPhysics.MassAPI(child)
        mass = massapi.GetMassAttr().Get()
        link_masses[path] = mass

    # Get all the joints
    def find_all_prim_children_with_type(prim_type, root_prim):
        found_prims = []
        for child in root_prim.GetChildren():
            if prim_type in child.GetTypeName():
                found_prims.append(child)
            found_prims += find_all_prim_children_with_type(
                prim_type=prim_type, root_prim=child
            )
        return found_prims

    prismatics = find_all_prim_children_with_type("PhysicsPrismaticJoint", prim)
    revolutes = find_all_prim_children_with_type("PhysicsRevoluteJoint", prim)
    fixeds = find_all_prim_children_with_type("PhysicsFixedJoint", prim)
    joints = prismatics + revolutes + fixeds
    joint_children = {}
    for joint in joints:
        body0 = joint.GetRelationship("physics:body0").GetTargets()[0].__str__()
        body1 = joint.GetRelationship("physics:body1").GetTargets()[0].__str__()
        G.add_edge(body0, body1)
        joint_children[joint] = body1

    def _get_subtree(G, n):
        return {n} | nx.descendants(G, n)

    movable_joints = prismatics + revolutes
    joint_inertias = {
        str(joint.GetPath()): sum(
            link_masses[link] for link in _get_subtree(G, joint_children[joint])
        )
        for joint in movable_joints
    }

    return joint_inertias


def main():
    input_usds = list(DATASET_PATH.glob("objects/*/*/usd/*.usd"))
    input_usds.sort(key=lambda x: x.parts[-3])
    print(len(input_usds))

    # Scale up
    futures = {}
    with tempfile.TemporaryDirectory() as tempdir:
        with ProcessPoolExecutor() as executor:
            for input_usd in tqdm(input_usds, desc="Queueing up jobs"):
                future = executor.submit(
                    get_joint_inertias_from_usd, input_usd, tempdir
                )
                futures[future] = input_usd

            # Gather the results (with a tqdm progress bar)
            results = {}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing results"
            ):
                input_usd = futures[future]
                results[input_usd.parts[-3]] = future.result()

    with open(DATASET_PATH / "joint_inertias.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
