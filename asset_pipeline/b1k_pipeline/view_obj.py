import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm

import pybullet as p
import pybullet_data


def main():
    # Load the scene into iGibson 2
    p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    p.loadURDF("plane.urdf")

    dataset_path = r"C:\Users\cgokmen\Downloads\urdf-test-2-3"
    cat = "ice_tray"
    obj = "gewlsk"
    urdf_path = os.path.join(dataset_path, "objects", cat, obj, "urdf", f"{obj}.urdf")

    p.loadURDF(urdf_path, useFixedBase=False, flags=p.URDF_USE_INERTIA_FROM_FILE)

    # Step the simulation by 5 seconds.
    while True:
        p.stepSimulation()


if __name__ == "__main__":
    main()
