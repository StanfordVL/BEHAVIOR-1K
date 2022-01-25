"""
Simple script to visualize the collision mesh of the VR hand in Pybullet
"""

import os
import time



from igibson import assets_path

p.connect(p.GUI)

vr_hand_col_path = os.path.join(
    assets_path, "models", "vr_agent", "vr_hand", "normal_color", "vr_hand_reduced_vis_coll.urdf"
)
p.loadURDF(vr_hand_col_path)

p.setRealTimeSimulation(0)
while True:
    time.sleep(0.01)
