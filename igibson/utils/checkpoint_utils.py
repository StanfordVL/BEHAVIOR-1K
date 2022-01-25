"""This file contains utils for BEHAVIOR demo replay checkpoints."""
import json
import os



from igibson.utils.utils import restoreState


def save_checkpoint(simulator, root_directory):
    bullet_path = os.path.join(root_directory, "%d.bullet" % simulator.frame_count)
    urdf_path = os.path.join(root_directory, "%d.urdf" % simulator.frame_count)
    simulator.scene.save(urdf_path=urdf_path, pybullet_filename=bullet_path)


def load_checkpoint(simulator, root_directory, frame):
    bullet_path = os.path.join(root_directory, "%d.bullet" % frame)
    urdf_path = os.path.join(root_directory, "%d.urdf" % frame)
    simulator.scene.restore(urdf_path=urdf_path, pybullet_filename=bullet_path)
