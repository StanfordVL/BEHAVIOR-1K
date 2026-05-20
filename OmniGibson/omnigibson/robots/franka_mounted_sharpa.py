"""
Constants for the Franka + SharpaWave dexterous hand robots.

The FrankaMountedSharpaRight / Left robot classes have been replaced by YAML
definitions in omnigibson/robots/definitions/franka_mounted_sharpa_{side}.yaml.
This file is kept so that external scripts can still import SHARPA_JOINT_LIMITS.
"""

import torch as th

# Joint limits for the 22 SharpaWave hand joints, in the order they appear in
# finger_joint_names (both right and left hands share the same limits):
#   thumb_CMC_FE, thumb_CMC_AA, thumb_MCP_FE, thumb_MCP_AA, thumb_IP,
#   index_MCP_FE, index_MCP_AA, index_PIP, index_DIP,
#   middle_MCP_FE, middle_MCP_AA, middle_PIP, middle_DIP,
#   ring_MCP_FE, ring_MCP_AA, ring_PIP, ring_DIP,
#   pinky_CMC, pinky_MCP_FE, pinky_MCP_AA, pinky_PIP, pinky_DIP
#
# Shape: (22, 2)  — column 0 = lower limit, column 1 = upper limit (radians)
SHARPA_JOINT_LIMITS = th.tensor([
    [-0.1745,  1.9199],  # thumb_CMC_FE
    [-0.3491,  0.3491],  # thumb_CMC_AA
    [-0.5236,  1.3963],  # thumb_MCP_FE
    [-0.3491,  0.3491],  # thumb_MCP_AA
    [ 0.0000,  1.7453],  # thumb_IP
    [-0.1745,  1.5708],  # index_MCP_FE
    [-0.3491,  0.3491],  # index_MCP_AA
    [ 0.0000,  1.7453],  # index_PIP
    [ 0.0000,  1.3963],  # index_DIP
    [-0.1745,  1.5708],  # middle_MCP_FE
    [-0.3491,  0.3491],  # middle_MCP_AA
    [ 0.0000,  1.7453],  # middle_PIP
    [ 0.0000,  1.3963],  # middle_DIP
    [-0.1745,  1.5708],  # ring_MCP_FE
    [-0.3491,  0.3491],  # ring_MCP_AA
    [ 0.0000,  1.7453],  # ring_PIP
    [ 0.0000,  1.3963],  # ring_DIP
    [ 0.0000,  0.2618],  # pinky_CMC
    [-0.1745,  1.5708],  # pinky_MCP_FE
    [-0.3491,  0.3491],  # pinky_MCP_AA
    [ 0.0000,  1.7453],  # pinky_PIP
    [ 0.0000,  1.3963],  # pinky_DIP
], dtype=th.float32)
