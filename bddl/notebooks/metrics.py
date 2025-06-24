# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import h5py
import matplotlib.pyplot as plt
import numpy as np

filename = "../../iGibson/putting_away_Christmas_decorations_filtered_0_Beechwood_0_int_2021-03-13_22-51-53.hdf5"

f = h5py.File(filename, "r")
print("Activity Name: ", f.attrs['/metadata/task_name'])
print("Activity Definition: ", f.attrs['/metadata/task_instance'])
print("Scene ID: ", f.attrs['/metadata/scene_id'])
print("Start time: ", f.attrs['/metadata/start_time'])

# # Satisfied and unsatisfied predicates

fig, ax = plt.subplots()
ax.plot(np.sum(f['goal_status']['satisfied'][:], axis=1))
ax.plot(np.sum(f['goal_status']['unsatisfied'][:], axis=1))

# +
left_position = f['vr']['vr_device_data']['left_controller'][:, 1:4]
right_position = f['vr']['vr_device_data']['right_controller'][:, 1:4]
body_position = f['vr']['vr_device_data']['vr_position_data'][:, 0:3]

left_delta_position = np.linalg.norm(left_position[1:-1] - left_position[0:-2], axis=1)
right_delta_position = np.linalg.norm(right_position[1:-1] - right_position[0:-2], axis=1)
body_delta_position = np.linalg.norm(body_position[1:-1] - body_position[0:-2], axis=1)

left_delta_position = np.clip(-0.2, 0.2, left_delta_position)
right_delta_position = np.clip(-0.2, 0.2, right_delta_position)
body_delta_position = np.clip(-0.2, 0.2, body_delta_position)
# -

# # Cumulative distance traveled

fig, ax = plt.subplots()
ax.plot(np.cumsum(left_delta_position))
ax.plot(np.cumsum(right_delta_position))
ax.plot(np.cumsum(body_delta_position))

# # Instantaneous velocity

fig, ax = plt.subplots()
ax.plot(left_delta_position)
ax.plot(right_delta_position)
ax.plot(body_delta_position)

# # Grasping

# +
left_grasp = f['vr']['vr_button_data']['left_controller'][:, 0]
right_grasp = f['vr']['vr_button_data']['right_controller'][:, 0]

left_grasp_engaged = left_grasp * (left_grasp > 0.8)
right_grasp_engaged = right_grasp * (right_grasp > 0.8)
# -

fig, ax = plt.subplots()
ax.plot(left_grasp_engaged)
ax.plot(right_grasp_engaged)

# +
right_grasp_on_filter = np.convolve(right_grasp_engaged, np.array([1,-1]))
right_grasp_on = np.where(np.isclose(right_grasp_on_filter, 1, 0.2))

right_grasp_off_filter = np.convolve(right_grasp_engaged, np.array([-1,1]))
right_grasp_off = np.where(np.isclose(right_grasp_off_filter, -1, 0.2))


# +
left_grasp_on_filter = np.convolve(left_grasp_engaged, np.array([1,-1]))
left_grasp_on = np.where(np.isclose(left_grasp_on_filter, 1, 0.2))

left_grasp_off_filter = np.convolve(left_grasp_engaged, np.array([-1,1]))
left_grasp_off = np.where(np.isclose(left_grasp_off_filter, -1, 0.2))
# -
# # Force/Work


left_force = np.sum(f['vr']['vr_device_data']['left_controller'][:, 17:23], axis=1)
right_force = np.sum(f['vr']['vr_device_data']['right_controller'][:, 17:23], axis=1)
body_force = np.sum(f['vr']['vr_device_data']['vr_position_data'][:, 6:12], axis=1)

fig, ax = plt.subplots()
ax.plot(left_force)
ax.plot(right_force)
ax.plot(body_force)


