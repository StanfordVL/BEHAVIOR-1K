---
icon: octicons/rocket-16
---

# 📷 **Sensor**

Sensors play a crucial role in OmniGibson, as they facilitate the robots' observation of their environment. We offer two main classes of sensors:

 - `ScanSensor`: This includes a 2D LiDAR range sensor and an occupancy grid sensor.
 - `VisionSensor`: This sensor type features a camera equipped with various modalities, including RGB, depth, normals, three types of segmentation, optical flow, 2D and 3D bounding boxes.

To obtain sensor readings, the `get_obs()` function can be invoked at multiple levels within our hierarchy:

 - From `Environment`: Provides
    1. All observations from all robots
    2. All task-related observations
    3. Observations from external sensors, if available
 - From `Robot`: Provides
    1. Readings from all sensors associated with the robot
    2. Proprioceptive observations for the robot (e.g., base pose, joint position, joint velocity)
 - From `Sensor`: Delivers all sensor readings based on the sensor's modalities. Additionally, our API allows for the simulation of real-world sensor behaviors by:
    1. Adding noise
    2. Dropping out sensor values to emulate missing data in sensor readings

Besides the actual data, `get_obs()` also returns a secondary dictionary containing information about the data, such as segmentation labels for vision sensors.

For instance, calling `get_obs()` on an environment with a single robot, which has all modalities enabled, might produce results similar to this:

<details>
<summary>Click to see code!</summary>
<pre><code>
data: 
{
    "robot0": {
        "robot0:laser_link:Lidar:0": {
            "scan": np.array(...),
            "occupancy_grid": np.array(...)
        },
        "robot0:eyes:Camera:0": {
            "rgb": np.array(...),
            "depth": np.array(...),
            "depth_linear": np.array(...),
            "normal": np.array(...),
            "flow": np.array(...),
            "bbox_2d_tight": np.array(...),
            "bbox_2d_loose": np.array(...),
            "bbox_3d": np.array(...),
            "seg_semantic": np.array(...),
            "seg_instance": np.array(...),
            "seg_instance_id": np.array(...)
        },
        "proprio": np.array(...)
    }
    "task": {
        "low_dim": np.array(...)
    }
}

info:
{
    'robot0': {
        'robot0:laser_link:Lidar:0': {}, 
        'robot0:eyes:Camera:0': {
            'seg_semantic': {'298104422': 'object', '764121901': 'background', '2814990211': 'agent'}, 
            'seg_instance': {...}, 
            'seg_instance_id': {...}
        }, 
        'proprio': {}
    }
}
</code></pre>
</details>

Here's a selection of camera modalities:

<table>
    <tr>
        <td valign="top">
            <strong>RGB</strong><br>  
            RGB image of the scene from the camera perspective.<br>
            Size: (height, width, 4), numpy.uint8
        </td>
        <td>
            <img src="../assets/camera_asset/rgb.png" alt="rgb">
        </td>
    </tr>
    <tr>
        <td valign="top">
            <strong>Depth Map</strong><br>  
            Distance between the camera and everything else in the scene.<br>
            Size: (height, width), numpy.float32
        </td>
        <td>
            <img src="../assets/camera_asset/depth.png" alt="depth">
        </td>
    </tr>
    <tr>
        <td valign="top">
            <strong>Semantic Segmentation</strong><br>  
            Each pixel is assigned a label, indicating the object category it belongs to (e.g. table, chair).<br> 
            Size: (height, width), numpy.uint32
        </td>
        <td>
            <img src="../assets/camera_asset/seg_semantic.png" alt="seg_semantic">
        </td>
    </tr>
    <tr>
        <td valign="top">
            <strong>Instance Segmentation</strong><br> 
            Each pixel is assigned a label, indicating the specific object instance this pixel belongs to (e.g. table1, chair2).<br> 
            Size: (height, width), numpy.uint32
        </td>
        <td>
            <img src="../assets/camera_asset/seg_instance.png" alt="seg_instance">
        </td>
    </tr>
    <tr>
        <td valign="top">
            <strong>2D Bounding Box</strong><br>
            Bounding boxes wrapping individual objects.<br>
            Size: a list of(<br>
            - semanticID, numpy.uint32<br>
            - x_min, numpy.int32<br>
            - y_min, numpy.int32<br>
            - x_max, numpy.int32<br>
            - y_max, numpy.int32<br>
            - occlusion_ratio, numpy.float32)
        </td>
        <td>
            <img src="../assets/camera_asset/bbox_2d_tight.png" alt="bbox_2d">
        </td>
    </tr>
</table>

