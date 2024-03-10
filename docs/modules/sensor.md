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

<details>
    <summary><strong>RGB</strong></summary>
    <p>RGB image of the scene from the camera perspective.</p>
    <p>Size: (height, width, 4), numpy.uint8</p>
    <img src="../assets/camera_asset/rgb.png" alt="rgb">
</details>

<details>
    <summary><strong>Depth</strong></summary>
    <p>Distance between the camera and everything else in the scene.</p>
    <p>Size: (height, width), numpy.float32</p>
    <img src="../assets/camera_asset/depth.png" alt="Depth Map">
</details>

<details>
    <summary><strong>Depth Linear</strong></summary>
    <p>Distance between the camera and everything else in the scene, where distance measurement is linearly proportional to the actual distance.</p>
    <p>Size: (height, width), numpy.float32</p>
    <img src="../assets/camera_asset/depth_linear.png" alt="Depth Map Linear">
</details>

<details>
    <summary><strong>Normal</strong></summary>
    <p>Surface normals - vectors perpendicular to the surface of objects in the scene.</p>
    <p>Size: (height, width, 4), numpy.float32</p>
    <img src="../assets/camera_asset/normal.png" alt="Normal">
</details>

<details>
    <summary><strong>Semantic Segmentation</strong></summary>
    <p>Each pixel is assigned a label, indicating the object category it belongs to (e.g., table, chair).</p>
    <p>Size: (height, width), numpy.uint32</p>
    <img src="../assets/camera_asset/seg_semantic.png" alt="Semantic Segmentation">
</details>

<details>
    <summary><strong>Instance Segmentation</strong></summary>
    <p>Each pixel is assigned a label, indicating the specific object instance it belongs to (e.g., table1, chair2).</p>
    <p>Size: (height, width), numpy.uint32</p>
    <img src="../assets/camera_asset/seg_instance.png" alt="Instance Segmentation">
</details>

<details>
    <summary><strong>Instance Segmentation ID</strong></summary>
    <p>Each pixel is assigned a label, indicating the specific object instance it belongs to (e.g., /World/table1/visuals, /World/chair2/visuals).</p>
    <p>Size: (height, width), numpy.uint32</p>
    <img src="../assets/camera_asset/seg_instance_id.png" alt="Instance Segmentation ID">
</details>

<details>
    <summary><strong>Optical Flow</strong></summary>
    <p>Optical flow - motion of pixels belonging to objects caused by the relative motion between the camera and the scene.</p>
    <p>Size: (height, width, 4), numpy.float32</p>
</details>

<details>
    <summary><strong>2D Bounding Box Tight</strong></summary>
    <p>2D bounding boxes wrapping individual objects, excluding any parts that are occluded.</p>
    <p>Size: a list of <br>
        semanticID, numpy.uint32;<br> 
        x_min, numpy.int32;<br> 
        y_min, numpy.int32;<br>  
        x_max, numpy.int32;<br> 
        y_max, numpy.int32;<br> 
        occlusion_ratio, numpy.float32</p>
    <img src="../assets/camera_asset/bbox_2d_tight.png" alt="2D Bounding Box Tight">
</details>

<details>
    <summary><strong>2D Bounding Box Loose</strong></summary>
    <p>2D bounding boxes wrapping individual objects, including occluded parts.</p>
    <p>Size: a list of <br>
        semanticID, numpy.uint32;<br> 
        x_min, numpy.int32;<br> 
        y_min, numpy.int32;<br>  
        x_max, numpy.int32;<br> 
        y_max, numpy.int32;<br> 
        occlusion_ratio, numpy.float32</p>
    <img src="../assets/camera_asset/bbox_2d_loose.png" alt="2D Bounding Box Loose">
</details>

<details>
    <summary><strong>3D Bounding Box</strong></summary>
    <p>3D bounding boxes wrapping individual objects.</p>
    <p>Size: a list of <br>
        semanticID, numpy.uint32;<br> 
        x_min, numpy.float32;<br>
        y_min, numpy.float32;<br>
        z_min, numpy.float32;<br>
        x_max, numpy.float32;<br>
        y_max, numpy.float32;<br>
        z_max, numpy.float32;<br>
        transform (4x4), numpy.float32;<br>
        occlusion_ratio, numpy.float32</p>
</details>
