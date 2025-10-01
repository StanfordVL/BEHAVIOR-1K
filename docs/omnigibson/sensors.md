# :material-camera-outline: **Sensor**

## Description

Sensors play a crucial role in OmniGibson, as they facilitate the robots' observation of their environment. We offer two main classes of sensors:

 - `ScanSensor`: This includes a 2D LiDAR range sensor and an occupancy grid sensor.
 - `VisionSensor`: This sensor type features a camera equipped with various modalities, including RGB, depth, normals, three types of segmentation, optical flow, 2D and 3D bounding boxes.

## Usage

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
<summary>Example observations</summary>
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
            "point_cloud": np.array(...)
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

## Types
**`OmniGibson`** currently supports two types of sensors (`VisionSensor`, `ScanSensor`), and three types of observations(vision, scan, low-dimensional). Below, we describe each of the types of observations:

### Vision

Vision observations are captured by the [`VisionSensor`](../reference/sensors/vision_sensor.md) class, which encapsulates a virtual pinhole camera sensor equipped with various modalities, including RGB, depth, normals, three types of segmentation, optical flow, 2D and 3D bounding boxes, shown below:

<table markdown="span">
    <tr>
        <td valign="top" width="60%">
            <strong>RGB</strong><br><br>  
            RGB image of the scene from the camera perspective.<br><br> 
            Size: (height, width, 4), numpy.uint8<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/rgb.png" alt="rgb">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>Depth</strong><br><br>  
            Distance between the camera and everything else in the scene.<br><br>
            Size: (height, width), numpy.float32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/depth.png" alt="Depth Map">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>Depth Linear</strong><br><br>  
            Distance between the camera and everything else in the scene, where distance measurement is linearly proportional to the actual distance.<br><br>
            Size: (height, width), numpy.float32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/depth_linear.png" alt="Depth Map Linear">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>Normal</strong><br><br>  
            Surface normals - vectors perpendicular to the surface of objects in the scene.<br><br>
            Size: (height, width, 4), numpy.float32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/normal.png" alt="Normal">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>Semantic Segmentation</strong><br><br>  
            Each pixel is assigned a label, indicating the object category it belongs to (e.g., table, chair).<br><br>
            Size: (height, width), numpy.uint32<br><br>
            We also provide a dictionary containing the mapping of semantic IDs to object categories. You can get this here: <br><br>
                from omnigibson.utils.constants import semantic_class_id_to_name
        </td>
        <td>
            <img src="../assets/sensor_asset/seg_semantic.png" alt="Semantic Segmentation">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>Instance Segmentation</strong><br><br>  
            Each pixel is assigned a label, indicating the specific object instance it belongs to (e.g., table1, chair2).<br><br>
            Size: (height, width), numpy.uint32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/seg_instance.png" alt="Instance Segmentation">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>Instance Segmentation ID</strong><br><br>  
            Each pixel is assigned a label, indicating the specific object instance it belongs to (e.g., /World/table1/visuals, /World/chair2/visuals).<br><br>
            Size: (height, width), numpy.uint32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/seg_instance_id.png" alt="Instance Segmentation ID">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>Optical Flow</strong><br><br>  
            Optical flow - motion of pixels belonging to objects caused by the relative motion between the camera and the scene.<br><br>
            Size: (height, width, 4), numpy.float32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/optical_flow.png" alt="Optical Flow">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>2D Bounding Box Tight</strong><br><br>  
            2D bounding boxes wrapping individual objects, excluding occluded parts.<br><br>
            Size: a list of <br>
            semanticID, numpy.uint32;<br> 
            x_min, numpy.int32;<br> 
            y_min, numpy.int32;<br>  
            x_max, numpy.int32;<br> 
            y_max, numpy.int32;<br> 
            occlusion_ratio, numpy.float32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/bbox_2d_tight.png" alt="2D Bounding Box Tight">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>2D Bounding Box Loose</strong><br><br>  
            2D bounding boxes wrapping individual objects, including occluded parts.<br><br>
            Size: a list of <br>
            semanticID, numpy.uint32;<br> 
            x_min, numpy.int32;<br> 
            y_min, numpy.int32;<br>  
            x_max, numpy.int32;<br> 
            y_max, numpy.int32;<br> 
            occlusion_ratio, numpy.float32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/bbox_2d_loose.png" alt="2D Bounding Box Loose">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>3D Bounding Box</strong><br><br>  
            3D bounding boxes wrapping individual objects.<br><br>
            Size: a list of <br>
            semanticID, numpy.uint32;<br> 
            x_min, numpy.float32;<br>
            y_min, numpy.float32;<br>
            z_min, numpy.float32;<br>
            x_max, numpy.float32;<br>
            y_max, numpy.float32;<br>
            z_max, numpy.float32;<br>
            transform (4x4), numpy.float32;<br>
            occlusion_ratio, numpy.float32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/bbox_3d.png" alt="3D Bounding Box">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>Point Cloud</strong><br><br>  
            Point cloud obtained from the camera.<br><br>
            Size: (height, width, 6), numpy.uint8<br>
            Last dimension is (r, g, b, x, y, z)<br>
            NOTE: coordinates are in camera frame<br>
        </td>
        <td>
            <img src="../assets/sensor_asset/pcd.png" alt="Point Cloud">
        </td>
    </tr>
</table>

### Range
Range observations are captured by the [`ScanSensor`](../reference/sensors/scan_sensor.md) class, which encapsulates a virtual 2D LiDAR range sensor with the following observations:

<table markdown="span">
    <tr>
        <td valign="top" width="60%">
            <strong>2D LiDAR</strong><br><br>  
            Distances to surrounding objects by emitting laser beams and detecting the reflected light.<br><br>
            Size: # of rays, numpy.float32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/lidar.png" alt="2D LiDAR">
        </td>
    </tr>
    <tr>
        <td valign="top" width="60%">
            <strong>Occupancy Grid</strong><br><br>  
            A representation of the environment as a 2D grid where each cell indicates the presence (or absence) of an obstacle.<br><br>
            Size: (grid resolution, grid resolution), numpy.float32<br><br>
        </td>
        <td>
            <img src="../assets/sensor_asset/occupancy_grid.png" alt="Occupancy Grid">
        </td>
    </tr>
</table>

### Low-Dimensional
Low-dimensional observations are not captured by any specific sensor, but are simply an aggregation of the underlying simulator state. There are two main types of low-dimensional observations: proprioception and task-relevant:


#### Proprioception
The following proprioceptive observations are supported off-the-shelf in **`OmniGibson`** (though additional ones may arbitrarily be added):

<table markdown="span">
    <tr>
        <td valign="top" width="100%">
            <strong>Joint Positions</strong><br><br>  
            Joint positions.<br><br>
            Size: # of joints, numpy.float64<br><br>
        </td>
        <td>
        </td>
    </tr>
    <tr>
        <td valign="top" width="100%">
            <strong>Joint Velocities</strong><br><br>  
            Joint velocities.<br><br>
            Size: # of joints, numpy.float64<br><br>
        </td>
        <td>
        </td>
    </tr>
    <tr>
        <td valign="top" width="100%">
            <strong>Joint Efforts</strong><br><br>  
            Torque measured at each joint.<br><br>
            Size: # of joints, numpy.float64<br><br>
        </td>
        <td>
        </td>
    </tr>
    <tr>
        <td valign="top" width="100%">
            <strong>Robot Position</strong><br><br>  
            Robot position in the world frame.<br><br>
            Size: (x, y, z), numpy.float64<br><br>
        </td>
        <td>
        </td>
    </tr>
    <tr>
        <td valign="top" width="100%">
            <strong>Robot Orientation</strong><br><br>  
            Robot global euler orientation.<br><br>
            Size: (roll, pitch, yaw), numpy.float64<br><br>
        </td>
        <td>
        </td>
    </tr>
    <tr>
        <td>
            <strong>Robot 2D Orientation</strong><br><br>  
            Robot orientation on the XY plane of the world frame.<br><br>
            Size: angle, numpy.float64<br><br>
        </td>
        <td>
        </td>
    </tr>
    <tr>
        <td valign="top" width="100%">
            <strong>Robot Linear Velocity</strong><br><br>  
            Robot linear velocity.<br><br>
            Size: (x_vel, y_vel, z_vel), numpy.float64<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top" width="100%">
            <strong>Robot Angular Velocity</strong><br><br>  
            Robot angular velocity.<br><br>
            Size: (x_vel, y_vel, z_vel), numpy.float64<br><br>
        </td>
        <td>
        </td>
    </tr>
</table>

#### Task-Relevant
Each task implements its own set of relevant observations:

<table markdown="span" style="width: 100%;">
    <tr>
        <td valign="top" width="100%">
            <strong>Low-dim task observation</strong><br><br>  
            Task-specific observation, e.g. navigation goal position.<br><br>
            Size: # of low-dim observation, numpy.float64<br><br>
        </td>
        <td>
        </td>
    </tr>
</table>

