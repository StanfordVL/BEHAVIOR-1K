import numpy as np
import time
import gym

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.sensors.sensor_base import BaseSensor
from omnigibson.systems.system_base import REGISTERED_SYSTEMS
from omnigibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT, MAX_VIEWER_SIZE, semantic_class_name_to_id, semantic_class_id_to_name
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.sim_utils import set_carb_setting
from omnigibson.utils.ui_utils import dock_window
from omnigibson.utils.vision_utils import Remapper


# Duplicate of simulator's render method, used so that this can be done before simulator is created!
def render():
    """
    Refreshes the Isaac Sim app rendering components including UI elements and view ports..etc.
    """
    set_carb_setting(og.app._carb_settings, "/app/player/playSimulations", False)
    og.app.update()
    set_carb_setting(og.app._carb_settings, "/app/player/playSimulations", True)

class VisionSensor(BaseSensor):
    """
    Vision sensor that handles a variety of modalities, including:

        - RGB (normal)
        - Depth (normal, linear)
        - Normals
        - Segmentation (semantic, instance)
        - Optical flow
        - 2D Bounding boxes (tight, loose)
        - 3D Bounding boxes
        - Camera state

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        modalities (str or list of str): Modality(s) supported by this sensor. Default is "all", which corresponds
            to all modalities being used. Otherwise, valid options should be part of cls.all_modalities.
            For this vision sensor, this includes any of:
                {rgb, depth, depth_linear, normal, seg_semantic, seg_instance, flow, bbox_2d_tight,
                bbox_2d_loose, bbox_3d, camera}
        enabled (bool): Whether this sensor should be enabled by default
        noise (None or BaseSensorNoise): If specified, sensor noise model to apply to this sensor.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this sensor's prim at runtime.
        image_height (int): Height of generated images, in pixels
        image_width (int): Width of generated images, in pixels
        focal_length (float): Focal length to set
        clipping_range (2-tuple): (min, max) viewing range of this vision sensor
        viewport_name (None or str): If specified, will link this camera to the specified viewport, overriding its
            current camera. Otherwise, creates a new viewport
    """

    ALL_MODALITIES = (
        "rgb",
        "depth",
        "depth_linear",
        "normal",
        "seg_semantic",  # Semantic segmentation shows the category each pixel belongs to
        "seg_instance",  # Instance segmentation shows the name of the object each pixel belongs to
        "seg_instance_id",  # Instance ID segmentation shows the prim path of the visual mesh each pixel belongs to
        "flow",
        "bbox_2d_tight",
        "bbox_2d_loose",
        "bbox_3d",
        "camera_params",
    )

    # Documentation for the different types of segmentation for particle systems:
    # - Cloth (e.g. `dishtowel`):
    #   - semantic: all shows up under one semantic label (e.g. `"4207839377": "dishtowel"`)
    #   - instance: entire cloth shows up under one label (e.g. `"87": "dishtowel_0"`)
    #   - instance id: entire cloth shows up under one label (e.g. `"31": "/World/dishtowel_0/base_link_cloth"`)
    # - MicroPhysicalParticleSystem - FluidSystem (e.g. `water`):
    #   - semantic: all shows up under one semantic label (e.g. `"3330677804": "water"`)
    #   - instance: all shows up under one instance label (e.g. `"21": "water"`)
    #   - instance id: all shows up under one instance ID label (e.g. `"36": "water"`)
    # - MicroPhysicalParticleSystem - GranularSystem (e.g. `sesame seed`):
    #   - semantic: all shows up under one semantic label (e.g. `"2975304485": "sesame_seed"`)
    #   - instance: all shows up under one instance label (e.g. `"21": "sesame_seed"`)
    #   - instance id: all shows up under one instance ID label (e.g. `"36": "sesame_seed"`)
    # - MacroPhysicalParticleSystem (e.g. `diced__carrot`):
    #   - semantic: all shows up under one semantic label (e.g. `"2419487146": "diced__carrot"`)
    #   - instance: all shows up under one instance label (e.g. `"21": "diced__carrot"`)
    #   - instance id: all shows up under one instance ID label (e.g. `"36": "diced__carrot"`)
    # - MacroVisualParticleSystem (e.g. `stain`):
    #   - semantic: all shows up under one semantic label (e.g. `"884110082": "stain"`)
    #   - instance: all shows up under one instance label (e.g. `"21": "stain"`)
    #   - instance id: all shows up under one instance ID label (e.g. `"36": "stain"`)

    # Persistent dictionary of sensors, mapped from prim_path to sensor
    SENSORS = dict()
    
    SEMANTIC_REMAPPER = Remapper()
    INSTANCE_REMAPPER = Remapper()
    INSTANCE_ID_REMAPPER = Remapper()
    INSTANCE_REGISTRY = {0: "background", 1: "unlabelled"}
    INSTANCE_ID_REGISTRY = {0: "background"}

    def __init__(
        self,
        prim_path,
        name,
        modalities="all",
        enabled=True,
        noise=None,
        load_config=None,
        image_height=128,
        image_width=128,
        focal_length=17.0,                          # Default 17.0 since this is roughly the human eye focal length
        clipping_range=(0.001, 10000000.0),
        viewport_name=None,
    ):
        # Create load config from inputs
        load_config = dict() if load_config is None else load_config
        load_config["image_height"] = image_height
        load_config["image_width"] = image_width
        load_config["focal_length"] = focal_length
        load_config["clipping_range"] = clipping_range
        load_config["viewport_name"] = viewport_name

        # Create variables that will be filled in later at runtime
        self._viewport = None       # Viewport from which to grab data
        self._annotators = None
        self._render_product = None
        
        self._RAW_SENSOR_TYPES = dict(
            rgb="rgb",
            depth="distance_to_camera",
            depth_linear="distance_to_image_plane",
            normal="normals",
            # Semantic segmentation shows the category each pixel belongs to
            seg_semantic="semantic_segmentation",
            # Instance segmentation shows the name of the object each pixel belongs to
            seg_instance="instance_segmentation",
            # Instance ID segmentation shows the prim path of the visual mesh each pixel belongs to
            seg_instance_id="instance_id_segmentation",
            flow="motion_vectors",
            bbox_2d_tight="bounding_box_2d_tight",
            bbox_2d_loose="bounding_box_2d_loose",
            bbox_3d="bounding_box_3d",
            camera_params="camera_params",
        )

        assert {key for key in self._RAW_SENSOR_TYPES.keys() if key != "camera_params"} == set(self.all_modalities), \
            "VisionSensor._RAW_SENSOR_TYPES must have the same keys as VisionSensor.all_modalities!"

        modalities = set([modalities]) if isinstance(modalities, str) else modalities

        # 1) seg_instance and seg_instance_id require seg_semantic to be enabled (for rendering particle systems)
        # 2) bounding box observations require seg_semantic to be enabled (for remapping bounding box semantic IDs)
        semantic_dependent_modalities = {"seg_instance", "seg_instance_id", "bbox_2d_loose", "bbox_2d_tight", "bbox_3d"}
        # if any of the semantic dependent modalities are enabled, then seg_semantic must be enabled
        if semantic_dependent_modalities.intersection(modalities) and "seg_semantic" not in modalities:
            modalities.add("seg_semantic")

        # Run super method
        super().__init__(
            prim_path=prim_path,
            name=name,
            modalities=modalities,
            enabled=enabled,
            noise=noise,
            load_config=load_config,
        )

    def _load(self):
        # Define a new camera prim at the current stage
        # Note that we can't use og.sim.stage here because the vision sensors get loaded first
        return lazy.pxr.UsdGeom.Camera.Define(lazy.omni.isaac.core.utils.stage.get_current_stage(), self._prim_path).GetPrim()

    def _post_load(self):
        # run super first
        super()._post_load()

        # Add this sensor to the list of global sensors
        self.SENSORS[self._prim_path] = self
        
        resolution = (self._load_config["image_width"], self._load_config["image_height"])
        self._render_product = lazy.omni.replicator.core.create.render_product(self._prim_path, resolution)

        # Create a new viewport to link to this camera or link to a pre-existing one
        viewport_name = self._load_config["viewport_name"]
        if viewport_name is not None:
            vp_names_to_handles = {vp.name: vp for vp in lazy.omni.kit.viewport.window.get_viewport_window_instances()}
            assert_valid_key(key=viewport_name, valid_keys=vp_names_to_handles, name="viewport name")
            viewport = vp_names_to_handles[viewport_name]
        else:
            viewport = lazy.omni.kit.viewport.utility.create_viewport_window()
            # Take a render step to make sure the viewport is generated before docking it
            render()
            # Grab the newly created viewport and dock it to the GUI
            # The first viewport is always the "main" global camera, and any additional cameras are auxiliary views
            # These auxiliary views will be stacked in a single column
            # Thus, the first auxiliary viewport should be generated to the left of the main dockspace, and any
            # subsequent viewports should be equally spaced according to the number of pre-existing auxiliary views
            n_auxiliary_sensors = len(self.SENSORS) - 1
            if n_auxiliary_sensors == 1:
                # This is the first auxiliary viewport, dock to the left of the main dockspace
                dock_window(space=lazy.omni.ui.Workspace.get_window("DockSpace"), name=viewport.name,
                            location=lazy.omni.ui.DockPosition.LEFT, ratio=0.25)
            elif n_auxiliary_sensors > 1:
                # This is any additional auxiliary viewports, dock equally-spaced in the auxiliary column
                # We also need to re-dock any prior viewports!
                for i in range(2, n_auxiliary_sensors + 1):
                    dock_window(space=lazy.omni.ui.Workspace.get_window(f"Viewport {i - 1}"), name=f"Viewport {i}",
                                location=lazy.omni.ui.DockPosition.BOTTOM, ratio=(1 + n_auxiliary_sensors - i) / (2 + n_auxiliary_sensors - i))

        self._viewport = viewport

        # Link the camera and viewport together
        self._viewport.viewport_api.set_active_camera(self._prim_path)

        # Requires 3 render updates to propagate changes
        for i in range(3):
            render()

        # Set the viewer size (requires taking one render step afterwards)
        self._viewport.viewport_api.set_texture_resolution(resolution)

        # Also update focal length and clipping range
        self.focal_length = self._load_config["focal_length"]
        self.clipping_range = self._load_config["clipping_range"]

        # Requires 3 render updates to propagate changes
        for i in range(3):
            render()

    def _initialize(self):
        # Run super first
        super()._initialize()
        
        self._annotators = {modality: None for modality in self._modalities}

        # Initialize sensors
        self.initialize_sensors(names=self._modalities)
        for _ in range(3):
            render()

    def initialize_sensors(self, names):
        """Initializes a raw sensor in the simulation.

        Args:
            names (str or list of str): Name of the raw sensor(s) to initialize.
                If they are not part of self._RAW_SENSOR_TYPES' keys, we will simply pass over them
        """
        names = {names} if isinstance(names, str) else set(names)
        for name in names:
            self._add_modality_to_backend(modality=name)

    def _get_obs(self):
        # Make sure we're initialized
        assert self.initialized, "Cannot grab vision observations without first initializing this VisionSensor!"

        # Run super first to grab any upstream obs
        obs, info = super()._get_obs()

        # Reorder modalities to ensure that seg_semantic is always ran before seg_instance or seg_instance_id
        if "seg_semantic" in self._modalities:
            reordered_modalities = ["seg_semantic"] + [modality for modality in self._modalities if modality != "seg_semantic"]
        else:
            reordered_modalities = self._modalities

        for modality in reordered_modalities:
            raw_obs = self._annotators[modality].get_data()
            # Obs is either a dictionary of {"data":, ..., "info": ...} or a direct array
            obs[modality] = raw_obs["data"] if isinstance(raw_obs, dict) else raw_obs
            if modality == "seg_semantic":
                id_to_labels = raw_obs["info"]["idToLabels"]
                obs[modality], info[modality] = self._remap_semantic_segmentation(obs[modality], id_to_labels)
            elif modality == "seg_instance":
                id_to_labels = raw_obs["info"]["idToLabels"]
                obs[modality], info[modality] = self._remap_instance_segmentation(
                    obs[modality], id_to_labels, obs["seg_semantic"], info["seg_semantic"], id=False)
            elif modality == "seg_instance_id":
                id_to_labels = raw_obs["info"]["idToLabels"]
                obs[modality], info[modality] = self._remap_instance_segmentation(
                    obs[modality], id_to_labels, obs["seg_semantic"], info["seg_semantic"], id=True)
            elif "bbox" in modality:
                obs[modality] = self._remap_bounding_box_semantic_ids(obs[modality])
        return obs, info
    
    def _remap_semantic_segmentation(self, img, id_to_labels):
        """
        Remap the semantic segmentation image to the class IDs defined in semantic_class_name_to_id().
        Also, correct the id_to_labels input with the labels from semantic_class_name_to_id() and return it.
        
        Args:
            img (np.ndarray): Semantic segmentation image to remap
            id_to_labels (dict): Dictionary of semantic IDs to class labels
        Returns:
            np.ndarray: Remapped semantic segmentation image
            dict: Corrected id_to_labels dictionary
        """
        # Preprocess id_to_labels to feed into the remapper
        replicator_mapping = {}
        for key, val in id_to_labels.items():
            key = int(key)
            replicator_mapping[key] = val["class"].lower()
            if "," in replicator_mapping[key]:
                # If there are multiple class names, grab the one that is a registered system
                # This happens with MacroVisual particles, e.g. {"11": {"class": "breakfast_table,stain"}}
                categories = [cat for cat in replicator_mapping[key].split(",") if cat in REGISTERED_SYSTEMS]
                assert len(categories) == 1, "There should be exactly one category that belongs to REGISTERED_SYSTEMS"
                replicator_mapping[key] = categories[0]

            assert replicator_mapping[key] in semantic_class_id_to_name().values(), f"Class {val['class']} does not exist in the semantic class name to id mapping!"

        assert set(np.unique(img)).issubset(set(replicator_mapping.keys())), "Semantic segmentation image does not match the original id_to_labels mapping."

        return VisionSensor.SEMANTIC_REMAPPER.remap(replicator_mapping, semantic_class_id_to_name(), img)

    def _remap_instance_segmentation(self, img, id_to_labels, semantic_img, semantic_labels, id=False):
        """
        Remap the instance segmentation image to our own instance IDs.
        Also, correct the id_to_labels input with our new labels and return it.
        
        Args:
            img (np.ndarray): Instance segmentation image to remap
            id_to_labels (dict): Dictionary of instance IDs to class labels
            semantic_img (np.ndarray): Semantic segmentation image to use for instance registry
            semantic_labels (dict): Dictionary of semantic IDs to class labels
            id (bool): Whether to remap for instance ID segmentation
        Returns:
            np.ndarray: Remapped instance segmentation image
            dict: Corrected id_to_labels dictionary
        """
        # Sometimes 0 and 1 show up in the image, but they are not in the id_to_labels mapping
        id_to_labels.update({"0": "BACKGROUND"})
        if not id:
            id_to_labels.update({"1": "UNLABELLED"})

        # Preprocess id_to_labels and update instance registry
        replicator_mapping = {}
        for key, value in id_to_labels.items():
            key = int(key)
            if value in ["BACKGROUND", "UNLABELLED"]:
                value = value.lower()
            else:
                assert "/" in value, f"Instance segmentation (ID) label {value} is not a valid prim path!"
                prim_name = value.split("/")[-1]
                # Hacky way to get the particles of MacroVisual/PhysicalParticleSystem
                # Remap instance segmentation and instance segmentation ID labels to system name
                if "Particle" in prim_name:
                    category_name = prim_name.split("Particle")[0]
                    assert category_name in REGISTERED_SYSTEMS, f"System name {category_name} is not in the registered systems!"
                    value = category_name
                else:
                    # Remap instance segmentation labels to object name
                    if not id:
                        # value is the prim path of the object
                        if value == "/World/groundPlane":
                            value = "groundPlane"
                        else:
                            obj = og.sim.scene.object_registry("prim_path", value)
                            # Remap instance segmentation labels from prim path to object name
                            assert obj is not None, f"Object with prim path {value} cannot be found in objct registry!"
                            value = obj.name

                    # Keep the instance segmentation ID labels intact (prim paths of visual meshes)
                    else:
                        pass

            self._register_instance(value, id=id)
            replicator_mapping[key] = value

        # Handle the cases for MicroPhysicalParticleSystem (FluidSystem, GranularSystem).
        # They show up in the image, but not in the info (id_to_labels).
        # We identify these values, find the corresponding semantic label (system name), and add the mapping.
        for key, img_idx in zip(*np.unique(img, return_index=True)):
            if str(key) not in id_to_labels:
                semantic_label = semantic_img.flatten()[img_idx]
                assert semantic_label in semantic_labels, f"Semantic map value {semantic_label} is not in the semantic labels!"
                category_name = semantic_labels[semantic_label]
                if category_name in REGISTERED_SYSTEMS:
                    value = category_name
                    self._register_instance(value, id=id)
                # If the category name is not in the registered systems, 
                # which happens because replicator sometimes returns segmentation map and id_to_labels that are not in sync,
                # we will label this as "unlabelled" for now
                else:
                    value = "unlabelled"
                replicator_mapping[key] = value

        registry = VisionSensor.INSTANCE_ID_REGISTRY if id else VisionSensor.INSTANCE_REGISTRY
        remapper = VisionSensor.INSTANCE_ID_REMAPPER if id else VisionSensor.INSTANCE_REMAPPER

        assert set(np.unique(img)).issubset(set(replicator_mapping.keys())), "Instance segmentation image does not match the original id_to_labels mapping."

        return remapper.remap(replicator_mapping, registry, img)

    def _register_instance(self, instance_name, id=False):
        registry = VisionSensor.INSTANCE_ID_REGISTRY if id else VisionSensor.INSTANCE_REGISTRY
        if instance_name not in registry.values():
            registry[len(registry)] = instance_name
    
    def _remap_bounding_box_semantic_ids(self, bboxes):
        """
        Remap the semantic IDs of the bounding boxes to our own semantic IDs.
        
        Args:
            bboxes (list of dict): List of bounding boxes to remap
        Returns:
            list of dict: Remapped list of bounding boxes
        """
        for bbox in bboxes:
            bbox["semanticId"] = VisionSensor.SEMANTIC_REMAPPER.remap_bbox(bbox["semanticId"])
        return bboxes
    
    def add_modality(self, modality):
        # Check if we already have this modality (if so, no need to initialize it explicitly)
        should_initialize = modality not in self._modalities

        # Run super
        super().add_modality(modality=modality)

        # We also need to initialize this new modality
        if should_initialize:
            self.initialize_sensors(names=modality)
    
    def remove_modality(self, modality):
        # Check if we don't have this modality (if not, no need to remove it explicitly)
        should_remove = modality in self._modalities

        # Run super
        super().remove_modality(modality=modality)

        if should_remove:
            self._remove_modality_from_backend(modality=modality)
    
    def _add_modality_to_backend(self, modality):
        """
        Helper function to add specified modality @modality to the omniverse Replicator backend so that its data is
        generated during get_obs()
        Args:
            modality (str): Name of the modality to add to the Replicator backend
        """
        if self._annotators.get(modality, None) is None:
            self._annotators[modality] = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(self._RAW_SENSOR_TYPES[modality])
            self._annotators[modality].attach([self._render_product])

    def _remove_modality_from_backend(self, modality):
        """
        Helper function to remove specified modality @modality from the omniverse Replicator backend so that its data is
        no longer generated during get_obs()
        Args:
            modality (str): Name of the modality to remove from the Replicator backend
        """
        if self._annotators.get(modality, None) is not None:
            self._annotators[modality].detach([self._render_product])
            self._annotators[modality] = None

    def remove(self):
        # Remove from global sensors dictionary
        self.SENSORS.pop(self._prim_path)

        # Remove viewport
        self._viewport.destroy()

        # Run super
        super().remove()
    
    @property
    def camera_parameters(self):
        """
        Returns a dictionary of keyword-mapped relevant intrinsic and extrinsic camera parameters for this vision sensor.
        The returned dictionary includes the following keys and their corresponding data types:

        - "cameraAperture": np.ndarray (float32) - Camera aperture dimensions.
        - "cameraApertureOffset": np.ndarray (float32) - Offset of the camera aperture.
        - "cameraFisheyeLensP": np.ndarray (float32) - Fisheye lens P parameter.
        - "cameraFisheyeLensS": np.ndarray (float32) - Fisheye lens S parameter.
        - "cameraFisheyeMaxFOV": float - Maximum field of view for fisheye lens.
        - "cameraFisheyeNominalHeight": int - Nominal height for fisheye lens.
        - "cameraFisheyeNominalWidth": int - Nominal width for fisheye lens.
        - "cameraFisheyeOpticalCentre": np.ndarray (float32) - Optical center for fisheye lens.
        - "cameraFisheyePolynomial": np.ndarray (float32) - Polynomial parameters for fisheye lens distortion.
        - "cameraFocalLength": float - Focal length of the camera.
        - "cameraFocusDistance": float - Focus distance of the camera.
        - "cameraFStop": float - F-stop value of the camera.
        - "cameraModel": str - Camera model identifier.
        - "cameraNearFar": np.ndarray (float32) - Near and far plane distances.
        - "cameraProjection": np.ndarray (float32) - Camera projection matrix.
        - "cameraViewTransform": np.ndarray (float32) - Camera view transformation matrix.
        - "metersPerSceneUnit": float - Scale factor from scene units to meters.
        - "renderProductResolution": np.ndarray (int32) - Resolution of the rendered product.

        Returns:
            dict: Keyword-mapped relevant intrinsic and extrinsic camera parameters for this vision sensor.
        """
        # Add the camera params modality if it doesn't already exist
        if "camera_params" not in self._annotators:
            self.initialize_sensors(names="camera_params")
            # Requires 3 render updates for camera params annotator to decome active
            for _ in range(3):
                render()

        # Grab and return the parameters
        return self._annotators["camera_params"].get_data()

    @property
    def viewer_visibility(self):
        """
        Returns:
            bool: Whether the viewer is visible or not
        """
        return self._viewport.visible

    @viewer_visibility.setter
    def viewer_visibility(self, visible):
        """
        Sets whether the viewer should be visible or not in the Omni UI

        Args:
            visible (bool): Whether the viewer should be visible or not
        """
        self._viewport.visible = visible
        # Requires 1 render update to propagate changes
        render()

    @property
    def image_height(self):
        """
        Returns:
            int: Image height of this sensor, in pixels
        """
        return self._viewport.viewport_api.get_texture_resolution()[1]

    @image_height.setter
    def image_height(self, height):
        """
        Sets the image height @height for this sensor

        Args:
            height (int): Image height of this sensor, in pixels
        """
        width, _ = self._viewport.viewport_api.get_texture_resolution()
        self._viewport.viewport_api.set_texture_resolution((width, height))
        # Requires 3 updates to propagate changes
        for i in range(3):
            render()

    @property
    def image_width(self):
        """
        Returns:
            int: Image width of this sensor, in pixels
        """
        return self._viewport.viewport_api.get_texture_resolution()[0]

    @image_width.setter
    def image_width(self, width):
        """
        Sets the image width @width for this sensor

        Args:
            width (int): Image width of this sensor, in pixels
        """
        _, height = self._viewport.viewport_api.get_texture_resolution()
        self._viewport.viewport_api.set_texture_resolution((width, height))
        # Requires 3 updates to propagate changes
        for i in range(3):
            render()

    @property
    def clipping_range(self):
        """
        Returns:
            2-tuple: [min, max] value of the sensor's clipping range, in meters
        """
        return np.array(self.get_attribute("clippingRange"))

    @clipping_range.setter
    def clipping_range(self, limits):
        """
        Sets the clipping range @limits for this sensor

        Args:
            limits (2-tuple): [min, max] value of the sensor's clipping range, in meters
        """
        self.set_attribute(attr="clippingRange", val=lazy.pxr.Gf.Vec2f(*limits))
        # In order for sensor changes to propagate, we must toggle its visibility
        self.visible = False
        # A single update step has to happen here before we toggle visibility for changes to propagate
        render()
        self.visible = True

    @property
    def horizontal_aperture(self):
        """
        Returns:
            float: horizontal aperture of this sensor, in mm
        """
        return self.get_attribute("horizontalAperture")

    @horizontal_aperture.setter
    def horizontal_aperture(self, length):
        """
        Sets the focal length @length for this sensor

        Args:
            length (float): horizontal aperture of this sensor, in meters
        """
        self.set_attribute("horizontalAperture", length)

    @property
    def focal_length(self):
        """
        Returns:
            float: focal length of this sensor, in mm
        """
        return self.get_attribute("focalLength")

    @focal_length.setter
    def focal_length(self, length):
        """
        Sets the focal length @length for this sensor

        Args:
            length (float): focal length of this sensor, in mm
        """
        self.set_attribute("focalLength", length)

    @property
    def intrinsic_matrix(self):
        """
        Returns:
            n-array: (3, 3) camera intrinsic matrix. Transforming point p (x,y,z) in the camera frame via K * p will
                produce p' (x', y', w) - the point in the image plane. To get pixel coordiantes, divide x' and y' by w
        """
        projection_matrix = self.camera_parameters["cameraProjection"]
        projection_matrix = np.array(projection_matrix).reshape(4, 4)

        fx = projection_matrix[0, 0]
        fy = projection_matrix[1, 1]
        cx = projection_matrix[0, 2]
        cy = projection_matrix[1, 2]
        s = projection_matrix[0, 1]  # Skew factor

        intrinsic_matrix = np.array([[fx, s, cx],
                                     [0.0, fy, cy],
                                     [0.0, 0.0, 1.0]])
        return intrinsic_matrix

    @property
    def _obs_space_mapping(self):
        # Generate the complex space types for special modalities:
        # {"bbox_2d_tight", "bbox_2d_loose", "bbox_3d"}
        bbox_3d_space = gym.spaces.Sequence(space=gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=MAX_CLASS_COUNT, shape=(), dtype=np.uint32),  # semanticId
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32), # x_min
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32), # y_min
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32), # z_min
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32), # x_max
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32), # y_max
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32), # z_max
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32), # transform
            gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),  # occlusion ratio
        )))

        bbox_2d_space = gym.spaces.Sequence(space=gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=MAX_CLASS_COUNT, shape=(), dtype=np.uint32),  # semanticId
            gym.spaces.Box(low=0, high=MAX_VIEWER_SIZE, shape=(), dtype=np.int32),  # x_min
            gym.spaces.Box(low=0, high=MAX_VIEWER_SIZE, shape=(), dtype=np.int32),  # y_min
            gym.spaces.Box(low=0, high=MAX_VIEWER_SIZE, shape=(), dtype=np.int32),  # x_max
            gym.spaces.Box(low=0, high=MAX_VIEWER_SIZE, shape=(), dtype=np.int32),  # y_max
            gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),  # occlusion ratio
        )))

        obs_space_mapping = dict(
            rgb=((self.image_height, self.image_width, 4), 0, 255, np.uint8),
            depth=((self.image_height, self.image_width), 0.0, np.inf, np.float32),
            depth_linear=((self.image_height, self.image_width), 0.0, np.inf, np.float32),
            normal=((self.image_height, self.image_width, 4), -1.0, 1.0, np.float32),
            seg_semantic=((self.image_height, self.image_width), 0, MAX_CLASS_COUNT, np.uint32),
            seg_instance=((self.image_height, self.image_width), 0, MAX_INSTANCE_COUNT, np.uint32),
            seg_instance_id=((self.image_height, self.image_width), 0, MAX_INSTANCE_COUNT, np.uint32),
            flow=((self.image_height, self.image_width, 4), -np.inf, np.inf, np.float32),
            bbox_2d_tight=bbox_2d_space,
            bbox_2d_loose=bbox_2d_space,
            bbox_3d=bbox_3d_space,
        )

        return obs_space_mapping

    @classmethod
    def clear(cls):
        """
        Clears all cached sensors that have been generated. Should be used when the simulator is completely reset; i.e.:
        all objects on the stage are destroyed
        """
        for sensor in cls.SENSORS.values():
            # Destroy any sensor that is not attached to the main viewport window
            if sensor._viewport.name != "Viewport":
                sensor._viewport.destroy()

        # Render to update
        render()

        cls.SENSORS = dict()
        cls.KNOWN_SEMANTIC_IDS = set()
        cls.KEY_ARRAY = None
        cls.INSTANCE_REGISTRY = {0: "background", 1: "unlabelled"}
        cls.INSTANCE_ID_REGISTRY = {0: "background"}

    @classproperty
    def all_modalities(cls):
        return {modality for modality in cls.ALL_MODALITIES if modality != "camera_params"}

    @classproperty
    def no_noise_modalities(cls):
        # bounding boxes and camera state should not have noise
        return {"bbox_2d_tight", "bbox_2d_loose", "bbox_3d"}
