import omnigibson.lazy as lazy
from omnigibson.renderer_settings.settings_base import SettingItem, SettingsBase, SubSettingsBase


class CommonSettings(SettingsBase):
    """
    Common setting group that handles a variety of sub-settings, including:
        - Rendering
        - Geometry
        - Materials
        - Lighting
        - Simple Fog
        - Flow
        - Debug View
    """

    def __init__(self):
        self.render_settings = RenderSettings()
        self.geometry_settings = GeometrySettings()
        self.materials_settings = MaterialsSettings()
        self.lighting_settings = LightingSettings()
        self.simple_fog_setting = SimpleFogSettings()
        self.flow_settings = FlowSettings()
        self.debug_view_settings = DebugViewSettings()

    @property
    def settings(self):
        settings = {}
        settings.update(self.render_settings.settings)
        settings.update(self.geometry_settings.settings)
        settings.update(self.materials_settings.settings)
        settings.update(self.lighting_settings.settings)
        settings.update(self.simple_fog_setting.settings)
        settings.update(self.flow_settings.settings)
        settings.update(self.debug_view_settings.settings)
        return settings


class RenderSettings(SubSettingsBase):
    def __init__(self):
        self.multi_threading_enabled = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.BOOL, "Multi-Threading", "/rtx/multiThreading/enabled"
        )

    @property
    def settings(self):
        return {
            "/rtx/multiThreading/enabled": self.multi_threading_enabled,
        }


class GeometrySettings(SubSettingsBase):
    def __init__(self):
        # Basic geometry settings.
        tbnMode = ["AUTO", "CPU", "GPU", "Force GPU"]
        self.tbn_frame_mode = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Normal & Tangent Space Generation Mode",
            "/rtx/hydra/TBNFrameMode",
            range_list=tbnMode,
        )
        self.face_culling_enabled = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.BOOL, "Back Face Culling", "/rtx/hydra/faceCulling/enabled"
        )
        # Wireframe settings.
        self.wireframe_thickness = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Wireframe Thickness",
            "/rtx/wireframe/wireframeThickness",
            range_from=0.1,
            range_to=100,
        )
        self.wireframe_thickness_world_space = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Wireframe World Space Thickness",
            "/rtx/wireframe/wireframeThicknessWorldSpace",
        )
        self.wireframe_shading_enabled = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.BOOL, "Shaded Wireframe", "/rtx/wireframe/shading/enabled"
        )
        # Subdivision settings.
        self.subdivision_refinement_level = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Subdivision Global Refinement Level",
            "/rtx/hydra/subdivision/refinementLevel",
            range_from=0,
            range_to=2,
        )
        self.subdivision_adaptive_refinement = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Subdivision Feature-adaptive Refinement",
            "/rtx/hydra/subdivision/adaptiveRefinement",
        )

        # if set to zero, override to scene unit, which means the scale factor would be 1
        self.renderMeterPerUnit = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Renderer-internal meters per unit ",
            "/rtx/scene/renderMeterPerUnit",
        )
        self.only_opaque_ray_flags = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Hide geometry that uses opacity (debug)",
            "/rtx/debug/onlyOpaqueRayFlags",
        )

    @property
    def settings(self):
        return {
            "/rtx/hydra/TBNFrameMode": self.tbn_frame_mode,
            "/rtx/hydra/faceCulling/enabled": self.face_culling_enabled,
            "/rtx/wireframe/wireframeThickness": self.wireframe_thickness,
            "/rtx/wireframe/wireframeThicknessWorldSpace": self.wireframe_thickness_world_space,
            "/rtx/wireframe/shading/enabled": self.wireframe_shading_enabled,
            "/rtx/hydra/subdivision/refinementLevel": self.subdivision_refinement_level,
            "/rtx/hydra/subdivision/adaptiveRefinement": self.subdivision_adaptive_refinement,
            "/rtx/scene/renderMeterPerUnit": self.renderMeterPerUnit,
            "/rtx/debug/onlyOpaqueRayFlags": self.only_opaque_ray_flags,
        }


class MaterialsSettings(SubSettingsBase):
    def __init__(self):
        self.skip_material_loading = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Disable Material Loading",
            "/app/renderer/skipMaterialLoading",
        )
        self.max_mip_count = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Textures: Mipmap Levels to Load",
            "/rtx-transient/resourcemanager/maxMipCount",
            range_from=2,
            range_to=15,
        )
        self.compression_mip_size_threshold = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Textures: Compression Mipmap Size Threshold (0 to disable) ",
            "/rtx-transient/resourcemanager/compressionMipSizeThreshold",
            0,
            8192,
        )
        self.enable_texture_streaming = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Textures: on-demand streaming (toggling requires scene reload)",
            "/rtx-transient/resourcemanager/enableTextureStreaming",
        )
        self.memory_budget = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Texture streaming memory budget (fraction of GPU memory)",
            "/rtx-transient/resourcemanager/texturestreaming/memoryBudget",
            0.01,
            1,
        )
        self.animation_time = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.FLOAT, "MDL Animation Time Override", "/rtx/animationTime"
        )
        self.animation_time_use_wallclock = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "MDL Animation Time Use Wallclock",
            "/rtx/animationTimeUseWallclock",
        )

    @property
    def settings(self):
        return {
            "/app/renderer/skipMaterialLoading": self.skip_material_loading,
            "/rtx-transient/resourcemanager/maxMipCount": self.max_mip_count,
            "/rtx-transient/resourcemanager/compressionMipSizeThreshold": self.compression_mip_size_threshold,
            "/rtx-transient/resourcemanager/enableTextureStreaming": self.enable_texture_streaming,
            "/rtx-transient/resourcemanager/texturestreaming/memoryBudget": self.memory_budget,
            "/rtx/animationTime": self.animation_time,
            "/rtx/animationTimeUseWallclock": self.animation_time_use_wallclock,
        }


class LightingSettings(SubSettingsBase):
    def __init__(self):
        # Basic light settings.
        show_lights_settings = {"Per-Light Enable": 0, "Force Enable": 1, "Force Disable": 2}
        self.show_lights = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Show Area Lights In Primary Rays",
            "/rtx/raytracing/showLights",
            range_dict=show_lights_settings,
        )
        self.shadow_bias = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Shadow Bias",
            "/rtx/raytracing/shadowBias",
            range_from=0.0,
            range_to=5.0,
        )
        self.skip_most_lights = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Use First Distant Light & First Dome Light Only",
            "/rtx/scenedb/skipMostLights",
        )
        # Demo light.
        dome_lighting_sampling_type = {
            "Upper & Lower Hemisphere": 0,
            # "Upper Visible & Sampled, Lower Is Black": 1,
            "Upper Hemisphere Visible & Sampled, Lower Is Only Visible": 2,
            "Use As Env Map": 3,
        }
        self.upper_lower_strategy = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Hemisphere Sampling",
            "/rtx/domeLight/upperLowerStrategy",
            range_dict=dome_lighting_sampling_type,
        )
        dome_texture_resolution_items = {
            "16": 16,
            "32": 32,
            "64": 64,
            "128": 128,
            "256": 256,
            "512": 512,
            "1024": 1024,
            "2048": 2048,
            "4096": 4096,
            "8192": 8192,
        }
        self.baking_resolution = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Baking Resolution",
            "/rtx/domeLight/baking/resolution",
            range_dict=dome_texture_resolution_items,
        )
        self.resolution_factor = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Dome Light Texture Resolution Factor",
            "/rtx/domeLight/resolutionFactor",
            range_from=0.01,
            range_to=4.0,
        )
        self.baking_spp = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Dome Light Material Baking SPP",
            "/rtx/domeLight/baking/spp",
            range_from=1,
            range_to=32,
        )

    @property
    def settings(self):
        return {
            "/rtx/raytracing/showLights": self.show_lights,
            "/rtx/raytracing/shadowBias": self.shadow_bias,
            "/rtx/scenedb/skipMostLights": self.skip_most_lights,
            "/rtx/domeLight/upperLowerStrategy": self.upper_lower_strategy,
            "/rtx/domeLight/baking/resolution": self.baking_resolution,
            "/rtx/domeLight/resolutionFactor": self.resolution_factor,
            "/rtx/domeLight/baking/spp": self.baking_spp,
        }


class SimpleFogSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.fog_color = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Color", "/rtx/fog/fogColor"
        )
        self.fog_color_intensity = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Intensity",
            "/rtx/fog/fogColorIntensity",
            range_from=1,
            range_to=1000000,
        )
        self.fog_z_up_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Height-based Fog - Use +Z Axis",
            "/rtx/fog/fogZup/enabled",
        )
        self.fog_start_height = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Height-based Fog - Plane Height",
            "/rtx/fog/fogStartHeight",
            range_from=-1000000,
            range_to=1000000,
        )
        self.fog_height_density = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Height Density",
            "/rtx/fog/fogHeightDensity",
            range_from=0,
            range_to=1,
        )
        self.fog_height_falloff = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Height Falloff",
            "/rtx/fog/fogHeightFalloff",
            range_from=0,
            range_to=1000,
        )
        self.fog_distance_density = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Distance Density",
            "/rtx/fog/fogDistanceDensity",
            range_from=0,
            range_to=1,
        )
        self.fog_start_dist = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Start Distance to Camera",
            "/rtx/fog/fogStartDist",
            range_from=0,
            range_to=1000000,
        )
        self.fog_end_dist = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "End Distance to Camera",
            "/rtx/fog/fogEndDist",
            range_from=0,
            range_to=1000000,
        )

    @property
    def settings(self):
        return {
            "/rtx/fog/fogColor": self.fog_color,
            "/rtx/fog/fogColorIntensity": self.fog_color_intensity,
            "/rtx/fog/fogZup/enabled": self.fog_z_up_enabled,
            "/rtx/fog/fogStartHeight": self.fog_height_density,
            "/rtx/fog/fogHeightDensity": self.fog_height_density,
            "/rtx/fog/fogHeightFalloff": self.fog_height_falloff,
            "/rtx/fog/fogDistanceDensity": self.fog_distance_density,
            "/rtx/fog/fogStartDist": self.fog_start_dist,
            "/rtx/fog/fogEndDist": self.fog_end_dist,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/fog/enabled"


class FlowSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.ray_traced_shadows_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Flow in Real-Time Ray Traced Shadows",
            "/rtx/flow/rayTracedShadowsEnabled",
        )
        self.ray_traced_reflections_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Flow in Real-Time Ray Traced Reflections",
            "/rtx/flow/rayTracedReflectionsEnabled",
        )
        self.path_tracing_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Flow in Path-Traced Mode",
            "/rtx/flow/pathTracingEnabled",
        )
        self.path_tracing_shadows_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Flow in Path-Traced Mode Shadows",
            "/rtx/flow/pathTracingShadowsEnabled",
        )
        self.composite_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Composite with Flow Library Renderer",
            "/rtx/flow/compositeEnabled",
        )
        self.use_flow_library_self_shadow = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Use Flow Library Self Shadow",
            "/rtx/flow/useFlowLibrarySelfShadow",
        )
        self.max_blocks = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.INT, "Max Blocks", "/rtx/flow/maxBlocks"
        )

    @property
    def settings(self):
        return {
            "/rtx/flow/rayTracedShadowsEnabled": self.ray_traced_shadows_enabled,
            "/rtx/flow/rayTracedReflectionsEnabled": self.ray_traced_reflections_enabled,
            "/rtx/flow/pathTracingEnabled": self.path_tracing_enabled,
            "/rtx/flow/pathTracingShadowsEnabled": self.path_tracing_shadows_enabled,
            "/rtx/flow/compositeEnabled": self.composite_enabled,
            "/rtx/flow/useFlowLibrarySelfShadow": self.use_flow_library_self_shadow,
            "/rtx/flow/maxBlocks": self.max_blocks,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/flow/enabled"


class DebugViewSettings(SubSettingsBase):
    def __init__(self):
        debug_view_items = {
            "Off": "",
            "Beauty Before Tonemap": "beautyPreTonemap",
            "Beauty After Tonemap": "beautyPostTonemap",
            "Timing Heat Map": "TimingHeatMap",
            "Depth": "depth",
            "World Position": "worldPosition",
            "Wireframe": "wire",
            "Barycentrics": "barycentrics",
            "Texture Coordinates 0": "texcoord0",
            "Tangent U": "tangentu",
            "Tangent V": "tangentv",
            "Interpolated Normal": "normal",
            "Triangle Normal": "triangleNormal",
            "Material Normal": "materialGeometryNormal",
            "Instance ID": "instanceId",
            "3D Motion Vectors": "targetMotion",
            "Shadow (last light)": "shadow",
            "Diffuse Reflectance": "diffuseReflectance",
            "Specular Reflectance": "reflectance",
            "Roughness": "roughness",
            "Ambient Occlusion": "ao",
            "Reflections": "reflections",
            "Reflections 3D Motion Vectors": "reflectionsMotion",
            "Translucency": "translucency",
            "Radiance": "radiance",
            "Diffuse GI": "indirectDiffuse",
            "Caustics": "caustics",
            "PT Noisy Result": "pathTracerNoisy",
            "PT Denoised Result": "pathTracerDenoised",
            "RT Noisy Sampled Lighting": "rtNoisySampledLighting",
            "RT Denoised Sampled Lighting": "rtDenoisedSampledLighting",
            "Developer Debug Texture": "developerDebug",
            "Noisy Dome Light": "noisyDomeLightingTex",
            "Denoised Dome Light": "denoisedDomeLightingTex",
            "RT Noisy Sampled Lighting Diffuse": "rtNoisySampledLightingDiffuse",  # ReLAX Only
            "RT Noisy Sampled Lighting Specular": "rtNoisySampledLightingSpecular",  # ReLAX Only
            "RT Denoised Sampled Lighting Diffuse": "rtDenoiseSampledLightingDiffuse",  # ReLAX Only
            "RT Denoised Sampled Lighting Specular": "rtDenoiseSampledLightingSpecular",  # ReLAX Only
            "Triangle Normal (OctEnc)": "triangleNormalOctEnc",  # ReLAX Only
            "Material Normal (OctEnc)": "materialGeometryNormalOctEnc",  # ReLAX Only
            # Targets not using RT accumulation
            "RT Noisy Sampled Lighting (Not Accumulated)": "rtNoisySampledLightingNonAccum",
            "RT Noisy Sampled Lighting Diffuse (Not Accumulated)": "sampledLightingDiffuseNonAccum",
            "RT Noisy Sampled Lighting Specular (Not Accumulated)": "sampledLightingSpecularNonAccum",
            "Reflections (Not Accumulated)": "reflectionsNonAccum",
            "Diffuse GI (Not Accumulated)": "indirectDiffuseNonAccum",
        }
        self.target = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Render Target",
            "/rtx/debugView/target",
            range_dict=debug_view_items,
        )
        self.scaling = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Output Value Scaling",
            "/rtx/debugView/scaling",
            range_from=-1000000,
            range_to=1000000,
        )

    @property
    def settings(self):
        return {
            "/rtx/debugView/target": self.target,
            "/rtx/debugView/scaling": self.scaling,
        }
