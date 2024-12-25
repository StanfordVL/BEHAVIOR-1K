import omnigibson.lazy as lazy
from omnigibson.renderer_settings.settings_base import SettingItem, SettingsBase, SubSettingsBase


class RealTimeSettings(SettingsBase):
    """
    Real-Time setting group that handles a variety of sub-settings, including:
        - Eco Mode
        - Anti Aliasing
        - Direct Lighting
        - Reflections
        - Translucency
        - Global Volumetric Effects
        - Caustics
        - Indirect Diffuse Lighting
        - RTMulti GPU (if multiple GPUs available)
    """

    def __init__(self):
        self.eco_mode_settings = EcoModeSettings()
        self.anti_aliasing_settings = AntiAliasingSettings()
        self.direct_lighting_settings = DirectLightingSettings()
        self.reflections_settings = ReflectionsSettings()
        self.translucency_settings = TranslucencySettings()
        self.global_volumetric_effects_settings = GlobalVolumetricEffectsSettings()
        self.caustics_settings = CausticsSettings()
        self.indirect_diffuse_lighting_settings = IndirectDiffuseLightingSettings()
        gpu_count = lazy.carb.settings.get_settings().get("/renderer/multiGpu/currentGpuCount")
        if gpu_count and gpu_count > 1:
            self.rt_multi_gpu_settings = RTMultiGPUSettings()

    @property
    def settings(self):
        settings = {}
        settings.update(self.eco_mode_settings.settings)
        settings.update(self.anti_aliasing_settings.settings)
        settings.update(self.direct_lighting_settings.settings)
        settings.update(self.reflections_settings.settings)
        settings.update(self.translucency_settings.settings)
        settings.update(self.global_volumetric_effects_settings.settings)
        settings.update(self.caustics_settings.settings)
        settings.update(self.indirect_diffuse_lighting_settings.settings)
        gpu_count = lazy.carb.settings.get_settings().get("/renderer/multiGpu/currentGpuCount")
        if gpu_count and gpu_count > 1:
            settings.update(self.rt_multi_gpu_settings.settings)
        return settings


class EcoModeSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.max_frames_without_change = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Stop Rendering After This Many Frames Without Changes",
            "/rtx/ecoMode/maxFramesWithoutChange",
            range_from=0,
            range_to=100,
        )

    @property
    def settings(self):
        return {
            "/rtx/ecoMode/maxFramesWithoutChange": self.max_frames_without_change,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/ecoMode/enabled"


class AntiAliasingSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        antialiasing_ops = ["Off", "TAA", "FXAA"]
        if self._carb_settings.get("/ngx/enabled") is True:
            antialiasing_ops.append("DLSS")
            antialiasing_ops.append("RTXAA")
        self.algorithm = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.STRING, "Algorithm", "/rtx/post/aa/op", antialiasing_ops
        )

        # antialiasing_op_idx == 1
        # TAA
        self.static_ratio = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Static scaling",
            "/rtx/post/scaling/staticRatio",
            range_from=0.33,
            range_to=1,
        )
        self.samples = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "TAA Samples",
            "/rtx/post/taa/samples",
            range_from=1,
            range_to=16,
        )
        self.alpha = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "TAA history scale",
            "/rtx/post/taa/alpha",
            range_from=0,
            range_to=1,
        )

        # antialiasing_op_idx == 2
        # FXAA
        self.quality_sub_pix = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Subpixel Quality",
            "/rtx/post/fxaa/qualitySubPix",
            range_from=0.0,
            range_to=1.0,
        )
        self.quality_edge_threshold = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Edge Threshold",
            "/rtx/post/fxaa/qualityEdgeThreshold",
            range_from=0.0,
            range_to=1.0,
        )
        self.quality_edge_threshold_min = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Edge Threshold Min",
            "/rtx/post/fxaa/qualityEdgeThresholdMin",
            range_from=0.0,
            range_to=1.0,
        )

        # antialiasing_op_idx == 3 or antialiasing_op_idx == 4
        # DLSS and RTXAA
        # if antialiasing_op_idx == 3
        dlss_opts = ["Performance", "Balanced", "Quality"]
        self.exec_mode = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Execution mode",
            "/rtx/post/dlss/execMode",
            dlss_opts,
        )

        self.sharpness = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Sharpness",
            "/rtx/post/aa/sharpness",
            range_from=0.0,
            range_to=1.0,
        )

        exposure_ops = ["Force self evaluated", "PostProcess Autoexposure", "Fixed"]
        self.auto_exposure_mode = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Exposure mode",
            "/rtx/post/aa/autoExposureMode",
            exposure_ops,
        )

        # auto_exposure_idx = self._carb_settings.get("/rtx/post/aa/autoExposureMode")
        # if auto_exposure_idx == 1
        self.exposure_multiplier = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Auto Exposure Multiplier",
            "/rtx/post/aa/exposureMultiplier",
            range_from=0.00001,
            range_to=10.0,
        )
        # if auto_exposure_idx == 2
        self.exposure = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Fixed Exposure Value",
            "/rtx/post/aa/exposure",
            range_from=0.00001,
            range_to=1.0,
        )

    @property
    def settings(self):
        settings = {
            "/rtx/post/aa/op": self.algorithm,
        }

        antialiasing_op_idx = self._carb_settings.get("/rtx/post/aa/op")
        if antialiasing_op_idx == 1:
            # TAA
            settings.update(
                {
                    "/rtx/post/scaling/staticRatio": self.static_ratio,
                    "/rtx/post/taa/samples": self.samples,
                    "/rtx/post/taa/alpha": self.alpha,
                }
            )
        elif antialiasing_op_idx == 2:
            # FXAA
            settings.update(
                {
                    "/rtx/post/fxaa/qualitySubPix": self.quality_sub_pix,
                    "/rtx/post/fxaa/qualityEdgeThreshold": self.quality_edge_threshold,
                    "/rtx/post/fxaa/qualityEdgeThresholdMin": self.quality_edge_threshold_min,
                }
            )
        elif antialiasing_op_idx == 3 or antialiasing_op_idx == 4:
            # DLSS and RTXAA
            if antialiasing_op_idx == 3:
                settings.update(
                    {
                        "/rtx/post/dlss/execMode": self.exec_mode,
                    }
                )
            settings.update(
                {
                    "/rtx/post/aa/sharpness": self.sharpness,
                    "/rtx/post/aa/autoExposureMode": self.auto_exposure_mode,
                }
            )

        auto_exposure_idx = self._carb_settings.get("/rtx/post/aa/autoExposureMode")
        if auto_exposure_idx == 1:
            settings.update(
                {
                    "/rtx/post/aa/exposureMultiplier": self.exposure_multiplier,
                }
            )
        elif auto_exposure_idx == 2:
            settings.update(
                {
                    "/rtx/post/aa/exposure": self.exposure,
                }
            )
        return settings


class DirectLightingSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.shadows_enabled = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.BOOL, "Enable Shadows", "/rtx/shadows/enabled"
        )

        self.sampled_lighting_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Sampled Direct Lighting",
            "/rtx/directLighting/sampledLighting/enabled",
        )
        self.sampled_lighting_auto_enable = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Auto-enable Sampled Lighting Above Light Count Threshold",
            "/rtx/directLighting/sampledLighting/autoEnable",
        )
        self.sampled_lighting_auto_enable_light_count_threshold = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Auto-enable Sampled Lighting:   Light Count Threshold",
            "/rtx/directLighting/sampledLighting/autoEnableLightCountThreshold",
        )

        # if not self._settings.get("/rtx/directLighting/sampledLighting/enabled"
        self.shadows_sample_count = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Shadow Samples per Pixel",
            "/rtx/shadows/sampleCount",
            range_from=1,
            range_to=16,
        )
        self.shadows_denoiser_quarter_res = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Lower Resolution Shadows Denoiser",
            "/rtx/shadows/denoiser/quarterRes",
        )
        self.dome_light_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Dome Lighting",
            "/rtx/directLighting/domeLight/enabled",
        )
        self.dome_light_sample_count = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Dome Light Samples per Pixel",
            "/rtx/directLighting/domeLight/sampleCount",
            range_from=0,
            range_to=32,
        )
        self.dome_light_enabled_in_reflections = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Dome Lighting in Reflections",
            "/rtx/directLighting/domeLight/enabledInReflections",
        )

        # if self._settings.get("/rtx/directLighting/sampledLighting/enabled")
        sampled_lighting_spp_items = {"1": 1, "2": 2, "4": 4, "8": 8}
        self.samples_per_pixel = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Samples per Pixel",
            "/rtx/directLighting/sampledLighting/samplesPerPixel",
            range_dict=sampled_lighting_spp_items,
        )
        self.clamp_samples_per_pixel_to_number_of_lights = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Clamp Sample Count to Light Count",
            "/rtx/directLighting/sampledLighting/clampSamplesPerPixelToNumberOfLights",
        )
        self.reflections_samples_per_pixel = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Reflections: Light Samples per Pixel",
            "/rtx/reflections/sampledLighting/samplesPerPixel",
            range_dict=sampled_lighting_spp_items,
        )
        self.reflections_clamp_samples_per_pixel_to_number_of_lights = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Reflections: Clamp Sample Count to Light Count",
            "/rtx/reflections/sampledLighting/clampSamplesPerPixelToNumberOfLights",
        )
        self.max_ray_intensity = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Max Ray Intensity",
            "/rtx/directLighting/sampledLighting/maxRayIntensity",
            range_from=0.0,
            range_to=1000000,
        )
        self.reflections_max_ray_intensity = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Reflections: Max Ray Intensity",
            "/rtx/reflections/sampledLighting/maxRayIntensity",
            range_from=0.0,
            range_to=1000000,
        )
        self.enabled_in_reflections = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Dome Lighting in Reflections",
            "/rtx/directLighting/domeLight/enabledInReflections",
        )
        firefly_filter_types = {"None": "None", "Median": "Cross-Bilateral Median", "RCRS": "Cross-Bilateral RCRS"}
        self.firefly_suppression_type = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Firefly Filter",
            "/rtx/lightspeed/ReLAX/fireflySuppressionType",
            range_dict=firefly_filter_types,
        )
        self.history_clamping_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "History Clamping",
            "/rtx/lightspeed/ReLAX/historyClampingEnabled",
        )
        self.denoiser_iterations = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Denoiser Iterations",
            "/rtx/lightspeed/ReLAX/aTrousIterations",
            range_from=1,
            range_to=10,
        )
        self.diffuse_backscattering_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Extended Diffuse Backscattering",
            "/rtx/directLighting/diffuseBackscattering/enabled",
        )
        self.shadow_offset = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Shadow Ray Offset",
            "/rtx/directLighting/diffuseBackscattering/shadowOffset",
            range_from=0.1,
            range_to=1000,
        )
        self.extinction = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Extinction",
            "/rtx/directLighting/diffuseBackscattering/extinction",
            range_from=0.001,
            range_to=100,
        )

    @property
    def settings(self):
        settings = {
            "/rtx/shadows/enabled": self.shadows_enabled,
            "/rtx/directLighting/sampledLighting/enabled": self.sampled_lighting_enabled,
            "/rtx/directLighting/sampledLighting/autoEnable": self.sampled_lighting_auto_enable,
            "/rtx/directLighting/sampledLighting/autoEnableLightCountThreshold": self.sampled_lighting_auto_enable_light_count_threshold,
        }
        if not self._carb_settings.get("/rtx/directLighting/sampledLighting/enabled"):
            settings.update(
                {
                    "/rtx/shadows/sampleCount": self.shadows_sample_count,
                    "/rtx/shadows/denoiser/quarterRes": self.shadows_denoiser_quarter_res,
                    "/rtx/directLighting/domeLight/enabled": self.dome_light_enabled,
                    "/rtx/directLighting/domeLight/sampleCount": self.dome_light_sample_count,
                    "/rtx/directLighting/domeLight/enabledInReflections": self.dome_light_enabled_in_reflections,
                }
            )
        else:
            settings.update(
                {
                    "/rtx/directLighting/sampledLighting/samplesPerPixel": self.samples_per_pixel,
                    "/rtx/directLighting/sampledLighting/clampSamplesPerPixelToNumberOfLights": self.clamp_samples_per_pixel_to_number_of_lights,
                    "/rtx/reflections/sampledLighting/samplesPerPixel": self.reflections_samples_per_pixel,
                    "/rtx/reflections/sampledLighting/clampSamplesPerPixelToNumberOfLights": self.reflections_clamp_samples_per_pixel_to_number_of_lights,
                    "/rtx/directLighting/sampledLighting/maxRayIntensity": self.max_ray_intensity,
                    "/rtx/reflections/sampledLighting/maxRayIntensity": self.reflections_max_ray_intensity,
                    "/rtx/directLighting/domeLight/enabledInReflections": self.enabled_in_reflections,
                    "/rtx/lightspeed/ReLAX/fireflySuppressionType": self.firefly_suppression_type,
                    "/rtx/lightspeed/ReLAX/historyClampingEnabled": self.history_clamping_enabled,
                    "/rtx/lightspeed/ReLAX/aTrousIterations": self.denoiser_iterations,
                    "/rtx/directLighting/diffuseBackscattering/enabled": self.diffuse_backscattering_enabled,
                    "/rtx/directLighting/diffuseBackscattering/shadowOffset": self.shadow_offset,
                    "/rtx/directLighting/diffuseBackscattering/extinction": self.extinction,
                }
            )
        return settings

    @property
    def enabled_setting_path(self):
        return "/rtx/directLighting/enabled"


class ReflectionsSettings(SettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.max_roughness = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Max Roughness",
            "/rtx/reflections/maxRoughness",
            range_from=0.0,
            range_to=1.0,
        )
        self.max_reflection_bounces = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max Reflection Bounces",
            "/rtx/reflections/maxReflectionBounces",
            range_from=0,
            range_to=100,
        )

    @property
    def settings(self):
        return {
            "/rtx/reflections/maxRoughness": self.max_roughness,
            "/rtx/reflections/maxReflectionBounces": self.max_reflection_bounces,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/reflections/enabled"


class TranslucencySettings(SettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.max_refraction_bounces = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max Refraction Bounces",
            "/rtx/translucency/maxRefractionBounces",
            range_from=0,
            range_to=100,
        )
        self.reflection_cutoff = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Secondary Bounce Roughness Cutoff",
            "/rtx/translucency/reflectionCutoff",
            range_from=0.0,
            range_to=1.0,
        )
        self.fractional_cutou_opacity = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Fractional Cutout Opacity",
            "/rtx/raytracing/fractionalCutoutOpacity",
        )
        self.virtual_depth = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Depth Correction for DoF",
            "/rtx/translucency/virtualDepth",
        )

    @property
    def settings(self):
        return {
            "/rtx/translucency/maxRefractionBounces": self.max_refraction_bounces,
            "/rtx/translucency/reflectionCutoff": self.reflection_cutoff,
            "/rtx/raytracing/fractionalCutoutOpacity": self.reflection_cutoff,
            "/rtx/translucency/virtualDepth": self.virtual_depth,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/translucency/enabled"


class GlobalVolumetricEffectsSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.max_accumulation_frames = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Accumulation Frames",
            "/rtx/raytracing/inscattering/maxAccumulationFrames",
            range_from=1,
            range_to=255,
        )
        self.depth_slices = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "# Depth Slices",
            "/rtx/raytracing/inscattering/depthSlices",
            range_from=16,
            range_to=1024,
        )
        self.pixel_ratio = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Pixel Density",
            "/rtx/raytracing/inscattering/pixelRatio",
            range_from=4,
            range_to=64,
        )
        self.max_distance = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Max inscattering Distance",
            "/rtx/raytracing/inscattering/maxDistance",
            range_from=10,
            range_to=100000,
        )
        self.atmosphere_height = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Atmosphere Height",
            "/rtx/raytracing/inscattering/atmosphereHeight",
            range_from=-100000,
            range_to=100000,
        )
        self.transmittance_color = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.COLOR3,
            "Transmittance Color",
            "/rtx/raytracing/inscattering/transmittanceColor",
        )
        self.transmittance_measurement_distance = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Transmittance Measurment Distance",
            "/rtx/raytracing/inscattering/transmittanceMeasurementDistance",
            range_from=0.0001,
            range_to=1000000,
        )
        self.single_scattering_albedo = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.COLOR3,
            "Single Scattering Albedo",
            "/rtx/raytracing/inscattering/singleScatteringAlbedo",
        )
        self.anisotropy_factor = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Anisotropy Factor",
            "/rtx/raytracing/inscattering/anisotropyFactor",
            range_from=-0.999,
            range_to=0.999,
        )
        self.slice_distribution_exponent = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Slice Distribution Exponent",
            "/rtx/raytracing/inscattering/sliceDistributionExponent",
            range_from=1,
            range_to=16,
        )
        self.blur_sigma = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Inscatter Blur Sigma",
            "/rtx/raytracing/inscattering/blurSigma",
            0.0,
            range_to=10.0,
        )
        self.dithering_scale = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Inscatter Dithering Scale",
            "/rtx/raytracing/inscattering/ditheringScale",
            range_from=0,
            range_to=100,
        )
        self.spatial_jitter_scale = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Spatial Sample Jittering Scale",
            "/rtx/raytracing/inscattering/spatialJitterScale",
            range_from=0.0,
            range_to=1,
        )
        self.temporal_jitter_scale = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Temporal Reprojection Jittering Scale",
            "/rtx/raytracing/inscattering/temporalJitterScale",
            range_from=0.0,
            range_to=1,
        )
        self.use_detail_noise = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Apply Density Noise",
            "/rtx/raytracing/inscattering/useDetailNoise",
        )
        self.detail_noise_scale = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Density Noise World Scale",
            "/rtx/raytracing/inscattering/detailNoiseScale",
            range_from=0.0,
            range_to=1,
        )
        self.noise_animation_speed_x = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Density Noise Animation Speed X",
            "/rtx/raytracing/inscattering/noiseAnimationSpeedX",
            range_from=-1.0,
            range_to=1.0,
        )
        self.noise_animation_speed_y = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Density Noise Animation Speed Y",
            "/rtx/raytracing/inscattering/noiseAnimationSpeedY",
            range_from=-1.0,
            range_to=1.0,
        )
        self.noise_animation_speed_z = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Density Noise Animation Speed Z",
            "/rtx/raytracing/inscattering/noiseAnimationSpeedZ",
            range_from=-1.0,
            range_to=1.0,
        )
        self.noise_scale_range_min = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Density Noise Scale Min",
            "/rtx/raytracing/inscattering/noiseScaleRangeMin",
            range_from=-1.0,
            range_to=5.0,
        )
        self.noise_scale_range_max = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Density Noise Scale Max",
            "/rtx/raytracing/inscattering/noiseScaleRangeMax",
            range_from=-1.0,
            range_to=5.0,
        )
        self.noise_num_octaves = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Density Noise Octave Count",
            "/rtx/raytracing/inscattering/noiseNumOctaves",
            range_from=1,
            range_to=8,
        )
        self.use_32bit_precision = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Use 32-bit Precision",
            "/rtx/raytracing/inscattering/use32bitPrecision",
        )

    @property
    def settings(self):
        return {
            "/rtx/raytracing/inscattering/maxAccumulationFrames": self.max_accumulation_frames,
            "/rtx/raytracing/inscattering/depthSlices": self.depth_slices,
            "/rtx/raytracing/inscattering/pixelRatio": self.pixel_ratio,
            "/rtx/raytracing/inscattering/maxDistance": self.max_distance,
            "/rtx/raytracing/inscattering/atmosphereHeight": self.atmosphere_height,
            "/rtx/raytracing/inscattering/transmittanceColor": self.transmittance_color,
            "/rtx/raytracing/inscattering/transmittanceMeasurementDistance": self.transmittance_measurement_distance,
            "/rtx/raytracing/inscattering/singleScatteringAlbedo": self.single_scattering_albedo,
            "/rtx/raytracing/inscattering/anisotropyFactor": self.anisotropy_factor,
            "/rtx/raytracing/inscattering/sliceDistributionExponent": self.slice_distribution_exponent,
            "/rtx/raytracing/inscattering/blurSigma": self.blur_sigma,
            "/rtx/raytracing/inscattering/ditheringScale": self.dithering_scale,
            "/rtx/raytracing/inscattering/spatialJitterScale": self.spatial_jitter_scale,
            "/rtx/raytracing/inscattering/temporalJitterScale": self.temporal_jitter_scale,
            "/rtx/raytracing/inscattering/useDetailNoise": self.use_detail_noise,
            "/rtx/raytracing/inscattering/detailNoiseScale": self.detail_noise_scale,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedX": self.noise_animation_speed_x,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedY": self.noise_animation_speed_y,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedZ": self.noise_animation_speed_z,
            "/rtx/raytracing/inscattering/noiseScaleRangeMin": self.noise_scale_range_min,
            "/rtx/raytracing/inscattering/noiseScaleRangeMax": self.noise_scale_range_max,
            "/rtx/raytracing/inscattering/noiseNumOctaves": self.noise_num_octaves,
            "/rtx/raytracing/inscattering/use32bitPrecision": self.use_32bit_precision,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/raytracing/globalVolumetricEffects/enabled"


class CausticsSettings(SettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.photon_count_nultiplier = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Photon Count Multiplier",
            "/rtx/raytracing/caustics/photonCountMultiplier",
            range_from=1,
            range_to=5000,
        )
        self.photon_max_bounces = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Photon Max Bounces",
            "/rtx/raytracing/caustics/photonMaxBounces",
            range_from=1,
            range_to=20,
        )
        self.positio_phi = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Position Phi",
            "/rtx/raytracing/caustics/positionPhi",
            range_from=0.1,
            range_to=50,
        )
        self.normal_phi = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Normal Phi",
            "/rtx/raytracing/caustics/normalPhi",
            range_from=0.3,
            range_to=1,
        )
        self.filtering_iterations = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Filter Iterations",
            "/rtx/raytracing/caustics/eawFilteringSteps",
            range_from=0,
            range_to=10,
        )

    @property
    def settings(self):
        return {
            "/rtx/raytracing/caustics/photonCountMultiplier": self.photon_count_nultiplier,
            "/rtx/raytracing/caustics/photonMaxBounces": self.photon_max_bounces,
            "/rtx/raytracing/caustics/positionPhi": self.positio_phi,
            "/rtx/raytracing/caustics/normalPhi": self.normal_phi,
            "/rtx/raytracing/caustics/eawFilteringSteps": self.filtering_iterations,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/caustics/enabled"


class IndirectDiffuseLightingSettings(SettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.ambient_light_color = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.COLOR3,
            "Ambient Light Color",
            "/rtx/sceneDb/ambientLightColor",
        )
        self.ambient_light_intensity = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Ambient Light Intensity",
            "/rtx/sceneDb/ambientLightIntensity",
            range_from=0.0,
            range_to=10.0,
        )
        self.ambient_occlusion_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Ambient Occlusion (AO)",
            "/rtx/ambientOcclusion/enabled",
        )

        # if self._carb_settings.get("/rtx/ambientOcclusion/enabled")
        self.ray_length_in_cm = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "AO: Ray Length (cm)",
            "/rtx/ambientOcclusion/rayLengthInCm",
            range_from=0.0,
            range_to=2000.0,
        )
        self.min_samples = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "AO: Minimum Samples per Pixel",
            "/rtx/ambientOcclusion/minSamples",
            range_from=1,
            range_to=16,
        )
        self.max_samples = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "AO: Maximum Samples per Pixel",
            "/rtx/ambientOcclusion/maxSamples",
            range_from=1,
            range_to=16,
        )
        self.aggressive_denoising = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "AO: Aggressive denoising",
            "/rtx/ambientOcclusion/aggressiveDenoising",
        )

        self.indirect_diffuse_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Indirect Diffuse GI",
            "/rtx/indirectDiffuse/enabled",
        )

        # if self._carb_settings.get("/rtx/indirectDiffuse/enabled")
        gi_denoising_techniques_ops = ["NVRTD", "NRD:Reblur"]
        self.fetch_sample_count = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Samples per Pixel",
            "/rtx/indirectDiffuse/fetchSampleCount",
            range_from=0,
            range_to=4,
        )
        self.max_bounces = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max Bounces",
            "/rtx/indirectDiffuse/maxBounces",
            range_from=0,
            range_to=16,
        )
        self.scaling_factor = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Intensity",
            "/rtx/indirectDiffuse/scalingFactor",
            range_from=0.0,
            range_to=20.0,
        )
        self.denoiser_method = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Denoising technique",
            "/rtx/indirectDiffuse/denoiser/method",
            range_list=gi_denoising_techniques_ops,
        )
        # if enabled and self._carb_settings.get("/rtx/indirectDiffuse/denoiser/method") == 0:
        self.kernel_radius = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Kernel radius",
            "/rtx/indirectDiffuse/denoiser/kernelRadius",
            range_from=1,
            range_to=64,
        )
        self.iterations = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Iteration count",
            "/rtx/indirectDiffuse/denoiser/iterations",
            range_from=1,
            range_to=10,
        )
        self.max_history = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max History Length",
            "/rtx/indirectDiffuse/denoiser/temporal/maxHistory",
            range_from=1,
            range_to=100,
        )
        # if enabled and self._carb_settings.get("/rtx/indirectDiffuse/denoiser/method") == 1:
        self.max_accumulated_frame_num = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Frames in History",
            "/rtx/lightspeed/NRD_ReblurDiffuse/maxAccumulatedFrameNum",
            range_from=0,
            range_to=63,
        )
        self.max_fast_accumulated_frame_num = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Frames in Fast History",
            "/rtx/lightspeed/NRD_ReblurDiffuse/maxFastAccumulatedFrameNum",
            range_from=0,
            range_to=63,
        )
        self.plane_distance_sensitivity = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Plane Distance Sensitivity",
            "/rtx/lightspeed/NRD_ReblurDiffuse/planeDistanceSensitivity",
            range_from=0,
            range_to=1,
        )
        self.blur_radius = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Blur Radius",
            "/rtx/lightspeed/NRD_ReblurDiffuse/blurRadius",
            range_from=0,
            range_to=100,
        )

    @property
    def settings(self):
        settings = {
            "/rtx/sceneDb/ambientLightColor": self.ambient_light_color,
            "/rtx/sceneDb/ambientLightIntensity": self.ambient_light_intensity,
            "/rtx/ambientOcclusion/enabled": self.ambient_occlusion_enabled,
            "/rtx/indirectDiffuse/enabled": self.indirect_diffuse_enabled,
        }
        if self._carb_settings.get("/rtx/ambientOcclusion/enabled"):
            settings.update(
                {
                    "/rtx/ambientOcclusion/rayLengthInCm": self.ray_length_in_cm,
                    "/rtx/ambientOcclusion/minSamples": self.min_samples,
                    "/rtx/ambientOcclusion/maxSamples": self.max_samples,
                    "/rtx/ambientOcclusion/aggressiveDenoising": self.aggressive_denoising,
                }
            )
        if self._carb_settings.get("/rtx/indirectDiffuse/enabled"):
            settings.update(
                {
                    "/rtx/indirectDiffuse/fetchSampleCount": self.max_bounces,
                    "/rtx/indirectDiffuse/maxBounces": self.ambient_light_color,
                    "/rtx/indirectDiffuse/scalingFactor": self.scaling_factor,
                    "/rtx/indirectDiffuse/denoiser/method": self.denoiser_method,
                }
            )
            if self._carb_settings.get("/rtx/indirectDiffuse/denoiser/method") == 0:
                settings.update(
                    {
                        "/rtx/indirectDiffuse/denoiser/kernelRadius": self.kernel_radius,
                        "/rtx/indirectDiffuse/denoiser/iterations": self.iterations,
                        "/rtx/indirectDiffuse/denoiser/temporal/maxHistory": self.max_history,
                    }
                )
            elif self._carb_settings.get("/rtx/indirectDiffuse/denoiser/method") == 1:
                settings.update(
                    {
                        "/rtx/lightspeed/NRD_ReblurDiffuse/maxAccumulatedFrameNum": self.max_accumulated_frame_num,
                        "/rtx/lightspeed/NRD_ReblurDiffuse/maxFastAccumulatedFrameNum": self.max_fast_accumulated_frame_num,
                        "/rtx/lightspeed/NRD_ReblurDiffuse/planeDistanceSensitivity": self.plane_distance_sensitivity,
                        "/rtx/lightspeed/NRD_ReblurDiffuse/blurRadius": self.blur_radius,
                    }
                )
        return settings


class RTMultiGPUSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        currentGpuCount = self._carb_settings.get("/renderer/multiGpu/currentGpuCount")
        self.tile_count = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Tile Count",
            "/rtx/realtime/mgpu/tileCount",
            range_from=2,
            range_to=currentGpuCount,
        )
        self.master_post_processOnly = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "GPU 0 Post Process Only",
            "/rtx/realtime/mgpu/masterPostProcessOnly",
        )
        self.tile_overlap = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Tile Overlap (Pixels)",
            "/rtx/realtime/mgpu/tileOverlap",
            range_from=0,
            range_to=256,
        )
        self.tile_overlap_blend_fraction = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Fraction of Overlap Pixels to Blend",
            "/rtx/realtime/mgpu/tileOverlapBlendFraction",
            range_from=0.0,
            range_to=1.0,
        )

    @property
    def settings(self):
        return {
            "/rtx/realtime/mgpu/tileCount": self.tile_count,
            "/rtx/realtime/mgpu/masterPostProcessOnly": self.master_post_processOnly,
            "/rtx/realtime/mgpu/tileOverlap": self.tile_overlap,
            "/rtx/realtime/mgpu/tileOverlapBlendFraction": self.tile_overlap_blend_fraction,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/realtime/mgpu/enabled"
