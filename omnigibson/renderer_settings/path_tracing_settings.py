import omnigibson.lazy as lazy
from omnigibson.renderer_settings.settings_base import SettingItem, SettingsBase, SubSettingsBase


class PathTracingSettings(SettingsBase):
    """
    Path-Tracing setting group that handles a variety of sub-settings, including:
        - Anti-Aliasing
        - Firefly Filter
        - Path-Tracing
        - Sampling & Caching
        - Denoising
        - Path-Traced Fog
        - Heterogeneous Volumes (Path Traced Volume)
        - Multi GPU (if multiple GPUs available)
    """

    def __init__(self):
        self.anti_aliasing_settings = AntiAliasingSettings()
        self.firefly_filter_settings = FireflyFilterSettings()
        self.path_tracing_settings = PathTracingSettings()
        self.sampling_and_caching_settings = SamplingAndCachingSettings()
        self.denoising_settings = DenoisingSettings()
        self.path_traced_fog_settings = PathTracedFogSettings()
        self.path_traced_volume_settings = PathTracedVolumeSettings()
        if lazy.carb.settings.get_settings().get("/renderer/multiGpu/currentGpuCount") > 1:
            self.multi_gpu_settings = MultiGPUSettings()

    @property
    def settings(self):
        settings = {}
        settings.update(self.anti_aliasing_settings.settings)
        settings.update(self.firefly_filter_settings.settings)
        settings.update(self.path_tracing_settings.settings)
        settings.update(self.sampling_and_caching_settings.settings)
        settings.update(self.denoising_settings.settings)
        settings.update(self.path_traced_fog_settings.settings)
        settings.update(self.path_traced_volume_settings.settings)
        if lazy.carb.settings.get_settings().get("/renderer/multiGpu/currentGpuCount") > 1:
            settings.update(self.multi_gpu_settings.settings)
        return settings


class AntiAliasingSettings(SubSettingsBase):
    def __init__(self):
        pt_aa_ops = ["Box", "Triangle", "Gaussian", "Uniform"]
        self.sample_pattern = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Anti-Aliasing Sample Pattern",
            "/rtx/pathtracing/aa/op",
            pt_aa_ops,
        )
        self.filter_radius = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Anti-Aliasing Radius",
            "/rtx/pathtracing/aa/filterRadius",
            range_from=0.0001,
            range_to=5.0,
        )

    @property
    def settings(self):
        return {
            "/rtx/pathtracing/aa/op": self.sample_pattern,
            "/rtx/pathtracing/aa/filterRadius": self.filter_radius,
        }


class FireflyFilterSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.max_intensity_per_sample = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Max Ray Intensity Glossy",
            "/rtx/pathtracing/fireflyFilter/maxIntensityPerSample",
            range_from=0,
            range_to=100000,
        )
        self.max_intensityper_sample_diffuse = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Max Ray Intensity Diffuse",
            "/rtx/pathtracing/fireflyFilter/maxIntensityPerSampleDiffuse",
            range_from=0,
            range_to=100000,
        )

    @property
    def settings(self):
        return {
            "/rtx/pathtracing/fireflyFilter/maxIntensityPerSample": self.max_intensity_per_sample,
            "/rtx/pathtracing/fireflyFilter/maxIntensityPerSampleDiffuse": self.max_intensityper_sample_diffuse,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/pathtracing/fireflyFilter/enabled"


class PathTracingSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.pathtracing_max_bounces = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max Bounces",
            "/rtx/pathtracing/maxBounces",
            range_from=0,
            range_to=64,
        )
        self.max_specular_and_transmission_bounces = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max Specular and Transmission Bounces",
            "/rtx/pathtracing/maxSpecularAndTransmissionBounces",
            range_from=1,
            range_to=128,
        )
        self.maxvolume_bounces = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max SSS Volume Scattering Bounces",
            "/rtx/pathtracing/maxVolumeBounces",
            range_from=0,
            range_to=1024,
        )
        self.ptfog_max_bounces = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max Fog Scattering Bounces",
            "/rtx/pathtracing/ptfog/maxBounces",
            range_from=1,
            range_to=10,
        )
        self.ptvol_max_bounces = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max Heterogeneous Volume Scattering Bounces",
            "/rtx/pathtracing/ptvol/maxBounces",
            range_from=0,
            range_to=1024,
        )

        clamp_spp = self._carb_settings.get("/rtx/pathtracing/clampSpp")
        if clamp_spp > 1:  # better 0, but setting range = (1,1) completely disables the UI control range
            self.spp = SettingItem(
                self,
                lazy.omni.kit.widget.settings.SettingType.INT,
                "Samples per Pixel per Frame (1 to {})".format(clamp_spp),
                "/rtx/pathtracing/spp",
                range_from=1,
                range_to=clamp_spp,
            )
        else:
            self.spp = SettingItem(
                self,
                lazy.omni.kit.widget.settings.SettingType.INT,
                "Samples per Pixel per Frame",
                "/rtx/pathtracing/spp",
                range_from=1,
                range_to=1048576,
            )
        self.total_spp = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Total Samples per Pixel (0 = inf)",
            "/rtx/pathtracing/totalSpp",
            range_from=0,
            range_to=1048576,
        )

        self.fractional_cutout_opacity = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Fractional Cutout Opacity",
            "/rtx/pathtracing/fractionalCutoutOpacity",
        )
        self.reset_pt_accum_on_anim_time_change = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Reset Accumulation on Time Change",
            "/rtx/resetPtAccumOnAnimTimeChange",
        )

    @property
    def settings(self):
        return {
            "/rtx/pathtracing/maxBounces": self.pathtracing_max_bounces,
            "/rtx/pathtracing/maxSpecularAndTransmissionBounces": self.max_specular_and_transmission_bounces,
            "/rtx/pathtracing/maxVolumeBounces": self.maxvolume_bounces,
            "/rtx/pathtracing/ptfog/maxBounces": self.ptfog_max_bounces,
            "/rtx/pathtracing/ptvol/maxBounces": self.ptvol_max_bounces,
            "/rtx/pathtracing/spp": self.spp,
            "/rtx/pathtracing/totalSpp": self.total_spp,
            "/rtx/pathtracing/fractionalCutoutOpacity": self.fractional_cutout_opacity,
            "/rtx/resetPtAccumOnAnimTimeChange": self.reset_pt_accum_on_anim_time_change,
        }


class SamplingAndCachingSettings(SubSettingsBase):
    def __init__(self):
        self.cached_enabled = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.BOOL, "Enable Caching", "/rtx/pathtracing/cached/enabled"
        )
        self.lightcache_cached_enabled = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Many-Light Sampling",
            "/rtx/pathtracing/lightcache/cached/enabled",
        )

    @property
    def settings(self):
        return {
            "/rtx/pathtracing/cached/enabled": self.cached_enabled,
            "/rtx/pathtracing/lightcache/cached/enabled": self.lightcache_cached_enabled,
        }


class DenoisingSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.blend_factor = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "OptiX Denoiser Blend Factor",
            "/rtx/pathtracing/optixDenoiser/blendFactor",
            range_from=0,
            range_to=1,
        )

    @property
    def settings(self):
        return {
            "/rtx/pathtracing/optixDenoiser/blendFactor": self.blend_factor,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/pathtracing/optixDenoiser/enabled"


class PathTracedFogSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.density = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Density",
            "/rtx/pathtracing/ptfog/density",
            range_from=0,
            range_to=1,
        )
        self.height = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Height",
            "/rtx/pathtracing/ptfog/height",
            range_from=-10,
            range_to=1000,
        )
        self.falloff = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Falloff",
            "/rtx/pathtracing/ptfog/falloff",
            range_from=0,
            range_to=100,
        )
        self.color = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.COLOR3,
            "Color",
            "/rtx/pathtracing/ptfog/color",
            range_from=0,
            range_to=1,
        )
        self.asymmetry = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Asymmetry (g)",
            "/rtx/pathtracing/ptfog/asymmetry",
            range_from=-0.99,
            range_to=0.99,
        )
        self.z_up = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.BOOL, "Use +Z Axis for Height", "/rtx/pathtracing/ptfog/ZUp"
        )

    @property
    def settings(self):
        return {
            "/rtx/pathtracing/ptfog/density": self.density,
            "/rtx/pathtracing/ptfog/height": self.height,
            "/rtx/pathtracing/ptfog/falloff": self.falloff,
            "/rtx/pathtracing/ptfog/color": self.color,
            "/rtx/pathtracing/ptfog/asymmetry": self.asymmetry,
            "/rtx/pathtracing/ptfog/ZUp": self.z_up,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/pathtracing/ptfog/enabled"


class PathTracedVolumeSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        pt_vol_tr_ops = ["Biased Ray Marching", "Ratio Tracking", "Brute-force Ray Marching"]
        self.transmittance_method = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Transmittance Method",
            "/rtx/pathtracing/ptvol/transmittanceMethod",
            range_list=pt_vol_tr_ops,
        )
        self.max_collision_count = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max Collision Count",
            "/rtx/pathtracing/ptvol/maxCollisionCount",
            range_from=0,
            range_to=1024,
        )
        self.max_light_collision_count = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Max Light Collision Count",
            "/rtx/pathtracing/ptvol/maxLightCollisionCount",
            range_from=0,
            range_to=1024,
        )
        self.max_density = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Max Density",
            "/rtx/pathtracing/ptvol/maxDensity",
            range_from=0,
            range_to=1000,
        )
        self.fast_vdb = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.BOOL, "Fast VDB", "/rtx/pathtracing/ptvol/fastVdb"
        )

        # if self._carb_settings.get("/rtx/pathtracing/ptvol/fastVdb")
        self.autoMajorant_vdb = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Fast VDB Auto majorant",
            "/rtx/pathtracing/ptvol/autoMajorantVdb",
        )

    @property
    def settings(self):
        settings = {
            "/rtx/pathtracing/ptvol/transmittanceMethod": self.transmittance_method,
            "/rtx/pathtracing/ptvol/maxCollisionCount": self.max_collision_count,
            "/rtx/pathtracing/ptvol/maxLightCollisionCount": self.max_light_collision_count,
            "/rtx/pathtracing/ptvol/maxDensity": self.max_density,
            "/rtx/pathtracing/ptvol/fastVdb": self.fast_vdb,
        }
        if self._carb_settings.get("/rtx/pathtracing/ptvol/fastVdb"):
            settings.update(
                {
                    "/rtx/pathtracing/ptvol/autoMajorantVdb": self.autoMajorant_vdb,
                }
            )
        return settings

    @property
    def enabled_setting_path(self):
        return "/rtx/pathtracing/ptvol/enabled"


class MultiGPUSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.weight_gpu0 = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "GPU 0 Weight",
            "/rtx/pathtracing/mgpu/weightGpu0",
            range_from=0,
            range_to=1,
        )
        self.compress_radiance = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Compress Radiance",
            "/rtx/pathtracing/mgpu/compressRadiance",
        )
        self.compress_albedo = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Compress Albedo",
            "/rtx/pathtracing/mgpu/compressAlbedo",
        )
        self.compress_normals = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Compress Normals",
            "/rtx/pathtracing/mgpu/compressNormals",
        )

    @property
    def settings(self):
        return {
            "/rtx/pathtracing/mgpu/weightGpu0": self.weight_gpu0,
            "/rtx/pathtracing/mgpu/compressRadiance": self.compress_radiance,
            "/rtx/pathtracing/mgpu/compressAlbedo": self.compress_albedo,
            "/rtx/pathtracing/mgpu/compressNormals": self.compress_normals,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/pathtracing/mgpu/enabled"
