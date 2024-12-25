import omnigibson.lazy as lazy
from omnigibson.renderer_settings.settings_base import SettingItem, SettingsBase, SubSettingsBase


class PostProcessingSettings(SettingsBase):
    """
    Post-Processing setting group that handles a variety of sub-settings, including:
        - Tone Mapping
        - Auto Exposure
        - Color Correction
        - Color Grading
        - XR Compositing
        - Chromatic Aberration
        - Depth Of Field Camera Overrides
        - Motion Blur
        - FTT Bloom
        - TV Noise & Film Grain
        - Reshade
    """

    def __init__(self):
        self.tone_mapping_settings = ToneMappingSettings()
        self.auto_exposure_settings = AutoExposureSettings()
        self.color_correction_settings = ColorCorrectionSettings()
        self.color_grading_settings = ColorGradingSettings()
        self.xr_compositing_settings = XRCompositingSettings()
        self.chromatic_aberration_settings = ChromaticAberrationSettings()
        self.depth_of_field_settings = DepthOfFieldSettings()
        self.motion_blur_settings = MotionBlurSettings()
        self.ftt_bloom_settings = FTTBloomSettings()
        self.tv_noise_grain_settings = TVNoiseGrainSettings()
        self.reshade_settings = ReshadeSettings()

    @property
    def settings(self):
        settings = {}
        settings.update(self.tone_mapping_settings.settings)
        settings.update(self.auto_exposure_settings.settings)
        settings.update(self.color_correction_settings.settings)
        settings.update(self.color_grading_settings.settings)
        settings.update(self.xr_compositing_settings.settings)
        settings.update(self.chromatic_aberration_settings.settings)
        settings.update(self.depth_of_field_settings.settings)
        settings.update(self.motion_blur_settings.settings)
        settings.update(self.ftt_bloom_settings.settings)
        settings.update(self.tv_noise_grain_settings.settings)
        settings.update(self.reshade_settings.settings)
        return settings


class ToneMappingSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        # The main tonemapping layout contains only the combo box. All the other options
        # are saved in a different layout which can be swapped out in case the tonemapper changes.
        tonemapper_ops = [
            "Clamp",
            "Linear",
            "Reinhard",
            "Reinhard (modified)",
            "HejiHableAlu",
            "HableUc2",
            "Aces",
            "Iray",
        ]
        self.tomemap_op = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Tone Mapping Operator",
            "/rtx/post/tonemap/op",
            range_list=tonemapper_ops,
        )

        # tonemap_op_idx = self._carb_settings.get("/rtx/post/tonemap/op")

        # Modified Reinhard
        # tonemap_op_idx == 3
        self.max_white_luminance = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Max White Luminance",
            "/rtx/post/tonemap/maxWhiteLuminance",
            range_from=0,
            range_to=100,
        )

        # HableUc2
        # tonemap_op_idx == 5
        self.white_scale = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "White Scale Value",
            "/rtx/post/tonemap/whiteScale",
            range_from=0,
            range_to=100,
        )

        self.cm2_factor = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "cm^2 Factor",
            "/rtx/post/tonemap/cm2Factor",
            range_from=0,
            range_to=2,
        )
        self.white_point = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "White Point", "/rtx/post/tonemap/whitepoint"
        )
        self.film_iso = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Film ISO",
            "/rtx/post/tonemap/filmIso",
            range_from=50,
            range_to=1600,
        )
        self.camera_shutter = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Camera Shutter",
            "/rtx/post/tonemap/cameraShutter",
            range_from=1,
            range_to=5000,
        )
        self.f_number = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "f-Number / f-Stop",
            "/rtx/post/tonemap/fNumber",
            range_from=1,
            range_to=20,
        )

        # Iray
        # tonemap_op_idx == 7
        self.crush_blacks = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Crush Blacks",
            "/rtx/post/tonemap/irayReinhard/crushBlacks",
            range_from=0,
            range_to=1,
        )
        self.burn_highlights = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Burn Highlights",
            "/rtx/post/tonemap/irayReinhard/burnHighlights",
            range_from=0,
            range_to=1,
        )
        self.burn_highlights_per_component = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Burn Highlights per Component",
            "/rtx/post/tonemap/irayReinhard/burnHighlightsPerComponent",
        )
        self.burn_highlights_max_component = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Burn Highlights max Component",
            "/rtx/post/tonemap/irayReinhard/burnHighlightsMaxComponent",
        )
        self.saturation = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Saturation",
            "/rtx/post/tonemap/irayReinhard/saturation",
            range_from=0,
            range_to=1,
        )

        # Clamp is never using srgb conversion
        # tonemap_op_idx != 0
        self.enable_srgb_to_gamma = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable SRGB To Gamma Conversion",
            "/rtx/post/tonemap/enableSrgbToGamma",
        )

        tonemapColorMode = ["sRGBLinear", "ACEScg"]
        self.color_mode = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Tonemapping Color Space",
            "/rtx/post/tonemap/colorMode",
            range_list=tonemapColorMode,
        )

        self.wrapvalue = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Wrap Value",
            "/rtx/post/tonemap/wrapValue",
            range_from=0,
            range_to=100000,
        )

    @property
    def settings(self):
        settings = {
            "/rtx/post/tonemap/op": self.tomemap_op,
            "/rtx/post/tonemap/cm2Factor": self.cm2_factor,
            "/rtx/post/tonemap/whitepoint": self.white_point,
            "/rtx/post/tonemap/filmIso": self.film_iso,
            "/rtx/post/tonemap/cameraShutter": self.camera_shutter,
            "/rtx/post/tonemap/fNumber": self.f_number,
            "/rtx/post/tonemap/colorMode": self.color_mode,
            "/rtx/post/tonemap/wrapValue": self.wrapvalue,
        }
        tonemap_op_idx = self._carb_settings.get("/rtx/post/tonemap/op")
        if tonemap_op_idx == 3:  # Modified Reinhard
            settings.update(
                {
                    "/rtx/post/tonemap/maxWhiteLuminance": self.max_white_luminance,
                }
            )
        if tonemap_op_idx == 5:  # HableUc2
            settings.update(
                {
                    "/rtx/post/tonemap/whiteScale": self.white_scale,
                }
            )
        if tonemap_op_idx == 7:  # Iray
            settings.update(
                {
                    "/rtx/post/tonemap/irayReinhard/crushBlacks": self.crush_blacks,
                    "/rtx/post/tonemap/irayReinhard/burnHighlights": self.burn_highlights,
                    "/rtx/post/tonemap/irayReinhard/burnHighlightsPerComponent": self.burn_highlights_per_component,
                    "/rtx/post/tonemap/irayReinhard/burnHighlightsMaxComponent": self.burn_highlights_max_component,
                    "/rtx/post/tonemap/irayReinhard/saturation": self.saturation,
                }
            )
        if tonemap_op_idx != 0:  # Clamp is never using srgb conversion
            settings.update(
                {
                    "/rtx/post/tonemap/enableSrgbToGamma": self.enable_srgb_to_gamma,
                }
            )
        return settings


class AutoExposureSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        histFilter_types = ["Median", "Average"]
        self.filter_type = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Histogram Filter",
            "/rtx/post/histogram/filterType",
            range_list=histFilter_types,
        )
        self.tau = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Adaptation Speed",
            "/rtx/post/histogram/tau",
            range_from=0.5,
            range_to=10.0,
        )
        self.white_scale = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "White Point Scale",
            "/rtx/post/histogram/whiteScale",
            range_from=0.01,
            range_to=80.0,
        )
        self.use_exposure_clamping = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Use Exposure Clamping",
            "/rtx/post/histogram/useExposureClamping",
        )
        self.min_ev = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Min EV",
            "/rtx/post/histogram/minEV",
            range_from=0.0,
            range_to=1000000.0,
        )
        self.max_ev = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Max EV",
            "/rtx/post/histogram/maxEV",
            range_from=0.0,
            range_to=1000000.0,
        )

    @property
    def settings(self):
        return {
            "/rtx/post/histogram/filterType": self.filter_type,
            "/rtx/post/histogram/tau": self.tau,
            "/rtx/post/histogram/whiteScale": self.white_scale,
            "/rtx/post/histogram/useExposureClamping": self.use_exposure_clamping,
            "/rtx/post/histogram/minEV": self.min_ev,
            "/rtx/post/histogram/maxEV": self.max_ev,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/post/histogram/enabled"


class ColorCorrectionSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        mode = ["ACES (Pre-Tonemap)", "Standard (Post-Tonemap)"]
        self.mode = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.STRING, "Mode", "/rtx/post/colorcorr/mode", range_list=mode
        )

        # ccMode = self._carb_settings.get("/rtx/post/colorcorr/mode")

        # ccMode == 0
        color_correction_mode = ["sRGBLinear", "ACEScg"]
        self.outputMode = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Output Color Space",
            "/rtx/post/colorcorr/outputMode",
            range_list=color_correction_mode,
        )

        self.saturation = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Saturation", "/rtx/post/colorcorr/saturation"
        )
        self.contrast = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Contrast", "/rtx/post/colorcorr/contrast"
        )
        self.gamma = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Gamma", "/rtx/post/colorcorr/gamma"
        )
        self.gain = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Gain", "/rtx/post/colorcorr/gain"
        )
        self.offset = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Offset", "/rtx/post/colorcorr/offset"
        )

    @property
    def settings(self):
        settings = {
            "/rtx/post/colorcorr/mode": self.mode,
            "/rtx/post/colorcorr/saturation": self.saturation,
            "/rtx/post/colorcorr/contrast": self.contrast,
            "/rtx/post/colorcorr/gamma": self.gamma,
            "/rtx/post/colorcorr/gain": self.gain,
            "/rtx/post/colorcorr/offset": self.offset,
        }
        cc_mode = self._carb_settings.get("/rtx/post/colorcorr/mode")
        if cc_mode == 0:
            settings.update(
                {
                    "/rtx/post/colorcorr/outputMode": self.outputMode,
                }
            )
        return settings

    @property
    def enabled_setting_path(self):
        return "/rtx/post/colorcorr/enabled"


class ColorGradingSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        mode = ["ACES (Pre-Tonemap)", "Standard (Post-Tonemap)"]
        self.mode = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.STRING, "Mode", "/rtx/post/colorgrad/mode", range_list=mode
        )

        cg_mode = self._carb_settings.get("/rtx/post/colorgrad/mode")
        if cg_mode == 0:
            colorGradingMode = ["sRGBLinear", "ACEScg"]
            self.output_mode = SettingItem(
                self,
                lazy.omni.kit.widget.settings.SettingType.STRING,
                "Output Color Space",
                "/rtx/post/colorgrad/outputMode",
                range_list=colorGradingMode,
            )

        self.blackpoint = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Black Point", "/rtx/post/colorgrad/blackpoint"
        )
        self.whitepoint = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "White Point", "/rtx/post/colorgrad/whitepoint"
        )
        self.contrast = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Contrast", "/rtx/post/colorgrad/contrast"
        )
        self.lift = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Lift", "/rtx/post/colorgrad/lift"
        )
        self.gain = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Gain", "/rtx/post/colorgrad/gain"
        )
        self.multiply = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Multiply", "/rtx/post/colorgrad/multiply"
        )
        self.offset = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Offset", "/rtx/post/colorgrad/offset"
        )
        self.gamma = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.COLOR3, "Gamma", "/rtx/post/colorgrad/gamma"
        )

    @property
    def settings(self):
        settings = {
            "/rtx/post/colorgrad/mode": self.mode,
            "/rtx/post/colorgrad/blackpoint": self.blackpoint,
            "/rtx/post/colorgrad/whitepoint": self.whitepoint,
            "/rtx/post/colorgrad/contrast": self.contrast,
            "/rtx/post/colorgrad/lift": self.lift,
            "/rtx/post/colorgrad/gain": self.gain,
            "/rtx/post/colorgrad/multiply": self.multiply,
            "/rtx/post/colorgrad/offset": self.offset,
            "/rtx/post/colorgrad/gamma": self.gamma,
        }
        cg_mode = self._carb_settings.get("/rtx/post/colorgrad/mode")
        if cg_mode == 0:
            settings.update(
                {
                    "/rtx/post/colorgrad/outputMode": self.output_mode,
                }
            )
        return settings

    @property
    def enabled_setting_path(self):
        return "/rtx/post/colorgrad/enabled"


class XRCompositingSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.apply_alpha_zero_pass_first = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Composite in Linear Space",
            "/rtx/post/backgroundZeroAlpha/ApplyAlphaZeroPassFirst",
        )
        self.backgroundComposite = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Composite in Editor",
            "/rtx/post/backgroundZeroAlpha/backgroundComposite",
        )
        # self.backplate_texture = SettingItem(self, "ASSET", "Default Backplate Texture", "/rtx/post/backgroundZeroAlpha/backplateTexture")
        self.background_default_color = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.COLOR3,
            "Default Backplate Color",
            "/rtx/post/backgroundZeroAlpha/backgroundDefaultColor",
        )
        self.enable_lens_distortion_correction = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Lens Distortion",
            "/rtx/post/backgroundZeroAlpha/enableLensDistortionCorrection",
        )
        # self.distortion_map = SettingItem(self, "ASSET", "Lens Distortion Map", "/rtx/post/lensDistortion/distortionMap")
        # self.undistortion_map = SettingItem(self, "ASSET", "Lens Undistortion Map", "/rtx/post/lensDistortion/undistortionMap")

    @property
    def settings(self):
        return {
            "/rtx/post/backgroundZeroAlpha/ApplyAlphaZeroPassFirst": self.apply_alpha_zero_pass_first,
            "/rtx/post/backgroundZeroAlpha/backgroundComposite": self.backgroundComposite,
            "/rtx/post/backgroundZeroAlpha/backgroundDefaultColor": self.background_default_color,
            "/rtx/post/backgroundZeroAlpha/enableLensDistortionCorrection": self.enable_lens_distortion_correction,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/post/backgroundZeroAlpha/enabled"


class ChromaticAberrationSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.strength_r = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Strength Red",
            "/rtx/post/chromaticAberration/strengthR",
            -1.0,
            1.0,
            0.01,
        )
        self.strength_g = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Strength Green",
            "/rtx/post/chromaticAberration/strengthG",
            -1.0,
            1.0,
            0.01,
        )
        self.strength_b = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Strength Blue",
            "/rtx/post/chromaticAberration/strengthB",
            -1.0,
            1.0,
            0.01,
        )
        chromatic_aberration_ops = ["Radial", "Barrel"]
        self.mode_r = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Algorithm Red",
            "/rtx/post/chromaticAberration/modeR",
            chromatic_aberration_ops,
        )
        self.mode_g = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Algorithm Green",
            "/rtx/post/chromaticAberration/modeG",
            chromatic_aberration_ops,
        )
        self.mode_b = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.STRING,
            "Algorithm Blue",
            "/rtx/post/chromaticAberration/modeB",
            chromatic_aberration_ops,
        )
        self.enable_lanczos = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Use Lanczos Sampler",
            "/rtx/post/chromaticAberration/enableLanczos",
        )

    @property
    def settings(self):
        return {
            "/rtx/post/chromaticAberration/strengthR": self.strength_r,
            "/rtx/post/chromaticAberration/strengthG": self.strength_g,
            "/rtx/post/chromaticAberration/strengthB": self.strength_b,
            "/rtx/post/chromaticAberration/modeR": self.mode_r,
            "/rtx/post/chromaticAberration/modeG": self.mode_g,
            "/rtx/post/chromaticAberration/modeB": self.mode_b,
            "/rtx/post/chromaticAberration/enableLanczos": self.enable_lanczos,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/post/chromaticAberration/enabled"


class DepthOfFieldSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.dof_enabled = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.BOOL, "Enable DOF", "/rtx/post/dof/enabled"
        )
        self.subject_distance = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Subject Distance",
            "/rtx/post/dof/subjectDistance",
            range_from=-10000,
            range_to=10000.0,
        )
        self.focal_length = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Focal Length (mm)",
            "/rtx/post/dof/focalLength",
            range_from=0,
            range_to=1000,
        )
        self.f_number = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "f-Number / f-Stop",
            "/rtx/post/dof/fNumber",
            range_from=0,
            range_to=1000,
        )
        self.anisotropy = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Anisotropy",
            "/rtx/post/dof/anisotropy",
            range_from=-1,
            range_to=1,
        )

    @property
    def settings(self):
        return {
            "/rtx/post/dof/enabled": self.dof_enabled,
            "/rtx/post/dof/subjectDistance": self.subject_distance,
            "/rtx/post/dof/focalLength": self.focal_length,
            "/rtx/post/dof/fNumber": self.f_number,
            "/rtx/post/dof/anisotropy": self.anisotropy,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/post/dof/overrideEnabled"


class MotionBlurSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.max_blur_diameter_fraction = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Blur Diameter Fraction",
            "/rtx/post/motionblur/maxBlurDiameterFraction",
            range_from=0,
            range_to=0.5,
        )
        self.num_samples = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Number of Samples",
            "/rtx/post/motionblur/numSamples",
            range_from=4,
            range_to=32,
        )
        self.exposure_fraction = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Exposure Fraction",
            "/rtx/post/motionblur/exposureFraction",
            range_from=0,
            range_to=5.0,
        )

    @property
    def settings(self):
        return {
            "/rtx/post/motionblur/maxBlurDiameterFraction": self.max_blur_diameter_fraction,
            "/rtx/post/motionblur/numSamples": self.num_samples,
            "/rtx/post/motionblur/exposureFraction": self.exposure_fraction,
        }

    @property
    def enabled_setting_path(self):
        return "/rtx/post/motionblur/enabled"


class FTTBloomSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.flare_scale = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Scale",
            "/rtx/post/lensFlares/flareScale",
            range_from=-1000,
            range_to=1000,
        )
        self.cutoff_point = SettingItem(
            self, lazy.omni.kit.widget.settings.SettingType.DOUBLE3, "Cutoff Point", "/rtx/post/lensFlares/cutoffPoint"
        )
        self.cutoff_fuzziness = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Cutoff Fuzziness",
            "/rtx/post/lensFlares/cutoffFuzziness",
            range_from=0.0,
            range_to=1.0,
        )
        self.energy_constraining_blend = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Energy Constrained",
            "/rtx/post/lensFlares/energyConstrainingBlend",
        )
        self.physical_settings = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Physical Settings",
            "/rtx/post/lensFlares/physicalSettings",
        )

        # fftbloom_use_physical_settings = self._carb_settings.get("/rtx/post/lensFlares/physicalSettings")
        # Physical settings
        self.blades = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.INT,
            "Blades",
            "/rtx/post/lensFlares/blades",
            range_from=0,
            range_to=10,
        )
        self.aperture_rotation = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Aperture Rotation",
            "/rtx/post/lensFlares/apertureRotation",
            range_from=-1000,
            range_to=1000,
        )
        self.sensor_diagonal = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Sensor Diagonal",
            "/rtx/post/lensFlares/sensorDiagonal",
            range_from=-1000,
            range_to=1000,
        )
        self.sensor_aspect_ratio = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Sensor Aspect Ratio",
            "/rtx/post/lensFlares/sensorAspectRatio",
            range_from=-1000,
            range_to=1000,
        )
        self.f_number = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "f-Number / f-Stop",
            "/rtx/post/lensFlares/fNumber",
            range_from=-1000,
            range_to=1000,
        )
        self.focal_length = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Focal Length (mm)",
            "/rtx/post/lensFlares/focalLength",
            range_from=-1000,
            range_to=1000,
        )

        # Non-physical settings
        self.halo_flare_radius = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.DOUBLE3,
            "Halo Radius",
            "/rtx/post/lensFlares/haloFlareRadius",
        )
        self.halo_flare_falloff = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.DOUBLE3,
            "Halo Flare Falloff",
            "/rtx/post/lensFlares/haloFlareFalloff",
        )
        self.halo_flare_weight = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Halo Flare Weight",
            "/rtx/post/lensFlares/haloFlareWeight",
            range_from=-1000,
            range_to=1000,
        )
        self.aniso_flare_falloff_y = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.DOUBLE3,
            "Aniso Falloff Y",
            "/rtx/post/lensFlares/anisoFlareFalloffY",
        )
        self.aniso_flare_falloff_x = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.DOUBLE3,
            "Aniso Falloff X",
            "/rtx/post/lensFlares/anisoFlareFalloffX",
        )
        self.aniso_flare_weight = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Aniso Flare Weight",
            "/rtx/post/lensFlares/anisoFlareWeight",
            range_from=-1000,
            range_to=1000,
        )
        self.isotropic_flare_falloff = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.DOUBLE3,
            "Isotropic Flare Falloff",
            "/rtx/post/lensFlares/isotropicFlareFalloff",
        )
        self.isotropic_flare_weight = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Isotropic Flare Weight",
            "/rtx/post/lensFlares/isotropicFlareWeight",
            range_from=-1000,
            range_to=1000,
        )

    @property
    def settings(self):
        settings = {
            "/rtx/post/lensFlares/flareScale": self.flare_scale,
            "/rtx/post/lensFlares/cutoffPoint": self.cutoff_point,
            "/rtx/post/lensFlares/cutoffFuzziness": self.cutoff_fuzziness,
            "/rtx/post/lensFlares/energyConstrainingBlend": self.energy_constraining_blend,
            "/rtx/post/lensFlares/physicalSettings": self.physical_settings,
        }
        fftbloom_use_physical_settings = self._carb_settings.get("/rtx/post/lensFlares/physicalSettings")
        if fftbloom_use_physical_settings:
            settings.update(
                {
                    "/rtx/post/lensFlares/blades": self.blades,
                    "/rtx/post/lensFlares/apertureRotation": self.aperture_rotation,
                    "/rtx/post/lensFlares/sensorDiagonal": self.sensor_diagonal,
                    "/rtx/post/lensFlares/sensorAspectRatio": self.sensor_aspect_ratio,
                    "/rtx/post/lensFlares/fNumber": self.f_number,
                    "/rtx/post/lensFlares/focalLength": self.focal_length,
                }
            )
        else:
            settings.update(
                {
                    "/rtx/post/lensFlares/haloFlareRadius": self.halo_flare_radius,
                    "/rtx/post/lensFlares/haloFlareFalloff": self.halo_flare_falloff,
                    "/rtx/post/lensFlares/haloFlareWeight": self.halo_flare_weight,
                    "/rtx/post/lensFlares/anisoFlareFalloffY": self.aniso_flare_falloff_y,
                    "/rtx/post/lensFlares/anisoFlareFalloffX": self.aniso_flare_falloff_x,
                    "/rtx/post/lensFlares/anisoFlareWeight": self.aniso_flare_weight,
                    "/rtx/post/lensFlares/isotropicFlareFalloff": self.isotropic_flare_falloff,
                    "/rtx/post/lensFlares/isotropicFlareWeight": self.isotropic_flare_weight,
                }
            )
        return settings

    @property
    def enabled_setting_path(self):
        return "/rtx/post/lensFlares/enabled"


class TVNoiseGrainSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        self.enable_scanlines = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Scanlines",
            "/rtx/post/tvNoise/enableScanlines",
        )
        self.scanline_spread = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Scanline Spreading",
            "/rtx/post/tvNoise/scanlineSpread",
            range_from=0.0,
            range_to=2.0,
        )
        self.enable_scroll_bug = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Scroll Bug",
            "/rtx/post/tvNoise/enableScrollBug",
        )
        self.enable_vignetting = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Vignetting",
            "/rtx/post/tvNoise/enableVignetting",
        )
        self.vignetting_size = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Vignetting Size",
            "/rtx/post/tvNoise/vignettingSize",
            range_from=0.0,
            range_to=255,
        )
        self.vignetting_strength = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Vignetting Strength",
            "/rtx/post/tvNoise/vignettingStrength",
            range_from=0.0,
            range_to=2.0,
        )
        self.enable_vignetting_flickering = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Vignetting Flickering",
            "/rtx/post/tvNoise/enableVignettingFlickering",
        )
        self.enable_ghost_flickering = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Ghost Flickering",
            "/rtx/post/tvNoise/enableGhostFlickering",
        )
        self.enable_wave_distortion = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Wavy Distortion",
            "/rtx/post/tvNoise/enableWaveDistortion",
        )
        self.enable_vertical_lines = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Vertical Lines",
            "/rtx/post/tvNoise/enableVerticalLines",
        )
        self.enable_random_splotches = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Random Splotches",
            "/rtx/post/tvNoise/enableRandomSplotches",
        )
        self.enable_film_grain = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.BOOL,
            "Enable Film Grain",
            "/rtx/post/tvNoise/enableFilmGrain",
        )

        # Filmgrain is a subframe in TV Noise
        # self._carb_settings.get("/rtx/post/tvNoise/enableFilmGrain"):
        self.grain_amount = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Grain Amount",
            "/rtx/post/tvNoise/grainAmount",
            range_from=0,
            range_to=0.2,
        )
        self.color_amount = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Color Amount",
            "/rtx/post/tvNoise/colorAmount",
            range_from=0,
            range_to=1.0,
        )
        self.lum_amount = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Luminance Amount",
            "/rtx/post/tvNoise/lumAmount",
            range_from=0,
            range_to=1.0,
        )
        self.grain_size = SettingItem(
            self,
            lazy.omni.kit.widget.settings.SettingType.FLOAT,
            "Grain Size",
            "/rtx/post/tvNoise/grainSize",
            range_from=1.5,
            range_to=2.5,
        )

    @property
    def settings(self):
        settings = {
            "/rtx/post/tvNoise/enableScanlines": self.enable_scanlines,
            "/rtx/post/tvNoise/scanlineSpread": self.scanline_spread,
            "/rtx/post/tvNoise/enableScrollBug": self.enable_scroll_bug,
            "/rtx/post/tvNoise/enableVignetting": self.enable_vignetting,
            "/rtx/post/tvNoise/vignettingSize": self.vignetting_size,
            "/rtx/post/tvNoise/vignettingStrength": self.vignetting_strength,
            "/rtx/post/tvNoise/enableVignettingFlickering": self.enable_vignetting_flickering,
            "/rtx/post/tvNoise/enableGhostFlickering": self.enable_ghost_flickering,
            "/rtx/post/tvNoise/enableWaveDistortion": self.enable_wave_distortion,
            "/rtx/post/tvNoise/enableVerticalLines": self.enable_vertical_lines,
            "/rtx/post/tvNoise/enableRandomSplotches": self.enable_random_splotches,
            "/rtx/post/tvNoise/enableFilmGrain": self.enable_film_grain,
        }
        if self._carb_settings.get("/rtx/post/tvNoise/enableFilmGrain"):
            settings.update(
                {
                    "/rtx/post/tvNoise/grainAmount": self.grain_amount,
                    "/rtx/post/tvNoise/colorAmount": self.color_amount,
                    "/rtx/post/tvNoise/lumAmount": self.lum_amount,
                    "/rtx/post/tvNoise/grainSize": self.grain_size,
                }
            )
        return settings

    @property
    def enabled_setting_path(self):
        return "/rtx/post/tvNoise/enabled"


class ReshadeSettings(SubSettingsBase):
    def __init__(self):
        self._carb_settings = lazy.carb.settings.get_settings()

        # self._add_setting("ASSET", "Preset path", "/rtx/reshade/presetFilePath")
        # widget = self._add_setting("ASSET", "Effect search dir path", "/rtx/reshade/effectSearchDirPath")
        # widget.is_folder=True
        # widget = self._add_setting("ASSET", "Texture search dir path", "/rtx/reshade/textureSearchDirPath")
        # widget.is_folder=True

    @property
    def settings(self):
        return {}

    @property
    def enabled_setting_path(self):
        return "/rtx/reshade/enable"
