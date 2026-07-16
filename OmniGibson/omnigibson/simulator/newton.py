"""Newton simulator implementation for native OmniGibson assets."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import math
import os
import warnings

# Newton's USD importer can initialize PXR while importing the Newton package.
# Keep this guard before ``import newton`` until the W1 load stress tests in
# docs/other/newton_migration.md pass with OpenUSD's default thread limit.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import newton
import numpy as np
import torch as th
import warp as wp
from newton._src.geometry import ShapeFlags
from newton._src.geometry.inertia import verify_and_correct_inertia

from omnigibson.newton.assets import (
    HIDDEN_METALINK_TYPES,
    prepared_dataset_object_usd,
    resolve_dataset_object_usd,
    resolve_robot_default_joint_positions,
    resolve_robot_controller_metadata,
    resolve_robot_asset,
)
from omnigibson.newton.entities import make_newton_entity_from_labels
from omnigibson.newton.visuals import add_usd_visual_shapes
from omnigibson.scenes.scene_base import NewtonObjectSpec, NewtonRobotSpec, NewtonSceneSpec
from omnigibson.runtime import EntityRegistry
from omnigibson.simulator.simulator import AbstractSimulator


@dataclass
class NewtonSimulationConfig:
    """Newton physics settings.

    This intentionally mirrors Isaac Lab's Newton split: OmniGibson owns the
    frame/substep timing, while MuJoCo-Warp-specific contact and solver knobs
    live here instead of being baked into scene/object import code.
    """

    fps: float = 50.0
    sim_substeps: int = 8
    device: str | None = None
    object_position: tuple[float, float, float] = (1.0, 0.0, 0.5)
    # Robots spawn at ground level like the legacy default. Spawning above the
    # ground drops mobile bases onto their chassis with the wheels sunk in the
    # floor, which pins them in place under wheel commands.
    robot_position: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    solver: str = "mujoco"
    # BEHAVIOR scenes are dense contact scenes imported from USD. These MuJoCo-
    # Warp defaults are deliberately more conservative than Newton's examples so
    # full interactive scenes stay finite during settling and moderate object /
    # articulation disturbance tests.
    solver_iterations: int = 150
    solver_ls_iterations: int = 80
    solver_ccd_iterations: int | None = 120
    solver_sdf_iterations: int | None = None
    solver_sdf_initpoints: int | None = None
    solver_method: str | None = None
    solver_integrator: str | None = None
    solver_cone: str | None = "elliptic"
    solver_jacobian: str | None = None
    solver_impratio: float | None = 5.0
    solver_tolerance: float | None = None
    solver_ls_tolerance: float | None = None
    solver_ccd_tolerance: float | None = None
    solver_ls_parallel: bool = False
    use_mujoco_cpu: bool = False
    enable_multiccd: bool = False
    disable_contacts: bool = False
    update_data_interval: int = 1
    # Match Isaac Lab's MJWarp default: MuJoCo-Warp handles contacts unless a
    # caller explicitly selects Newton's collision pipeline. Keep the fallback
    # exposed because it is useful while isolating contact importer issues.
    use_mujoco_contacts: bool = True
    njmax: int = 32_768
    nconmax: int = 250_000
    mesh_maxhullvert: int = 64
    skip_mesh_approximation: bool = True
    load_visual_shapes: bool = False
    default_shape_ke: float = 1.0e2
    default_shape_kd: float = 5.0e1
    default_shape_kf: float = 1.0e2
    default_shape_mu: float = 0.9

    def solver_kwargs(self):
        kwargs = {
            "iterations": self.solver_iterations,
            "ls_iterations": self.solver_ls_iterations,
            "njmax": self.njmax,
            "nconmax": self.nconmax,
            "ccd_iterations": self.solver_ccd_iterations,
            "sdf_iterations": self.solver_sdf_iterations,
            "sdf_initpoints": self.solver_sdf_initpoints,
            "solver": self.solver_method,
            "integrator": self.solver_integrator,
            "cone": self.solver_cone,
            "jacobian": self.solver_jacobian,
            "impratio": self.solver_impratio,
            "tolerance": self.solver_tolerance,
            "ls_tolerance": self.solver_ls_tolerance,
            "ccd_tolerance": self.solver_ccd_tolerance,
            "use_mujoco_cpu": self.use_mujoco_cpu,
            "enable_multiccd": self.enable_multiccd,
            "disable_contacts": self.disable_contacts,
            "update_data_interval": self.update_data_interval,
            "ls_parallel": self.solver_ls_parallel,
            "use_mujoco_contacts": self.use_mujoco_contacts,
        }
        return {key: value for key, value in kwargs.items() if value is not None}

    def apply_default_shape_cfg(self, builder):
        builder.default_shape_cfg.ke = self.default_shape_ke
        builder.default_shape_cfg.kd = self.default_shape_kd
        builder.default_shape_cfg.kf = self.default_shape_kf
        builder.default_shape_cfg.mu = self.default_shape_mu


class NewtonSceneSimulator(AbstractSimulator):
    """Build and run an OmniGibson scene with Newton."""

    def __init__(
        self,
        scene,
        *,
        data_path=None,
        config=None,
    ):
        self.scene = scene
        self.data_path = data_path
        self.config = config or NewtonSimulationConfig()
        self.viewer = None
        self.viewer_camera = _NewtonViewerCamera(self)
        self.entity_registry = EntityRegistry()

        if self.config.device:
            wp.set_device(self.config.device)

        self.frame_dt = 1.0 / self.config.fps
        self.sim_dt = self.frame_dt / self.config.sim_substeps
        self.sim_time = 0.0

        self._dataset_usd_contexts = []
        self._prepared_dataset_usd_paths = []
        self._prepared_dataset_usd_cache = {}
        self._object_builder_cache = {}
        self._visual_mesh_sources = []
        self._low_friction_chassis_bodies = set()
        self.model = None
        self.solver = None
        self.state_0 = None
        self.state_1 = None
        self.control = None
        self.contacts = None

    def __enter__(self):
        self.build()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @property
    def resolved_dataset_usd_paths(self):
        return tuple(self._resolve_dataset_usd_path(obj) for obj in self.scene.object_specs)

    @property
    def resolved_robot_asset_paths(self):
        return tuple(self._resolve_robot_asset_path(robot) for robot in self.scene.robot_specs)

    @classmethod
    def from_environment_configs(cls, configs):
        from omnigibson.newton.config import load_newton_config, scene_from_config, simulation_config_from_config

        merged = {}
        for config in configs:
            _merge_dicts(merged, load_newton_config(config))
        return cls(
            scene_from_config(merged),
            config=simulation_config_from_config(merged),
        )

    @staticmethod
    def should_auto_build_environment(configs):
        from omnigibson.newton.config import load_newton_config

        merged = {}
        for config in configs:
            _merge_dicts(merged, load_newton_config(config))
        return merged.get("newton", {}).get("environment", {}).get("auto_build", True)

    def build_environment(self):
        import omnigibson as og
        from omnigibson.macros import gm

        if self.model is None:
            self.build()
        og.sim = self
        # Isaac Sim creates a visible application at launch in non-headless
        # runs. Newton's OpenGL viewer is explicit, so attach it here to keep
        # existing examples visible even when they only call env.step().
        if not gm.HEADLESS and self.viewer is None:
            self.attach_viewer()
        print_welcome(self)
        return self

    def apply_environment_action(self, action):
        if action is None or _is_empty_action(action):
            return

        robots = self.robots
        if not robots:
            return

        if isinstance(action, dict):
            for robot in robots:
                if robot.name in action and hasattr(robot, "apply_action"):
                    robot.apply_action(action[robot.name])
            return

        if len(robots) == 1:
            robots[0].apply_action(action)
            return

        cursor = 0
        for robot in robots:
            action_dim = robot.action_dim
            robot.apply_action(action[cursor : cursor + action_dim])
            cursor += action_dim

    def build(self):
        builder = newton.ModelBuilder()
        pending_entities = []
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        self.config.apply_default_shape_cfg(builder)

        if self.scene.use_ground_plane:
            builder.add_ground_plane()

        pending_visual_imports = []
        hidden_collision_shape_indices = []

        # Import robot articulations before object resources accumulate. This
        # ordering remains until workaround W2 in the migration record passes
        # equivalent robot-last stress tests.
        for robot in self.scene.robot_specs:
            if robot.asset.asset_format != "usd":
                raise ValueError("The Newton-first OmniGibson path imports robots through USD only.")
            robot_asset_path = self._resolve_robot_asset_path(robot)
            robot_import_info = builder.add_usd(
                str(robot_asset_path),
                xform=_xform(robot.position, robot.orientation),
                floating=not robot.fixed_base,
                enable_self_collisions=False,
                collapse_fixed_joints=False,
                skip_mesh_approximation=self.config.skip_mesh_approximation,
                load_visual_shapes=False,
                hide_collision_shapes=False,
            )
            _disable_imported_self_collisions(builder, robot_import_info)
            _apply_authored_robot_mass_properties(builder, robot_asset_path, robot_import_info)
            self._low_friction_chassis_bodies |= _apply_chassis_caster_friction(builder, robot_import_info)
            pending_visual_imports.append(
                {
                    "usd_path": robot_asset_path,
                    "root_xform": _xform(robot.position, robot.orientation),
                    "root_scale": None,
                    "label_prefix": robot.name,
                    "import_info": robot_import_info,
                }
            )
            pending_entities.append(
                {
                    "name": robot.name,
                    "category": "robot",
                    "kind": "robot",
                    "source_path": robot_asset_path,
                    "label_prefix": _import_label_prefix(robot_import_info, robot.asset.model),
                    "default_joint_pos": resolve_robot_default_joint_positions(robot.asset, self.data_path),
                    "controller_metadata": resolve_robot_controller_metadata(robot.asset, self.data_path),
                    "action_normalize": robot.action_normalize,
                    "obs_modalities": tuple(getattr(robot, "obs_modalities", ("rgb",))),
                }
            )

        object_builder_keys = [self._object_builder_cache_key(obj) for obj in self.scene.object_specs]
        repeated_object_builder_keys = {key for key, count in Counter(object_builder_keys).items() if count > 1}

        for obj, object_builder_key in zip(self.scene.object_specs, object_builder_keys):
            object_source_path, object_import_path, object_builder, object_import_info = self._prepare_object_builder(
                obj,
                cache_builder=object_builder_key in repeated_object_builder_keys,
                cache_key=object_builder_key,
            )
            start_body_idx = builder.body_count
            start_shape_idx = builder.shape_count
            builder.add_builder(object_builder, xform=_xform(obj.position, obj.orientation), label_prefix=obj.name)
            pending_visual_imports.append(
                {
                    "usd_path": object_import_path,
                    "root_xform": _xform(obj.position, obj.orientation),
                    "root_scale": obj.scale,
                    "label_prefix": obj.name,
                    "import_info": _offset_import_info(object_import_info, start_body_idx, start_shape_idx),
                }
            )

            category = getattr(obj, "category", None) or (obj.asset.category if obj.asset is not None else "usd_object")
            pending_entities.append(
                {
                    "name": obj.name,
                    "category": category,
                    "kind": "object",
                    "source_path": object_source_path,
                    "label_prefix": obj.name,
                    "scale": obj.scale,
                    "scale_baked": obj.object_type == "DatasetObject",
                }
            )

        # Combined visual and physics import has produced native failures in full
        # BEHAVIOR scenes. Finish physics imports first, then add visible-only
        # meshes as documented by workaround W3 in the migration record.
        for visual_import in pending_visual_imports:
            visual_result = add_usd_visual_shapes(
                builder,
                visual_import["usd_path"],
                visual_import["import_info"],
                root_xform=visual_import["root_xform"],
                root_scale=visual_import["root_scale"],
                label_prefix=visual_import["label_prefix"],
            )
            hidden_collision_shape_indices.extend((visual_import["import_info"].get("path_shape_map") or {}).values())
            self._visual_mesh_sources.extend(visual_result.mesh_sources)

        if self.config.solver == "mujoco":
            _ensure_mujoco_moving_body_mass_properties(builder)
        _refresh_builder_mass_properties(builder)
        if os.environ.get("OMNIGIBSON_NEWTON_VALIDATE_INERTIA_DETAILED"):
            # Opt-in diagnostic for migration work: Newton's fast GPU inertia
            # validation only reports a correction count. The detailed path
            # names the offending bodies without changing normal runtime logs.
            builder.validate_inertia_detailed = True

        self.model = builder.finalize()
        _hide_model_shapes(self.model, hidden_collision_shape_indices)
        self.entity_registry.clear()
        for entity_info in pending_entities:
            scale = entity_info.pop("scale", None)
            scale_baked = entity_info.pop("scale_baked", False)
            entity = make_newton_entity_from_labels(simulator=self, **entity_info)
            self.entity_registry.add(entity)
            if scale is not None:
                if scale_baked:
                    entity.set_loaded_scale(scale)
                else:
                    entity.scale = scale

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if self.model.joint_count > 0:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        for robot in self.robots:
            if hasattr(robot, "reset"):
                robot.reset()
        if hasattr(self.scene, "bind_simulator"):
            self.scene.bind_simulator(self)

        if self.model.joint_count == 0:
            self.solver = None
        elif self.config.solver == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=self.config.solver_iterations)
        elif self.config.solver == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                **self.config.solver_kwargs(),
            )
        else:
            raise ValueError(f"Unsupported Newton solver: {self.config.solver}")

        if self.solver is not None and self.config.solver == "mujoco" and self.config.use_mujoco_contacts:
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        else:
            self.contacts = self.model.contacts()

        if self._low_friction_chassis_bodies and isinstance(self.solver, newton.solvers.SolverMuJoCo):
            _elevate_mjc_geom_priority(self.solver, self.model, self._low_friction_chassis_bodies)

        return self

    @property
    def entities(self):
        return self.entity_registry.values()

    @property
    def objects(self):
        return self.entity_registry.by_kind("object")

    @property
    def robots(self):
        return self.entity_registry.by_kind("robot")

    def attach_viewer(self, viewer=None):
        if self.model is None:
            raise RuntimeError("Build the Newton model before attaching a viewer.")
        if self.viewer is not None:
            if viewer is None or viewer is self.viewer:
                self.viewer_camera.apply_pending()
                return self.viewer
            self.viewer.close()
        self.viewer = viewer or newton.viewer.ViewerGL(headless=False)
        self.viewer.set_model(self.model)
        self._apply_scene_lights()
        self.viewer_camera.apply_pending()
        return self.viewer

    def enable_viewer_camera_teleoperation(self):
        if self.viewer is None:
            self.attach_viewer()
        return self.viewer_camera

    def step(self, render=False):
        if self.solver is None:
            self.sim_time += self.frame_dt
            if render:
                self.render()
            return

        for _ in range(self.config.sim_substeps):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            if self.config.solver != "mujoco" or not self.config.use_mujoco_contacts:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.config.solver == "mujoco" and self.config.use_mujoco_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)
        self.sim_time += self.frame_dt
        if render:
            self.render()

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def run(self, num_frames, render=False):
        for _ in range(num_frames):
            self.step()
            if render:
                self.render()

    def summary(self):
        robot_assets = self.resolved_robot_asset_paths
        return {
            "scene": self.scene.name,
            "dataset_object_usds": [str(path) for path in self._prepared_dataset_usd_paths],
            "robot_assets": [str(path) for path in robot_assets],
            "dataset_object_usd": str(self._prepared_dataset_usd_paths[0])
            if self._prepared_dataset_usd_paths
            else None,
            "robot_asset": str(robot_assets[0]) if robot_assets else None,
            "body_count": self.model.body_count,
            "joint_count": self.model.joint_count,
            "shape_count": self.model.shape_count,
            "entities": [entity.summary() for entity in self.entities],
        }

    _STATE_SNAPSHOT_ARRAYS = ("body_q", "body_qd", "joint_q", "joint_qd")
    _CONTROL_SNAPSHOT_ARRAYS = ("joint_f", "joint_target_q", "joint_target_qd", "joint_act")

    def dump_state(self, serialized=False):
        if self.model is None:
            raise RuntimeError("Build the Newton model before dumping simulator state.")
        state = {
            "format": 1,
            "sim_time": float(self.sim_time),
            "state": {
                name: _snapshot_wp_array(getattr(self.state_0, name, None)) for name in self._STATE_SNAPSHOT_ARRAYS
            },
            "control": {
                name: _snapshot_wp_array(getattr(self.control, name, None)) for name in self._CONTROL_SNAPSHOT_ARRAYS
            },
        }
        return self.serialize(state) if serialized else state

    def load_state(self, state, serialized=False):
        if self.model is None:
            raise RuntimeError("Build the Newton model before loading simulator state.")
        if serialized:
            state = self.deserialize(state)
        if state.get("format") != 1:
            raise ValueError(f"Unsupported Newton simulator state format: {state.get('format')!r}")

        for sim_state in (self.state_0, self.state_1):
            for name in self._STATE_SNAPSHOT_ARRAYS:
                _restore_wp_array(getattr(sim_state, name, None), state["state"].get(name), name)
        for name in self._CONTROL_SNAPSHOT_ARRAYS:
            _restore_wp_array(getattr(self.control, name, None), state["control"].get(name), name)

        # MuJoCo derives body poses from joint coordinates on the next step;
        # refresh them immediately so entity pose queries reflect the restored
        # state without stepping.
        if self.state_0.joint_q is not None:
            for sim_state in (self.state_0, self.state_1):
                newton.eval_fk(self.model, sim_state.joint_q, sim_state.joint_qd, sim_state)

        # Clear solver buffers that persist between steps (contact warm-starts,
        # applied forces, actuator activations). With flags=0 the solver leaves
        # the restored joint coordinates untouched and only clears buffers.
        if isinstance(self.solver, newton.solvers.SolverMuJoCo):
            self.solver.reset(self.state_0, flags=newton.StateFlags(0))

        # Controller adapters accumulate joint targets across steps; drop the
        # caches so post-restore commands start from the restored state.
        for robot in self.robots:
            if hasattr(robot, "_last_joint_target"):
                robot._last_joint_target = None

        self.sim_time = float(state["sim_time"])

    def serialize(self, state):
        parts = [th.tensor([float(state["format"]), float(state["sim_time"])], dtype=th.float32)]
        for group, names in (("state", self._STATE_SNAPSHOT_ARRAYS), ("control", self._CONTROL_SNAPSHOT_ARRAYS)):
            for name in names:
                tensor = state[group].get(name)
                if tensor is not None:
                    parts.append(tensor.detach().reshape(-1).to(dtype=th.float32, device="cpu"))
        return th.cat(parts)

    def deserialize(self, state):
        flat = state.detach().reshape(-1).to(dtype=th.float32, device="cpu")
        result = {"format": int(flat[0].item()), "sim_time": float(flat[1].item()), "state": {}, "control": {}}
        cursor = 2
        for group, owner, names in (
            ("state", self.state_0, self._STATE_SNAPSHOT_ARRAYS),
            ("control", self.control, self._CONTROL_SNAPSHOT_ARRAYS),
        ):
            for name in names:
                view = _wp_array_view(getattr(owner, name, None))
                if view is None:
                    result[group][name] = None
                    continue
                count = view.numel()
                result[group][name] = flat[cursor : cursor + count].reshape(view.shape).to(view.device)
                cursor += count
        if cursor != flat.numel():
            raise ValueError(f"Serialized state has {flat.numel()} values, expected {cursor}.")
        return result

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        self.contacts = None
        self.control = None
        self.state_0 = None
        self.state_1 = None
        self.solver = None
        self.model = None
        self.entity_registry.clear()
        for dataset_usd_context in reversed(self._dataset_usd_contexts):
            dataset_usd_context.__exit__(None, None, None)
        self._dataset_usd_contexts.clear()
        self._prepared_dataset_usd_paths.clear()
        self._prepared_dataset_usd_cache.clear()
        self._object_builder_cache.clear()
        self._visual_mesh_sources.clear()
        if hasattr(self.scene, "unbind_simulator"):
            self.scene.unbind_simulator(self)
        import omnigibson as og

        if og.sim is self:
            og.sim = None

    def _resolve_dataset_usd_path(self, obj):
        if obj.usd_path is not None:
            return obj.usd_path
        if obj.asset is None:
            raise ValueError(f"Object {obj.name!r} does not define a USD path or DatasetObject asset.")
        return resolve_dataset_object_usd(obj.asset, self.data_path)

    def _resolve_robot_asset_path(self, robot):
        if robot.asset_path is not None:
            return robot.asset_path
        return resolve_robot_asset(robot.asset, self.data_path)

    def _prepare_object_usd(self, obj):
        source_path = self._resolve_dataset_usd_path(obj)
        if obj.object_type == "DatasetObject":
            cache_key = (source_path, _scale_cache_key(obj.scale))
            prepared_dataset_usd_path = self._prepared_dataset_usd_cache.get(cache_key)
            if prepared_dataset_usd_path is None:
                dataset_usd_context = prepared_dataset_object_usd(source_path, self.data_path)
                prepared_dataset_usd_path = dataset_usd_context.__enter__()
                self._dataset_usd_contexts.append(dataset_usd_context)
                self._prepared_dataset_usd_paths.append(prepared_dataset_usd_path)
                self._prepared_dataset_usd_cache[cache_key] = prepared_dataset_usd_path
            return prepared_dataset_usd_path, prepared_dataset_usd_path
        return source_path, source_path

    def _object_builder_cache_key(self, obj):
        return (
            self._resolve_dataset_usd_path(obj),
            _scale_cache_key(obj.scale),
            obj.fixed_base,
            obj.visual_only,
            self.config.skip_mesh_approximation,
            self.config.load_visual_shapes,
        )

    def _prepare_object_builder(self, obj, *, cache_builder, cache_key):
        object_source_path, object_import_path = self._prepare_object_usd(obj)
        cached_entry = self._object_builder_cache.get(cache_key) if cache_builder else None
        if cached_entry is not None:
            return object_source_path, object_import_path, cached_entry["builder"], cached_entry["import_info"]
        object_builder = None
        if object_builder is None:
            object_builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(object_builder)
            self.config.apply_default_shape_cfg(object_builder)
            object_import_info = object_builder.add_usd(
                str(object_import_path),
                floating=not obj.fixed_base,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                skip_mesh_approximation=self.config.skip_mesh_approximation,
                load_visual_shapes=False,
                hide_collision_shapes=False,
            )
            _scale_imported_builder(object_builder, obj.scale)
            _disable_imported_self_collisions(object_builder, object_import_info)
            _add_passive_object_joint_damping(object_builder)
            if obj.fixed_base:
                _insert_fixed_base_anchor_body(object_builder, obj.name)
            _hide_imported_metalink_shapes(object_builder, object_import_info)
            if obj.visual_only:
                _disable_imported_collision_shapes(object_builder, object_import_info)
            if cache_builder:
                self._object_builder_cache[cache_key] = {
                    "builder": object_builder,
                    "import_info": object_import_info,
                }
        return object_source_path, object_import_path, object_builder, object_import_info

    def _apply_scene_lights(self):
        if self.viewer is None or not self.scene.lights:
            return
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is None:
            return
        light = self.scene.lights[0]
        if hasattr(renderer, "_light_color"):
            value = max(light.intensity / 1.0e5, 0.1)
            renderer._light_color = (value, value, value)
        if hasattr(renderer, "_sun_direction"):
            direction = th.tensor(light.position, dtype=th.float32)
            norm = th.linalg.norm(direction)
            if norm > 0:
                renderer._sun_direction = (direction / norm).numpy()

    @property
    def viewer_visibility(self):
        return self.viewer is not None

    @viewer_visibility.setter
    def viewer_visibility(self, visible):
        if visible and self.viewer is None:
            self.attach_viewer()
        elif not visible and self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @property
    def viewer_width(self):
        return None

    @viewer_width.setter
    def viewer_width(self, width):
        pass

    @property
    def viewer_height(self):
        return None

    @viewer_height.setter
    def viewer_height(self, height):
        pass

    def get_sim_step_dt(self):
        return self.frame_dt

    def get_physics_dt(self):
        return self.sim_dt

    def get_rendering_dt(self):
        return self.frame_dt

    @property
    def gravity(self):
        """Gravity magnitude along -z, matching the legacy simulator attribute."""
        if self.model is None or self.model.gravity is None:
            return 9.81
        return float(-_wp_array_view(self.model.gravity).reshape(-1)[2])

    @property
    def n_physics_timesteps_per_render(self):
        return self.config.sim_substeps

    @property
    def current_time(self):
        return self.sim_time

    @property
    def current_time_step_index(self):
        return int(round(self.sim_time / self.sim_dt))

    def step_physics(self):
        self.step(render=False)

    def play(self):
        pass

    def pause(self):
        pass

    def stop(self):
        pass

    def is_playing(self):
        return True

    def is_stopped(self):
        return False

    def is_paused(self):
        return False

    @property
    def scenes(self):
        return (self.scene,)

    @property
    def floor_plane(self):
        return None

    @property
    def skybox(self):
        return None

    @property
    def device(self):
        return self.config.device or "cuda:0"

    @property
    def initial_physics_dt(self):
        return self.sim_dt

    @property
    def initial_rendering_dt(self):
        return self.frame_dt


class NewtonObjectRobotSimulator(NewtonSceneSimulator):
    """Compatibility wrapper for callers that provide one object and one robot."""

    def __init__(
        self,
        dataset_object,
        robot,
        *,
        data_path=None,
        dataset_usd_path=None,
        robot_asset_path=None,
        config=None,
    ):
        sim_config = config or NewtonSimulationConfig()
        scene = NewtonSceneSpec(
            objects=(
                NewtonObjectSpec(
                    name=f"{dataset_object.category}_{dataset_object.model}",
                    object_type="DatasetObject",
                    asset=dataset_object,
                    position=sim_config.object_position,
                    usd_path=Path(dataset_usd_path).expanduser().resolve() if dataset_usd_path else None,
                ),
            ),
            robots=(
                NewtonRobotSpec(
                    name=f"robot_{robot.model}",
                    asset=robot,
                    position=sim_config.robot_position,
                    asset_path=Path(robot_asset_path).expanduser().resolve() if robot_asset_path else None,
                ),
            ),
        )
        super().__init__(scene, data_path=data_path, config=sim_config)


def _wp_array_view(array):
    if array is None:
        return None
    return wp.to_torch(array)


def _snapshot_wp_array(array):
    view = _wp_array_view(array)
    return None if view is None else view.detach().clone()


def _restore_wp_array(array, saved, name):
    view = _wp_array_view(array)
    if view is None and saved is None:
        return
    if view is None or saved is None:
        raise ValueError(f"Simulator state entry {name!r} does not match the current model.")
    if tuple(saved.shape) != tuple(view.shape):
        raise ValueError(
            f"Simulator state entry {name!r} has shape {tuple(saved.shape)}, expected {tuple(view.shape)}."
        )
    view.copy_(saved.to(device=view.device, dtype=view.dtype))


def _xform(position, orientation=(0.0, 0.0, 0.0, 1.0)):
    return wp.transform(wp.vec3(*position), wp.quat(*orientation))


def _scale_cache_key(scale):
    if isinstance(scale, th.Tensor):
        values = scale.detach().cpu().flatten().tolist()
    elif isinstance(scale, (int, float)):
        values = [float(scale)] * 3
    else:
        values = list(scale)
    if len(values) == 1:
        values = values * 3
    return tuple(round(float(value), 8) for value in values[:3])


def _scale_imported_builder(builder, scale):
    scale = _scale_cache_key(scale)
    if scale == (1.0, 1.0, 1.0):
        return

    for body_idx, body_q in enumerate(builder.body_q):
        builder.body_q[body_idx] = _scaled_transform_position(body_q, scale)
    for body_idx, body_com in enumerate(builder.body_com):
        # Preserve authored MassAPI mass. Isaac/PhysX treats an authored mass as
        # overriding density, so object scale changes geometry but not total
        # mass; inertia is recomputed below from the scaled collision bounds.
        builder.body_com[body_idx] = _scaled_vec3(body_com, scale)

    for joint_idx, joint_xform in enumerate(builder.joint_X_p):
        builder.joint_X_p[joint_idx] = _scaled_transform_position(joint_xform, scale)
    for joint_idx, joint_xform in enumerate(builder.joint_X_c):
        builder.joint_X_c[joint_idx] = _scaled_transform_position(joint_xform, scale)

    for shape_idx, shape_xform in enumerate(builder.shape_transform):
        builder.shape_transform[shape_idx] = _scaled_transform_position(shape_xform, scale)
        shape_scale = builder.shape_scale[shape_idx]
        builder.shape_scale[shape_idx] = (
            float(shape_scale[0]) * scale[0],
            float(shape_scale[1]) * scale[1],
            float(shape_scale[2]) * scale[2],
        )

    for joint_idx, joint_type in enumerate(builder.joint_type):
        q_start = builder.joint_q_start[joint_idx]
        q_stop = (
            builder.joint_q_start[joint_idx + 1] if joint_idx + 1 < len(builder.joint_q_start) else len(builder.joint_q)
        )
        qd_start = builder.joint_qd_start[joint_idx]
        qd_stop = (
            builder.joint_qd_start[joint_idx + 1]
            if joint_idx + 1 < len(builder.joint_qd_start)
            else len(builder.joint_qd)
        )
        if _enum_value(joint_type) == newton.JointType.FREE.value and q_stop - q_start >= 3:
            for axis_idx in range(3):
                builder.joint_q[q_start + axis_idx] *= scale[axis_idx]
        elif _enum_value(joint_type) == newton.JointType.PRISMATIC.value and qd_stop > qd_start:
            axis = builder.joint_axis[qd_start]
            axis_scale = _axis_scale(axis, scale)
            if q_stop > q_start:
                builder.joint_q[q_start] *= axis_scale
            builder.joint_qd[qd_start] *= axis_scale
            builder.joint_target_pos[qd_start] *= axis_scale
            builder.joint_target_vel[qd_start] *= axis_scale
            builder.joint_limit_lower[qd_start] *= axis_scale
            builder.joint_limit_upper[qd_start] *= axis_scale

    _add_scaled_joint_stabilization(builder)
    _recompute_scaled_body_inertias(builder)
    _refresh_builder_mass_properties(builder)


def _scaled_transform_position(xform, scale):
    return wp.transform(_scaled_vec3(xform.p, scale), xform.q)


def _scaled_vec3(vec, scale):
    return wp.vec3(float(vec[0]) * scale[0], float(vec[1]) * scale[1], float(vec[2]) * scale[2])


def _recompute_scaled_body_inertias(builder):
    """Assign stable inertia tensors after non-uniform scene-object scale.

    Scaling an inertia matrix by a single scalar is not physically valid for
    non-uniform object scales. Use the scaled collision geometry bounds for each
    body and approximate the body as a solid box with the imported/scaled mass.
    This is conservative, deterministic, and avoids tiny invalid inertias that
    make MuJoCo explode when users apply forces.
    """
    body_bounds = {}
    shape_body = getattr(builder, "shape_body", ())
    for shape_idx, body_idx in enumerate(shape_body):
        body_idx = int(body_idx)
        if body_idx < 0 or body_idx >= len(builder.body_mass) or float(builder.body_mass[body_idx]) <= 0.0:
            continue
        bounds = _shape_body_bounds(builder, shape_idx)
        if bounds is None:
            continue
        lower, upper = bounds
        if body_idx not in body_bounds:
            body_bounds[body_idx] = [lower, upper]
        else:
            body_bounds[body_idx][0] = np.minimum(body_bounds[body_idx][0], lower)
            body_bounds[body_idx][1] = np.maximum(body_bounds[body_idx][1], upper)

    min_extent = 1.0e-3
    min_inertia = 1.0e-6
    for body_idx, (lower, upper) in body_bounds.items():
        mass = float(builder.body_mass[body_idx])
        extent = np.maximum(upper - lower, min_extent)
        ix = max(mass * (extent[1] * extent[1] + extent[2] * extent[2]) / 12.0, min_inertia)
        iy = max(mass * (extent[0] * extent[0] + extent[2] * extent[2]) / 12.0, min_inertia)
        iz = max(mass * (extent[0] * extent[0] + extent[1] * extent[1]) / 12.0, min_inertia)
        builder.body_inertia[body_idx] = wp.mat33(ix, 0.0, 0.0, 0.0, iy, 0.0, 0.0, 0.0, iz)


def _shape_body_bounds(builder, shape_idx):
    local_bounds = _shape_local_bounds(builder, shape_idx)
    if local_bounds is None:
        return None

    lower, upper = local_bounds
    corners = np.array(
        [[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])],
        dtype=np.float64,
    )

    if shape_idx < len(builder.shape_transform):
        xform = builder.shape_transform[shape_idx]
        rotation = _quat_to_matrix(xform.q)
        translation = _vec3_to_np(xform.p)
        corners = corners @ rotation.T + translation
    return corners.min(axis=0), corners.max(axis=0)


def _shape_local_bounds(builder, shape_idx):
    scale = np.maximum(np.abs(_vec3_to_np(builder.shape_scale[shape_idx])), 1.0e-6)
    source = builder.shape_source[shape_idx] if shape_idx < len(builder.shape_source) else None
    vertices = getattr(source, "vertices", None)
    if vertices is not None:
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim == 2 and vertices.shape[0] > 0 and vertices.shape[1] >= 3:
            points = vertices[:, :3] * scale
            return points.min(axis=0), points.max(axis=0)

    # Primitive importers encode their dimensions in shape_scale. Treat this as
    # a local half-extent fallback; overestimating inertia is preferable to the
    # near-zero inertias that destabilize awakened scaled objects.
    return -scale, scale


def _vec3_to_np(value):
    return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def _quat_to_matrix(quat):
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _refresh_builder_mass_properties(builder):
    """Keep Newton builder mass/inertia arrays internally consistent and valid.

    Newton's USD importer stores mass, inertia, inverse mass, and inverse inertia
    as separate builder lists. Any runtime scale or MuJoCo mass floor applied by
    OmniGibson must update the inverse arrays as well; leaving stale inverse
    inertia values can destabilize MuJoCo several steps after scene load. Run
    Newton's inertia validation on the builder too, because finalization repairs
    only the model arrays and otherwise leaves later builder merges/caches with
    the original invalid tensors.
    """
    for body_idx, mass in enumerate(builder.body_mass):
        mass = float(mass)
        if body_idx >= len(builder.body_inertia):
            builder.body_inv_mass[body_idx] = 1.0 / mass if mass > 0.0 else 0.0
            continue
        inertia = _symmetrized_mat33(builder.body_inertia[body_idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mass, inertia, _ = verify_and_correct_inertia(
                mass,
                inertia,
                builder.balance_inertia,
                builder.bound_mass,
                builder.bound_inertia,
            )
        builder.body_mass[body_idx] = mass
        builder.body_inv_mass[body_idx] = 1.0 / mass if mass > 0.0 else 0.0
        builder.body_inertia[body_idx] = inertia
        builder.body_inv_inertia[body_idx] = wp.inverse(inertia) if _mat33_has_values(inertia) else wp.mat33(0.0)


def _symmetrized_mat33(matrix):
    xy = 0.5 * (float(matrix[0, 1]) + float(matrix[1, 0]))
    xz = 0.5 * (float(matrix[0, 2]) + float(matrix[2, 0]))
    yz = 0.5 * (float(matrix[1, 2]) + float(matrix[2, 1]))
    return wp.mat33(
        float(matrix[0, 0]),
        xy,
        xz,
        xy,
        float(matrix[1, 1]),
        yz,
        xz,
        yz,
        float(matrix[2, 2]),
    )


def _mat33_has_values(matrix):
    return any(abs(float(matrix[row, col])) > 0.0 for row in range(3) for col in range(3))


def _axis_scale(axis, scale):
    x = float(axis[0]) * scale[0]
    y = float(axis[1]) * scale[1]
    z = float(axis[2]) * scale[2]
    return max(math.sqrt(x * x + y * y + z * z), 1.0e-6)


def _add_scaled_joint_stabilization(builder):
    # Newton/MuJoCo-Warp keeps imported free bodies fully active, so scaled
    # scene clutter needs small generalized armature/friction to keep dense
    # resting contacts from injecting unbounded velocity.
    armature_floor = 1.0e-2
    friction_floor = 2.0e-1
    fixed_joint_type = newton.JointType.FIXED.value
    for joint_idx, joint_type in enumerate(builder.joint_type):
        if _enum_value(joint_type) == fixed_joint_type:
            continue
        qd_start = builder.joint_qd_start[joint_idx]
        qd_stop = (
            builder.joint_qd_start[joint_idx + 1]
            if joint_idx + 1 < len(builder.joint_qd_start)
            else len(builder.joint_qd)
        )
        for dof_idx in range(qd_start, qd_stop):
            if dof_idx < len(builder.joint_armature):
                builder.joint_armature[dof_idx] = max(float(builder.joint_armature[dof_idx]), armature_floor)
            if dof_idx < len(builder.joint_friction):
                builder.joint_friction[dof_idx] = max(float(builder.joint_friction[dof_idx]), friction_floor)


def _add_passive_object_joint_damping(builder):
    """Add passive damping for non-free DatasetObject articulation joints.

    Isaac keeps settled scene articulations quiet through PhysX joint damping
    and friction. Newton/MuJoCo imports many BEHAVIOR object joints as
    zero-gain effort DOFs, so cabinet/door links can drift under contact and
    gravity even when the scene is supposed to start at rest. Use a velocity
    damper only for unactuated non-free object joints; free bases and authored
    drives are left unchanged so objects remain interactable.
    """
    passive_kd = 5.0
    velocity_mode = newton.JointTargetMode.VELOCITY.value
    unactuated_modes = {newton.JointTargetMode.NONE.value, newton.JointTargetMode.EFFORT.value}
    skipped_types = {newton.JointType.FIXED.value, newton.JointType.FREE.value}

    for joint_idx, joint_type in enumerate(builder.joint_type):
        if _enum_value(joint_type) in skipped_types:
            continue
        qd_start = builder.joint_qd_start[joint_idx]
        qd_stop = (
            builder.joint_qd_start[joint_idx + 1]
            if joint_idx + 1 < len(builder.joint_qd_start)
            else len(builder.joint_qd)
        )
        for dof_idx in range(qd_start, qd_stop):
            has_authored_drive = (
                float(builder.joint_target_ke[dof_idx]) > 0.0 or float(builder.joint_target_kd[dof_idx]) > 0.0
            )
            if has_authored_drive or _enum_value(builder.joint_target_mode[dof_idx]) not in unactuated_modes:
                continue
            builder.joint_target_mode[dof_idx] = velocity_mode
            builder.joint_target_vel[dof_idx] = 0.0
            builder.joint_target_kd[dof_idx] = max(float(builder.joint_target_kd[dof_idx]), passive_kd)


def _insert_fixed_base_anchor_body(builder, object_name):
    """Insert a world-fixed base body for fixed-base articulated objects.

    Isaac/PhysX represents a fixed-base object as a base link fixed to the
    scene, with child joints such as drawers and doors attached to that base.
    Some fixed-base BEHAVIOR imports leave movable root joints parented directly
    to world instead of preserving one fixed base and its child articulation.
    Insert one local zero-mass base before imported bodies, add a fixed joint
    from world to that base, then reparent only imported root movable joints to
    it. Child links stay dynamic and interactable. Do not use
    ``BodyFlags.KINEMATIC`` here: MuJoCo-Warp skips kinematic bodies while
    building child joint mappings. See workaround W7 in the migration record.
    """
    skipped_types = {newton.JointType.FIXED.value, newton.JointType.FREE.value}
    root_joint_indices = [
        joint_idx
        for joint_idx, joint_type in enumerate(builder.joint_type)
        if int(builder.joint_parent[joint_idx]) == -1 and _enum_value(joint_type) not in skipped_types
    ]
    if not root_joint_indices:
        return None

    builder.body_inertia.insert(0, wp.mat33(0.0))
    builder.body_mass.insert(0, 0.0)
    builder.body_inv_inertia.insert(0, wp.mat33(0.0))
    builder.body_inv_mass.insert(0, 0.0)
    builder.body_com.insert(0, wp.vec3())
    builder.body_lock_inertia.insert(0, True)
    builder.body_flags.insert(0, int(newton.BodyFlags.DYNAMIC))
    builder.body_q.insert(0, wp.transform())
    builder.body_qd.insert(0, wp.spatial_vector())
    builder.body_label.insert(0, f"{object_name}//fixed_base_anchor")
    builder.body_world.insert(0, builder.current_world)

    builder.shape_body = [body_idx + 1 if body_idx >= 0 else body_idx for body_idx in builder.shape_body]
    shifted_body_shapes = {-1: list(builder.body_shapes.get(-1, [])), 0: []}
    for body_idx, shape_indices in builder.body_shapes.items():
        if body_idx == -1:
            continue
        shifted_body_shapes[int(body_idx) + 1] = shape_indices
    builder.body_shapes = shifted_body_shapes

    for joint_idx in range(len(builder.joint_parent)):
        if int(builder.joint_parent[joint_idx]) >= 0:
            builder.joint_parent[joint_idx] += 1
        if int(builder.joint_child[joint_idx]) >= 0:
            builder.joint_child[joint_idx] += 1
    for joint_idx in root_joint_indices:
        builder.joint_parent[joint_idx] = 0

    _insert_fixed_base_joint(builder, object_name)
    _rebuild_builder_joint_maps(builder)
    return 0


def _insert_fixed_base_joint(builder, object_name):
    """Insert the world-to-base fixed joint before imported child joints."""
    builder.joint_type.insert(0, newton.JointType.FIXED)
    builder.joint_parent.insert(0, -1)
    builder.joint_child.insert(0, 0)
    builder.joint_X_p.insert(0, wp.transform())
    builder.joint_X_c.insert(0, wp.transform())
    builder.joint_label.insert(0, f"{object_name}//fixed_base_joint")
    builder.joint_dof_dim.insert(0, (0, 0))
    builder.joint_enabled.insert(0, True)
    builder.joint_collision_filter_parent.insert(0, True)
    builder.joint_world.insert(0, builder.current_world)
    builder.joint_articulation.insert(0, 0)
    builder.joint_q_start.insert(0, 0)
    builder.joint_qd_start.insert(0, 0)
    builder.joint_cts_start.insert(0, 0)
    # The inserted fixed base and all repaired root joints now form a single
    # articulation tree for this fixed-base object.
    builder.articulation_start = [0] if builder.joint_count else []


def _rebuild_builder_joint_maps(builder):
    builder.joint_parents.clear()
    builder.joint_children.clear()
    for joint_idx, (parent_idx, child_idx) in enumerate(zip(builder.joint_parent, builder.joint_child, strict=True)):
        builder.joint_parents.setdefault(child_idx, []).append((parent_idx, joint_idx))
        builder.joint_children.setdefault(parent_idx, []).append((child_idx, joint_idx))


def _disable_imported_self_collisions(builder, import_info):
    """Filter collision pairs between shapes from the same USD import.

    Newton's USD importer honors ``enable_self_collisions=False`` while parsing
    a single builder, but fixed-base shapes may be represented as world-body
    shapes after import/merge. MuJoCo then sees contacts such as a cabinet door
    colliding with its own frame. Preserve OmniGibson's default self-collision
    behavior by explicitly filtering intra-import shape pairs.
    """
    shape_indices = sorted(set((import_info.get("path_shape_map") or {}).values()))
    if len(shape_indices) < 2:
        return

    collision_mask = ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES
    collidable_shape_indices = [
        shape_idx for shape_idx in shape_indices if builder.shape_flags[shape_idx] & collision_mask
    ]
    for i, shape_a in enumerate(collidable_shape_indices):
        body_a = builder.shape_body[shape_a]
        for shape_b in collidable_shape_indices[i + 1 :]:
            if body_a == builder.shape_body[shape_b]:
                continue
            builder.add_shape_collision_filter_pair(shape_a, shape_b)


def _disable_imported_collision_shapes(builder, import_info):
    shape_indices = set((import_info.get("path_shape_map") or {}).values())
    collision_mask = ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES
    for shape_idx in shape_indices:
        builder.shape_flags[shape_idx] &= ~collision_mask


def _offset_import_info(import_info, body_offset, shape_offset):
    def offset_body(body_idx):
        return body_idx if body_idx == -1 else body_idx + body_offset

    return {
        **import_info,
        "path_body_map": {
            path: offset_body(body_idx) for path, body_idx in (import_info.get("path_body_map") or {}).items()
        },
        "path_shape_map": {
            path: shape_idx + shape_offset for path, shape_idx in (import_info.get("path_shape_map") or {}).items()
        },
    }


def _hide_model_shapes(model, shape_indices):
    if not shape_indices:
        return
    shape_flags = model.shape_flags.numpy()
    for shape_idx in shape_indices:
        shape_flags[shape_idx] &= ~ShapeFlags.VISIBLE
    model.shape_flags = wp.array(shape_flags, dtype=wp.int32, device=model.device)


def _hide_imported_metalink_shapes(builder, import_info):
    collision_mask = ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES
    for path, shape_idx in (import_info.get("path_shape_map") or {}).items():
        path = str(path).lower()
        if "meta" not in path:
            continue
        if not any(metalink_type in path for metalink_type in HIDDEN_METALINK_TYPES):
            continue
        builder.shape_flags[shape_idx] &= ~(collision_mask | ShapeFlags.VISIBLE)


def _ensure_mujoco_moving_body_mass_properties(builder):
    # MuJoCo rejects moving bodies whose mass or inertia is at / below mjMINVAL.
    # Newton's USD importer represents floating-base chains as intermediate
    # prismatic / revolute bodies, and some robot assets such as R1Pro author
    # those adapter bodies with zero mass because PhysX tolerated them. Give
    # only non-fixed joint children a tiny physically valid floor before model
    # finalization; fixed marker links can stay massless.
    moving_body_indices = set()
    fixed_joint_type = newton.JointType.FIXED.value
    for joint_idx, joint_type in enumerate(getattr(builder, "joint_type", ())):
        if _enum_value(joint_type) == fixed_joint_type:
            continue
        child_idx = int(builder.joint_child[joint_idx])
        if child_idx >= 0:
            moving_body_indices.add(child_idx)

    min_mass = 1.0e-3
    min_inertia = 1.0e-6
    inertia_floor = wp.mat33(min_inertia, 0.0, 0.0, 0.0, min_inertia, 0.0, 0.0, 0.0, min_inertia)
    for body_idx in sorted(moving_body_indices):
        if body_idx >= len(builder.body_mass):
            continue
        if float(builder.body_mass[body_idx]) <= min_mass:
            builder.body_mass[body_idx] = min_mass
        if body_idx < len(builder.body_inertia) and _diagonal_min(builder.body_inertia[body_idx]) <= min_inertia:
            builder.body_inertia[body_idx] = inertia_floor


def _enum_value(value):
    return value.value if hasattr(value, "value") else int(value)


def _diagonal_min(matrix):
    try:
        return min(float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2]))
    except Exception:
        return min(float(matrix[0][0]), float(matrix[1][1]), float(matrix[2][2]))


class _NewtonViewerCamera:
    def __init__(self, simulator):
        self._simulator = simulator
        self._pending_camera = None

    def set_position_orientation(self, position, orientation):
        pos = _to_float_tuple(position, 3)
        quat = _to_float_tuple(orientation, 4)
        pitch, yaw = _quat_xyzw_to_newton_pitch_yaw(quat)
        self._pending_camera = (pos, pitch, yaw)
        if self._simulator.viewer is not None:
            self.apply_pending()

    def apply_pending(self):
        if self._pending_camera is None or self._simulator.viewer is None:
            return
        pos, pitch, yaw = self._pending_camera
        viewer = self._simulator.viewer
        viewer.set_camera(pos=wp.vec3(*pos), pitch=pitch, yaw=yaw)
        camera = getattr(viewer, "camera", None)
        if camera is not None and hasattr(camera, "look_at"):
            camera.look_at(_scene_camera_target(self._simulator))

    def _ensure_viewer(self):
        if self._simulator.viewer is None:
            self._simulator.attach_viewer()
        return self._simulator.viewer


def _to_float_tuple(value, length):
    if isinstance(value, th.Tensor):
        value = value.detach().cpu().tolist()
    result = tuple(float(v) for v in value)
    if len(result) != length:
        raise ValueError(f"Expected {length} values, got {result}.")
    return result


def _quat_xyzw_to_newton_pitch_yaw(quat):
    x, y, z, w = quat
    # USD/Isaac cameras look along local -Z. Convert that world direction into
    # Newton ViewerGL's Z-up pitch/yaw parameterization.
    forward = (
        -2.0 * (x * z + w * y),
        -2.0 * (y * z - w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    )
    fx, fy, fz = forward
    norm = math.sqrt(fx * fx + fy * fy + fz * fz)
    if norm <= 0:
        return 0.0, -180.0
    fx, fy, fz = fx / norm, fy / norm, fz / norm
    pitch = math.degrees(math.asin(max(min(fz, 1.0), -1.0)))
    yaw = math.degrees(math.atan2(fy, fx))
    return pitch, yaw


def _scene_camera_target(simulator):
    entities = simulator.robots or simulator.objects
    if not entities:
        return (0.0, 0.0, 0.0)
    centers = [entity.aabb_center for entity in entities]
    center = th.stack(centers).mean(dim=0)
    return _to_float_tuple(center, 3)


def _apply_authored_robot_mass_properties(builder, usd_path, import_info):
    """Align robot body inertia and center of mass with the authored USD mass.

    Newton's importer keeps the authored ``UsdPhysics.MassAPI`` mass when
    inertia is not authored, but computes the inertia tensor from collision
    geometry at the default density. The later mass/inertia consistency repair
    then raises the mass to match the oversized inertia — Fetch inflated to
    3.5x its authored weight and saturated its actuators. Rebuild each robot
    body's inertia as a solid box at the authored mass (the same conservative
    approximation used for scaled scene objects) so mass and inertia stay
    consistent through validation. See workaround W13 in
    docs/other/newton_migration.md.
    """
    from pxr import Usd, UsdPhysics

    body_map = import_info.get("path_body_map") or {}
    if not body_map:
        return
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return

    body_indices = {int(idx) for idx in body_map.values()}
    body_bounds = {}
    for shape_idx, body_idx in enumerate(getattr(builder, "shape_body", ())):
        body_idx = int(body_idx)
        if body_idx not in body_indices:
            continue
        bounds = _shape_body_bounds(builder, shape_idx)
        if bounds is None:
            continue
        lower, upper = bounds
        if body_idx not in body_bounds:
            body_bounds[body_idx] = [lower, upper]
        else:
            body_bounds[body_idx][0] = np.minimum(body_bounds[body_idx][0], lower)
            body_bounds[body_idx][1] = np.maximum(body_bounds[body_idx][1], upper)

    min_extent = 1.0e-3
    min_inertia = 1.0e-6
    for path, body_idx in body_map.items():
        body_idx = int(body_idx)
        if body_idx < 0 or body_idx >= len(builder.body_mass):
            continue
        prim = stage.GetPrimAtPath(str(path))
        if not prim or not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        if not mass_api.GetMassAttr().HasAuthoredValue():
            continue
        mass = float(mass_api.GetMassAttr().Get())
        if mass <= 0.0:
            continue

        builder.body_mass[body_idx] = mass
        builder.body_inv_mass[body_idx] = 1.0 / mass

        if mass_api.GetDiagonalInertiaAttr().HasAuthoredValue():
            di = mass_api.GetDiagonalInertiaAttr().Get()
            inertia = wp.mat33(float(di[0]), 0.0, 0.0, 0.0, float(di[1]), 0.0, 0.0, 0.0, float(di[2]))
        elif body_idx in body_bounds:
            lower, upper = body_bounds[body_idx]
            extent = np.maximum(upper - lower, min_extent)
            ix = max(mass * (extent[1] * extent[1] + extent[2] * extent[2]) / 12.0, min_inertia)
            iy = max(mass * (extent[0] * extent[0] + extent[2] * extent[2]) / 12.0, min_inertia)
            iz = max(mass * (extent[0] * extent[0] + extent[1] * extent[1]) / 12.0, min_inertia)
            inertia = wp.mat33(ix, 0.0, 0.0, 0.0, iy, 0.0, 0.0, 0.0, iz)
        else:
            inertia = _symmetrized_mat33(builder.body_inertia[body_idx])
        builder.body_inertia[body_idx] = inertia
        builder.body_inv_inertia[body_idx] = wp.inverse(inertia) if _mat33_has_values(inertia) else wp.mat33(0.0)

        if mass_api.GetCenterOfMassAttr().HasAuthoredValue():
            com = mass_api.GetCenterOfMassAttr().Get()
            builder.body_com[body_idx] = wp.vec3(float(com[0]), float(com[1]), float(com[2]))


def _apply_chassis_caster_friction(builder, import_info):
    """Give wheeled-robot chassis shapes near-zero contact friction.

    Robot USDs bake caster pads into the base link's collision mesh instead of
    modeling caster links. Those pads rest on the ground with default friction
    and pin the base in place, so drive-wheel commands cannot move the robot.
    Real casters roll freely; approximate them by dropping friction on the
    root-body shapes of robots that have wheel joints, while the wheels keep
    their default traction. See workaround W14 in
    docs/other/newton_migration.md.
    """
    joint_map = import_info.get("path_joint_map") or {}
    wheel_joints = [int(idx) for path, idx in joint_map.items() if "wheel" in str(path).lower()]
    if not wheel_joints:
        return set()

    # The chassis carrying the caster pads is the parent body of the wheels.
    joint_parent = getattr(builder, "joint_parent", ())
    chassis_bodies = {int(joint_parent[j]) for j in wheel_joints if 0 <= j < len(joint_parent)}
    chassis_bodies.discard(-1)
    if not chassis_bodies:
        return set()

    for shape_idx, body_idx in enumerate(getattr(builder, "shape_body", ())):
        if int(body_idx) in chassis_bodies and shape_idx < len(builder.shape_material_mu):
            builder.shape_material_mu[shape_idx] = 0.02
    return chassis_bodies


def _elevate_mjc_geom_priority(solver, model, body_indices):
    """Make low-friction chassis geoms win MuJoCo's contact-parameter mix.

    MuJoCo combines contact friction from the two geoms' values (the larger
    wins at equal priority), so a 0.02 chassis pad against a 0.9 ground plane
    still produces a 0.9 contact. Raising the chassis geom priority makes its
    friction authoritative; MuJoCo-Warp honors ``geom_priority`` in its
    contact kernels.
    """
    import mujoco

    mj_model = solver.mj_model
    if mj_model is None:
        return
    labels = [str(label) for label in model.body_label]
    names = {labels[int(i)].replace("/", "_") for i in body_indices if 0 <= int(i) < len(labels)}
    changed = False
    for geom_idx in range(mj_model.ngeom):
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, int(mj_model.geom_bodyid[geom_idx])) or ""
        if body_name in names:
            mj_model.geom_priority[geom_idx] = 1
            changed = True
    if changed and getattr(solver, "mjw_model", None) is not None:
        solver.mjw_model.geom_priority.assign(mj_model.geom_priority)


def _import_label_prefix(import_info, fallback_name):
    for key in ("path_body_map", "path_shape_map", "path_joint_map"):
        mapping = import_info.get(key) or {}
        if mapping:
            path = next(iter(mapping))
            parts = [part for part in path.split("/") if part]
            if parts:
                return f"/{parts[0]}"
    return f"/{fallback_name}"


Simulator = NewtonSceneSimulator


def launch_app(*args, **kwargs):
    """Newton does not require an Isaac/Kit app; keep the launch phase for API parity."""
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise ValueError(f"Unsupported Newton app launch arguments: {unsupported}")
    return None


def _launch_simulator(
    config=None,
    *,
    data_path=None,
    dataset_usd_path=None,
    robot_asset_path=None,
    build=True,
    **kwargs,
):
    """Launch the native Newton simulator and assign it to ``omnigibson.sim``.

    Args:
        config (None or str or dict): Newton config. Defaults to an empty scene.
        data_path (None or str): Optional OmniGibson datasets directory.
        dataset_usd_path (None or str): Optional prepared DatasetObject USD.
        robot_asset_path (None or str): Optional robot USD/URDF path.
        build (bool): Whether to build the Newton model immediately.
        **kwargs: Reserved for future compatibility with the PhysX launcher.
    """
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise ValueError(f"Unsupported Newton simulator launch arguments: {unsupported}")

    import omnigibson as og
    from omnigibson.newton.config import simulator_from_config

    if og.sim is not None:
        return og.sim

    sim = simulator_from_config(
        config or {"scene": {"type": "Scene"}},
        data_path=data_path,
        dataset_usd_path=dataset_usd_path,
        robot_asset_path=robot_asset_path,
    )
    if build:
        sim.build()
    og.sim = sim
    print_welcome(sim)
    return sim


_launch_newton_simulator = _launch_simulator


def logo_small():
    return "OmniGibson Newton"


def print_welcome(sim=None):
    if sim is not None and getattr(sim, "_welcome_printed", False):
        return

    import omnigibson as og

    _print_logo()
    og.log.info(f"{'-' * 10} Welcome to {logo_small()}! {'-' * 10}")
    if sim is not None:
        sim._welcome_printed = True


def _print_logo():
    print()
    _print_icon()
    _print_wordmark()
    print()


def _print_icon():
    from termcolor import colored

    raw_texts = [
        ("                   ___________", "", "", "", "", "", "_"),
        ("                  /          ", "", "", "", "", "", "/ \\"),
        ("                 /          ", "", "", "", "/ /", "__", ""),
        ("                /          ", "", "", "", "", "", "/ /  /\\"),
        ("               /", "__________", "", "", "/ /", "__", "/  \\"),
        ("               ", "\\   _____  ", "", "", "\\ \\", "__", "\\  /"),
        ("                ", "\\  \\  ", "/ ", "\\  ", "", "", "\\ \\_/ /"),
        ("                 ", "\\  \\", "/", "___\\  ", "", "", "\\   /"),
        ("                  ", "\\__________", "", "", "", "", "\\_/  "),
    ]
    for lgrey0, grey0, lgrey1, grey1, red0, lgrey2, red1 in raw_texts:
        print(
            colored(lgrey0, "light_grey", attrs=["bold"])
            + colored(grey0, "light_grey", attrs=["bold", "dark"])
            + colored(lgrey1, "light_grey", attrs=["bold"])
            + colored(grey1, "light_grey", attrs=["bold", "dark"])
            + colored(red0, "light_red", attrs=["bold"])
            + colored(lgrey2, "light_grey", attrs=["bold"])
            + colored(red1, "light_red", attrs=["bold"])
        )


def _print_wordmark():
    from termcolor import colored

    raw_texts = [
        ("       ___                  _", "  ____ _ _                     "),
        (r"      / _ \ _ __ ___  _ __ (_)", r"/ ___(_) |__  ___  ___  _ __  "),
        (r"     | | | | '_ ` _ \| '_ \| |", r" |  _| | '_ \/ __|/ _ \| '_ \ "),
        (r"     | |_| | | | | | | | | | |", r" |_| | | |_) \__ \ (_) | | | |"),
        (r"      \___/|_| |_| |_|_| |_|_|", r"\____|_|_.__/|___/\___/|_| |_|"),
    ]
    for grey_text, red_text in raw_texts:
        print(colored(grey_text, "light_grey", attrs=["bold", "dark"]) + colored(red_text, "light_red", attrs=["bold"]))


def _merge_dicts(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dicts(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _is_empty_action(action):
    if hasattr(action, "numel"):
        return action.numel() == 0
    if hasattr(action, "__len__"):
        return len(action) == 0
    return False


__all__ = [
    "NewtonObjectRobotSimulator",
    "NewtonSceneSimulator",
    "NewtonSimulationConfig",
    "Simulator",
    "launch_app",
    "print_welcome",
    "_launch_newton_simulator",
    "_launch_simulator",
    "logo_small",
]
