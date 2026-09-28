"""
Utilities for recording an OmniGibson environment through one of Newton's own standalone viewers
(``ViewerRTX`` or ``ViewerGL``) instead of Omniverse Kit's renderer.

This path only makes sense when ``gm.PHYSICS_BACKEND == "newton"`` and ``gm.RENDER_BACKEND ==
"none"`` -- running Newton physics in the same process as a live Kit render (``RENDER_BACKEND ==
"kit"``) hits an upstream OpenUSD 25.11 heap-corruption bug in Isaac Sim's own bundled USD library
(see ``docs/other/newton_migration.md``'s known-workarounds section). Callers must set those two
``gm`` flags -- and, for the GL viewer, ``pyglet.options["headless"] = True`` -- *before* importing
``omnigibson`` at all.

Under Kit rendering, OmniGibson's Newton physics model carries only collision geometry (rendering
reads the original USD's separate visual mesh subtree directly, via Kit). With ``RENDER_BACKEND ==
"none"``, ``physics_backends.newton_backend.NewtonBackend`` bakes real, textured visual-only shapes
into the model itself instead (see ``physics_backends/newton_visuals.py``), so ``viewer.log_state()``
below renders them automatically -- this module only needs to build the viewer, place the camera, and
step/capture frames.
"""

import time

import numpy as np
import torch as th

import omnigibson as og
from omnigibson.controllers import ControllerView
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


def reposition_robot_into_furnished_room(env, robot, room_type="living_room"):
    """Teleport the robot into a random point of the given room type, if the scene has room
    segmentation and a room of that type exists; otherwise a no-op.

    Even a smart, furniture-clearance-aware camera (:func:`_pick_camera_pose`) can only frame
    whatever is actually near wherever the robot happens to be. A scene's default spawn point is
    frequently a bare entryway/hallway near the origin, not a furnished room (confirmed empirically:
    Rs_int's default Fetch spawn is ~(0, 0, 0), an empty area with no furniture within camera range
    at all) -- so a camera built only to avoid clipping through geometry, with nothing furnished
    nearby to begin with, just frames more bare wall. Moving the robot into an actual furnished room
    first is what makes the "closest safe point to the robot" camera strategy in
    :func:`_pick_camera_pose` land somewhere worth looking at.
    """
    seg_map = getattr(env.scene, "seg_map", None)
    if seg_map is None:
        return
    try:
        room_names = sorted(n for n in seg_map.room_ins_name_to_ins_id if n.startswith(room_type))
        if not room_names:
            return
        target_ins_id = seg_map.room_ins_name_to_ins_id[room_names[0]]
        ins_map_t = th.as_tensor(np.asarray(seg_map.room_ins_map))
        ys, xs = th.where(ins_map_t == target_ins_id)
        if ys.numel() == 0:
            return
        idx = int(th.randint(0, ys.numel(), (1,)).item())
        point_map = th.tensor([float(ys[idx].item()), float(xs[idx].item())])
        x, y = seg_map.map_to_world(point_map)
        _, cur_quat = robot.get_position_orientation()
        robot.set_position_orientation(position=th.tensor([x.item(), y.item(), 0.1]), orientation=cur_quat)
        for _ in range(5):
            og.sim.step()
        log.info(f"Repositioned robot into room_instance={room_names[0]}")
    except Exception as e:
        log.warning(f"Could not reposition robot into a furnished room, leaving it at its spawn point: {e}")


def _pick_camera_pose(env, robot):
    """Pick a camera position/look-at for recording. If the scene has room segmentation
    (InteractiveTraversableScene), anchor the camera at a point in the robot's current room that (a)
    keeps a safety margin from every wall and every object's footprint, so it doesn't clip through
    geometry (confirmed empirically: an earlier wall-only-clearance version anchored 0.28m from a
    kitchen cabinet, reading as a flat, unlit black mass filling the frame -- the camera was
    essentially inside it), and (b) among the safe candidates, is the one CLOSEST to the robot --
    not the one farthest from everything. Maximizing clearance instead systematically picks the
    emptiest, most boring spot in the room (the middle of a bare wall), which is exactly the wrong
    bias: a real room's *interesting*, furniture-filled area is near where an occupant (the robot)
    actually is, not in the open middle. Otherwise falls back to a fixed offset behind and above the
    robot. Does not move the robot itself."""
    robot_pos, _ = robot.get_position_orientation()
    rx, ry, rz = robot_pos.tolist()
    seg_map = getattr(env.scene, "seg_map", None)

    if seg_map is not None:
        try:
            xy_map = seg_map.world_to_map(robot_pos[:2])
            ins_id = int(seg_map.room_ins_map[int(xy_map[0]), int(xy_map[1])].item())
            if ins_id != 0:
                from scipy.ndimage import distance_transform_edt

                ins_mask_np = np.asarray(seg_map.room_ins_map) == ins_id
                ys_room, xs_room = np.where(ins_mask_np)
                room_pixels_map = th.stack(
                    [th.as_tensor(ys_room, dtype=th.float32), th.as_tensor(xs_room, dtype=th.float32)], dim=1
                )
                room_pixels_world = seg_map.map_to_world(room_pixels_map)

                obstacle_centers = []
                obstacle_radii = []
                for obj in env.scene.objects:
                    if getattr(obj, "category", "") in ("walls", "floors", "ceilings"):
                        continue
                    try:
                        lo, hi = obj.aabb
                    except Exception:
                        continue
                    lo, hi = lo.cpu(), hi.cpu()
                    obstacle_centers.append((lo[:2] + hi[:2]) / 2)
                    obstacle_radii.append(float(((hi[:2] - lo[:2]) / 2).norm()))

                margin = 0.3  # meters of extra clearance to keep from every object's footprint
                if obstacle_centers:
                    d = th.cdist(room_pixels_world, th.stack(obstacle_centers))
                    furniture_clear = ~(d < (th.tensor(obstacle_radii) + margin)).any(dim=1).numpy()
                else:
                    furniture_clear = np.ones(len(ys_room), dtype=bool)

                wall_dist = distance_transform_edt(ins_mask_np)[ys_room, xs_room]
                min_wall_clearance = 0.3  # meters -- just enough to not clip into a wall
                safe = furniture_clear & (wall_dist * seg_map.map_resolution > min_wall_clearance)

                robot_dist_sq = (room_pixels_world[:, 0] - rx) ** 2 + (room_pixels_world[:, 1] - ry) ** 2
                min_robot_standoff_sq = 1.2**2  # meters -- enough standoff to actually frame the robot
                far_enough = robot_dist_sq.numpy() > min_robot_standoff_sq

                candidates = np.where(safe & far_enough)[0]
                if len(candidates) == 0:
                    candidates = np.where(safe)[0]
                if len(candidates) == 0:
                    candidates = np.where(furniture_clear)[0]
                if len(candidates) == 0:
                    candidates = np.arange(len(ys_room))

                best = candidates[np.argmin(robot_dist_sq.numpy()[candidates])]
                pm = th.tensor([float(ys_room[best]), float(xs_room[best])])
                wx, wy = seg_map.map_to_world(pm)
                return (wx.item(), wy.item(), rz + 0.6), (rx, ry, rz + 0.4)
        except Exception as e:
            log.warning(f"Room-aware camera placement failed, falling back to a fixed offset: {e}")

    return (rx - 2.5, ry - 2.5, rz + 2.0), (rx, ry, rz + 0.4)


# UsdLux light prim types worth copying into the RTX viewer's stage (see build_viewer). Every one of
# these is a real emitter OVRTX renders; MeshLight/GeometryLight variants don't appear in these assets.
_USD_LIGHT_TYPES = ("SphereLight", "RectLight", "DiskLight", "DistantLight", "DomeLight", "CylinderLight")


def _copy_scene_lights_into_rtx_viewer(viewer):
    """Copy every visible UsdLux light authored on the live OmniGibson stage into ``viewer``'s own USD
    stage (under ``/root/_OGSceneLights/``), baking each light's full world transform onto the copy.

    Rationale (see the call site in :func:`build_viewer`): the Newton model itself carries only
    geometry -- the scene's interior lights (ceiling DiskLights etc.) live purely as USD prims on
    OmniGibson's stage, so without this the RTX viewer's enclosed-interior renders are near-black.

    ``/root`` in the viewer stage carries only a uniform ``scaling`` op (1.0 for every caller here),
    so world-space transforms parented under it land where they should. Lights are copied with their
    authored attributes (``inputs:*`` namespace: intensity, color, radius, texture file, ...)
    verbatim.

    Static-only by design: light poses are snapshotted at viewer-build time and never updated per
    frame -- fine for scene lighting (ceiling fixtures don't move), and updating them later isn't
    safe anyway (OVRTX crashes on post-init light edits, see build_viewer).
    """
    from pxr import Gf, Usd, UsdGeom

    if og.sim is None:
        return 0

    src_stage = og.sim.stage
    copied = 0
    for prim in src_stage.Traverse():
        if prim.GetTypeName() not in _USD_LIGHT_TYPES:
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue

        dst = viewer.stage.DefinePrim(f"/root/_OGSceneLights/light_{copied}", prim.GetTypeName())
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if not name.startswith("inputs:"):
                continue
            value = attr.Get()
            if value is None:
                continue
            dst.CreateAttribute(name, attr.GetTypeName()).Set(value)

        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        dst_xf = UsdGeom.Xformable(dst)
        dst_xf.ClearXformOpOrder()
        dst_xf.AddTransformOp().Set(Gf.Matrix4d(world))
        copied += 1

    log.info(f"Copied {copied} scene light(s) into the RTX viewer stage")
    return copied


def build_viewer(renderer, model, cam_pos, look_at, width=640, height=480, headless=True):
    """Build and camera-pose a Newton viewer. ``renderer`` is ``"rtx"`` or ``"gl"``.

    ``headless=True`` (the default) creates no visible window -- frames are still readable via
    :func:`capture_frame`, but nothing is shown on screen; this is the only mode that works without
    a real display (it uses pyglet's EGL surfaceless backend for GL). ``headless=False`` opens an
    actual on-screen window and requires a real ``DISPLAY`` (a local X server or X11-forwarded SSH
    session) -- for ``"gl"`` specifically, the caller must NOT have set
    ``pyglet.options["headless"] = True`` (that forces pyglet's EGL backend globally, which
    pre-empts a real window regardless of this ``headless`` argument); leave that option unset and
    let pyglet use its normal X11 backend against the caller's own ``DISPLAY``.
    """
    import newton
    from pyglet.math import Vec3 as PyVec3

    if renderer == "gl":
        viewer = newton.viewer.ViewerGL(width=width, height=height, headless=headless)
    elif renderer == "rtx":
        # Built the same minimal way as ViewerGL above (no `environment="studio"`, no
        # `num_frames`): the "studio" light rig's extra light prims are what
        # boost_rtx_interior_lighting() used to mutate post-construction, which reliably crashed
        # the OVRTX renderer process (see git history) -- staying on the "default" environment
        # avoids ever touching those prims in the first place.
        viewer = newton.viewer.ViewerRTX(width=width, height=height, headless=headless)
        # ViewerRTX is a real ray tracer: its own default lighting is only a dome (sky) + distant
        # (sun) rig, which an enclosed interior scene (walls + ceilings, e.g. any
        # InteractiveTraversableScene) physically blocks -- confirmed empirically that a full Rs_int
        # render comes out near-black (mean ~1.5/255) regardless of how much those two lights are
        # boosted, while an open-air scene renders fine. (ViewerGL is unaffected: a plain rasterizer
        # with no shadowing, so occlusion never darkens anything.) The scene's own interior lights
        # (e.g. Rs_int's 8 ceiling DiskLights) exist as UsdLux prims on the OmniGibson stage but are
        # never part of the Newton model -- copy them into the viewer's stage so the interior is lit
        # the same way Kit lights it. Must happen HERE, before the first end_frame(): OVRTX
        # serializes the stage once at _init_ovrtx(), and light-prim edits after that are the exact
        # post-construction mutations that crashed it (see comment above).
        _copy_scene_lights_into_rtx_viewer(viewer)
    else:
        raise ValueError(f"Unknown renderer {renderer!r}, expected 'rtx' or 'gl'.")

    viewer.set_model(model)
    viewer.camera.pos = PyVec3(*cam_pos)
    viewer.camera.look_at(look_at)
    viewer._camera_dirty = True
    viewer.show_collision = False  # collision shapes are non-visual; the model's baked-in visual
    # shapes (physics_backends/newton_visuals.py) are what should render.
    viewer.show_visual = True
    return viewer


def capture_frame(viewer, renderer):
    """Grab the last-rendered frame as an (H, W, 3) uint8 numpy array."""
    if renderer == "gl":
        return viewer.get_frame().numpy()
    return np.array(viewer._capture_screenshot_pixels())[..., :3]


def record_with_newton_viewer(env, robot, renderer, output_path=None, num_steps=100, fps=30):
    """Step the environment with a random action (resampled every 30 steps, matching
    ``robot_control_example.py``'s ``control_mode="random"`` behavior), driving it through the given
    Newton viewer.

    If ``output_path`` is given, runs headless and writes the recorded frames there as an mp4 (the
    only mode that works without a real display). If ``output_path`` is ``None``, opens a live,
    on-screen window instead (paced to roughly ``fps``) and does not save anything -- this requires
    a real ``DISPLAY`` (a local X server or X11-forwarded SSH session); for ``renderer="gl"`` the
    caller must not have set ``pyglet.options["headless"] = True`` in this case (see
    :func:`build_viewer`).

    Requires ``gm.PHYSICS_BACKEND == "newton"`` and ``gm.RENDER_BACKEND == "none"`` to already be in
    effect (set before ``omnigibson`` was imported).
    """
    headless = output_path is not None

    reposition_robot_into_furnished_room(env, robot)

    backend = og.sim.physics_backend
    model = backend._model

    cam_pos, look_at = _pick_camera_pose(env, robot)
    viewer = build_viewer(renderer, model, cam_pos, look_at, headless=headless)

    action_dim = robot.action_dim
    frames = [] if headless else None
    random_action = None
    frame_dt = 1.0 / fps
    t0 = time.time()

    for i in range(num_steps):
        step_t0 = time.time()
        if i % 30 == 0:
            random_action = (th.rand(action_dim, device=og.sim.device) * 2 - 1) * 0.05
        env.step(action=random_action)

        viewer.begin_frame(i / fps)
        viewer.log_state(backend._state_0)
        viewer.end_frame()

        if headless:
            frames.append(capture_frame(viewer, renderer))
        else:
            # Pace to roughly real time so a live window is actually watchable rather than
            # blitzing through frames as fast as the GPU allows.
            elapsed = time.time() - step_t0
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
        if i % 15 == 0:
            log.info(f"step {i} ({time.time() - t0:.1f}s elapsed)")

    viewer.close()
    if headless:
        import imageio

        log.info(f"Captured {len(frames)} frames, writing to {output_path}")
        imageio.mimsave(output_path, frames, fps=fps, quality=8)


class _PygletTeleopController:
    """Keyboard teleop for a robot, driven directly by a Newton viewer's own pyglet window.

    This is a carb-independent reimplementation of
    :class:`~omnigibson.utils.ui_utils.KeyboardRobotController`'s control scheme (arrow-key/letter-key
    axis control per arm, IJKL differential-drive, bracket-key joint stepping, T gripper toggle) --
    ``lazy.carb`` (Kit's input module) is not importable at all under ``RENDER_BACKEND="none"``
    (confirmed empirically: ``omnigibson.lazy`` has no ``carb`` attribute in that configuration, unlike
    the ``"kit"`` renderer where a live Kit app provides it), so ``KeyboardRobotController`` itself
    can't be reused here -- its ``populate_keypress_mapping()`` builds ``self.keypress_mapping`` keyed
    by ``lazy.carb.input.KeyboardInput`` members unconditionally, not just in its (skippable)
    Kit-registration step. Same key bindings, just keyed by ``pyglet.window.key`` symbols instead.
    """

    def __init__(self, robot):
        import pyglet

        self._key = pyglet.window.key
        key = self._key

        self.robot = robot
        self.action_dim = robot.action_dim
        self.controller_info = dict()
        self.joint_idx_to_group_key = dict()
        idx = 0
        for name, (group_key, _) in robot.controllers.items():
            self.controller_info[name] = {
                "name": ControllerView.get_controller_type_str(group_key),
                "start_idx": idx,
                "dofs": ControllerView.get_dof_idx(group_key),
                "command_dim": ControllerView.get_command_dim(group_key),
            }
            idx += ControllerView.get_command_dim(group_key)
            for i in ControllerView.get_dof_idx(group_key).tolist():
                self.joint_idx_to_group_key[i] = group_key

        self.joint_names = list(robot.joints.keys())
        self.joint_types = [joint.joint_type for joint in robot.joints.values()]
        self.joint_command_idx = []
        self.joint_control_idx = []
        self.active_joint_command_idx_idx = 0
        self.ik_arms = []
        self.active_arm_idx = 0
        self.binary_grippers = []
        self.active_gripper_idx = 0
        self.gripper_direction = {}
        self.persistent_gripper_action = {}
        self.keypress_mapping = {}
        self.current_keypress = None
        self.active_action = None
        self.toggling_gripper = False

        self.keypress_mapping[key.BRACKETRIGHT] = {"idx": None, "val": 0.1}
        self.keypress_mapping[key.BRACKETLEFT] = {"idx": None, "val": -0.1}

        for component, info in self.controller_info.items():
            if info["name"] in ("JointController", "HolonomicBaseJointController"):
                for i in range(info["command_dim"]):
                    self.joint_command_idx.append(info["start_idx"] + i)
                self.joint_control_idx += info["dofs"].tolist()
            elif info["name"] == "DifferentialDriveController":
                self.keypress_mapping[key.I] = {"idx": info["start_idx"] + 0, "val": 0.4}
                self.keypress_mapping[key.K] = {"idx": info["start_idx"] + 0, "val": -0.4}
                self.keypress_mapping[key.L] = {"idx": info["start_idx"] + 1, "val": -0.2}
                self.keypress_mapping[key.J] = {"idx": info["start_idx"] + 1, "val": 0.2}
            elif info["name"] in ("InverseKinematicsController", "OperationalSpaceController"):
                self.ik_arms.append(component)
                step = 0.1 if info["name"] == "InverseKinematicsController" else 0.25
                self.keypress_mapping.update(self._generate_axis_keypress_mapping(info, step))
            elif info["name"] == "MultiFingerGripperController":
                if info["command_dim"] > 1:
                    for i in range(info["command_dim"]):
                        self.joint_command_idx.append(info["start_idx"] + i)
                    self.joint_control_idx += info["dofs"].tolist()
                else:
                    self.keypress_mapping[key.T] = {"idx": info["start_idx"], "val": 1.0}
                    self.gripper_direction[component] = 1.0
                    self.persistent_gripper_action[component] = 1.0
                    self.binary_grippers.append(component)
            elif info["name"] == "NullJointController":
                self.keypress_mapping[key.T] = {"idx": None, "val": None}
            else:
                raise ValueError(f"Unknown controller name received: {info['name']}")

    def _generate_axis_keypress_mapping(self, info, step):
        key = self._key
        s = info["start_idx"]
        return {
            key.UP: {"idx": s + 0, "val": step},
            key.DOWN: {"idx": s + 0, "val": -step},
            key.RIGHT: {"idx": s + 1, "val": -step},
            key.LEFT: {"idx": s + 1, "val": step},
            key.P: {"idx": s + 2, "val": step},
            key.SEMICOLON: {"idx": s + 2, "val": -step},
            key.N: {"idx": s + 3, "val": step},
            key.B: {"idx": s + 3, "val": -step},
            key.O: {"idx": s + 4, "val": step},
            key.U: {"idx": s + 4, "val": -step},
            key.V: {"idx": s + 5, "val": step},
            key.C: {"idx": s + 5, "val": -step},
        }

    def on_key_press(self, symbol, modifiers):
        key = self._key
        if symbol in (key._1, key._2) and len(self.joint_control_idx) > 1:
            self.active_joint_command_idx_idx = (
                max(0, self.active_joint_command_idx_idx - 1)
                if symbol == key._1
                else min(len(self.joint_control_idx) - 1, self.active_joint_command_idx_idx + 1)
            )
            print(
                f"Now controlling joint {self.joint_names[self.joint_control_idx[self.active_joint_command_idx_idx]]}"
            )
        elif symbol in (key._3, key._4) and len(self.ik_arms) > 1:
            self.active_arm_idx = (
                max(0, self.active_arm_idx - 1)
                if symbol == key._3
                else min(len(self.ik_arms) - 1, self.active_arm_idx + 1)
            )
            new_arm = self.ik_arms[self.active_arm_idx]
            info = self.controller_info[new_arm]
            step = 0.1 if info["name"] == "InverseKinematicsController" else 0.25
            self.keypress_mapping.update(self._generate_axis_keypress_mapping(info, step))
            print(f"Now controlling arm {new_arm} EEF")
        elif symbol in (key._5, key._6) and len(self.binary_grippers) > 1:
            self.active_gripper_idx = (
                max(0, self.active_gripper_idx - 1)
                if symbol == key._5
                else min(len(self.binary_grippers) - 1, self.active_gripper_idx + 1)
            )
            print(f"Now controlling gripper {self.binary_grippers[self.active_gripper_idx]} with binary toggling")
        else:
            self.active_action = self.keypress_mapping.get(symbol, None)

        self.current_keypress = symbol
        if symbol == key.T:
            self.toggling_gripper = True

    def on_key_release(self, symbol, modifiers):
        if symbol == self.current_keypress:
            self.active_action = None
            self.current_keypress = None

    def get_teleop_action(self):
        action = th.zeros(self.action_dim, device=og.sim.device)

        if self.active_action is not None:
            idx, val = self.active_action["idx"], self.active_action["val"]
            if val is not None:
                if idx is None and len(self.joint_command_idx) != 0:
                    idx = self.joint_command_idx[self.active_joint_command_idx_idx]
                    joint_idx = self.joint_control_idx[self.active_joint_command_idx_idx]

                    from omnigibson.utils.constants import JointType

                    gk = self.joint_idx_to_group_key[joint_idx]
                    if (
                        self.joint_types[joint_idx] == JointType.JOINT_PRISMATIC
                        and ControllerView.get_use_delta_commands(gk)
                        and ControllerView.get_motor_type(gk) == "position"
                    ):
                        val *= 0.2
                if idx is not None:
                    action[idx] = val

        if len(self.binary_grippers) > 0 and self.keypress_mapping[self._key.T]["val"] is not None:
            for i, binary_gripper in enumerate(self.binary_grippers):
                if self.toggling_gripper and i == self.active_gripper_idx:
                    self.gripper_direction[binary_gripper] *= -1.0
                    self.persistent_gripper_action[binary_gripper] = (
                        self.keypress_mapping[self._key.T]["val"] * self.gripper_direction[binary_gripper]
                    )
                    self.toggling_gripper = False
                action[self.controller_info[binary_gripper]["start_idx"]] = self.persistent_gripper_action[
                    binary_gripper
                ]

        return action

    def print_teleop_info(self):
        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print()
        print("*" * 30)
        print("Controlling the Robot Using the Keyboard")
        print("*" * 30)
        print()
        print("Joint Control")
        print_command("1, 2", "decrement / increment the joint to control")
        print_command("[, ]", "move the joint backwards, forwards, respectively")
        print()
        print("Differential Drive Control")
        print_command("j, l", "turn left, right")
        print_command("i, k", "move forward, backwards")
        print()
        print("EEF Control (IK/OSC)")
        print_command("3, 4", "decrement / increment the arm to control")
        print_command("up, down", "move the EEF in the +/- x direction")
        print_command("left, right", "move the EEF in the +/- y direction")
        print_command("p, ;", "move the EEF in the +/- z direction")
        print_command("n, b", "rotate the EEF about the +/- x axis")
        print_command("o, u", "rotate the EEF about the +/- y axis")
        print_command("v, c", "rotate the EEF about the +/- z axis")
        print()
        print("Grasping")
        print_command("5, 6", "decrement / increment the gripper to control")
        print_command("t", "toggle gripper (open/close)")
        print()


def attach_teleop_keyboard(viewer, renderer, kb_controller):
    """Wire a Newton viewer's own keyboard events into ``kb_controller``'s ``on_key_press``/
    ``on_key_release``, so :meth:`_PygletTeleopController.get_teleop_action` reflects live input.

    Args:
        viewer: a viewer built by :func:`build_viewer` with ``headless=False`` (teleop needs a real,
            visible window to read input from).
        renderer (str): ``"gl"`` or ``"rtx"`` -- the two backends expose key-event registration
            differently (see below).
        kb_controller (_PygletTeleopController): controller built by :func:`run_teleop_with_newton_viewer`.
    """
    if renderer == "gl":
        # Public API: ViewerGL's renderer supports multiple simultaneous key-press/release listeners
        # (appended to a list), so this coexists with its own built-in camera-fly-key handling.
        viewer.renderer.register_key_press(kb_controller.on_key_press)
        viewer.renderer.register_key_release(kb_controller.on_key_release)
    elif renderer == "rtx":
        # No public registration API exists on ViewerRTX (its own on_key_press/on_key_release are
        # bound directly via `@self._window.event`, a single-slot decorator) -- push_handlers() adds
        # ours on top of pyglet's normal handler stack instead of replacing that slot, so both fire
        # (confirmed via pyglet's EventDispatcher: a handler returning None/False falls through to the
        # next one down the stack). This reaches into ViewerRTX's private `_window` because there's no
        # other way in.
        viewer._window.push_handlers(
            on_key_press=kb_controller.on_key_press, on_key_release=kb_controller.on_key_release
        )
    else:
        raise ValueError(f"Unknown renderer {renderer!r}, expected 'rtx' or 'gl'.")


def run_teleop_with_newton_viewer(env, robot, renderer, fps=30):
    """Interactively teleoperate ``robot`` through a live, on-screen Newton viewer (``"rtx"`` or
    ``"gl"``), using the same keyboard scheme as the Kit-renderer path's
    :class:`~omnigibson.utils.ui_utils.KeyboardRobotController` (see :class:`_PygletTeleopController`).

    Always opens a live window (``headless=False``) since teleop is inherently interactive; there's no
    headless/recording variant of this function (see :func:`record_with_newton_viewer` for that). Runs
    until the window is closed (via its 'X' button or ESC, both handled by newton's own viewer GUI).

    Requires ``gm.PHYSICS_BACKEND == "newton"`` and ``gm.RENDER_BACKEND == "none"`` to already be in
    effect (set before ``omnigibson`` was imported), same as :func:`record_with_newton_viewer`.
    """
    reposition_robot_into_furnished_room(env, robot)

    backend = og.sim.physics_backend
    model = backend._model

    cam_pos, look_at = _pick_camera_pose(env, robot)
    viewer = build_viewer(renderer, model, cam_pos, look_at, headless=False)

    # ViewerRTX defers creating its pyglet window until the first end_frame() call
    # (_init_ovrtx()) -- prime one frame here so `viewer._window` exists before
    # attach_teleop_keyboard() reaches into it below. Harmless no-op timing-wise for ViewerGL,
    # whose window/renderer already exist from build_viewer() above.
    viewer.begin_frame(0.0)
    viewer.log_state(backend._state_0)
    viewer.end_frame()

    kb_controller = _PygletTeleopController(robot)
    attach_teleop_keyboard(viewer, renderer, kb_controller)
    kb_controller.print_teleop_info()
    print("Running demo.")
    print("Close the window (or press ESC) to quit.")

    frame_dt = 1.0 / fps
    i = 0
    while viewer.is_running():
        step_t0 = time.time()
        action = kb_controller.get_teleop_action()
        env.step(action=action)

        viewer.begin_frame(i / fps)
        viewer.log_state(backend._state_0)
        viewer.end_frame()

        # Pace to roughly real time, matching record_with_newton_viewer's live-window branch.
        elapsed = time.time() - step_t0
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)
        i += 1

    viewer.close()
