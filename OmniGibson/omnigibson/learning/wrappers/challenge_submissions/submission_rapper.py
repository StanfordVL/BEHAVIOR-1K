from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
import omnigibson.utils.transform_utils as T
import torch as th
from typing import List, Optional
import os
import csv
import ast
import re
import math

from omnigibson.maps.map_base import BaseMap
from omnigibson.robots import BaseRobot

logger = create_module_logger(__name__)


class RAPPERWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env: Environment,
        target_sequence: Optional[List] = None,
        arrival_threshold: float = 0.001,
        waypoint_arrival_threshold: float = 0.01,
        task_name: str = None,
        erosion_radius: float = 0.4,  # Erosion radius for map (meters), includes robot size + safety
    ):
        super().__init__(env=env)
        self.task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.planner_dir = os.path.join(self.base_dir, "planner")
        list_path = os.path.join(self.planner_dir, f"{self.task_idx:04d}.txt")
        with open(list_path, "r", encoding="utf-8") as f:
            self.lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        self.csv_index = 0
        self.csv_path = self._resolve_csv_from_list(self.lines, self.csv_index)

        # Load replace_range.csv
        self.replace_range = {}  # key: (task_idx:int, skill:str, target:str) -> replace_length:int
        self._load_replace_range()

        if target_sequence is not None:
            self._target_sequence: List[dict] = list(target_sequence)
        else:
            self._target_sequence: List[dict] = self._load_planner_sequence()

        if target_sequence is not None:
            self._target_sequence: List[dict] = list(target_sequence)
        else:
            self._target_sequence: List[dict] = self._load_planner_sequence()

        self._cur_target_idx: int = 0
        self._arrival_threshold: float = float(arrival_threshold)
        self.only_replace_this_list = []

        # Erosion parameter for dynamic map
        self.erosion_radius: float = float(erosion_radius)

        # Action-skill replay
        self._action_active_idx: int = -1
        self._action_step_count: int = 0
        self._action_total_steps: int = 0
        self.moveto_max = 500
        self._done = False

        # A* path plan var
        self._planned_path: Optional[List[th.Tensor]] = None
        self._path_idx: int = 0
        self._waypoint_arrival_threshold: float = float(waypoint_arrival_threshold)
        self._final_target_pos_3d: Optional[th.Tensor] = None
        self._final_target_yaw: Optional[float] = None

        self._defer_ticks_after_reset = 0

        env.load_observation_space()
        print("[MyWrapper] Initialized and reloaded observation space.")

    def _resolve_csv_from_list(self, lines, csv_index: int) -> str:
        csv_path = os.path.expanduser(lines[csv_index])
        return csv_path

    def _load_replace_range(self) -> None:
        """
        Load planner_dir/replace_range.csv file and store in self.replace_range.
        CSV columns: task_idx(int), skill(str), target(str), replace_length(int)
        Key format: (task_idx, skill_lower, target_stripped)
        """
        path = os.path.join(self.planner_dir, "replace_range.csv")
        self.replace_range = {}
        if not os.path.exists(path):
            logger.info(f"[MyWrapper] replace_range.csv not found at {path}; using empty replace_range.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                required = {"task_idx", "skill", "target", "replace_length"}
                if not required.issubset(set(h.strip().lower() for h in reader.fieldnames or [])):
                    logger.warning("[MyWrapper] replace_range.csv header missing required columns; ignoring file.")
                    return
                for i, row in enumerate(reader, start=1):
                    try:
                        t_idx = int(str(row.get("task_idx", "")).strip())
                        skill = str(row.get("skill", "")).strip().lower()
                        target = str(row.get("target", "")).strip()
                        repl = int(str(row.get("replace_length", "")).strip())
                        if not skill or not target:
                            continue
                        self.replace_range[(t_idx, skill, target)] = repl
                    except Exception as e:
                        logger.warning(f"[MyWrapper] replace_range.csv line {i} parse error: {e}")
        except Exception as e:
            logger.warning(f"[MyWrapper] Failed to read replace_range.csv: {e}")

    def _load_planner_sequence(self, csv_path: Optional[str] = None) -> List[dict]:
        steps: List[dict] = []
        csv_path = csv_path if csv_path is not None else self.csv_path
        print(f"[MyWrapper] Using CSV: {csv_path}")
        csv_path = os.path.join(self.planner_dir, csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    skill = str((row.get("skill") or "")).strip().lower()
                    if not skill:
                        continue
                    target_obj = str((row.get("target_obj") or "")).strip()
                    target_robot_pos = (row.get("robot_pos") or "").strip()
                    target_obj_pos = (row.get("obj_pos") or "").strip()
                    action_path_raw = (row.get("action_path") or "").strip()
                    repeat = (row.get("repeat") or "").strip()
                    info = {}
                    if skill == "moveto":
                        info["target"] = target_obj
                        if target_robot_pos and target_obj_pos:
                            target_robot_pos_cleaned = re.sub(r",\s*\]", "]", target_robot_pos)
                            target_obj_pos_cleaned = re.sub(r",\s*\]", "]", target_obj_pos)
                            target_robot_pos_vec = ast.literal_eval(target_robot_pos_cleaned)
                            target_obj_pos_vec = ast.literal_eval(target_obj_pos_cleaned)
                            info["target_robot_pos_vec"] = [float(x) for x in target_robot_pos_vec[:9]]
                            info["target_obj_pos_vec"] = [float(x) for x in target_obj_pos_vec[:9]]
                            info["repeat"] = int(repeat)
                    elif skill == "action":
                        if not action_path_raw:
                            continue
                        info["action_path"] = action_path_raw
                        action_path = action_path_raw
                        if not os.path.isabs(action_path):
                            action_path = os.path.normpath(os.path.join(os.path.dirname(csv_path), action_path))
                        if os.path.exists(action_path):
                            info["action_path_resolved"] = action_path
                            length = self._npz_action_length(action_path)
                            if isinstance(length, int) and length > 0:
                                info["action_total_steps"] = length
                        info["repeat"] = int(repeat)
                    elif skill == "moveto_pose":
                        info["target"] = target_obj
                        if target_robot_pos:
                            target_robot_pos_cleaned = re.sub(r",\s*\]", "]", target_robot_pos)
                            target_robot_pos_vec = ast.literal_eval(target_robot_pos_cleaned)
                            info["target_robot_pos_vec"] = [float(x) for x in target_robot_pos_vec[:9]]
                            info["repeat"] = int(repeat)
                    if skill == "action":
                        desc = f"action {os.path.basename(info.get('action_path', ''))}"
                    elif skill == "moveto":
                        desc = f"move to {info.get('target', '?')}"
                    elif skill == "moveto_pose":
                        desc = f"move to {info.get('target_robot_pos_vec', '?')}"
                    info["desc"] = desc
                    step = {"skill": skill}
                    step.update(info)
                    steps.append(step)
                except Exception:
                    continue
        return steps

    def _switch_to_next_csv_after_current(
        self, only_replace_this: bool = False, replace_length: Optional[int] = None
    ) -> bool:
        if not self._target_sequence:
            return False
        cur = self._target_sequence[self._cur_target_idx % len(self._target_sequence)]
        cur_skill = str(cur.get("skill", "")).strip().lower()
        cur_target = str(cur.get("target", "")).strip()
        cur_repeat = int(cur.get("repeat", 0))

        next_index = self.csv_index + 1

        if next_index >= len(self.lines):
            logger.warning("[MyWrapper] No more CSVs to switch to.")
            return False

        next_csv_rel = self.lines[next_index]
        try:
            next_steps = self._load_planner_sequence(next_csv_rel)
        except Exception as e:
            logger.warning(f"[MyWrapper] Failed to load next CSV: {e}")
            return False

        match_idx = -1
        for i, st in enumerate(next_steps):
            if (
                str(st.get("skill", "")).strip().lower() == cur_skill
                and str(st.get("target", "")).strip() == cur_target
                and int(st.get("repeat", 0)) == cur_repeat
            ):
                match_idx = i
                break

        if match_idx == -1:
            logger.warning("[MyWrapper] Could not align with next CSV (no match).")
            return False

        # -------- 선택적 부분 교체 (replace_length 우선) --------
        if isinstance(replace_length, int) and replace_length > 0:
            # 현재 실패 스텝부터 replace_length 길이만큼만 교체
            slice_end = min(match_idx + replace_length, len(next_steps))
            new_chunk = next_steps[match_idx:slice_end]
            if not new_chunk:
                logger.warning("[MyWrapper] Replacement slice is empty.")
                return False

            old_tail = self._target_sequence[
                self._cur_target_idx + replace_length :
            ]  # 기존 꼬리는 동일 길이만큼 건너뛰고 이어붙임
            self.csv_index = next_index
            self.csv_path = next_csv_rel
            self._target_sequence = list(new_chunk) + old_tail
            self._cur_target_idx = 0

            # 상태 리셋
            self._planned_path = None
            self._path_idx = 0
            self._final_target_pos_3d = None
            self._final_target_yaw = None
            self._action_step_count = 0
            self._action_total_steps = 0
            self._action_active_idx = -1
            self._done = False
            print(
                f"[MyWrapper] Replaced {len(new_chunk)} step(s) from next CSV starting at the failed step; kept the remaining original tail."
            )
            return True

        if only_replace_this:
            # 현재 스텝을 next_steps의 매칭 스텝으로 '대체'하고,
            # 그 이후의 스텝은 기존 시퀀스의 (현재+1)부터 그대로 이어감
            old_tail = self._target_sequence[self._cur_target_idx + 1 :]
            replaced_current = next_steps[match_idx]
            self.csv_index = next_index
            self.csv_path = next_csv_rel
            self._target_sequence = [replaced_current] + old_tail
            self._cur_target_idx = 0
            # 상태 리셋
            self._planned_path = None
            self._path_idx = 0
            self._final_target_pos_3d = None
            self._final_target_yaw = None
            self._action_step_count = 0
            self._action_total_steps = 0
            self._action_active_idx = -1
            self._done = False
            print(
                "[MyWrapper] Replaced only the current step from next CSV; kept the following steps from the original plan."
            )
            return True

        # -------- 기본 동작: 매칭 지점부터 끝까지 전부 교체 --------
        if match_idx + 1 >= len(next_steps):
            logger.warning("[MyWrapper] Could not align or nothing after match.")
            return False
        self.csv_index = next_index
        self.csv_path = next_csv_rel
        self._target_sequence = next_steps[match_idx:]
        self._cur_target_idx = 0
        # 상태 리셋
        self._planned_path = None
        self._path_idx = 0
        self._final_target_pos_3d = None
        self._final_target_yaw = None
        self._action_step_count = 0
        self._action_total_steps = 0
        self._action_active_idx = -1
        self._done = False
        print(
            f"[MyWrapper] Switched to next CSV (entire): {self.csv_path} (start from {next_steps[match_idx].get('skill')} - {next_steps[match_idx].get('target')})"
        )
        return True

    def _npz_action_length(self, path: str) -> Optional[int]:
        try:
            import numpy as np

            p = os.path.expanduser(path)
            if not os.path.exists(p):
                return None
            arr = np.load(p, allow_pickle=True)
            if hasattr(arr, "files"):
                key = "actions" if "actions" in arr.files else (arr.files[0] if arr.files else None)
                if key is not None:
                    a = arr[key]
                    return int(a.shape[0]) if hasattr(a, "shape") and len(a.shape) > 0 else None
        except Exception:
            return None
        return None

    def step(self, action, n_render_iterations=1):
        obs, reward, terminated, truncated, info = self.env.step(action, n_render_iterations=n_render_iterations)
        if self._done:
            return obs, reward, True, truncated, info
        self._inject_robot_and_target(obs)
        return obs, reward, terminated, truncated, info

    def reset(self):
        self._done = False
        self._cur_target_idx = 0
        self._action_active_idx = -1
        self._action_step_count = 0
        self._action_total_steps = 0

        self.csv_index = 0
        self.csv_path = self._resolve_csv_from_list(self.lines, self.csv_index)
        self._target_sequence = self._load_planner_sequence(self.csv_path)

        self._defer_ticks_after_reset = 1

        self._planned_path = None
        self._path_idx = 0
        self._final_target_pos_3d = None
        self._final_target_yaw = None

        ret = self.env.reset()
        self._inject_robot_and_target(ret[0])
        return ret

    def _inject_robot_and_target(self, obs: dict) -> None:
        for k in (
            "aux::target_pos",
            "aux::target_yaw",
            "aux::action_path",
            "aux::target_robot_pos_vec",
            "aux::target_obj_pos_vec",
        ):
            obs.pop(k, None)
        if getattr(self, "_defer_ticks_after_reset", 0) > 0:
            self._defer_ticks_after_reset -= 1
            return
        robot = self.env.robots[0]
        robot_pos, robot_orn = robot.get_position_orientation()
        if not isinstance(robot_pos, th.Tensor):
            robot_pos = th.tensor(robot_pos, dtype=th.float32)
        obs["aux::robot_pos"] = robot_pos.float()
        eulers = T.quat2euler(robot_orn)
        yaw = eulers[2]
        if isinstance(yaw, th.Tensor):
            yaw = float(yaw.view(-1)[0])
        yaw_t = th.tensor(yaw, dtype=th.float32)
        obs["aux::robot_yaw"] = yaw_t
        current_step = self._target_sequence[self._cur_target_idx % len(self._target_sequence)]
        skill = str(current_step.get("skill", "")).strip().lower()
        info = {k: v for k, v in current_step.items() if k != "skill"}
        obs["aux::current_skill"] = skill
        obs["aux::current_spec"] = info.get("desc", str(current_step))
        if skill == "moveto":
            self._do_moveto_planned(info, obs, relative=True)
        elif skill == "moveto_pose":
            self._do_moveto_planned(info, obs, relative=False)
        elif skill == "action":
            act_path = info.get("action_path_resolved") or info.get("action_path")
            if isinstance(act_path, str) and len(act_path) > 0:
                obs["aux::action_path"] = act_path
            self._do_action(info)
        else:
            pass

    def _resolve_target_object(self, spec: str):
        idx = 1
        base = spec
        if "(" in spec and ")" in spec:
            try:
                prefix, rest = spec.split("(", 1)
                sel = rest.split(")", 1)[0].strip()
                base = prefix.strip()
                if sel.lower() == "another":
                    idx = 2
                else:
                    val = int(sel)
                    idx = max(1, val)
            except Exception:
                base = spec
        matches = []
        seen = set()

        def add_unique(objs):
            for o in objs:
                oid = getattr(o, "name", None)
                if oid is None:
                    oid = id(o)
                if oid in seen:
                    continue
                seen.add(oid)
                matches.append(o)

        all_objs = sorted(self.env.scene.objects, key=lambda o: getattr(o, "name", ""))
        base_lower = base.lower()
        add_unique([o for o in all_objs if getattr(o, "name", "") == base])
        add_unique([o for o in all_objs if base_lower in getattr(o, "name", "").lower()])
        add_unique(
            sorted(list(self.env.scene.object_registry("category", base, [])), key=lambda o: getattr(o, "name", ""))
        )
        add_unique([o for o in all_objs if base_lower in getattr(o, "category", "").lower()])
        if not matches:
            return None
        sel_idx = min(len(matches) - 1, idx - 1)
        return matches[sel_idx]

    def _advance_step(self):
        if self._cur_target_idx + 1 == len(self._target_sequence):
            self._done = True
        self._cur_target_idx = (self._cur_target_idx + 1) % len(self._target_sequence)

        self._planned_path = None
        self._path_idx = 0
        self._final_target_pos_3d = None
        self._final_target_yaw = None
        self._action_step_count = 0
        self._action_total_steps = 0
        self._action_active_idx = -1

        current_step = self._target_sequence[self._cur_target_idx % len(self._target_sequence)]
        skill = str(current_step.get("skill", "")).strip().lower()
        desc = str(current_step.get("desc", "")).strip().lower()
        print(f"---------- Doing {skill}: {desc} ----------")

    def _do_moveto_planned(self, info: dict, obs: dict, relative: bool) -> None:
        """
        A* 경로를 생성하고, Policy가 다음 경유지를 쫓아가도록
        Dynamic map을 사용하여 movable objects를 고려함
        """
        robot_pos = obs.get("aux::robot_pos")
        robot: BaseRobot = self.env.robots[0]
        current_map: BaseMap = self.env.scene.trav_map

        if self._planned_path is None:
            self._action_active_idx = self._cur_target_idx
            self._action_step_count = 0
            self._action_total_steps = self.moveto_max
            if relative:
                try:

                    def _yaw_from_pose9(vec9):
                        cos_y, sin_y = float(vec9[5]), float(vec9[8])
                        return math.atan2(sin_y, cos_y)

                    def _pos_from_pose9(vec9):
                        return float(vec9[0]), float(vec9[1]), float(vec9[2])

                    def _rot2d(a):
                        ca, sa = math.cos(a), math.sin(a)
                        return ca, -sa, sa, ca

                    target_name = info.get("target")
                    r_pose9 = info.get("target_robot_pos_vec", None)
                    o_pose9 = info.get("target_obj_pos_vec", None)
                    rx, ry, rz = _pos_from_pose9(r_pose9)
                    ox, oy, oz = _pos_from_pose9(o_pose9)
                    yaw_r_ref = _yaw_from_pose9(r_pose9)
                    yaw_o_ref = _yaw_from_pose9(o_pose9)
                    drx_ref, dry_ref, drz_ref = rx - ox, ry - oy, rz - oz
                    dyaw_ref = yaw_r_ref - yaw_o_ref
                    target_obj = self._resolve_target_object(target_name)
                    if target_obj is None:
                        print(f"[MyWrapper] Target object '{target_name}' not found!")
                        self._planned_path = []
                        return
                    o_pos_w, o_quat_w = target_obj.get_position_orientation()
                    if not isinstance(o_pos_w, th.Tensor):
                        o_pos_w = th.tensor(o_pos_w, dtype=th.float32)
                    yaw_o_w = float(T.quat2euler(o_quat_w)[2])
                    delta = yaw_o_w - yaw_o_ref
                    ca, nsa, sa, ca2 = _rot2d(delta)
                    dx_w = ca * drx_ref + nsa * dry_ref
                    dy_w = sa * drx_ref + ca2 * dry_ref
                    dz_w = drz_ref
                    target_pos = th.stack(
                        [
                            o_pos_w[0] + th.tensor(dx_w, dtype=th.float32),
                            o_pos_w[1] + th.tensor(dy_w, dtype=th.float32),
                            o_pos_w[2] + th.tensor(dz_w, dtype=th.float32),
                        ]
                    ).float()
                    self._final_target_pos_3d = (
                        target_pos.clone().detach().to(dtype=th.float32)
                        if isinstance(target_pos, th.Tensor)
                        else th.tensor(target_pos, dtype=th.float32)
                    )
                    target_yaw = yaw_o_w + dyaw_ref
                    self._final_target_yaw = th.tensor(float(target_yaw), dtype=th.float32)
                except Exception as e:
                    print(f"[MyWrapper] Failed to calculate relative target: {e}")
                    self._planned_path = []
                    return
            else:
                r_pose9 = info.get("target_robot_pos_vec")
                self._final_target_pos_3d = th.tensor(r_pose9[:3], dtype=th.float32)
                cy, sy = float(r_pose9[3 + 2]), float(r_pose9[6 + 2])
                target_yaw = math.atan2(sy, cy)
                self._final_target_yaw = th.tensor(float(target_yaw), dtype=th.float32)

            # ========== Dynamic Map 생성 + A* path plan ==========
            if self._final_target_pos_3d is not None and self._planned_path is None:
                print(f"[MyWrapper] Planning A* path with dynamic map...")
                print(f"            Start: {robot_pos[:2].tolist()}")
                print(f"            Goal:  {self._final_target_pos_3d[:2].tolist()}")

                # 1. Static map 가져오기
                static_map = th.clone(current_map.floor_map[0])

                # 2. Dynamic map 생성 (movable objects 반영)
                dynamic_map = self._update_dynamic_map(static_map, robot)
                print(f"[MyWrapper] Dynamic map updated with movable objects")

                # Save dynamic map visualization immediately after creation
                try:
                    step_desc = info.get("target", "unknown")
                    filename = f"dynamic_map_step_{self._cur_target_idx:03d}_{step_desc}.png"
                    self._save_map_visualization(
                        dynamic_map, robot_pos, self._final_target_pos_3d, None, filename
                    )  # No path yet
                except Exception as e:
                    print(f"[MyWrapper] Map visualization failed: {e}")

                # 3. Erosion 적용 (로봇 크기 고려) - use custom erosion radius
                eroded_map = self._erode_map_custom(dynamic_map, robot)
                try:
                    step_desc = info.get("target", "unknown")
                    filename = f"eroded_map_step_{self._cur_target_idx:03d}_{step_desc}.png"
                    self._save_map_visualization(
                        eroded_map, robot_pos, self._final_target_pos_3d, None, filename
                    )  # No path yet
                except Exception as e:
                    print(f"[MyWrapper] Map visualization failed: {e}")

                # 4. A* 실행
                from omnigibson.utils.motion_planning_utils import astar

                source_map = tuple(current_map.world_to_map(robot_pos[:2]).tolist())
                target_map = tuple(current_map.world_to_map(self._final_target_pos_3d[:2]).tolist())

                path_map = astar(eroded_map, source_map, target_map)

                if path_map is None or len(path_map) == 0:
                    print(f"[MyWrapper] A* failed! Trying to switch to next CSV aligned with current progress...")

                    # --- replace_range 조회: (task_idx, skill, target) -> replace_length ---
                    try:
                        cur_step = self._target_sequence[self._cur_target_idx % len(self._target_sequence)]
                        cur_skill_lookup = str(cur_step.get("skill", "")).strip().lower()
                    except Exception:
                        cur_skill_lookup = "moveto" if relative else "moveto_pose"
                    cur_target_lookup = str(info.get("target", "")).strip()
                    repl_len = self.replace_range.get((self.task_idx, cur_skill_lookup, cur_target_lookup), None)

                    # only_replace_this_list에 해당하고 repl_len이 없으면 현재 스텝만 교체
                    if self.task_idx in getattr(self, "only_replace_this_list", []) and repl_len is None:
                        print("[MyWrapper] Only replace this action (policy list hit, no replace_length override).")
                        possible = self._switch_to_next_csv_after_current(only_replace_this=True)
                    else:
                        # repl_len이 있으면 부분 교체, 없으면 전체 교체
                        if isinstance(repl_len, int) and repl_len > 0:
                            print(
                                f"[MyWrapper] Using replace_length={repl_len} from replace_range for ({self.task_idx}, {cur_skill_lookup}, {cur_target_lookup})."
                            )
                        possible = self._switch_to_next_csv_after_current(
                            only_replace_this=False, replace_length=repl_len
                        )
                    if possible:
                        print(f"[MyWrapper] Switched to next CSV. Will re-inject on next step.")
                        new_step = self._target_sequence[self._cur_target_idx % len(self._target_sequence)]
                        new_skill = str(new_step.get("skill", "")).strip().lower()
                        new_info = {k: v for k, v in new_step.items() if k != "skill"}
                        # 이전 타깃 흔적 정리
                        obs.pop("aux::target_pos", None)
                        obs.pop("aux::target_yaw", None)
                        # 새 스텝을 같은 틱에 주입
                        if new_skill == "moveto":
                            self._do_moveto_planned(new_info, obs, relative=True)
                        elif new_skill == "moveto_pose":
                            self._do_moveto_planned(new_info, obs, relative=False)
                        elif new_skill == "action":
                            self._do_action(new_info)
                        return  # 같은 틱 내 재주입 완료

                    print(f"[MyWrapper] A* failed! Using straight line fallback")
                    distance = th.norm(self._final_target_pos_3d[:2] - robot_pos[:2])
                    num_waypoints = max(30, int(distance / 0.05))  # 5cm 간격
                    current_z = robot_pos[2]
                    self._planned_path = []
                    for i in range(num_waypoints + 1):
                        t = i / num_waypoints
                        x = float(robot_pos[0] * (1 - t) + self._final_target_pos_3d[0] * t)
                        y = float(robot_pos[1] * (1 - t) + self._final_target_pos_3d[1] * t)
                        self._planned_path.append(th.tensor([x, y, current_z], dtype=th.float32))
                    print(f"  Created {len(self._planned_path)} interpolated waypoints")
                else:
                    # A* success - convert map coordinates to world coordinates
                    path_world = current_map.map_to_world(path_map)
                    geo_dist = th.sum(th.norm(path_world[1:] - path_world[:-1], dim=1))

                    current_z = robot_pos[2]
                    self._planned_path = [
                        th.tensor([float(p[0]), float(p[1]), current_z], dtype=th.float32) for p in path_world
                    ]
                    self._planned_path[-1] = self._final_target_pos_3d.clone()
                    print(f"            A* success! {len(self._planned_path)} waypoints, {geo_dist:.2f}m")

                self._path_idx = 0

        # step count
        self._action_step_count += 1
        # follow path
        if not self._planned_path or len(self._planned_path) == 0:
            obs["aux::target_pos"] = self._final_target_pos_3d.float()
            obs["aux::target_yaw"] = th.tensor(float(self._final_target_yaw), dtype=th.float32)
        elif self._path_idx == len(self._planned_path) - 1:
            obs["aux::target_pos"] = self._final_target_pos_3d.float()
            obs["aux::target_yaw"] = th.tensor(float(self._final_target_yaw), dtype=th.float32)
        else:
            waypoint_to_chase = self._planned_path[self._path_idx]
            obs["aux::target_pos"] = waypoint_to_chase.float()
            obs["aux::target_yaw"] = th.tensor(float(self._final_target_yaw), dtype=th.float32)
            # waypoint arrive
            dx = float(waypoint_to_chase[0] - robot_pos[0])
            dy = float(waypoint_to_chase[1] - robot_pos[1])
            dist_to_waypoint = math.sqrt(dx * dx + dy * dy)
            if dist_to_waypoint < self._waypoint_arrival_threshold and self._path_idx < len(self._planned_path) - 1:
                self._action_step_count = 0
                self._path_idx += 1
                print(f"[MyWrapper] Waypoint {self._path_idx}/{len(self._planned_path)}")

        # target arrive
        if self._final_target_pos_3d is None:
            if self._action_step_count >= 100:
                print(f"[MyWrapper] Moveto failed (no target)")
                self._advance_step()
            return

        dx = float(self._final_target_pos_3d[0] - robot_pos[0])
        dy = float(self._final_target_pos_3d[1] - robot_pos[1])
        final_dist = math.sqrt(dx * dx + dy * dy)

        cur_yaw = float(obs.get("aux::robot_yaw", th.tensor(0.0)))
        des_yaw = float(self._final_target_yaw)
        yaw_err = des_yaw - cur_yaw
        while yaw_err > math.pi:
            yaw_err -= 2 * math.pi
        while yaw_err < -math.pi:
            yaw_err += 2 * math.pi
        yaw_ok = abs(yaw_err) <= 0.005  # 0.10 rad ~ 5.7 deg

        if final_dist <= self._arrival_threshold and yaw_ok:
            print(f"[MyWrapper] ✓ Moveto SUCCESS ({final_dist:.3f}m, {yaw_err:.3f}, {self._action_step_count} steps)")
            self._advance_step()
        elif self._action_step_count >= self._action_total_steps:
            print(f"[MyWrapper] ⏱ Moveto TIMEOUT ({final_dist:.3f}m, {yaw_err:.3f})")
            self._advance_step()

    def _do_action(self, info: dict) -> None:
        """Gate an 'action' skill by replay length from an npz action file."""
        action_path = info.get("action_path_resolved") or info.get("action_path")
        # Initialize replay counters on step entry
        if self._action_active_idx != self._cur_target_idx:
            total = info.get("action_total_steps")
            if not isinstance(total, int) or total <= 0:
                total = self._npz_action_length(action_path) or 1
                info["action_total_steps"] = total
            self._action_active_idx = self._cur_target_idx
            self._action_step_count = 0
            self._action_total_steps = int(total)
        # Advance replay counter
        self._action_step_count += 1
        if self._action_step_count >= self._action_total_steps:
            print(f"[MyWrapper] Action(replay) finished after {self._action_step_count} steps !!\n")
            self._action_step_count = 0
            self._action_total_steps = 0
            self._action_active_idx = -1
            self._advance_step()

    def _update_dynamic_map(self, static_map: th.Tensor, robot: BaseRobot) -> th.Tensor:
        """
        Update dynamic traversability map with all objects in real-time
        Overlays current scene objects on top of static map

        Args:
            static_map: Original static traversability map
            robot: Robot object (excluded from obstacles)

        Returns:
            dynamic_map: Map with all objects reflected
        """
        from omnigibson.utils.constants import GROUND_CATEGORIES, STRUCTURE_CATEGORIES

        dynamic_map = static_map.clone()
        current_map: BaseMap = self.env.scene.trav_map

        # Get robot name to exclude it from obstacles
        robot_name = robot.name if hasattr(robot, "name") else None

        # Combine categories to exclude (already in static map)
        exclude_categories = GROUND_CATEGORIES | STRUCTURE_CATEGORIES

        # Additional outdoor/structure objects to exclude
        additional_exclude = {
            "paver",  # Paving stones (covers large areas)
            "rail_fence",  # Fencing (already in static map)
            "bush",  # Bushes (landscaping, already in map)
            "tree",  # Trees (already in static map)
            "downlight",  # Ceiling lights
            "track_light",  # Track lighting
            "room_light",  # Ceiling lights
            "garden_light",  # Garden lights (pole-mounted)
            "wall_mounted_light",  # Wall lights
            "electric_switch",  # Light switches
            "wall_socket",  # Electrical outlets
            "motion_sensor",  # Sensors
            "mirror",  # Wall mirrors
            "decorative_sign",  # Wall signs
            "hook",  # Wall hooks
        }

        exclude_categories = exclude_categories | additional_exclude

        # Iterate through all objects
        for obj in self.env.scene.objects:
            try:
                # Skip the robot itself to avoid blocking start position
                obj_name = getattr(obj, "name", None)
                if obj_name is not None and obj_name == robot_name:
                    continue

                if obj is robot:
                    continue

                # Skip structure and ground categories
                obj_category = getattr(obj, "category", None)
                if obj_category in exclude_categories:
                    continue

                # Get AABB (axis-aligned bounding box)
                try:
                    bbox_center = obj.aabb_center
                    bbox_extent = obj.aabb_extent
                    if bbox_center is None or bbox_extent is None:
                        continue

                    # Shrink AABB slightly to reduce over-blocking (95% of original size)
                    shrink_factor = 0.95
                    shrunk_extent = bbox_extent * shrink_factor

                except Exception:
                    continue

                # Convert world coordinates to map coordinates
                lower_corner = bbox_center - shrunk_extent / 2.0
                upper_corner = bbox_center + shrunk_extent / 2.0

                min_xy_world = lower_corner[:2]
                max_xy_world = upper_corner[:2]

                min_xy_map = current_map.world_to_map(min_xy_world)
                max_xy_map = current_map.world_to_map(max_xy_world)

                # Mark obstacle region in map
                map_h, map_w = dynamic_map.shape
                min_r = max(0, int(min_xy_map[0]))
                max_r = min(map_h, int(max_xy_map[0]))
                min_c = max(0, int(min_xy_map[1]))
                max_c = min(map_w, int(max_xy_map[1]))

                # Ensure minimum 1 pixel is marked (for very small objects)
                if max_r <= min_r:
                    max_r = min_r + 1
                if max_c <= min_c:
                    max_c = min_c + 1

                # Set as obstacle (0 = obstacle)
                dynamic_map[min_r:max_r, min_c:max_c] = 0

            except Exception:
                continue

        return dynamic_map

    def _erode_map_custom(self, trav_map: th.Tensor, robot: BaseRobot) -> th.Tensor:
        """
        Erode traversability map with custom erosion radius

        Args:
            trav_map: Traversability map to erode
            robot: Robot object (unused, kept for signature compatibility)

        Returns:
            Eroded traversability map
        """
        import cv2
        import math

        current_map: BaseMap = self.env.scene.trav_map

        # Use only the configured erosion radius (no robot size calculation)
        erosion_radius = self.erosion_radius

        # Convert to pixels
        radius_pixels = int(math.ceil(erosion_radius / current_map.map_resolution))

        print(f"[MyWrapper] Erosion: radius={erosion_radius:.3f}m ({radius_pixels}px)")

        # Apply erosion
        if radius_pixels > 0:
            kernel = th.ones((radius_pixels, radius_pixels))
            eroded_map = th.tensor(cv2.erode(trav_map.cpu().numpy(), kernel.cpu().numpy()))
        else:
            eroded_map = trav_map.clone()

        return eroded_map

    def _save_map_visualization(
        self,
        map_tensor: th.Tensor,
        robot_pos: th.Tensor,
        target_pos: th.Tensor,
        path: Optional[List[th.Tensor]],
        filename: str = "debug_map.png",
    ) -> None:
        """
        Save map as PNG for debugging

        Args:
            map_tensor: Traversability map (2D tensor)
            robot_pos: Robot position (world coordinates)
            target_pos: Target position (world coordinates)
            path: A* path (world coordinates list)
            filename: Filename to save
        """
        try:
            import cv2
            import numpy as np

            current_map: BaseMap = self.env.scene.trav_map

            # Convert tensor to numpy array
            map_np = map_tensor.cpu().numpy().astype(np.uint8)

            # Convert grayscale to BGR (for color visualization)
            map_bgr = cv2.cvtColor(map_np, cv2.COLOR_GRAY2BGR)

            # Mark robot position (green circle scaled to robot size)
            robot_map = current_map.world_to_map(robot_pos[:2])
            robot_r, robot_c = int(robot_map[0]), int(robot_map[1])

            # Calculate robot radius in pixels (approximate robot base radius ~0.2m)
            robot_radius_world = 0.2  # meters
            robot_radius_pixels = int(robot_radius_world / current_map.map_resolution)
            cv2.circle(map_bgr, (robot_c, robot_r), robot_radius_pixels, (0, 255, 0), 2)  # Green outline

            # Mark target position (red)
            target_map = current_map.world_to_map(target_pos[:2])
            target_r, target_c = int(target_map[0]), int(target_map[1])
            cv2.circle(map_bgr, (target_c, target_r), 5, (0, 0, 255), -1)  # Red dot

            # Mark path (blue)
            if path is not None and len(path) > 0:
                for i, wp in enumerate(path):
                    wp_map = current_map.world_to_map(wp[:2])
                    wp_r, wp_c = int(wp_map[0]), int(wp_map[1])
                    cv2.circle(map_bgr, (wp_c, wp_r), 3, (255, 0, 0), -1)  # Blue

                    # Draw path line
                    if i > 0:
                        prev_wp = path[i - 1]
                        prev_map = current_map.world_to_map(prev_wp[:2])
                        prev_r, prev_c = int(prev_map[0]), int(prev_map[1])
                        cv2.line(map_bgr, (prev_c, prev_r), (wp_c, wp_r), (255, 100, 0), 2)  # Blue line

            # Save file to map_debug directory
            map_debug_dir = os.path.join(self.base_dir, "map_debug")
            os.makedirs(map_debug_dir, exist_ok=True)
            save_path = os.path.join(map_debug_dir, filename)
            cv2.imwrite(save_path, map_bgr)
            print(f"[MyWrapper] Map saved to {save_path}")

        except Exception as e:
            print(f"[MyWrapper] Failed to save map visualization: {e}")


WRAPPER_CLASS = RAPPERWrapper
