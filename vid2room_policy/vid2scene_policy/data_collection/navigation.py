import math
import logging
import heapq

import torch as th
import numpy as np

from .robot_context import RobotContext, MODE_NAVIGATION, MODE_MANIPULATION

logger = logging.getLogger(__name__)

DEBUG_NAV = False


def _astar_pathfind(floor_map, start_rc, goal_rc, max_iterations=50000):
    """A* pathfinding on a binary floor map. Returns list of (row, col) or None."""
    rows, cols = floor_map.shape
    sr, sc = start_rc
    gr, gc = goal_rc

    if not (0 <= sr < rows and 0 <= sc < cols and floor_map[sr, sc] == 255):
        return None
    if not (0 <= gr < rows and 0 <= gc < cols and floor_map[gr, gc] == 255):
        return None

    def heuristic(r, c):
        return math.sqrt((r - gr)**2 + (c - gc)**2)

    open_set = [(heuristic(sr, sc), 0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0}
    closed = set()

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    iterations = 0
    while open_set and iterations < max_iterations:
        iterations += 1
        _, g, r, c = heapq.heappop(open_set)

        if (r, c) in closed:
            continue
        closed.add((r, c))

        if r == gr and c == gc:
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            return path[::-1]

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and floor_map[nr, nc] == 255 and (nr, nc) not in closed:
                move_cost = 1.414 if dr != 0 and dc != 0 else 1.0
                new_g = g + move_cost
                if (nr, nc) not in g_score or new_g < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = new_g
                    f = new_g + heuristic(nr, nc)
                    heapq.heappush(open_set, (f, new_g, nr, nc))
                    came_from[(nr, nc)] = (r, c)

    return None


class NavigationController:
    ANGLE_THRESHOLD = 0.10
    DIST_THRESHOLD = 0.15

    def __init__(self, ctx: RobotContext):
        self.ctx = ctx
        self._eroded_map_cache = None
        self._raw_map_cache = None
        self._debug_counter = 0

    def _save_path_visualization(self, path: list, robot_pos: tuple, target_pos: tuple):
        if not DEBUG_NAV:
            return
        try:
            import cv2
            trav_map = self.ctx.env.scene._trav_map
            floor_map = self._get_eroded_map()
            img = np.zeros((floor_map.shape[0], floor_map.shape[1], 3), dtype=np.uint8)
            floor_np = floor_map.cpu().numpy() if hasattr(floor_map, 'cpu') else np.array(floor_map)
            img[floor_np == 255] = [255, 255, 255]

            for i, wp in enumerate(path):
                if wp is None:
                    continue
                wp_map = trav_map.world_to_map(th.tensor([wp[0], wp[1]]))
                r, c = int(wp_map[0].item()), int(wp_map[1].item())
                cv2.circle(img, (c, r), 3, (255, 0, 0), -1)
                if i > 0 and path[i-1] is not None:
                    prev_map = trav_map.world_to_map(th.tensor([path[i-1][0], path[i-1][1]]))
                    pr, pc = int(prev_map[0].item()), int(prev_map[1].item())
                    cv2.line(img, (pc, pr), (c, r), (255, 0, 0), 1)

            robot_map = trav_map.world_to_map(th.tensor([robot_pos[0], robot_pos[1]]))
            cv2.circle(img, (int(robot_map[1].item()), int(robot_map[0].item())), 8, (0, 255, 0), -1)

            target_map = trav_map.world_to_map(th.tensor([target_pos[0], target_pos[1]]))
            cv2.circle(img, (int(target_map[1].item()), int(target_map[0].item())), 8, (0, 0, 255), -1)

            self._debug_counter += 1
            cv2.imwrite(f"/tmp/nav_path_{self._debug_counter:03d}.png", img)
        except Exception:
            pass

    def _get_eroded_map(self):
        if self._eroded_map_cache is None:
            import cv2
            trav_map = self.ctx.env.scene._trav_map
            erosion_radius = 0.325
            radius_pixel = int(math.ceil(erosion_radius / trav_map.map_resolution))
            floor_map = trav_map.floor_map[0].clone()
            self._eroded_map_cache = th.tensor(cv2.erode(floor_map.cpu().numpy(), th.ones((radius_pixel, radius_pixel)).cpu().numpy()))
        return self._eroded_map_cache

    def _get_raw_map(self):
        if self._raw_map_cache is None:
            self._raw_map_cache = self.ctx.env.scene._trav_map.floor_map[0]
        return self._raw_map_cache

    def invalidate_map_cache(self):
        self._eroded_map_cache = None
        self._raw_map_cache = None

    def _find_path_on_eroded_map(self, source_world, target_world):
        """Find path using A* on our custom eroded map. Returns list of (x,y) world coords or None."""
        trav_map = self.ctx.env.scene._trav_map
        eroded_map = self._get_eroded_map()
        eroded_np = eroded_map.cpu().numpy() if hasattr(eroded_map, 'cpu') else np.array(eroded_map)

        source_map = trav_map.world_to_map(th.tensor([source_world[0], source_world[1]]))
        target_map = trav_map.world_to_map(th.tensor([target_world[0], target_world[1]]))
        sr, sc = int(source_map[0].item()), int(source_map[1].item())
        tr, tc = int(target_map[0].item()), int(target_map[1].item())

        path_rc = _astar_pathfind(eroded_np, (sr, sc), (tr, tc))
        if path_rc is None:
            return None

        # Simplify path - keep only waypoints where direction changes significantly
        simplified = []
        for i, (r, c) in enumerate(path_rc):
            if i == 0 or i == len(path_rc) - 1:
                world_pt = trav_map.map_to_world(th.tensor([r, c]))
                simplified.append((world_pt[0].item(), world_pt[1].item()))
            elif i % 10 == 0:  # Sample every 10th point
                world_pt = trav_map.map_to_world(th.tensor([r, c]))
                simplified.append((world_pt[0].item(), world_pt[1].item()))

        # Always include final point
        if len(simplified) < 2:
            world_pt = trav_map.map_to_world(th.tensor([path_rc[-1][0], path_rc[-1][1]]))
            simplified.append((world_pt[0].item(), world_pt[1].item()))

        return simplified

    def _find_nearest_traversable_point(self, target_pos_2d: th.Tensor, max_search_dist: float = 2.0, use_eroded: bool = True) -> th.Tensor | None:
        trav_map = self.ctx.env.scene._trav_map
        floor_map = self._get_eroded_map() if use_eroded else self._get_raw_map()
        target = th.tensor([target_pos_2d[0].item(), target_pos_2d[1].item()])
        target_map = trav_map.world_to_map(target)
        r, c = int(target_map[0].item()), int(target_map[1].item())

        if 0 <= r < floor_map.shape[0] and 0 <= c < floor_map.shape[1]:
            val = floor_map[r, c].item() if hasattr(floor_map[r, c], 'item') else floor_map[r, c]
            if val == 255:
                return target

        for radius in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            if radius > max_search_dist:
                break
            for angle in range(0, 360, 15):
                rad = math.radians(angle)
                test_point = th.tensor([target[0].item() + radius * math.cos(rad), target[1].item() + radius * math.sin(rad)])
                test_map = trav_map.world_to_map(test_point)
                tr, tc = int(test_map[0].item()), int(test_map[1].item())
                if 0 <= tr < floor_map.shape[0] and 0 <= tc < floor_map.shape[1]:
                    val = floor_map[tr, tc].item() if hasattr(floor_map[tr, tc], 'item') else floor_map[tr, tc]
                    if val == 255:
                        return test_point
        return None

    def find_approach_point(self, object_pos_2d: th.Tensor, approach_dist: float = 0.25, support=None) -> tuple[th.Tensor, bool] | tuple[None, bool]:
        trav_map = self.ctx.env.scene._trav_map
        robot_pos, _ = self.ctx.robot.get_position_orientation()
        obj_x, obj_y = object_pos_2d[0].item(), object_pos_2d[1].item()
        robot_x, robot_y = robot_pos[0].item(), robot_pos[1].item()

        print(f"[Nav] Finding approach: obj=({obj_x:.2f}, {obj_y:.2f}), robot=({robot_x:.2f}, {robot_y:.2f}), dist={approach_dist:.2f}", flush=True)

        angle_to_robot = math.atan2(robot_y - obj_y, robot_x - obj_x)
        MAX_APPROACH_DIST = 0.80
        distances = [approach_dist + d for d in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40] if approach_dist + d <= MAX_APPROACH_DIST]
        angle_offsets = [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 120, -120, 150, -150, 180]

        eroded_map = self._get_eroded_map()
        # Try all angles at each distance - find closest traversable point at any angle
        for offset in angle_offsets:
            for dist in distances:
                angle = angle_to_robot + math.radians(offset)
                test_x, test_y = obj_x + dist * math.cos(angle), obj_y + dist * math.sin(angle)
                test_map = trav_map.world_to_map(th.tensor([test_x, test_y]))
                tr, tc = int(test_map[0].item()), int(test_map[1].item())
                if 0 <= tr < eroded_map.shape[0] and 0 <= tc < eroded_map.shape[1]:
                    val = eroded_map[tr, tc].item() if hasattr(eroded_map[tr, tc], 'item') else eroded_map[tr, tc]
                    if val == 255:
                        print(f"[Nav] Approach point: ({test_x:.2f}, {test_y:.2f}) at {dist:.2f}m, {offset}° from robot dir", flush=True)
                        return th.tensor([test_x, test_y]), True

        raw_map = self._get_raw_map()
        raw_np = raw_map.cpu().numpy() if hasattr(raw_map, 'cpu') else np.array(raw_map)
        robot_map = trav_map.world_to_map(th.tensor([robot_x, robot_y]))
        rr, rc = int(robot_map[0].item()), int(robot_map[1].item())
        for offset in angle_offsets:
            for dist in distances:
                angle = angle_to_robot + math.radians(offset)
                test_x, test_y = obj_x + dist * math.cos(angle), obj_y + dist * math.sin(angle)
                test_map = trav_map.world_to_map(th.tensor([test_x, test_y]))
                tr, tc = int(test_map[0].item()), int(test_map[1].item())
                if 0 <= tr < raw_map.shape[0] and 0 <= tc < raw_map.shape[1]:
                    val = raw_np[tr, tc]
                    if val == 255:
                        # Check connectivity with A*
                        path_rc = _astar_pathfind(raw_np, (rr, rc), (tr, tc), max_iterations=10000)
                        if path_rc is not None:
                            print(f"[Nav] Approach point (raw): ({test_x:.2f}, {test_y:.2f}) at {dist:.2f}m, {offset}°", flush=True)
                            return th.tensor([test_x, test_y]), False

        print(f"[Nav] No approach point found for object at ({obj_x:.2f}, {obj_y:.2f})", flush=True)
        return None, False

    def _rotate_to_face(self, target_x: float, target_y: float, angle_threshold: float, max_steps: int, keep_gripper_closed: bool, mode: float) -> tuple[bool, list, list]:
        observations, actions = [], []

        for step in range(max_steps):
            robot_pos, _ = self.ctx.robot.get_position_orientation()
            dx, dy = target_x - robot_pos[0].item(), target_y - robot_pos[1].item()
            target_yaw = math.atan2(dy, dx)
            yaw_error = self.ctx.normalize_angle(target_yaw - self.ctx.get_robot_yaw())

            if step % 20 == 0:
                print(f"[Rot] step={step}, yaw_err={math.degrees(yaw_error):.1f}°", flush=True)

            if abs(yaw_error) < angle_threshold:
                return True, observations, actions

            action = self.ctx.empty_action(mode=mode)
            action[self.ctx.base_idx[0]] = 0.0
            action[self.ctx.base_idx[1]] = self.ctx.KP_ANGLE_VEL * (1.0 if yaw_error > 0 else -1.0)
            if keep_gripper_closed:
                action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())
            if terminated or truncated:
                return False, observations, actions

        return True, observations, actions

    def _drive_forward(self, target_x: float, target_y: float, dist_threshold: float, max_steps: int, keep_gripper_closed: bool, mode: float) -> tuple[bool, list, list]:
        observations, actions = [], []

        for step in range(max_steps):
            robot_pos, _ = self.ctx.robot.get_position_orientation()
            dx, dy = target_x - robot_pos[0].item(), target_y - robot_pos[1].item()
            dist = math.sqrt(dx * dx + dy * dy)

            if step % 30 == 0:
                print(f"[Drive] step={step}, dist={dist:.2f}m", flush=True)

            if dist < dist_threshold:
                return True, observations, actions

            yaw_error = self.ctx.normalize_angle(math.atan2(dy, dx) - self.ctx.get_robot_yaw())

            if abs(yaw_error) > self.ANGLE_THRESHOLD:
                success, obs, acts = self._rotate_to_face(target_x, target_y, self.ANGLE_THRESHOLD, 50, keep_gripper_closed, mode)
                observations.extend(obs)
                actions.extend(acts)
                if not success:
                    return False, observations, actions
                continue

            action = self.ctx.empty_action(mode=mode)
            action[self.ctx.base_idx[0]] = self.ctx.KP_LIN_VEL
            if abs(yaw_error) > 0.02:
                action[self.ctx.base_idx[1]] = self.ctx.KP_ANGLE_VEL * 0.5 * (1.0 if yaw_error > 0 else -1.0)
            if keep_gripper_closed:
                action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())
            if terminated or truncated:
                return False, observations, actions

        return False, observations, actions

    def navigate_to_target(self, target_pos_2d: th.Tensor, approach_dist: float = 0.5, keep_gripper_closed: bool = False, use_eroded_map: bool = True) -> tuple[bool, list, list]:
        observations, actions = [], []
        max_steps = self.ctx.config.max_navigation_steps

        robot_pos, _ = self.ctx.robot.get_position_orientation()
        source = (robot_pos[0].item(), robot_pos[1].item())
        original_target = (target_pos_2d[0].item(), target_pos_2d[1].item())

        trav_map = self.ctx.env.scene._trav_map
        path = None

        if use_eroded_map:
            eroded_map = self._get_eroded_map()
            source_map = trav_map.world_to_map(th.tensor(source))
            target_map = trav_map.world_to_map(th.tensor(original_target))
            sr, sc = int(source_map[0].item()), int(source_map[1].item())
            tr, tc = int(target_map[0].item()), int(target_map[1].item())

            eroded_source, eroded_target = source, original_target

            if not (0 <= sr < eroded_map.shape[0] and 0 <= sc < eroded_map.shape[1] and eroded_map[sr, sc].item() == 255):
                nearest = self._find_nearest_traversable_point(th.tensor(source), 3.0, True)
                if nearest is not None:
                    eroded_source = (nearest[0].item(), nearest[1].item())

            if not (0 <= tr < eroded_map.shape[0] and 0 <= tc < eroded_map.shape[1] and eroded_map[tr, tc].item() == 255):
                nearest = self._find_nearest_traversable_point(th.tensor(original_target), 3.0, True)
                if nearest is not None:
                    eroded_target = (nearest[0].item(), nearest[1].item())

            # Use custom A* pathfinding on the eroded map
            path = self._find_path_on_eroded_map(eroded_source, eroded_target)
            if path is not None:
                print(f"[Nav] A* path found: {len(path)} waypoints on eroded map", flush=True)

        if path is None:
            # Fallback to raw map with A*
            floor_map = self._get_raw_map()
            raw_np = floor_map.cpu().numpy() if hasattr(floor_map, 'cpu') else np.array(floor_map)
            source_map = trav_map.world_to_map(th.tensor(source))
            target_map = trav_map.world_to_map(th.tensor(original_target))
            sr, sc = int(source_map[0].item()), int(source_map[1].item())
            tr, tc = int(target_map[0].item()), int(target_map[1].item())

            raw_source, raw_target = source, original_target

            if not (0 <= sr < floor_map.shape[0] and 0 <= sc < floor_map.shape[1] and floor_map[sr, sc].item() == 255):
                nearest = self._find_nearest_traversable_point(th.tensor(source), 3.0, False)
                if nearest is not None:
                    raw_source = (nearest[0].item(), nearest[1].item())

            if not (0 <= tr < floor_map.shape[0] and 0 <= tc < floor_map.shape[1] and floor_map[tr, tc].item() == 255):
                nearest = self._find_nearest_traversable_point(th.tensor(original_target), 3.0, False)
                if nearest is not None:
                    raw_target = (nearest[0].item(), nearest[1].item())

            raw_source_map = trav_map.world_to_map(th.tensor([raw_source[0], raw_source[1]]))
            raw_target_map = trav_map.world_to_map(th.tensor([raw_target[0], raw_target[1]]))
            path_rc = _astar_pathfind(raw_np, (int(raw_source_map[0].item()), int(raw_source_map[1].item())), (int(raw_target_map[0].item()), int(raw_target_map[1].item())))
            if path_rc is not None:
                path = []
                for i, (r, c) in enumerate(path_rc):
                    if i == 0 or i == len(path_rc) - 1 or i % 10 == 0:
                        world_pt = trav_map.map_to_world(th.tensor([r, c]))
                        path.append((world_pt[0].item(), world_pt[1].item()))
                print(f"[Nav] A* path found: {len(path)} waypoints on raw map", flush=True)

        if path is None:
            print(f"[Nav] No valid path found from {source} to {original_target}", flush=True)
            return False, observations, actions

        self._save_path_visualization(path, source, original_target)

        step_count = 0
        for waypoint_idx, waypoint in enumerate(path):
            if waypoint is None:
                continue

            robot_pos, _ = self.ctx.robot.get_position_orientation()
            robot_x, robot_y = robot_pos[0].item(), robot_pos[1].item()

            final_dx, final_dy = original_target[0] - robot_x, original_target[1] - robot_y
            dist_to_target = math.sqrt(final_dx * final_dx + final_dy * final_dy)

            if dist_to_target < approach_dist:
                print(f"[Nav] Reached target at step {step_count}", flush=True)
                return True, observations, actions

            wp_x, wp_y = waypoint[0], waypoint[1]
            dx, dy = wp_x - robot_x, wp_y - robot_y
            dist_to_waypoint = math.sqrt(dx * dx + dy * dy)

            if dist_to_waypoint < self.DIST_THRESHOLD and waypoint_idx < len(path) - 1:
                continue

            if step_count % 50 == 0:
                print(f"[Nav] Step {step_count}: wp={waypoint_idx}/{len(path)}, wp_dist={dist_to_waypoint:.2f}, tgt_dist={dist_to_target:.2f}", flush=True)

            yaw_error = self.ctx.normalize_angle(math.atan2(dy, dx) - self.ctx.get_robot_yaw())

            if abs(yaw_error) > self.ANGLE_THRESHOLD:
                success, obs, acts = self._rotate_to_face(wp_x, wp_y, self.ANGLE_THRESHOLD, 100, keep_gripper_closed, MODE_NAVIGATION)
                observations.extend(obs)
                actions.extend(acts)
                step_count += len(acts)
                if not success or step_count >= max_steps:
                    return False, observations, actions

            success, obs, acts = self._drive_forward(wp_x, wp_y, self.DIST_THRESHOLD, 200, keep_gripper_closed, MODE_NAVIGATION)
            observations.extend(obs)
            actions.extend(acts)
            step_count += len(acts)

            if step_count >= max_steps:
                print(f"[Nav] Max steps reached at {step_count}", flush=True)
                break

        robot_pos, _ = self.ctx.robot.get_position_orientation()
        final_dist = math.sqrt((original_target[0] - robot_pos[0].item())**2 + (original_target[1] - robot_pos[1].item())**2)
        success = final_dist < approach_dist + 0.1
        print(f"[Nav] Final dist={final_dist:.2f}, success={success}", flush=True)
        return success, observations, actions

    def orient_to_object(self, object_pos: th.Tensor, keep_gripper_closed: bool = False) -> tuple[bool, list, list]:
        robot_pos, _ = self.ctx.robot.get_position_orientation()
        dx, dy = object_pos[0].item() - robot_pos[0].item(), object_pos[1].item() - robot_pos[1].item()
        target_yaw = math.atan2(dy, dx)
        current_yaw = self.ctx.get_robot_yaw()
        rotation_needed = self.ctx.normalize_angle(target_yaw - current_yaw)
        print(f"[Nav] Orient: current={math.degrees(current_yaw):.0f}°, target={math.degrees(target_yaw):.0f}°, rotation={math.degrees(rotation_needed):.0f}°", flush=True)
        return self._rotate_to_face(object_pos[0].item(), object_pos[1].item(), 0.15, 100, keep_gripper_closed, MODE_NAVIGATION)

    def rotate_to_align_wrist_with_object(self, target_obj) -> tuple[bool, list, list]:
        from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
        grasp_poses = get_grasp_poses_for_object_sticky(target_obj)
        grasp_pos, _ = grasp_poses[0]

        GRIPPER_OFFSET = -math.radians(100)
        robot_pos, _ = self.ctx.robot.get_position_orientation()
        dx, dy = grasp_pos[0].item() - robot_pos[0].item(), grasp_pos[1].item() - robot_pos[1].item()
        angle_to_grasp = math.atan2(dy, dx)

        current_yaw = self.ctx.get_robot_yaw()
        target_yaw = self.ctx.normalize_angle(angle_to_grasp - GRIPPER_OFFSET)
        rotation_needed = self.ctx.normalize_angle(target_yaw - current_yaw)
        print(f"[Nav] Wrist align: current={math.degrees(current_yaw):.0f}°, target={math.degrees(target_yaw):.0f}°, rotation={math.degrees(rotation_needed):.0f}°", flush=True)

        return self._rotate_to_target_yaw(target_yaw, 0.10, 250, MODE_MANIPULATION)

    def rotate_to_align_wrist_with_position(self, target_pos: th.Tensor, keep_gripper_closed: bool = False) -> tuple[bool, list, list]:
        GRIPPER_OFFSET = -math.radians(100)
        robot_pos, _ = self.ctx.robot.get_position_orientation()
        dx, dy = target_pos[0].item() - robot_pos[0].item(), target_pos[1].item() - robot_pos[1].item()
        target_yaw = self.ctx.normalize_angle(math.atan2(dy, dx) - GRIPPER_OFFSET)
        return self._rotate_to_target_yaw(target_yaw, 0.10, 250, MODE_MANIPULATION, keep_gripper_closed)

    def _rotate_to_target_yaw(self, target_yaw: float, angle_threshold: float, max_steps: int, mode: float, keep_gripper_closed: bool = False) -> tuple[bool, list, list]:
        observations, actions = [], []

        for step in range(max_steps):
            yaw_error = self.ctx.normalize_angle(target_yaw - self.ctx.get_robot_yaw())

            if abs(yaw_error) < angle_threshold:
                return True, observations, actions

            action = self.ctx.empty_action(mode=mode)
            action[self.ctx.base_idx[0]] = 0.0
            action[self.ctx.base_idx[1]] = self.ctx.KP_ANGLE_VEL * 0.9 * (1.0 if yaw_error > 0 else -1.0)
            if keep_gripper_closed:
                action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())
            if terminated or truncated:
                return False, observations, actions

        return True, observations, actions
