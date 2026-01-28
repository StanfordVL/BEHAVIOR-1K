import math
import logging

import cv2
import torch as th
import numpy as np

from .robot_context import RobotContext, MODE_NAVIGATION, MODE_MANIPULATION

logger = logging.getLogger(__name__)


class NavigationController:
    """Navigation controller using BEHAVIOR-1K's built-in trav_map functions."""

    ANGLE_THRESHOLD = 0.10
    DIST_THRESHOLD = 0.15

    def __init__(self, ctx: RobotContext):
        self.ctx = ctx
        self._room_ins_id = None  # Current room instance ID filter
        self._eroded_room_map = None  # Pre-computed: trav AND room_mask with 0.35m erosion
        self._component_labels = None  # Connected component labels

    def set_room_filter(self, room_ins_id: int | None):
        """Set room filter and compute connected components.

        Uses scene._seg_map.room_ins_map (already at trav_map resolution).
        Pre-computes eroded map and connected components for efficient lookups.

        Args:
            room_ins_id: Room instance ID from scene._seg_map.room_ins_map (0 = no filter)
        """
        if room_ins_id is None or room_ins_id == 0:
            self._room_ins_id = None
            self._eroded_room_map = None
            self._component_labels = None
            return

        self._room_ins_id = room_ins_id
        scene = self.ctx.env.scene
        trav_map = scene._trav_map
        seg_map = scene._seg_map

        # Get maps (already same resolution)
        floor_np = trav_map.floor_map[0].cpu().numpy()
        room_ins_np = seg_map.room_ins_map.cpu().numpy()

        # Create room mask and AND with trav_map
        room_mask = ((room_ins_np == room_ins_id) * 255).astype(np.uint8)
        room_trav = np.minimum(floor_np, room_mask)

        # Apply 0.3m erosion
        erosion_radius_m = 0.35
        erosion_radius_px = int(math.ceil(erosion_radius_m / trav_map.map_resolution))
        if erosion_radius_px > 0:
            kernel = np.ones((erosion_radius_px, erosion_radius_px), dtype=np.uint8)
            room_trav = cv2.erode(room_trav, kernel)

        self._eroded_room_map = room_trav

        # Compute connected components
        binary = (room_trav == 255).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary)
        self._component_labels = labels

        print(f"[Nav] Room filter: ins_id={room_ins_id}, {num_labels - 1} components", flush=True)

    def _get_component_at(self, x: float, y: float) -> int:
        """Get connected component label at world position."""
        if self._component_labels is None:
            return 0
        trav_map = self.ctx.env.scene._trav_map
        map_pos = trav_map.world_to_map(th.tensor([x, y]))
        r, c = int(map_pos[0].item()), int(map_pos[1].item())
        h, w = self._component_labels.shape
        if 0 <= r < h and 0 <= c < w:
            return int(self._component_labels[r, c])
        return 0

    def _is_in_robot_component(self, x: float, y: float) -> bool:
        """Check if position is in same connected component as robot."""
        if self._component_labels is None:
            # No room filter - fall back to simple traversability
            return self._is_traversable(x, y)

        robot_pos, _ = self.ctx.robot.get_position_orientation()
        robot_comp = self._get_component_at(robot_pos[0].item(), robot_pos[1].item())
        if robot_comp == 0:
            return False
        point_comp = self._get_component_at(x, y)
        return point_comp == robot_comp

    def _get_path(self, source_2d: tuple, target_2d: tuple) -> tuple[th.Tensor | None, float]:
        """Get shortest path using A* on our pre-computed eroded map.

        Uses distance-to-obstacle cost to prefer paths away from objects.
        Returns: (path_tensor, distance) or (None, 0) if no path found
        """
        import heapq

        trav_map = self.ctx.env.scene._trav_map

        # Use pre-computed eroded map
        if self._eroded_room_map is not None:
            floor_np = self._eroded_room_map
        else:
            floor_np = trav_map.floor_map[0].cpu().numpy()
            erosion_radius_m = 0.35
            erosion_radius_px = int(math.ceil(erosion_radius_m / trav_map.map_resolution))
            if erosion_radius_px > 0:
                floor_np = cv2.erode(floor_np, np.ones((erosion_radius_px, erosion_radius_px), dtype=np.uint8))

        # Compute distance transform (distance to nearest obstacle)
        # Higher values = further from obstacles = safer
        binary_map = (floor_np == 255).astype(np.uint8)
        dist_transform = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
        max_dist = dist_transform.max() + 1e-6  # Avoid division by zero

        # Convert world to map coordinates
        src_map = trav_map.world_to_map(th.tensor([source_2d[0], source_2d[1]]))
        tgt_map = trav_map.world_to_map(th.tensor([target_2d[0], target_2d[1]]))
        start = (int(src_map[0].item()), int(src_map[1].item()))
        goal = (int(tgt_map[0].item()), int(tgt_map[1].item()))

        h, w = floor_np.shape
        src_val = floor_np[start[0], start[1]] if 0 <= start[0] < h and 0 <= start[1] < w else -1
        tgt_val = floor_np[goal[0], goal[1]] if 0 <= goal[0] < h and 0 <= goal[1] < w else -1
        print(f"[Nav] Path: src {start}={src_val}, tgt {goal}={tgt_val}", flush=True)

        # A* search with obstacle avoidance cost
        def heuristic(node):
            return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

        def is_valid(cell):
            r, c = cell
            return 0 <= r < h and 0 <= c < w and floor_np[r, c] == 255

        def obstacle_cost(cell):
            """Higher cost for cells closer to obstacles"""
            r, c = cell
            dist_to_obstacle = dist_transform[r, c]
            # Inverse: closer to obstacle = higher cost
            # Scale factor controls how much to penalize being near obstacles
            return 2.0 * (1.0 - dist_to_obstacle / max_dist)

        if not is_valid(start) or not is_valid(goal):
            print(f"[Nav] Start or goal not traversable", flush=True)
            return None, 0

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.insert(0, current)
                    current = came_from[current]
                path.insert(0, start)

                # Convert to world coordinates and subsample
                path_world = []
                for i, (r, c) in enumerate(path):
                    if i % 3 == 0 or i == len(path) - 1:  # Subsample
                        world_pos = trav_map.map_to_world(th.tensor([r, c]))
                        path_world.append([world_pos[0].item(), world_pos[1].item()])

                dist = len(path) * trav_map.map_resolution
                print(f"[Nav] Path found: {len(path)} steps, {len(path_world)} waypoints", flush=True)
                return th.tensor(path_world), dist

            # 8-connected neighbors
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if not is_valid(neighbor) or neighbor in visited:
                    continue
                # Base movement cost + obstacle proximity penalty
                move_cost = 1.414 if dr != 0 and dc != 0 else 1.0
                total_cost = move_cost + obstacle_cost(neighbor)
                tentative_g = g_score[current] + total_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))

        print(f"[Nav] No path found (A* exhausted)", flush=True)
        return None, 0

    def _is_traversable(self, x: float, y: float, debug: bool = False) -> bool:
        """Check if a world position is traversable (using pre-computed eroded room map)."""
        if self._eroded_room_map is not None:
            floor_np = self._eroded_room_map
        else:
            # Fallback: compute on-the-fly
            trav_map = self.ctx.env.scene._trav_map
            floor_np = trav_map.floor_map[0].cpu().numpy()
            erosion_radius_m = 0.35
            erosion_radius_px = int(math.ceil(erosion_radius_m / trav_map.map_resolution))
            if erosion_radius_px > 0:
                floor_np = cv2.erode(floor_np, np.ones((erosion_radius_px, erosion_radius_px), dtype=np.uint8))

        trav_map = self.ctx.env.scene._trav_map
        map_pos = trav_map.world_to_map(th.tensor([x, y]))
        r, c = int(map_pos[0].item()), int(map_pos[1].item())

        if 0 <= r < floor_np.shape[0] and 0 <= c < floor_np.shape[1]:
            val = floor_np[r, c]
            if debug:
                comp = self._get_component_at(x, y) if self._component_labels is not None else -1
                print(f"[Nav] _is_traversable ({x:.2f}, {y:.2f}) -> ({r},{c}), val={val}, comp={comp}", flush=True)
            return val == 255
        if debug:
            print(f"[Nav] _is_traversable ({x:.2f}, {y:.2f}) -> out of bounds", flush=True)
        return False

    def find_approach_point(self, object_pos_2d: th.Tensor, approach_dist: float = 0.25,
                             support=None) -> tuple[th.Tensor | None, bool]:
        """Find traversable approach point near object using BFS.

        Uses pixel-based BFS on eroded map (accounts for robot size).
        Guarantees finding a valid point if one exists in robot's component.

        Returns: (approach_point, use_eroded) - use_eroded is always True
        """
        obj_x, obj_y = object_pos_2d[0].item(), object_pos_2d[1].item()
        robot_pos, _ = self.ctx.robot.get_position_orientation()
        robot_x, robot_y = robot_pos[0].item(), robot_pos[1].item()

        trav_map = self.ctx.env.scene._trav_map
        resolution = trav_map.map_resolution

        # Get robot's connected component
        robot_comp = self._get_component_at(robot_x, robot_y)
        print(f"[Nav] Finding approach: obj=({obj_x:.2f},{obj_y:.2f}), robot comp={robot_comp}", flush=True)

        if robot_comp == 0 and self._component_labels is not None:
            print(f"[Nav] WARNING: Robot not in any component", flush=True)
            return None, True

        # Get eroded map (accounts for robot size)
        if self._eroded_room_map is not None:
            floor_map = self._eroded_room_map
        else:
            floor_map = trav_map.floor_map[0].cpu().numpy()

        # Convert object position to map coordinates
        obj_map = trav_map.world_to_map(th.tensor([obj_x, obj_y]))
        obj_r, obj_c = int(obj_map[0].item()), int(obj_map[1].item())

        # Distance thresholds in pixels
        min_dist_px = int(approach_dist / resolution)
        max_dist_px = int((approach_dist + 0.3) / resolution)  # Search up to +0.3m (prefer closer)

        h, w = floor_map.shape

        # BFS from object position outward
        from collections import deque
        visited = set()
        queue = deque([(obj_r, obj_c, 0)])  # (row, col, distance)
        visited.add((obj_r, obj_c))

        candidates = []  # (world_x, world_y, dist_to_robot)

        while queue:
            r, c, dist = queue.popleft()

            # Stop if beyond max search distance
            if dist > max_dist_px:
                continue

            # Check if this pixel is valid approach point
            if dist >= min_dist_px:
                if 0 <= r < h and 0 <= c < w and floor_map[r, c] == 255:
                    # Check if in robot's component
                    world_pos = trav_map.map_to_world(th.tensor([r, c]))
                    wx, wy = world_pos[0].item(), world_pos[1].item()

                    if self._component_labels is not None:
                        point_comp = self._get_component_at(wx, wy)
                        if point_comp == robot_comp:
                            # Calculate distance to object (prefer closer to object)
                            dist_to_obj = math.sqrt((wx - obj_x)**2 + (wy - obj_y)**2)
                            candidates.append((wx, wy, dist_to_obj))
                    else:
                        dist_to_obj = math.sqrt((wx - obj_x)**2 + (wy - obj_y)**2)
                        candidates.append((wx, wy, dist_to_obj))

            # Expand to 8-connected neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and 0 <= nr < h and 0 <= nc < w:
                    visited.add((nr, nc))
                    new_dist = dist + (1.414 if dr != 0 and dc != 0 else 1)
                    queue.append((nr, nc, int(new_dist)))

        if candidates:
            # Sort by distance to object, pick closest to object
            candidates.sort(key=lambda x: x[2])
            best_x, best_y, best_dist = candidates[0]
            print(f"[Nav] BFS found {len(candidates)} candidates, best: ({best_x:.2f},{best_y:.2f}), dist_to_obj={best_dist:.2f}m", flush=True)
            return th.tensor([best_x, best_y]), True

        print(f"[Nav] BFS found no approach point for ({obj_x:.2f}, {obj_y:.2f})", flush=True)
        return None, True

    def _rotate_to_face(self, target_x: float, target_y: float, angle_threshold: float,
                        max_steps: int, keep_gripper_closed: bool, mode: float) -> tuple[bool, list, list]:
        """Rotate in place to face target point."""
        observations, actions = [], []
        MIN_ANG_VEL = 2.5
        last_yaw_error = None
        stuck_count = 0

        for step in range(max_steps):
            robot_pos, _ = self.ctx.robot.get_position_orientation()
            dx = target_x - robot_pos[0].item()
            dy = target_y - robot_pos[1].item()
            target_yaw = math.atan2(dy, dx)
            yaw_error = self.ctx.normalize_angle(target_yaw - self.ctx.get_robot_yaw())

            if step % 20 == 0:
                print(f"[Rot] step={step}, yaw_err={math.degrees(yaw_error):.1f}°", flush=True)

            if abs(yaw_error) < angle_threshold:
                return True, observations, actions

            if last_yaw_error is not None and abs(abs(yaw_error) - abs(last_yaw_error)) < 0.01:
                stuck_count += 1
                if stuck_count >= 120:
                    print(f"[Rot] STUCK at yaw_err={math.degrees(yaw_error):.1f}°", flush=True)
                    return False, observations, actions
            else:
                stuck_count = 0
            last_yaw_error = yaw_error

            action = self.ctx.empty_action(mode=mode)
            action[self.ctx.base_idx[0]] = 0.0
            proportional = self.ctx.KP_ANGLE_VEL * abs(yaw_error)
            ang_vel = min(MIN_ANG_VEL + proportional, self.ctx.KP_ANGLE_VEL)
            if yaw_error < 0:
                ang_vel = -ang_vel
            action[self.ctx.base_idx[1]] = ang_vel

            if keep_gripper_closed:
                action[self.ctx.gripper_idx] = -1.0

            obs, _, _, _, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

        return False, observations, actions

    def _drive_to_point(self, target_x: float, target_y: float, dist_threshold: float,
                        max_steps: int, keep_gripper_closed: bool, mode: float) -> tuple[bool, list, list]:
        """Drive forward to reach target point."""
        observations, actions = [], []
        last_dist = None
        stuck_count = 0

        for step in range(max_steps):
            robot_pos, _ = self.ctx.robot.get_position_orientation()
            dx = target_x - robot_pos[0].item()
            dy = target_y - robot_pos[1].item()
            dist = math.sqrt(dx * dx + dy * dy)

            if step % 30 == 0:
                print(f"[Drive] step={step}, dist={dist:.2f}m", flush=True)

            if dist < dist_threshold:
                return True, observations, actions

            if last_dist is not None and abs(dist - last_dist) < 0.005:
                stuck_count += 1
                if stuck_count >= 30:
                    print(f"[Drive] STUCK at dist={dist:.2f}m", flush=True)
                    return False, observations, actions
            else:
                stuck_count = 0
            last_dist = dist

            target_yaw = math.atan2(dy, dx)
            yaw_error = self.ctx.normalize_angle(target_yaw - self.ctx.get_robot_yaw())

            action = self.ctx.empty_action(mode=mode)
            action[self.ctx.base_idx[0]] = self.ctx.KP_LIN_VEL
            action[self.ctx.base_idx[1]] = self.ctx.KP_ANGLE_VEL * yaw_error

            if keep_gripper_closed:
                action[self.ctx.gripper_idx] = -1.0

            obs, _, _, _, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

        return False, observations, actions

    def orient_to_object(self, obj_pos: th.Tensor) -> tuple[bool, list, list]:
        """Rotate to face the object."""
        return self._rotate_to_face(obj_pos[0].item(), obj_pos[1].item(),
                                     self.ANGLE_THRESHOLD, max_steps=200,
                                     keep_gripper_closed=False, mode=MODE_NAVIGATION)

    def rotate_to_align_wrist_with_object(self, target_obj) -> tuple[bool, list, list]:
        """Rotate so arm points toward object (arm is 90° offset from robot heading)."""
        obj_pos, _ = target_obj.get_position_orientation()
        return self._rotate_for_arm_alignment(obj_pos[0].item(), obj_pos[1].item(),
                                               keep_gripper_closed=False)

    def rotate_to_align_wrist_with_position(self, target_pos: th.Tensor,
                                             keep_gripper_closed: bool = False) -> tuple[bool, list, list]:
        """Rotate so arm points toward target position."""
        return self._rotate_for_arm_alignment(target_pos[0].item(), target_pos[1].item(),
                                               keep_gripper_closed=keep_gripper_closed)

    def _rotate_for_arm_alignment(self, target_x: float, target_y: float,
                                   keep_gripper_closed: bool) -> tuple[bool, list, list]:
        """Rotate robot so its arm (which is 90° offset) points toward target."""
        observations, actions = [], []
        mode = MODE_MANIPULATION if keep_gripper_closed else MODE_NAVIGATION

        for step in range(300):
            robot_pos, _ = self.ctx.robot.get_position_orientation()
            dx = target_x - robot_pos[0].item()
            dy = target_y - robot_pos[1].item()

            # Arm direction is robot_yaw - 90° (arm on right side of Stretch)
            desired_arm_angle = math.atan2(dy, dx)
            desired_robot_yaw = desired_arm_angle + math.pi / 2
            yaw_error = self.ctx.normalize_angle(desired_robot_yaw - self.ctx.get_robot_yaw())

            if step % 30 == 0:
                print(f"[ArmAlign] step={step}, yaw_err={math.degrees(yaw_error):.1f}°", flush=True)

            if abs(yaw_error) < 0.05:
                return True, observations, actions

            action = self.ctx.empty_action(mode=mode)
            action[self.ctx.base_idx[0]] = 0.0
            ang_vel = 2.0 + self.ctx.KP_ANGLE_VEL * abs(yaw_error)
            ang_vel = min(ang_vel, self.ctx.KP_ANGLE_VEL)
            if yaw_error < 0:
                ang_vel = -ang_vel
            action[self.ctx.base_idx[1]] = ang_vel

            if keep_gripper_closed:
                action[self.ctx.gripper_idx] = -1.0

            obs, _, _, _, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

        return False, observations, actions

    def navigate_to_target(self, target_pos_2d: th.Tensor, approach_dist: float = 0.5,
                           keep_gripper_closed: bool = False,
                           use_eroded_map: bool = True) -> tuple[bool, list, list]:
        """Navigate to target position using BEHAVIOR-1K path planning."""
        robot_pos, _ = self.ctx.robot.get_position_orientation()
        source = (robot_pos[0].item(), robot_pos[1].item())
        target = (target_pos_2d[0].item(), target_pos_2d[1].item())

        print(f"[Nav] Navigate: ({source[0]:.2f}, {source[1]:.2f}) -> ({target[0]:.2f}, {target[1]:.2f})", flush=True)

        # Get path using BEHAVIOR-1K
        path, dist = self._get_path(source, target)
        if path is None:
            print(f"[Nav] No path found", flush=True)
            return False, [], []

        print(f"[Nav] Path found: {len(path)} waypoints, dist={dist:.2f}m", flush=True)

        all_obs, all_acts = [], []
        mode = MODE_NAVIGATION

        # Follow waypoints
        for i, waypoint in enumerate(path):
            if waypoint is None:
                continue
            wp_x, wp_y = waypoint[0].item(), waypoint[1].item()

            # Check if close enough to final target
            robot_pos, _ = self.ctx.robot.get_position_orientation()
            dist_to_target = math.sqrt((target[0] - robot_pos[0].item())**2 +
                                       (target[1] - robot_pos[1].item())**2)
            if dist_to_target < approach_dist:
                print(f"[Nav] Close enough to target: {dist_to_target:.2f}m", flush=True)
                break

            # Rotate to face waypoint
            success, obs, acts = self._rotate_to_face(wp_x, wp_y, self.ANGLE_THRESHOLD,
                                                       max_steps=300, keep_gripper_closed=keep_gripper_closed,
                                                       mode=mode)
            all_obs.extend(obs)
            all_acts.extend(acts)

            if not success:
                continue

            # Drive to waypoint
            success, obs, acts = self._drive_to_point(wp_x, wp_y, self.DIST_THRESHOLD,
                                                       max_steps=200, keep_gripper_closed=keep_gripper_closed,
                                                       mode=mode)
            all_obs.extend(obs)
            all_acts.extend(acts)

        # Final check
        robot_pos, _ = self.ctx.robot.get_position_orientation()
        final_dist = math.sqrt((target[0] - robot_pos[0].item())**2 +
                               (target[1] - robot_pos[1].item())**2)
        success = final_dist < approach_dist * 1.5

        print(f"[Nav] Done: final_dist={final_dist:.2f}m, success={success}", flush=True)
        return success, all_obs, all_acts
