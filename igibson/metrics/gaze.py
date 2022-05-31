from collections import defaultdict

import numpy as np


from igibson.metrics.metric_base import MetricBase


class GazeVizMarker(object):
    """
    Spherical visual marker that can be used to visualize gaze.
    Does not load into PyBullet, so shouldn't affect determinism.
    """

    def __init__(self, s, radius, color=[1, 0, 0]):
        self.s = s
        self.radius = radius
        self.color = color
        self.marker_instance = self.s.load_visual_sphere(self.radius, color=self.color)

    def set_pos(self, pos):
        self.marker_instance.set_position(pos)


class GazeMetric(MetricBase):
    def __init__(self):
        self.target_obj = -1
        self.disallowed_categories = ["walls", "floors", "ceilings"]
        self.gaze_max_distance = 100.0
        self.object_gaze_time_map = defaultdict(int)
        self.task_obj_info = None
        self.obj_info_map = None
        self.gaze_marker = None

    def start_callback(self, env, _):
        self.name_to_category = {obj.name: obj.category for obj in env.scene.objects_by_name.values()}
        self.task_obj_info = {obj.name: obj.category for obj in env.task.object_scope.values()}

        self.gaze_marker = GazeVizMarker(env.simulator, 0.02)

    def step_callback(self, env, log_reader):
        s = env.simulator
        eye_data = log_reader.get_vr_data().query("eye_data")
        if eye_data[0]:
            if self.target_obj in s.scene.objects_by_id:
                s.scene.objects_by_id[self.target_obj].unhighlight()

            origin = eye_data[1]
            direction = eye_data[2]
            intersection = p.rayTest(origin, np.array(origin) + (np.array(direction) * self.gaze_max_distance))
            self.target_obj = intersection[0][0]

            if self.target_obj in s.scene.objects_by_id:
                obj = s.scene.objects_by_id[self.target_obj]
                if obj.category not in self.disallowed_categories:
                    obj.highlight()
                    self.gaze_marker.set_pos(intersection[0][3])
                    self.object_gaze_time_map[obj.name] += 1

    def gather_results(self):
        return {
            "object_gaze_time_map": self.object_gaze_time_map,
            "task_obj_info": self.task_obj_info,
            "object_info_map": self.obj_info_map,
            "name_to_category": self.name_to_category,
        }
