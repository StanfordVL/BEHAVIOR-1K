import json

import numpy as np

from igibson.scene_graphs.enum_obj_states import BinaryStatesEnum, UnaryStatesEnum
from igibson.scene_graphs.graph_builder import SceneGraphBuilder


class SceneGraphExporter(SceneGraphBuilder):
    def __init__(self, h5py_file, num_frames_to_save=None, full_obs=False, **kwargs):
        """
        @param h5py_file: file to store exported scene graph
        @param num_frames_to_save: total number of equidistant frames to save. None if you want to set all frames.
        """
        if num_frames_to_save is None:
            assert full_obs, "You can only use full observability mode if you're subsampling frames."

        super(SceneGraphExporter, self).__init__(only_true=True, full_obs=full_obs, **kwargs)
        self.h5py_file = h5py_file

        self.num_frames_to_save = num_frames_to_save
        self.frame_idxes_to_save = None

    def start(self, activity, log_reader):
        super(SceneGraphExporter, self).start(activity, log_reader)

        scene = activity.simulator.scene
        objs = set(scene.get_objects()) | set(activity.object_scope.values())
        self.num_obj = len(objs)
        self.dim = 34  # 34: dimension of all relevant information
        # create a dictionary that maps objs to ids
        self.obj_to_id = {obj: i for i, obj in enumerate(objs)}

        self.h5py_file.attrs["id_to_category"] = json.dumps(
            {self.obj_to_id[obj]: obj.category for obj in self.obj_to_id}
        )
        self.h5py_file.attrs["id_to_name"] = json.dumps({self.obj_to_id[obj]: obj.name for obj in self.obj_to_id})

        if self.num_frames_to_save is not None:
            key_presses = log_reader.hf["agent_actions"]["vr_robot"][:, [19, 27]]
            assert len(key_presses) > 0, "No key press found: too few frames."
            any_key_press = np.max(key_presses[200:], axis=1)
            first_frame = np.argmax(any_key_press) + 200
            assert np.any(key_presses[first_frame] == 1), "No key press found: robot never activated."
            self.frame_idxes_to_save = set(
                np.linspace(
                    first_frame, log_reader.total_frame_num - 1, self.num_frames_to_save, endpoint=True, dtype=int
                )
            )

    def step(self, activity, log_reader):
        frame_count = activity.simulator.frame_count
        if self.num_frames_to_save is not None and frame_count not in self.frame_idxes_to_save:
            return

        super(SceneGraphExporter, self).step(activity, log_reader)
        print("Frame: %s" % frame_count)

        nodes_t = np.zeros((self.num_obj, self.dim), dtype=np.float32)
        for obj in self.G.nodes:
            states = self.G.nodes[obj]["states"]
            unary_states = np.full(len(UnaryStatesEnum), -1, dtype=np.int8)
            for state_name in states:
                unary_states[UnaryStatesEnum[state_name].value] = states[state_name]

            nodes_t[self.obj_to_id[obj]] = (
                [item for tupl in self.G.nodes[obj]["pose"] for item in tupl]
                + [item for tupl in self.G.nodes[obj]["bbox_pose"] for item in tupl]
                + [item for item in self.G.nodes[obj]["bbox_extent"]]
                + list(unary_states)
            )

        edges_t = []
        for edge in self.G.edges:
            from_obj_id, to_obj_id = self.obj_to_id[edge[0]], self.obj_to_id[edge[1]]
            edges_t.append([BinaryStatesEnum[edge[2]].value, from_obj_id, to_obj_id])

        fc = str(frame_count)
        self.h5py_file.create_dataset("/nodes/" + fc, data=nodes_t, compression="gzip")
        self.h5py_file.create_dataset("/edges/" + fc, data=edges_t, compression="gzip")

        # profiler.stop()

        # html = profiler.output_html()
        # html_path = "profile.html"
        # with open(html_path, "w") as f:
        #     f.write(html)

        # import pdb
        # pdb.set_trace()
