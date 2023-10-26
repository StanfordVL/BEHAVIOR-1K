import json

import numpy as np

from omnigibson.scene_graphs.graph_builder import SceneGraphBuilder


class SceneGraphExporter(SceneGraphBuilder):
    def __init__(self, h5py_file, **kwargs):
        """
        A class that exports a scene graph into a hdf5 file while generating it.

        Params:
            h5py_file: an h5py file object to write to
        """
        super(SceneGraphExporter, self).__init__(only_true=True, **kwargs)
        self.h5py_file = h5py_file

    def start(self, activity, log_reader):
        super(SceneGraphExporter, self).start(activity, log_reader)

        scene = activity.simulator.scene
        objs = set(scene.objects()) | set(activity.object_scope.values())
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
        super(SceneGraphExporter, self).step(activity, log_reader)

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
