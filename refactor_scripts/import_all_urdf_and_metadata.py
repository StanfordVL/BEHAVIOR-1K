import os
from refactor_scripts.import_urdfs_from_scene import import_obj_urdf
from refactor_scripts.import_metadata import import_obj_metadata

from igibson import app

## For test_states.py
test_objects = [("sink", "sink_1"),
                ("microwave", "7128"),
                ("stove", "101908"),
                ("fridge", "12252"),
                ("bottom_cabinet", "46380"),
                ("apple", "00_0"),
                ("apple", "00_1"),
                ("milk", "milk_000"),
                ("table_knife", "1")]
for category, model in test_objects:
    import_obj_urdf(obj_category=category, obj_model=model, skip_if_exist=False)
    import_obj_metadata(obj_category=category, obj_model=model, name=None)

# for category in os.listdir(os.path.join(igibson.ig_dataset_path, "objects")):
#     for model in os.listdir(os.path.join(igibson.ig_dataset_path, "objects", category)):
#         print(category)
#         print(model)
#         import_obj_urdf(obj_category=category, obj_model=model, skip_if_exist=False)
#         import_obj_metadata(obj_category=category, obj_model=model, name=None)

app.close()

