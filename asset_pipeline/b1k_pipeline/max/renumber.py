import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name

import pymxs

rt = pymxs.runtime

from collections import defaultdict

objs_by_id = defaultdict(set)
for obj in rt.objects:
    m = parse_name(obj.name)
    if m:
        objs_by_id[m.group("model_id")].add(obj)

# Check if model 0 is here. If not, add B
for objs in objs_by_id.values():
    if not any(int(parse_name(obj.name).group("instance_id")) == 0 for obj in objs):
        for obj in objs:
            if not obj.name.startswith("B-"):
                obj.name = "B-" + obj.name

# Now renumber each group of objects.
for m, objs in objs_by_id.items():
    i = 0
    while len(objs) != 0:
        # if it has a 0 instance id then keep it at 0 so that its metalinks remain id 0
        obj = None
        objs_id_0 = [
            ob for ob in objs if int(parse_name(ob.name).group("instance_id")) == 0
        ]
        if objs_id_0:
            obj = objs_id_0[0]
            objs.remove(obj)
        else:
            obj = objs.pop()

        # renumber
        existing_id = parse_name(obj.name).group("instance_id")
        key_from = f"{m}-{existing_id}"
        assert key_from in obj.name, f"{key_from} not in {obj.name}"
        key_to = f"{m}-{i}"
        obj.name = obj.name.replace(key_from, key_to)

        # if there is a duplicate name. e.g. light, metalinks then process it now to keep id same as parent and remove from iterable
        duplicates = set()
        for ob in objs:
            if parse_name(ob.name).group("instance_id") == existing_id:
                duplicates.add(ob)

        for duplicate in duplicates:
            objs.remove(duplicate)
            duplicate.name = duplicate.name.replace(key_from, key_to)

        i += 1
