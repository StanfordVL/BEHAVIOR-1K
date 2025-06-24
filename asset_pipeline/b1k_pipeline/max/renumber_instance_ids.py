import re
import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name

import pymxs

rt = pymxs.runtime

from collections import defaultdict

objs_by_model_and_instance = defaultdict(lambda: defaultdict(set))
for obj in rt.objects:
    m = parse_name(obj.name)
    if m:
        objs_by_model_and_instance[m.group("model_id")][
            int(m.group("instance_id"))
        ].add(obj)

# Now renumber each group of objects.
for m, objs_by_instance in objs_by_model_and_instance.items():
    # Check if the instance keys are complete
    instance_keys = set(objs_by_instance.keys())
    if instance_keys == set(range(len(instance_keys))):
        continue

    print("Found discontinuous model", m)

    # Renumber the instances
    old_instance_ids = sorted(objs_by_instance.keys())
    new_instance_ids = list(range(len(old_instance_ids)))
    for oin, nin in zip(old_instance_ids, new_instance_ids):
        print("Renaming", m, oin, "to", nin)
        find_regex = re.compile(f"{m}-{oin}(-|$)")
        replace_regex = lambda match: f"{m}-{nin}{match.group(1)}"
        for obj in objs_by_instance[oin]:
            assert find_regex.search(obj.name), f"{m}-{oin} not in {obj.name}"
            obj.name = find_regex.sub(replace_regex, obj.name)
