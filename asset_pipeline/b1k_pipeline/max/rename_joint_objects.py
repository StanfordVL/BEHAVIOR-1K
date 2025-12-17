from b1k_pipeline.utils import parse_name
from collections import defaultdict


def fix_group_names():
    groups = defaultdict(list)
    for obj in rt.selection:
        match = parse_name(obj.name)
        if not match:
            continue

        key = f"{match.group('category')}-{match.group('model_id')}-{match.group('instance_id')}"
        groups[key].append(obj)

    for i, (name, group_objs) in enumerate(groups.items()):
        cat, model, _ = name.split("-")
        new_name = f"{cat}-{model}-{i}"
        for x in group_objs:
            x.name = x.name.replace(name, new_name)


if __name__ == "__main__":
    fix_group_names()
