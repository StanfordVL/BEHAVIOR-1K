from b1k_pipeline.utils import parse_name
from collections import defaultdict


def group_selection():
    groups = defaultdict(list)
    for obj in rt.selection:
        match = parse_name(obj.name)
        if not match:
            continue

        key = f"{match.group('category')}-{match.group('model_id')}-{match.group('instance_id')}"
        groups[key].append(obj)

    for name, group_objs in groups.items():
        if len(group_objs) <= 1:
            continue

        # Pick base link
        base_links = [
            x
            for x in group_objs
            if parse_name(x.name).group("link_name") in (None, "base_link")
        ]
        assert len(base_links) == 1, base_links
        (base_link,) = base_links
        others = [x for x in group_objs if x != base_link]

        # Create group
        group = rt.group([base_link], name=name)
        for x in others:
            x.parent = group

        assert np.allclose(group.position, base_link.position)


if __name__ == "__main__":
    group_selection()
