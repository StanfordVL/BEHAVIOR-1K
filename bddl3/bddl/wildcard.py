"""Wildcard expansion for BDDL problem files.

Some BDDL problem files use ``*`` in instance names (e.g.
``cabinet.n.01_*``) to mean "match as many scene objects of this type as
exist in the given room." The number of expanded instances is a property
of the environment, not the task.

:func:`expand_wildcards` takes a raw BDDL string and a scene layout dict,
and returns an expanded BDDL string with wildcards replaced by concrete
numbered instances.
"""

from copy import deepcopy


def expand_wildcards(raw_bddl_str, scene_layout, kb):
    """Expand wildcard instances in a BDDL problem string.

    Args:
        raw_bddl_str: The raw BDDL problem file contents as a string.
        scene_layout: Dict mapping ``room_type -> {category: count}``.
            For example::

                {
                    "kitchen": {"cabinet": 4, "microwave": 1},
                    "bathroom": {"sink": 2},
                }

        kb: A :class:`~bddl.knowledge_base.KnowledgeBase` instance, used
            to resolve synsets to categories.

    Returns:
        str: The expanded BDDL string with wildcards replaced.

    Raises:
        AssertionError: If wildcards appear in goal conditions, if a
            wildcard synset is not a sceneObject, or if the scene doesn't
            have enough matching objects.
    """
    raw_bddl = raw_bddl_str.splitlines(keepends=True)

    # First pass: find wildcard declarations and their metadata
    swap_info = {}
    in_goal = False
    start_init_idx = None
    end_init_idx = None

    for idx, line in enumerate(raw_bddl):
        if "*" in line:
            assert not in_goal, (
                "Found wildcard in BDDL goal conditions, "
                "but only expected in object_scope and init conditions!"
            )

            if "-" in line:
                # Object scope line: "inst_1 inst_* - synset"
                instances, synset = line.strip(" \n\t").split(" - ")
                instances = instances.split(" ")

                synset_obj = kb.get_synset(synset)
                assert synset_obj is not None, f"Synset {synset} not found in knowledge base!"
                assert "sceneObject" in synset_obj.abilities, (
                    f"Wildcard can only be used on sceneObject synsets, "
                    f"but got synset: {synset}"
                )

                # Collect all valid categories from synset subtree
                categories = set()
                for s in [synset_obj] + sorted(
                    synset_obj.descendants, key=lambda x: x.name
                ):
                    if s.is_leaf:
                        for c in s.categories:
                            categories.add(c.name)

                wildcard_instance = instances[-1]
                assert "*" in wildcard_instance, (
                    f"Expected wildcard in final instance: {line}"
                )
                assert wildcard_instance not in swap_info, (
                    f"Duplicate wildcard for synset {synset}!"
                )

                swap_info[wildcard_instance] = {
                    "object_scope_idx": idx,
                    "n_minimum_instances": len(instances) - 1,
                    "categories": categories,
                    "synset": synset,
                }

            else:
                # Init condition line: "(inroom inst_* room_type)"
                tokens = line.strip(" ()\n\t").split(" ")
                assert len(tokens) == 3, (
                    f"Expected 3 tokens for wildcard init condition: {line}"
                )
                assert tokens[0] == "inroom", (
                    f"Only inroom supported for wildcard init condition: {tokens[0]}"
                )
                _, wildcard_instance, room = tokens
                assert wildcard_instance in swap_info, (
                    f"Wildcard instance {wildcard_instance} not declared in object scope!"
                )
                swap_info[wildcard_instance]["room"] = room
                swap_info[wildcard_instance]["init_cond_idx"] = idx

        elif ":init" in line:
            start_init_idx = idx
        elif ":goal" in line:
            end_init_idx = idx
            in_goal = True

    # If no wildcards found, return as-is
    if not swap_info:
        return raw_bddl_str

    # Second pass: expand wildcards using scene_layout
    raw_bddl_init_cond_lines = deepcopy(raw_bddl[start_init_idx:end_init_idx])
    new_init_cond_lines = []

    for line in raw_bddl_init_cond_lines:
        if "*" in line:
            # Find which wildcard instance this line references
            tokens = line.split(" ")
            wildcard_instance = None
            for token in tokens:
                if "*" in token:
                    wildcard_instance = token.strip("()")
                    break
            assert wildcard_instance is not None, (
                f"Expected wildcard token in line: {line}"
            )

            info = swap_info[wildcard_instance]
            n_min = info["n_minimum_instances"]
            synset = info["synset"]
            room_type = info["room"]

            # Count matching objects in the room from scene_layout
            room_categories = scene_layout.get(room_type, {})
            n_valid = sum(
                count for cat, count in room_categories.items()
                if cat in info["categories"]
            )

            assert n_valid >= n_min, (
                f"BDDL requires at least {n_min} instances of synset {synset}, "
                f"but only found {n_valid} in rooms of type {room_type}!"
            )

            # Generate extra instance names
            extra_instances = [
                f"{synset}_{i + 1}" for i in range(n_min, n_valid)
            ]

            # Replace wildcard in object scope line
            obj_scope_idx = info["object_scope_idx"]
            raw_bddl[obj_scope_idx] = raw_bddl[obj_scope_idx].replace(
                wildcard_instance, " ".join(extra_instances)
            )

            # Generate extra init condition lines
            init_cond_line = raw_bddl[info["init_cond_idx"]]
            for extra_inst in extra_instances:
                new_init_cond_lines.append(
                    init_cond_line.replace(wildcard_instance, extra_inst)
                )
        else:
            new_init_cond_lines.append(line)

    # Reassemble the BDDL
    raw_bddl = raw_bddl[:start_init_idx] + new_init_cond_lines + raw_bddl[end_init_idx:]
    return "".join(raw_bddl)
