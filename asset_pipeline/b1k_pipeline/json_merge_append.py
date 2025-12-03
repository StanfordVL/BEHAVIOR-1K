import itertools
import json
import os
import sys


def combine_list(ancestor, ours, theirs):
    assert (
        isinstance(ancestor, list)
        and isinstance(ours, list)
        and isinstance(theirs, list)
    )

    # Check that all the elements in ancestor are in ours and theirs
    for x in ancestor:
        assert x in ours and x in theirs

    # Combine by appending the elements in ours and theirs that are not in ancestor
    for x in theirs + ours:
        if x not in ancestor:
            ancestor.append(x)

    return ancestor


def combine_obj(ancestor, ours, theirs):
    assert (
        isinstance(ancestor, dict)
        and isinstance(ours, dict)
        and isinstance(theirs, dict)
    )

    # Check that all the keys in ancestor are in ours and theirs
    for k in ancestor.keys():
        assert k in ours and k in theirs

    # Check that ours and theirs have the same value for overlapping keys
    collision_keys = (set(ours.keys()) & set(theirs.keys())) - set(ancestor.keys())
    assert not collision_keys, f"Collision keys: {collision_keys}"

    # Combine by appending the elements in ours and theirs that are not in ancestor
    for k, v in itertools.chain(theirs.items(), ours.items()):
        if k not in ancestor:
            ancestor[k] = v

    return ancestor


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python json_merge.py <list|obj> <ancestor_json> <our_json> <their_json>"
        )
        return

    mode, ancestor_fn, ours_fn, theirs_fn = sys.argv[1:]
    assert mode in ["list", "obj"]

    ancestor = [] if mode == "list" else {}
    if os.path.getsize(ancestor_fn) > 0:
        with open(ancestor_fn, "r") as f:
            ancestor = json.load(f)
    with open(ours_fn, "r") as f:
        ours = json.load(f)
    with open(theirs_fn, "r") as f:
        theirs = json.load(f)

    combined = (
        combine_list(ancestor, ours, theirs)
        if mode == "list"
        else combine_obj(ancestor, ours, theirs)
    )

    with open(ours_fn, "w") as f:
        json.dump(combined, f, indent=4)


if __name__ == "__main__":
    main()
