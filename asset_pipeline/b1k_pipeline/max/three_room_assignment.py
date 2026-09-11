import itertools

import pymxs

rt = pymxs.runtime


def three_room_assignment(assignment_objects, target1, target2, target3):
    # Check that targets are in layers that have single-room names (e.g. no comma)
    layers = []
    for target in [target1, target2, target3]:
        assert (
            "," not in target.layer.name and target.layer.name != "0"
        ), f"Target object {target.name} is in layer {target.layer.name} which does not have a single-room name"
        layers.append(target.layer)

    # Check if the layermanager has the combined layer in any of the permutations
    layers.sort(key=lambda layer: layer.name)
    canonical_combined_layer_name = ",".join(layer.name for layer in layers)
    permutation_names = [",".join(p) for p in itertools.permutations(layer.name for layer in layers)]
    existing_layers = [
        (name, rt.LayerManager.getLayerFromName(name))
        for name in permutation_names
    ]
    existing_layers = [(name, layer) for name, layer in existing_layers if layer]
    assert len(existing_layers) <= 1, (
        f"Multiple combined layers exist: {[name for name, _ in existing_layers]}, which is unexpected"
    )

    # If the layer does not exist, create it using the canonical sorted name.
    if not existing_layers:
        combined_layer = rt.LayerManager.NewLayerFromName(canonical_combined_layer_name)
    else:
        combined_layer = existing_layers[0][1]

    # Move the assignment objects to the combined layer
    for obj in assignment_objects:
        combined_layer.addNode(obj)

    print(f"Moved {len(assignment_objects)} objects to layer {combined_layer.name}")


def select_and_three_room_assignment():
    assignment_objects = list(rt.selection)

    # Ask the user to pick three target objects
    target1 = rt.pickobject()
    target2 = rt.pickobject()
    target3 = rt.pickobject()

    # Merge the collision objects
    three_room_assignment(assignment_objects, target1, target2, target3)


def three_room_assignment_button():
    try:
        select_and_three_room_assignment()
        # rt.messageBox("Success!")
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return


if __name__ == "__main__":
    three_room_assignment_button()
