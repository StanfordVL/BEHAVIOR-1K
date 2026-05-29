import pymxs

rt = pymxs.runtime


def two_room_assignment(assignment_objects, target1, target2):
    # Check that target1 and target2 are in layers that have single-room names (e.g. no comma)
    layers = []
    for target in [target1, target2]:
        assert (
            "," not in target.layer.name and target.layer.name != "0"
        ), f"Target object {target.name} is in layer {target.layer.name} which does not have a single-room name"
        layers.append(target.layer)

    # Check if the layermanager has the combined layer in either formation
    layers.sort(key=lambda layer: layer.name)
    combined_layer_name1 = f"{layers[0].name},{layers[1].name}"
    combined_layer_name2 = f"{layers[1].name},{layers[0].name}"
    combined_layer_1 = rt.LayerManager.getLayerFromName(combined_layer_name1)
    combined_layer_2 = rt.LayerManager.getLayerFromName(combined_layer_name2)
    assert not (combined_layer_1 and combined_layer_2), f"Both combined layers {combined_layer_name1} and {combined_layer_name2} exist, which is unexpected"

    # If the layer does not exist, create it.
    if not combined_layer_1 and not combined_layer_2:
        combined_layer = rt.LayerManager.NewLayerFromName(combined_layer_name1)
    elif combined_layer_1:
        combined_layer = combined_layer_1
    else:
        combined_layer = combined_layer_2

    # Move the assignment objects to the combined layer
    for obj in assignment_objects:
        combined_layer.addNode(obj)

    print(f"Moved {len(assignment_objects)} objects to layer {combined_layer.name}")


def select_and_two_room_assignment():
    assignment_objects = list(rt.selection)

    # Ask the user to pick two target objects
    target1 = rt.pickobject()
    target2 = rt.pickobject()

    # Merge the collision objects
    two_room_assignment(assignment_objects, target1, target2)


def two_room_assignment_button():
    try:
        select_and_two_room_assignment()
        # rt.messageBox("Success!")
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return


if __name__ == "__main__":
    two_room_assignment_button()
