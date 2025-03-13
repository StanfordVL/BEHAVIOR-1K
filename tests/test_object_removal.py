import pytest
from utils import og_test

import omnigibson as og
from omnigibson.objects import DatasetObject


@og_test
def test_removal_and_readdition(env):
    # Add an apple
    apple = DatasetObject(
        name="apple_unique",
        category="apple",
        model="agveuv",
    )

    # Import it into the scene
    env.scene.add_object(apple)

    # Check that apple exists
    assert env.scene.object_registry("name", "apple_unique") is not None

    # Step a few times
    for _ in range(5):
        og.sim.step()

    # Remove the apple
    env.scene.remove_object(obj=apple)

    # Check that NAMES is the same as before
    assert env.scene.object_registry("name", "apple_unique") is None

    # Importing should work now
    apple2 = DatasetObject(
        name="apple_unique",
        category="apple",
        model="agveuv",
    )
    env.scene.add_object(apple2)
    og.sim.step()

    # Clear the stuff we added
    env.scene.remove_object(apple2)


@og_test
def test_readdition(env):
    # Add an apple
    apple = DatasetObject(
        name="apple_unique",
        category="apple",
        model="agveuv",
    )

    # Import it into the scene
    env.scene.add_object(apple)

    # Check that apple exists
    assert env.scene.object_registry("name", "apple_unique") is not None

    # Step a few times
    for _ in range(5):
        og.sim.step()

    # Creating and importing a new apple should fail
    with pytest.raises(AssertionError):
        apple2 = DatasetObject(
            name="apple_unique",
            category="apple",
            model="agveuv",
        )
        env.scene.add_object(apple2)

    # Check that apple exists
    assert env.scene.object_registry("name", "apple_unique") is not None

    # Clear the stuff we added
    env.scene.remove_object(apple)
