import pytest
from utils import og_test

import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.utils.python_utils import NAMES


@og_test
def test_removal_and_readdition(env):
    # Make a copy of NAMES
    initial_names = NAMES.copy()

    # Add an apple
    apple = DatasetObject(
        name="apple_unique",
        category="apple",
        model="agveuv",
    )

    # Import it into the scene
    env.scene.add_object(apple)

    # Check that NAMES has changed
    assert NAMES != initial_names

    # Step a few times
    for _ in range(5):
        og.sim.step()

    # Remove the apple
    og.sim.remove_object(obj=apple)

    # Check that NAMES is the same as before
    extra_names = NAMES - initial_names
    assert len(extra_names) == 0, f"Extra names: {extra_names}"

    # Importing should work now
    apple2 = DatasetObject(
        name="apple_unique",
        category="apple",
        model="agveuv",
    )
    env.scene.add_object(apple2)
    og.sim.step()

    # Clear the stuff we added
    og.sim.remove_object(apple2)


@og_test
def test_readdition(env):
    # Make a copy of NAMES
    initial_names = NAMES.copy()

    # Add an apple
    apple = DatasetObject(
        name="apple_unique",
        category="apple",
        model="agveuv",
    )

    # Import it into the scene
    env.scene.add_object(apple)

    # Check that NAMES has changed
    new_names = NAMES.copy()
    assert new_names != initial_names

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

    # Check that NAMES has not changed
    assert NAMES == new_names

    # Clear the stuff we added
    og.sim.remove_object(apple)
