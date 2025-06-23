import numpy as np
import pytest

from gello.dynamixel.driver import FakeDynamixelDriver


@pytest.fixture
def fake_driver():
    return FakeDynamixelDriver(ids=[1, 2])


def test_set_joints(fake_driver):
    fake_driver.set_torque_mode(True)
    fake_driver.set_joints([np.pi / 2, np.pi / 2])
    assert np.allclose(fake_driver.get_joints(), [np.pi / 2, np.pi / 2])


def test_set_joints_wrong_length(fake_driver):
    with pytest.raises(ValueError):
        fake_driver.set_joints([np.pi / 2])


def test_set_joints_torque_disabled(fake_driver):
    with pytest.raises(RuntimeError):
        fake_driver.set_joints([np.pi / 2, np.pi / 2])


def test_torque_enabled(fake_driver):
    assert not fake_driver.torque_enabled()
    fake_driver.set_torque_mode(True)
    assert fake_driver.torque_enabled()


def test_get_joints(fake_driver):
    assert np.allclose(fake_driver.get_joints(), [0, 0])
