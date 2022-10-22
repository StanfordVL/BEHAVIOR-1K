from omnigibson.utils.python_utils import assert_valid_key

from omnigibson.sensors.sensor_base import BaseSensor, REGISTERED_SENSORS, ALL_SENSOR_MODALITIES
from omnigibson.sensors.scan_sensor import ScanSensor
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.sensors.sensor_noise_base import BaseSensorNoise, REGISTERED_SENSOR_NOISES
from omnigibson.sensors.dropout_sensor_noise import DropoutSensorNoise


# Map sensor prim names to corresponding sensor classes
SENSOR_PRIMS_TO_SENSOR_CLS = {
    "Lidar": ScanSensor,
    "Camera": VisionSensor,
}


def create_sensor(
        sensor_type,
        prim_path,
        name,
        modalities="all",
        sensor_kwargs=None,
        noise_type=None,
        noise_kwargs=None
):
    """
    Create a sensor of type @sensor_type with optional keyword args @sensor_kwargs that should be passed to the
    constructor. Also, additionally send noise of type @noise_type with corresponding keyword args @noise_kwargs
    that should be passed to the noise constructor.

    Args:
        sensor_type (str): Type of sensor to create. Should be either one of SENSOR_PRIM_TO_SENSOR.keys() or
            one of REGISTERED_SENSORS (i.e.: the string name of the desired class to create)
        prim_path (str): prim path of the Sensor to encapsulate or create.
        name (str): Name for the sensor. Names need to be unique per scene.
        modalities (str or list of str): Modality(s) supported by this sensor. Valid options are part of
            sensor.all_modalities. Default is "all", which corresponds to all modalities being used
        sensor_kwargs (dict): Any keyword kwargs to pass to the constructor
        noise_type (str): Type of sensor to create. Should be one of REGISTERED_SENSOR_NOISES
            (i.e.: the string name of the desired class to create)
        noise_kwargs (dict): Any keyword kwargs to pass to the constructor

    Returns:
        BaseSensor: Created sensor with specified params
    """
    # Run basic sanity check
    assert isinstance(sensor_type, str), "Inputted sensor_type must be a string!"

    # Grab the requested sensor class
    if sensor_type in SENSOR_PRIMS_TO_SENSOR_CLS:
        sensor_cls = SENSOR_PRIMS_TO_SENSOR_CLS[sensor_type]
    elif sensor_type in REGISTERED_SENSORS:
        sensor_cls = REGISTERED_SENSORS[sensor_type]
    else:
        # This is an error, we didn't find the requested sensor ):
        raise ValueError(f"No sensor found with corresponding sensor_type: {sensor_type}")

    # Create the noise, and sanity check to make sure it's a valid type
    noise = None
    if noise_type is not None:
        assert_valid_key(key=noise_type, valid_keys=REGISTERED_SENSOR_NOISES, name="sensor noise type")
        noise_kwargs = dict() if noise_kwargs is None else noise_kwargs
        noise = REGISTERED_SENSOR_NOISES[noise_type](**noise_kwargs)

    # Create the sensor
    sensor_kwargs = dict() if sensor_kwargs is None else sensor_kwargs
    sensor = sensor_cls(prim_path=prim_path, name=name, modalities=modalities, noise=noise, **sensor_kwargs)

    return sensor
