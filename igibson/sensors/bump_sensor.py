from igibson.sensors.sensor_base import BaseSensor


class BumpSensor(BaseSensor):
    """
    Bump sensor
    """

    def __init__(self, env):
        super(BumpSensor, self).__init__(env)

    def get_obs(self, env):
        """
        Get Bump sensor reading

        :return: Bump sensor reading
        """
        has_collision = float(len(env.current_collisions) > 0)
        return has_collision
