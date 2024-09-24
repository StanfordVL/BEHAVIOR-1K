import torch as th

from omnigibson.sensors.sensor_noise_base import BaseSensorNoise


class DropoutSensorNoise(BaseSensorNoise):
    """
    Naive dropout sensor noise model

    Args:
        dropout_prob (float): Value in [0.0, 1.0] representing fraction of a single observation to be replaced
            with @dropout_value
        dropout_value (float): Value in [0.0, 1.0] to replace observations selected to be dropped out
        enabled (bool): Whether this sensor should be enabled by default
    """

    def __init__(
        self,
        dropout_prob=0.05,
        dropout_value=1.0,
        enabled=True,
    ):
        # Store args, and make sure values are in acceptable range
        for name, val in zip(("dropout_prob", "dropout_value"), (dropout_prob, dropout_value)):
            assert 0.0 <= val <= 1.0, f"{name} should be in range [0.0, 1.0], got: {val}"
        self._dropout_prob = dropout_prob
        self._dropout_value = dropout_value

        # Run super method
        super().__init__(enabled=enabled)

    def _corrupt(self, obs):
        # If our noise rate is 0, we just return the obs
        if self._dropout_prob == 0.0:
            return obs

        # Corrupt with randomized dropout
        valid_mask = th.bernoulli(th.full(obs.shape, 1.0 - self._dropout_prob)).to(th.int64)
        obs[valid_mask == 0] = self._dropout_value
        return obs

    @property
    def dropout_prob(self):
        """
        Returns:
            float: Value in [0.0, 1.0] representing fraction of a single observation to be replaced
                with self.dropout_value
        """
        return self._dropout_prob

    @dropout_prob.setter
    def dropout_prob(self, p):
        """
        Set the dropout probability for this noise model.

        Args:
            p (float): Value in [0.0, 1.0] representing fraction of a single observation to be replaced
                with self.dropout_value
        """
        assert 0.0 <= p <= 1.0, f"dropout_prob should be in range [0.0, 1.0], got: {p}"
        self._dropout_prob = p

    @property
    def dropout_value(self):
        """
        Returns:
            float: Value in [0.0, 1.0] to replace observations selected to be dropped out
        """
        return self._dropout_value

    @dropout_value.setter
    def dropout_value(self, val):
        """
        Set the dropout value for this noise model.

        Args:
            val (float): Value in [0.0, 1.0] to replace observations selected to be dropped out
        """
        assert 0.0 <= val <= 1.0, f"dropout_value should be in range [0.0, 1.0], got: {val}"
        self._dropout_value = val
