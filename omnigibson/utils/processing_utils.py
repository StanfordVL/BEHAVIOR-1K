import torch as th

from omnigibson.utils.python_utils import Serializable


class Filter(Serializable):
    """
    A base class for filtering a noisy data stream in an online fashion.
    """

    def estimate(self, observation):
        """
        Takes an observation and returns a de-noised estimate.

        Args:
            observation (n-array): A current observation.

        Returns:
            n-array: De-noised estimate.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets this filter. Default is no-op.
        """
        pass

    def _dump_state(self):
        # Default is no state (empty dict)
        return dict()

    def _load_state(self, state):
        # Default is no state (empty dict), so this is a no-op
        pass

    def serialize(self, state):
        # Default is no state, so do nothing
        return th.empty(0, dtype=th.float32)

    def deserialize(self, state):
        # Default is no state, so do nothing
        return dict(), 0


class MovingAverageFilter(Filter):
    """
    This class uses a moving average to de-noise a noisy data stream in an online fashion.
    This is a FIR filter.
    """

    def __init__(self, obs_dim, filter_width):
        """

        Args:
            obs_dim (int): The dimension of the points to filter.
            filter_width (int): The number of past samples to take the moving average over.
        """
        self.obs_dim = obs_dim
        assert filter_width > 0, f"MovingAverageFilter must have a non-zero size! Got: {filter_width}"
        self.filter_width = filter_width
        self.past_samples = th.zeros((filter_width, obs_dim))
        self.current_idx = 0
        self.fully_filled = False  # Whether the entire filter buffer is filled or not

        super().__init__()

    def estimate(self, observation):
        """
        Do an online hold for state estimation given a recent observation.

        Args:
            observation (n-array): New observation to hold internal estimate of state.

        Returns:
            n-array: New estimate of state.
        """
        # Write the newest observation at the appropriate index
        self.past_samples[self.current_idx, :] = observation

        # Compute value based on whether we're fully filled or not
        if not self.fully_filled:
            val = self.past_samples[: self.current_idx + 1, :].mean(dim=0)
            # Denote that we're fully filled if we're at the end of the buffer
            if self.current_idx == self.filter_width - 1:
                self.fully_filled = True
        else:
            val = self.past_samples.mean(dim=0)

        # Increment the index to write the next sample to
        self.current_idx = (self.current_idx + 1) % self.filter_width

        return val

    def reset(self):
        # Clear internal state
        self.past_samples *= 0.0
        self.current_idx = 0
        self.fully_filled = False

    def _dump_state(self):
        # Run super init first
        state = super()._dump_state()

        # Add info from this filter
        state["past_samples"] = self.past_samples
        state["current_idx"] = self.current_idx
        state["fully_filled"] = self.fully_filled

        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # Load relevant info for this filter
        self.past_samples = state["past_samples"]
        self.current_idx = state["current_idx"]
        self.fully_filled = state["fully_filled"]

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        # Serialize state for this filter
        return th.cat(
            [
                state_flat,
                state["past_samples"].flatten(),
                th.tensor([state["current_idx"]]),
                th.tensor([state["fully_filled"]]),
            ]
        )

    def deserialize(self, state):
        # Run super first
        state_dict, idx = super().deserialize(state=state)

        # Deserialize state for this filter
        samples_len = self.filter_width * self.obs_dim
        state_dict["past_samples"] = state[idx : idx + samples_len].reshape(self.filter_width, self.obs_dim)
        state_dict["current_idx"] = int(state[idx + samples_len])
        state_dict["fully_filled"] = bool(state[idx + samples_len + 1])

        return state_dict, idx + samples_len + 2


class ExponentialAverageFilter(Filter):
    """
    This class uses an exponential average of the form y_n = alpha * x_n + (1 - alpha) * y_{n - 1}.
    This is an IIR filter.
    """

    def __init__(self, obs_dim, alpha=0.9):
        """

        Args:
            obs_dim (int): The dimension of the points to filter.
            alpha (float): The relative weighting of new samples relative to older samples
        """
        self.obs_dim = obs_dim
        self.avg = th.zeros(obs_dim)
        self.num_samples = 0
        self.alpha = alpha

        super().__init__()

    def estimate(self, observation):
        """
        Do an online hold for state estimation given a recent observation.

        Args:
            observation (n-array): New observation to hold internal estimate of state.

        Returns:
            n-array: New estimate of state.
        """
        self.avg = self.alpha * observation + (1.0 - self.alpha) * self.avg
        self.num_samples += 1

        return th.tensor(self.avg)

    def reset(self):
        # Clear internal state
        self.avg *= 0.0
        self.num_samples = 0

    def _dump_state(self):
        # Run super init first
        state = super()._dump_state()

        # Add info from this filter
        state["avg"] = th.tensor(self.avg)
        state["num_samples"] = self.num_samples

        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # Load relevant info for this filter
        self.avg = th.tensor(state["avg"])
        self.num_samples = state["num_samples"]

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        # Serialize state for this filter
        return th.cat(
            [
                state_flat,
                state["avg"],
                [state["num_samples"]],
            ]
        )

    def deserialize(self, state):
        # Run super first
        state_dict, idx = super().deserialize(state=state)

        # Deserialize state for this filter
        state_dict["avg"] = state[idx : idx + self.obs_dim]
        state_dict["num_samples"] = int(state[idx + self.obs_dim])

        return state_dict, idx + self.obs_dim + 1


class Subsampler:
    """
    A base class for subsampling a data stream in an online fashion.
    """

    def subsample(self, observation):
        """
        Takes an observation and returns the observation, or None, which
        corresponds to deleting the observation.

        Args:
            observation (n-array): A current observation.

        Returns:
            None or n-array: No observation if subsampled, otherwise the observation
        """
        raise NotImplementedError


class UniformSubsampler(Subsampler):
    """
    A class for subsampling a data stream uniformly in time in an online fashion.
    """

    def __init__(self, T):
        """
        Args:
            T (int): Pick one every T observations.
        """
        self.T = T
        self.counter = 0

        super(UniformSubsampler, self).__init__()

    def subsample(self, observation):
        """
        Returns an observation once every T observations, None otherwise.

        Args:
            observation (n-array): A current observation.

        Returns:
            None or n-array: The observation, or None.
        """
        self.counter += 1
        if self.counter == self.T:
            self.counter = 0
            return observation
        return None


if __name__ == "__main__":
    f = MovingAverageFilter(3, 10)
    a = th.tensor([1, 1, 1])
    for i in range(500):
        print(f.estimate(a + th.randn_like(a) * 0.1))
