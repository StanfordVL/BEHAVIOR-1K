import torch as th

from omnigibson.utils.backend_utils import _compute_backend as cb
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

    @property
    def state_size(self):
        """
        Size of the serialized state of this filter
        """
        raise NotImplementedError


class MovingAverageFilter(Filter):
    """
    This class uses a moving average to de-noise a noisy data stream in an online fashion.
    This is a FIR filter.

    Supports a batch of n_members independent filter rows. Each member has its own circular
    buffer row; estimate() targets one member, estimate_batch() processes all rows at once
    using matrix operations and broadcasting with no intermediate large allocations.
    """

    def __init__(self, obs_dim, filter_width, n_members):
        """
        Args:
            obs_dim (int): The dimension of the points to filter.
            filter_width (int): The number of past samples to take the moving average over.
            n_members (int): Number of independent filter rows (one per controller member).
        """
        self.obs_dim = obs_dim
        assert filter_width > 0, f"MovingAverageFilter must have a non-zero size! Got: {filter_width}"
        self.filter_width = filter_width
        self.n_members = n_members
        # (n_members, filter_width, obs_dim) — unfilled slots stay zero so sum/count gives correct mean
        self.past_samples = th.zeros(n_members, filter_width, obs_dim)
        self.current_idx = th.zeros(n_members, dtype=th.long)
        # bool tensor enables in-place updates and broadcasting in estimate_batch
        self.fully_filled = th.zeros(n_members, dtype=th.bool)
        # cached row indices to avoid re-allocating in estimate_batch
        self._member_arange = th.arange(n_members)

        super().__init__()

    def add_member(self, slot):
        """Register a member at the given slot index.

        If slot < n_members the slot is being reused (tombstone reuse): its buffer is cleared
        in-place and n_members is unchanged. If slot == n_members a new row is appended.

        Args:
            slot (int): Slot index as determined by the controller's add_member (either a
                previously tombstoned index or the next new index == current n_members).
        """
        if slot < self.n_members:
            # Reuse: clear the slot so it starts fresh
            self.past_samples[slot].zero_()
            self.current_idx[slot] = 0
            self.fully_filled[slot] = False
            # n_members and _member_arange stay the same — slot count is unchanged
            return
        # New slot: append a fresh row
        self.past_samples = th.cat([self.past_samples, th.zeros(1, self.filter_width, self.obs_dim)], dim=0)
        self.current_idx = th.cat([self.current_idx, th.zeros(1, dtype=th.long)])
        self.fully_filled = th.cat([self.fully_filled, th.zeros(1, dtype=th.bool)])
        # rebuild arange to include the new member
        self._member_arange = th.arange(self.n_members + 1)
        self.n_members += 1

    def unregister_member(self, member_idx):
        """Zero a member's buffer when it is unregistered (tombstoned at the controller level).

        Args:
            member_idx (int): Index of the member to unregister.
        """
        self.past_samples[member_idx].zero_()
        self.current_idx[member_idx] = 0
        self.fully_filled[member_idx] = False

    def estimate(self, member_idx, observation):
        """
        Do an online hold for state estimation given a recent observation for one member.

        Args:
            member_idx (int): Index of the controller member whose row to update.
            observation (n-array): New observation of shape (obs_dim,).

        Returns:
            n-array: New estimate of state.
        """
        idx = self.current_idx[member_idx].item()
        self.past_samples[member_idx, idx, :] = observation

        # Compute value based on whether we're fully filled or not
        if not self.fully_filled[member_idx]:
            val = self.past_samples[member_idx, : idx + 1, :].mean(dim=0)
            # Denote that we're fully filled if we're at the end of the buffer
            if idx == self.filter_width - 1:
                self.fully_filled[member_idx] = True
        else:
            val = self.past_samples[member_idx].mean(dim=0)

        # Increment the index to write the next sample to
        self.current_idx[member_idx] = (idx + 1) % self.filter_width
        return val

    def estimate_batch(self, observations):
        """
        Process all N member rows at once using batched matrix operations.

        Unfilled slots in past_samples are zero, so sum(dim=1) / fill_count gives the correct
        per-member mean without any masking matrix. Broadcasting is used throughout to avoid
        allocating intermediate large tensors.

        Args:
            observations (Tensor): (N, obs_dim) new observations for all members.

        Returns:
            Tensor: (N, obs_dim) smoothed estimates.
        """
        # Write new observations in-place via advanced indexing — no new large tensor
        self.past_samples[self._member_arange, self.current_idx] = observations

        # fill_count: filter_width for fully-filled rows, current_idx+1 for partial rows
        # th.where produces a (N,) tensor — unavoidable but small
        fill_count = th.where(self.fully_filled, self.filter_width, self.current_idx + 1)

        # Sum over the filter dimension; unfilled slots contribute 0, so result is correct.
        # unsqueeze(-1) broadcasts fill_count (N,) against (N, obs_dim) — no copy
        vals = self.past_samples.sum(dim=1) / fill_count.unsqueeze(-1)

        # Update fully_filled in-place for any rows that just hit the last slot
        self.fully_filled |= self.current_idx == self.filter_width - 1

        # Advance circular buffer pointer in-place
        self.current_idx += 1
        self.current_idx %= self.filter_width

        return vals

    def reset(self, member_idx=None):
        """
        Reset one member's filter row, or all rows if member_idx is None.

        Args:
            member_idx (int or None): Member to reset. Resets all members if None.
        """
        if member_idx is None:
            self.past_samples.zero_()
            self.current_idx.zero_()
            self.fully_filled.zero_()
        else:
            self.past_samples[member_idx].zero_()
            self.current_idx[member_idx] = 0
            self.fully_filled[member_idx] = False

    @property
    def state_size(self):
        return self.filter_width * self.obs_dim + 2

    def dump_state(self, controller_idx, serialized=False):
        state = self._dump_state(controller_idx)
        return self.serialize(state, controller_idx) if serialized else state

    def _dump_state(self, controller_idx):
        return {
            "past_samples": self.past_samples[controller_idx].clone(),
            "current_idx": self.current_idx[controller_idx].clone(),
            "fully_filled": self.fully_filled[controller_idx].clone(),
        }

    def load_state(self, controller_idx, state, serialized=False):
        if serialized:
            orig_state_len = len(state)
            state, deserialized_items = self.deserialize(state, controller_idx)
            assert deserialized_items == orig_state_len, (
                f"Invalid state deserialization occurred! Expected {orig_state_len} total "
                f"values to be deserialized, only {deserialized_items} were."
            )
        self._load_state(controller_idx, state)

    def _load_state(self, controller_idx, state):
        self.past_samples[controller_idx] = state["past_samples"]
        self.current_idx[controller_idx] = state["current_idx"]
        self.fully_filled[controller_idx] = state["fully_filled"]

    def serialize(self, state, controller_idx):
        return th.cat(
            [
                state["past_samples"].flatten(),
                state["current_idx"].float().reshape(1),
                state["fully_filled"].float().reshape(1),
            ]
        )

    def deserialize(self, state, controller_idx):
        samples_len = self.filter_width * self.obs_dim
        state_dict = {
            "past_samples": state[:samples_len].reshape(self.filter_width, self.obs_dim),
            "current_idx": state[samples_len].long(),
            "fully_filled": state[samples_len + 1].bool(),
        }
        return state_dict, samples_len + 2


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
        self.avg = cb.zeros(obs_dim)
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

        return cb.copy(self.avg)

    def reset(self):
        # Clear internal state
        self.avg *= 0.0
        self.num_samples = 0

    @property
    def state_size(self):
        # This is the size of the internal value as well as a num samples
        return len(self.avg) + 1

    def _dump_state(self):
        # Run super init first
        state = super()._dump_state()

        # Add info from this filter
        state["avg"] = cb.to_torch(self.avg)
        state["num_samples"] = self.num_samples

        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # Load relevant info for this filter
        self.avg = cb.from_torch(state["avg"])
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
