import numpy as np
from typing import Tuple


class ReplayBuffer:
    """A simple replay buffer for storing and sampling experiences.

    Parameters
    ----------
    buffer_size : int
        The maximum number of experiences the buffer can hold.
    state_dim : int
        The dimension of the state space.
    action_dim : int
        The dimension of the action space.

    Attributes
    ----------
    states : np.ndarray
        The array storing states.
    actions : np.ndarray
        The array storing actions.
    rewards : np.ndarray
        The array storing rewards.
    next_states : np.ndarray
        The array storing next states.
    terminateds : np.ndarray
        The array storing terminated flags.
    truncateds : np.ndarray
        The array storing truncated flags.
    """

    def __init__(self, buffer_size: int, state_dim: int, action_dim: int) -> None:
        self.buffer_size = buffer_size
        self.pointer = 0
        self.size = 0

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        if action_dim == 0:
            self.actions = np.zeros((buffer_size, ), dtype=np.int64)
        else:
            self.actions = np.zeros((buffer_size, action_dim), dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.terminateds = np.zeros(buffer_size, dtype=bool)
        self.truncateds = np.zeros(buffer_size, dtype=bool)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Add a new experience to the buffer.

        Parameters
        ----------
        state : np.ndarray
            The observed state.
        action : np.ndarray
            The action taken.
        reward : float
            The observed reward.
        next_state : np.ndarray
            The next observed state.
        terminated : bool
            Whether the episode has finished due to termination.
        truncated : bool
            Whether the episode has finished due to truncation.
        """
        idx = self.pointer % self.buffer_size

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.terminateds[idx] = terminated
        self.truncateds[idx] = truncated

        self.pointer += 1
        self.size = min(self.size + 1, self.buffer_size)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing batches of states, actions, rewards, next states, terminateds, and truncateds.
        """
        idx = np.random.randint(0, self.size, size=batch_size)

        batch_states = self.states[idx]
        batch_actions = self.actions[idx]
        batch_rewards = self.rewards[idx]
        batch_next_states = self.next_states[idx]
        batch_terminateds = self.terminateds[idx]
        batch_truncateds = self.truncateds[idx]

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_terminateds,
            batch_truncateds,
        )
