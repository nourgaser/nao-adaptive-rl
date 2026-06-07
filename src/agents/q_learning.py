"""
q_learning.py
=============

Plain tabular Q-learning (Watkins & Dayan, 1992). Deliberately small and
dependency-light: the Q-table is a NumPy array of shape (n_states, n_actions).

Update rule:
    Q[s,a] <- Q[s,a] + alpha * ( r + gamma * max_a' Q[s',a'] - Q[s,a] )

Action selection is epsilon-greedy with exponential decay: explore early,
exploit later.

Tabular methods are the right tool here because the state space is tiny
(12 states). No GPU, no neural network --- this runs on a laptop CPU in
milliseconds per episode. Function approximation (e.g. a Deep Q-Network) would
only be needed if the state space became large or continuous.
"""

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.5,      # learning rate
        gamma: float = 0.9,      # discount factor
        epsilon: float = 1.0,    # initial exploration rate
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.999,
        seed: int | None = None,
    ):
        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(seed)

    def act(self, state: int, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))   # explore
        return int(np.argmax(self.Q[state]))                 # exploit

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
