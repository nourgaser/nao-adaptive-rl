"""
mock_robot.py
=============

A laptop-only implementation of `RobotInterface`. It contains a simple
*simulated person* so that the full RL loop can run with no hardware and no
NAOqi SDK installed --- this is how we satisfy the "implementation has begun"
requirement before robot access is confirmed.

The simulated person has a hidden comfort preference (a comfortable distance
and greeting intensity). The robot does NOT know these; it only receives
social signals (a head touch when the person is comfortable, or the person
leaving when crowded). The learning agent must discover the comfortable
configuration purely from those signals --- exactly the real task.

Sensor noise is injected on purpose: it "hardens" the learned policy and makes
the eventual sim-to-real gap smaller (see docs/architecture.md).
"""

import random
from .robot_interface import RobotInterface, Signals

# Distance the robot moves per step (metres).
_STEP = 0.30
# Physical bounds on robot-person distance (metres).
_MIN_D, _MAX_D = 0.20, 2.20


class MockRobot(RobotInterface):
    def __init__(self, sensor_noise: float = 0.03, seed: int | None = None):
        self._rng = random.Random(seed)
        self._noise = sensor_noise
        self._distance = 1.5          # start far away
        self._intensity = 0           # start at low intensity
        self._present = True
        # --- hidden person preferences (unknown to the agent) ------------
        self._ideal_lo, self._ideal_hi = 0.45, 0.90   # "personal" zone
        self._ideal_intensity = 1                      # medium gesture
        self._too_close = 0.35                         # crowding threshold

    # --- actuation --------------------------------------------------------
    def say(self, text: str) -> None:
        pass  # no-op on the mock; real NAO would speak

    def perform(self, action: str) -> None:
        if action == "closer":
            self._distance = max(_MIN_D, self._distance - _STEP)
        elif action == "back":
            self._distance = min(_MAX_D, self._distance + _STEP)
        elif action == "intensity_up":
            self._intensity = min(2, self._intensity + 1)
        elif action == "intensity_down":
            self._intensity = max(0, self._intensity - 1)
        elif action == "hold":
            pass
        else:
            raise ValueError(f"unknown action: {action}")

        # The simulated person may leave if the robot crowds them.
        if self._distance < self._too_close and self._rng.random() < 0.6:
            self._present = False

    # --- sensing ----------------------------------------------------------
    def sense(self) -> Signals:
        noisy_d = self._distance + self._rng.gauss(0, self._noise)
        noisy_d = max(_MIN_D, min(_MAX_D, noisy_d))

        comfortable = (
            self._ideal_lo <= self._distance < self._ideal_hi
            and self._intensity == self._ideal_intensity
        )
        if comfortable:
            touch = self._rng.random() < 0.90        # clear approval
        elif self._ideal_lo <= self._distance < self._ideal_hi:
            touch = self._rng.random() < 0.20        # close-ish
        else:
            touch = False

        return Signals(distance=noisy_d, touch=touch, present=self._present)

    def get_intensity(self) -> int:
        return self._intensity

    # --- episode lifecycle ------------------------------------------------
    def reset_interaction(self) -> None:
        self._distance = 1.5
        self._intensity = 0
        self._present = True
