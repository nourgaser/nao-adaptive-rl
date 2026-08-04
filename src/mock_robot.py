"""
mock_robot.py
=============

A laptop-only implementation of `RobotInterface`. It contains a *simulated person*
so that the full RL loop runs with no hardware and no NAOqi SDK installed.

The person's comfort preference now lives in a `UserProfile` (see `user_profile.py`)
rather than being hard-coded, so the same robot code can be pointed at different
simulated people. That is what turns this from an adaptation demo into a
personalization demo.

The robot does NOT observe the profile. It receives only social signals --- a head
touch when the person is comfortable, or the person leaving when crowded --- and
must infer the preference from those alone.

Sensor noise is injected on purpose: it hardens the learned policy and shrinks the
eventual sim-to-real gap (see docs/architecture.md).
"""

import random

from .robot_interface import RobotInterface, Signals
from .user_profile import UserProfile, NEUTRAL

# Distance the robot moves per step (metres).
_STEP = 0.30
# Physical bounds on robot-person distance (metres).
_MIN_D, _MAX_D = 0.20, 2.20
# Where every interaction starts.
_START_D = 1.5


class MockRobot(RobotInterface):
    def __init__(
        self,
        profile: UserProfile = NEUTRAL,
        sensor_noise: float = 0.03,
        seed: int | None = None,
    ):
        self.profile = profile
        self._rng = random.Random(seed)
        self._noise = sensor_noise
        self._distance = _START_D
        self._intensity = 0
        self._present = True

    # --- actuation --------------------------------------------------------
    def say(self, text: str) -> None:
        pass  # no-op on the mock; a real NAO would speak

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

        # A crowded person may end the interaction. How readily depends on
        # the profile: `reserved` disengages far sooner than `sociable`.
        p = self.profile
        if self._distance < p.too_close and self._rng.random() < p.leave_prob:
            self._present = False

    # --- sensing ----------------------------------------------------------
    def sense(self) -> Signals:
        noisy_d = self._distance + self._rng.gauss(0, self._noise)
        noisy_d = max(_MIN_D, min(_MAX_D, noisy_d))

        p = self.profile
        in_zone = p.ideal_lo <= self._distance < p.ideal_hi
        if in_zone and self._intensity == p.ideal_intensity:
            touch = self._rng.random() < p.approve_prob    # clear approval
        elif in_zone:
            touch = self._rng.random() < p.partial_prob    # right place, wrong manner
        else:
            touch = False

        return Signals(distance=noisy_d, touch=touch, present=self._present)

    def get_intensity(self) -> int:
        return self._intensity

    # --- episode lifecycle ------------------------------------------------
    def reset_interaction(self) -> None:
        self._distance = _START_D
        self._intensity = 0
        self._present = True

    # --- personalization hook --------------------------------------------
    def set_profile(self, profile: UserProfile) -> None:
        """Swap the simulated person. Used to train one policy across a
        population, and to test a policy against a user it never met."""
        self.profile = profile
        self.reset_interaction()
