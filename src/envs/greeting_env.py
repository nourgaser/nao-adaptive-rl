"""
greeting_env.py
===============

A Gym-style environment that turns *any* `RobotInterface` into the greeting /
social-distance MDP. It follows the standard `reset()` / `step(action)`
contract that every RL algorithm expects, so it is compatible with the wider
Gymnasium ecosystem without depending on it.

This is the "illustrative" experiment (greeting/proxemics). The same env
skeleton --- read signals, build a discrete state, compute reward from social
signals --- is what the tutoring/pace and engagement-maximization experiments
would reuse; only the state features, actions, and reward wiring change.

MDP definition
--------------
State  S = (distance_bin, intensity_level)
         distance_bin in {0:intimate, 1:personal, 2:social, 3:far}   (sensed)
         intensity_level in {0:low, 1:med, 2:high}                    (commanded)
         -> 4 x 3 = 12 discrete states
Action A in {closer, back, intensity_up, intensity_down, hold}        (5 actions)
Reward R from social signals only (the robot never sees the hidden preference):
         +1.0   head touch (the person "approves")
         -2.0   the person leaves (crowded), and the episode ends
         -0.05  per-step cost (encourages settling into the comfortable zone)
"""

from typing import Tuple
from ..robot_interface import RobotInterface

ACTIONS = ["closer", "back", "intensity_up", "intensity_down", "hold"]
N_DISTANCE_BINS = 4
N_INTENSITY = 3
N_STATES = N_DISTANCE_BINS * N_INTENSITY
N_ACTIONS = len(ACTIONS)


def _distance_bin(d: float) -> int:
    if d < 0.45:
        return 0   # intimate (too close)
    if d < 0.90:
        return 1   # personal (the comfortable zone, in our mock person)
    if d < 1.50:
        return 2   # social
    return 3       # far


class GreetingEnv:
    def __init__(self, robot: RobotInterface, max_steps: int = 15):
        self.robot = robot
        self.max_steps = max_steps
        self._steps = 0

    # --- state encoding ---------------------------------------------------
    def _state(self) -> int:
        sig = self.robot.sense()
        db = _distance_bin(sig["distance"])
        il = self.robot.get_intensity()
        return db * N_INTENSITY + il

    # --- Gym-style API ----------------------------------------------------
    def reset(self) -> int:
        self.robot.reset_interaction()
        self._steps = 0
        return self._state()

    def step(self, action_idx: int) -> Tuple[int, float, bool, dict]:
        self.robot.perform(ACTIONS[action_idx])
        self._steps += 1

        sig = self.robot.sense()
        reward = -0.05                       # per-step cost
        done = False
        if not sig["present"]:               # person left (crowded)
            reward += -2.0
            done = True
        elif sig["touch"]:                   # approval
            reward += 1.0

        if self._steps >= self.max_steps:
            done = True

        return self._state(), reward, done, {"signals": sig}
