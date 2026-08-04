"""
greeting_env.py
===============

A Gym-style environment that turns *any* `RobotInterface` into the greeting /
social-distance MDP. It follows the standard `reset()` / `step(action)` contract
that every RL algorithm expects, so it is compatible with the wider Gymnasium
ecosystem without depending on it.

The environment is deliberately preference-agnostic: it does not know, and cannot
see, which `UserProfile` the underlying robot is simulating. The same MDP therefore
serves every user in the population, and any behavioural difference between users
comes from learning, not from the environment being told the answer.

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

The distance bins line up with Hall's proxemic zones, and each of the three user
profiles has its comfortable band inside a *different* bin -- so the bins are
expressive enough to represent every user's optimum, without encoding any of them.
"""

from typing import Tuple

from ..robot_interface import RobotInterface, Signals

ACTIONS = ["closer", "back", "intensity_up", "intensity_down", "hold"]
N_DISTANCE_BINS = 4
N_INTENSITY = 3
N_STATES = N_DISTANCE_BINS * N_INTENSITY
N_ACTIONS = len(ACTIONS)

# Reward constants, named so the shaping is auditable rather than magic.
R_APPROVAL = 1.0
R_LEFT = -2.0
R_STEP_COST = -0.05


def _distance_bin(d: float) -> int:
    if d < 0.45:
        return 0   # intimate  -- where `sociable` is comfortable
    if d < 0.90:
        return 1   # personal  -- where `neutral` is comfortable
    if d < 1.50:
        return 2   # social    -- where `reserved` is comfortable
    return 3       # far       -- comfortable for nobody; the start state


class GreetingEnv:
    def __init__(self, robot: RobotInterface, max_steps: int = 15):
        self.robot = robot
        self.max_steps = max_steps
        self._steps = 0

    # --- state encoding ---------------------------------------------------
    def _state_from(self, sig: Signals) -> int:
        """Encode an *already-taken* sensor reading.

        Taking the reading as an argument matters: `sense()` re-rolls sensor
        noise and the approval draw on every call, so sensing separately for the
        reward and for the state would let the two disagree about the same
        instant. One reading per step, used for both.
        """
        return _distance_bin(sig["distance"]) * N_INTENSITY + self.robot.get_intensity()

    # --- Gym-style API ----------------------------------------------------
    def reset(self) -> int:
        self.robot.reset_interaction()
        self._steps = 0
        return self._state_from(self.robot.sense())

    def step(self, action_idx: int) -> Tuple[int, float, bool, dict]:
        self.robot.perform(ACTIONS[action_idx])
        self._steps += 1

        sig = self.robot.sense()          # the single reading for this step
        reward = R_STEP_COST
        done = False
        if not sig["present"]:            # person left (crowded)
            reward += R_LEFT
            done = True
        elif sig["touch"]:                # approval
            reward += R_APPROVAL

        if self._steps >= self.max_steps:
            done = True

        return self._state_from(sig), reward, done, {"signals": sig}
