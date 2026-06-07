"""
robot_interface.py
==================

The single most important file in the project architecturally.

`RobotInterface` is an *abstract* contract. Every concrete robot --- the
laptop `MockRobot`, a Webots/Choregraphe virtual NAO, or a real NAO via the
NAOqi SDK --- implements these same methods. The learning code (the Gym-style
environment and the Q-learning agent) only ever talks to this interface, never
to a concrete robot.

Why this matters: it means the *identical* reinforcement-learning loop runs
unchanged whether we are prototyping on a laptop or fine-tuning on hardware.
Swapping `MockRobot()` for `NaoRobot(ip=...)` is a one-line change. This is the
"common plumbing" that every candidate experiment (greeting/proxemics,
tutoring/pace, engagement-maximization) shares.
"""

from abc import ABC, abstractmethod
from typing import TypedDict


class Signals(TypedDict):
    """Whatever the robot can sense about the person, each step.

    On the MockRobot these come from a simulated person model.
    On a real NAO they come from sonar (`ALSonar`), the head tactile sensors
    (via `ALMemory`), and presence/engagement from `ALFaceDetection`.
    """
    distance: float   # metres to the person (NAO sonar range ~0.25-2.55 m)
    touch: bool       # head tactile sensor pressed == person "approves"
    present: bool     # is the person still engaged / in front of the robot


class RobotInterface(ABC):
    """The contract shared by mock, simulator, and real NAO."""

    # --- actuation --------------------------------------------------------
    @abstractmethod
    def say(self, text: str) -> None:
        """Speak a line (real NAO: ALTextToSpeech.say)."""

    @abstractmethod
    def perform(self, action: str) -> None:
        """Apply one greeting action.

        Shared action vocabulary across experiments:
          'closer'         - step toward the person
          'back'           - step away
          'intensity_up'   - bigger gesture / louder
          'intensity_down' - smaller gesture / quieter
          'hold'           - do nothing this step
        (real NAO: ALMotion.moveTo / gesture amplitude + ALTextToSpeech volume)
        """

    # --- sensing ----------------------------------------------------------
    @abstractmethod
    def sense(self) -> Signals:
        """Read the current social signals (state + reward inputs)."""

    @abstractmethod
    def get_intensity(self) -> int:
        """Current greeting-intensity level {0,1,2}.

        This is a *commanded* action-state, not a sensor reading, so it is
        valid on the real robot too (we store the last commanded value).
        """

    # --- episode lifecycle ------------------------------------------------
    @abstractmethod
    def reset_interaction(self) -> None:
        """Begin a fresh interaction (new simulated person, or re-home NAO)."""
