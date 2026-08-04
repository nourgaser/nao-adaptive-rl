"""
user_profile.py
===============

The difference between *adaptation* and *personalization*.

The original `MockRobot` had a single hard-coded comfort preference, so the agent
learned one comfortable configuration and that was that. It demonstrated that a
robot can adapt --- but not that it can adapt *to whom it is talking to*, which is
the subject of our review (Wang et al. 2025; Tozadore et al. 2018).

A `UserProfile` is the hidden internal state of a simulated person. The robot never
observes any of these fields. It only ever receives the social signals defined in
`RobotInterface.sense()` --- a head touch when the person is comfortable, or the
person walking away when crowded --- and must infer the preference from those alone.
That is exactly the real problem: preferences are not announced, they are displayed.

The three profiles are deliberately chosen so that a policy which is optimal for one
is actively *wrong* for another --- a robot that has learned to stand close and
gesture broadly for `sociable` will drive `reserved` away. If a single policy could
serve all three, personalization would not be worth doing, and the experiment would
have nothing to show.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class UserProfile:
    """Hidden preferences of a simulated person. Never visible to the agent."""

    name: str

    # Comfortable interpersonal distance, in metres. Hall's proxemic zones:
    # intimate <0.45, personal 0.45-1.2, social 1.2-3.6.
    ideal_lo: float
    ideal_hi: float

    # Preferred greeting intensity level {0: low, 1: medium, 2: high}.
    ideal_intensity: int

    # Below this distance the person feels crowded and may end the interaction.
    too_close: float

    # Probability of leaving on a given step while crowded. Higher = less tolerant.
    leave_prob: float

    # Probability of signalling approval (head touch) when fully comfortable.
    # Below 1.0 because real approval signals are noisy and intermittent.
    approve_prob: float = 0.90

    # Probability of approval when the distance is right but the intensity is not.
    # This partial-credit signal is what makes the problem learnable rather than
    # a sparse-reward needle-in-a-haystack.
    partial_prob: float = 0.20

    def describe(self) -> str:
        return (
            f"{self.name:9s} | comfortable {self.ideal_lo:.2f}-{self.ideal_hi:.2f} m "
            f"| intensity {self.ideal_intensity} | crowded below {self.too_close:.2f} m"
        )


# The population. Distinct enough that one policy cannot serve all three.
RESERVED = UserProfile(
    name="reserved",
    ideal_lo=0.90, ideal_hi=1.50,   # keeps the robot at social distance
    ideal_intensity=0,              # prefers a quiet, small greeting
    too_close=0.60,                 # feels crowded early
    leave_prob=0.70,                # and is quick to disengage
)

NEUTRAL = UserProfile(
    name="neutral",
    ideal_lo=0.45, ideal_hi=0.90,   # the original hard-coded preference
    ideal_intensity=1,
    too_close=0.35,
    leave_prob=0.60,
)

SOCIABLE = UserProfile(
    name="sociable",
    ideal_lo=0.20, ideal_hi=0.45,   # happy to be approached closely
    ideal_intensity=2,              # enjoys a big, expressive greeting
    too_close=0.20,                 # rarely feels crowded
    leave_prob=0.40,
)

POPULATION = [RESERVED, NEUTRAL, SOCIABLE]
BY_NAME = {p.name: p for p in POPULATION}
