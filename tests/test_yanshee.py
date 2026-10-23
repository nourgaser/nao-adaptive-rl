"""
Tests for the Yanshee port, run against the fake YanAPI.

They check the wiring: that `YansheeRobot` satisfies the same environment
contract as `MockRobot`, that the learned policies behave on it as they do in
simulation, and that the safety and fallback paths work. They cannot check the
real robot's function names or sensor readings; that is the robot session.

    python -m pytest tests/test_yanshee.py -v
"""

import pytest

from experiments.personalization import train_personalized
from src.envs.greeting_env import ACTIONS, N_ACTIONS, N_STATES, GreetingEnv
from src.user_profile import RESERVED, SOCIABLE
from src.yanshee_fake import FakeYanAPI
from src.yanshee_robot import (
    KeyboardOperator, YanAPIError, YansheeRobot, _first_number,
)


def make(profile, operator=None, seed=0):
    api = FakeYanAPI(profile, seed=seed)
    robot = YansheeRobot(api=api, operator=operator, settle_s=0,
                         on_reset=api.reset, verbose=False)
    return robot, api


def greedy_episode(env, agent):
    state, total, done, left = env.reset(), 0.0, False, False
    while not done:
        state, r, done, info = env.step(agent.act(state, greedy=True))
        total += r
        left = left or not info["signals"]["present"]
    return total, left


@pytest.fixture(scope="module")
def policies():
    return {p.name: train_personalized(p, 6000, 0)[0] for p in (RESERVED, SOCIABLE)}


# --------------------------------------------------------------------------
# Same contract as the simulated robot
# --------------------------------------------------------------------------
def test_environment_contract_holds_on_yanshee():
    robot, _ = make(RESERVED)
    env = GreetingEnv(robot, max_steps=15)
    for a in range(N_ACTIONS):
        s = env.reset()
        assert 0 <= s < N_STATES
        s, r, done, _ = env.step(a)
        assert 0 <= s < N_STATES and isinstance(r, float) and isinstance(done, bool)


def test_start_state_is_far_and_low_intensity():
    robot, _ = make(RESERVED)
    env = GreetingEnv(robot)
    assert env.reset() == 3 * 3 + 0   # far bin, intensity 0


# --------------------------------------------------------------------------
# The result carries over: matched policies work, a mismatched one fails
# --------------------------------------------------------------------------
def test_matched_policies_earn_reward_on_yanshee(policies):
    for profile in (RESERVED, SOCIABLE):
        robot, _ = make(profile)
        total, left = greedy_episode(GreetingEnv(robot), policies[profile.name])
        assert total > 5 and not left, profile.name


def test_sociable_policy_drives_reserved_person_away(policies):
    robot, _ = make(RESERVED)
    total, left = greedy_episode(GreetingEnv(robot), policies["sociable"])
    assert left and total < 0


# --------------------------------------------------------------------------
# Hardware safety and fallbacks
# --------------------------------------------------------------------------
def test_robot_does_not_walk_into_the_person():
    robot, api = make(SOCIABLE)
    env = GreetingEnv(robot, max_steps=15)
    env.reset()
    for _ in range(15):
        env.step(ACTIONS.index("closer"))
    walks = [c for c in api.calls if c[:2] == ("motion", "walk")]
    # 1.5 m start, 0.3 m per walk: four walks reach 0.3 m, then the guard holds.
    assert len(walks) == 4


def test_intensity_sets_volume_and_speaks():
    robot, api = make(RESERVED)
    robot.reset_interaction()
    robot.perform("intensity_up")
    robot.perform("intensity_up")
    robot.perform("intensity_up")          # clamps at 2
    volumes = [c[1] for c in api.calls if c[0] == "volume"]
    assert volumes[-1] == 100 and robot.get_intensity() == 2
    assert any(c[0] == "tts" for c in api.calls)


def test_operator_mode_reads_zone_touch_and_leaving():
    answers = iter(["", "s t", "i l"])     # reset confirmation, then two readings
    op = KeyboardOperator(prompt=lambda _: next(answers))
    robot, _ = make(RESERVED, operator=op)
    robot.reset_interaction()
    sig = robot.sense()
    assert sig["touch"] and sig["present"] and 0.9 <= sig["distance"] < 1.5
    sig = robot.sense()
    assert not sig["present"]


def test_missing_yanapi_function_names_the_fix():
    class Partial:
        def yan_api_init(self, ip):
            pass
    robot = YansheeRobot(api=Partial(), settle_s=0, verbose=False)
    with pytest.raises(YanAPIError, match="CALLS"):
        robot.sense()


def test_robot_error_code_is_not_read_as_a_distance():
    with pytest.raises(YanAPIError):
        _first_number({"code": 1, "data": {"value": 40}})
    assert _first_number({"code": 0, "data": {"infrared": [{"id": 1, "value": 42}]}}) == 42
