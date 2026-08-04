"""
Tests for the personalization experiment.

These exist so that "it works" is demonstrable rather than asserted -- the demo is
run live, and a broken assumption discovered in front of the committee is much
more expensive than a test suite.

    python -m pytest tests/ -v
"""

import numpy as np
import pytest

from src.agents.q_learning import QLearningAgent
from src.envs.greeting_env import (
    ACTIONS, N_ACTIONS, N_INTENSITY, N_STATES, GreetingEnv, _distance_bin,
)
from src.mock_robot import MockRobot
from src.user_profile import NEUTRAL, POPULATION, RESERVED, SOCIABLE


# --------------------------------------------------------------------------
# The environment contract
# --------------------------------------------------------------------------
def test_state_is_always_in_range():
    env = GreetingEnv(MockRobot(seed=0))
    s = env.reset()
    assert 0 <= s < N_STATES
    for a in range(N_ACTIONS):
        env.reset()
        s, r, done, info = env.step(a)
        assert 0 <= s < N_STATES
        assert isinstance(r, float)
        assert isinstance(done, bool)


def test_episode_terminates_within_max_steps():
    env = GreetingEnv(MockRobot(seed=1), max_steps=15)
    env.reset()
    for i in range(15):
        _, _, done, _ = env.step(ACTIONS.index("hold"))
        if done:
            break
    assert done, "episode must terminate by max_steps"


def test_unknown_action_rejected():
    with pytest.raises(ValueError):
        MockRobot(seed=0).perform("teleport")


def test_single_sense_per_step():
    """Regression: the env used to call sense() twice per step, once for the
    reward and once for the state, so the two could disagree about the same
    instant. Exactly one reading per step."""
    robot = MockRobot(seed=0)
    calls = {"n": 0}
    original = robot.sense

    def counting():
        calls["n"] += 1
        return original()

    robot.sense = counting
    env = GreetingEnv(robot)
    env.reset()
    calls["n"] = 0
    env.step(ACTIONS.index("hold"))
    assert calls["n"] == 1, f"expected 1 sense() per step, got {calls['n']}"


# --------------------------------------------------------------------------
# Profiles: the premise of the whole experiment
# --------------------------------------------------------------------------
def test_profiles_occupy_distinct_distance_bins():
    """The experiment only means something if the users genuinely disagree.
    Each profile's comfortable band must fall in a different bin, otherwise a
    single policy could serve everyone and personalization would be pointless."""
    bins = {p.name: _distance_bin((p.ideal_lo + p.ideal_hi) / 2)
            for p in POPULATION}
    assert len(set(bins.values())) == len(POPULATION), bins


def test_profiles_prefer_different_intensities():
    assert len({p.ideal_intensity for p in POPULATION}) == len(POPULATION)


def test_robot_cannot_observe_the_profile():
    """The agent's whole input is Signals. If a preference field leaked into
    the observation the task would be trivial and the result meaningless."""
    sig = MockRobot(profile=RESERVED, seed=0).sense()
    assert set(sig.keys()) == {"distance", "touch", "present"}


def test_crowding_ends_interaction_for_reserved():
    """`reserved` disengages when approached too closely."""
    robot = MockRobot(profile=RESERVED, seed=0)
    for _ in range(10):
        robot.perform("closer")
    assert robot.sense()["present"] is False


def test_sociable_tolerates_close_approach():
    """`sociable` should survive an approach that drives `reserved` away.
    Averaged over seeds because leaving is stochastic."""
    survived = 0
    for seed in range(30):
        robot = MockRobot(profile=SOCIABLE, seed=seed)
        for _ in range(4):            # down to ~0.3 m
            robot.perform("closer")
        survived += robot.sense()["present"]
    assert survived > 15, f"sociable left too often: {survived}/30"


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------
def test_agent_learns_to_beat_scripted():
    """The core claim: learning beats the scripted baseline on `neutral`."""
    from experiments.personalization import evaluate, scripted, train_personalized

    agent, _ = train_personalized(NEUTRAL, episodes=2000, seed=0)
    env = GreetingEnv(MockRobot(profile=NEUTRAL, seed=1000))
    assert evaluate(env, agent, n=200) > scripted(env, n=200) + 2.0


def test_personalized_policies_actually_differ():
    """The headline claim. If every user produced the same policy there would
    be no personalization to demonstrate."""
    from experiments.personalization import train_personalized

    policies = {}
    for p in POPULATION:
        agent, _ = train_personalized(p, episodes=2000, seed=0)
        policies[p.name] = tuple(int(np.argmax(agent.Q[s])) for s in range(N_STATES))
    assert len(set(policies.values())) == len(POPULATION), \
        f"expected a distinct policy per user, got {policies}"


def test_q_learning_update_is_correct():
    """Q[s,a] <- Q[s,a] + alpha * (r + gamma * max Q[s'] - Q[s,a])"""
    agent = QLearningAgent(2, 2, alpha=0.5, gamma=0.9, seed=0)
    agent.Q[1] = np.array([1.0, 3.0])
    agent.update(0, 0, r=1.0, s_next=1, done=False)
    assert agent.Q[0, 0] == pytest.approx(0.5 * (1.0 + 0.9 * 3.0))


def test_terminal_update_ignores_bootstrap():
    agent = QLearningAgent(2, 2, alpha=1.0, gamma=0.9, seed=0)
    agent.Q[1] = np.array([99.0, 99.0])
    agent.update(0, 0, r=2.0, s_next=1, done=True)
    assert agent.Q[0, 0] == pytest.approx(2.0)


def test_greedy_action_is_deterministic():
    agent = QLearningAgent(N_STATES, N_ACTIONS, seed=0)
    agent.Q[3] = np.arange(N_ACTIONS, dtype=float)
    assert {agent.act(3, greedy=True) for _ in range(20)} == {N_ACTIONS - 1}


def test_same_seed_reproduces():
    """Every number in the report must be re-runnable."""
    from experiments.personalization import train_personalized

    a, _ = train_personalized(NEUTRAL, episodes=300, seed=7)
    b, _ = train_personalized(NEUTRAL, episodes=300, seed=7)
    assert np.array_equal(a.Q, b.Q)
