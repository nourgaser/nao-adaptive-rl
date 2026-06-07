"""
train_greeting.py
=================

Runs the full reinforcement-learning loop on the greeting/proxemics MDP using
the laptop `MockRobot`. Demonstrates that tabular Q-learning converges from
social signals alone, then prints:
  * a learning curve (average reward over time),
  * a comparison against a fixed scripted baseline,
  * the learned greedy policy for every state.

Run:
    python -m experiments.train_greeting
    python -m experiments.train_greeting --episodes 4000 --seed 7

Swapping to a (virtual or real) NAO later is a one-line change:
    from src.nao_robot import NaoRobot
    robot = NaoRobot(ip="<robot-ip>")
The environment and agent code below do not change at all.
"""

import argparse
import numpy as np

from src.mock_robot import MockRobot
from src.envs.greeting_env import (
    GreetingEnv, ACTIONS, N_STATES, N_ACTIONS,
    N_INTENSITY, _distance_bin,
)
from src.agents.q_learning import QLearningAgent

_DIST_NAMES = ["intimate", "personal", "social", "far"]
_INT_NAMES = ["low", "med", "high"]


def run_episode(env, agent, train=True, greedy=False):
    state = env.reset()
    total = 0.0
    done = False
    while not done:
        action = agent.act(state, greedy=greedy)
        nxt, reward, done, _ = env.step(action)
        if train:
            agent.update(state, action, reward, nxt, done)
        state = nxt
        total += reward
    if train:
        agent.decay_epsilon()
    return total


def scripted_baseline(env, n=200):
    """A fixed, non-learning greeter: always 'closer' then 'hold'.
    Represents the scripted behaviour we compare the learned policy against."""
    totals = []
    for _ in range(n):
        env.reset()
        total, done, step = 0.0, False, 0
        while not done:
            action = ACTIONS.index("closer") if step < 3 else ACTIONS.index("hold")
            _, r, done, _ = env.step(action)
            total += r
            step += 1
        totals.append(total)
    return float(np.mean(totals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    robot = MockRobot(seed=args.seed)
    env = GreetingEnv(robot)
    agent = QLearningAgent(N_STATES, N_ACTIONS, seed=args.seed)

    print(f"Training tabular Q-learning for {args.episodes} episodes "
          f"({N_STATES} states x {N_ACTIONS} actions)\n")

    window, curve = [], []
    for ep in range(1, args.episodes + 1):
        r = run_episode(env, agent, train=True)
        window.append(r)
        if ep % 200 == 0:
            avg = float(np.mean(window))
            curve.append((ep, avg))
            bar = "#" * int(max(0, avg))
            print(f"  ep {ep:5d} | eps {agent.epsilon:0.3f} | "
                  f"avg reward {avg:6.2f} | {bar}")
            window = []

    # --- evaluation: learned (greedy) vs scripted baseline ---------------
    learned = float(np.mean([run_episode(env, agent, train=False, greedy=True)
                             for _ in range(200)]))
    baseline = scripted_baseline(env)
    print("\nEvaluation (avg reward over 200 interactions):")
    print(f"  scripted baseline : {baseline:6.2f}")
    print(f"  learned policy    : {learned:6.2f}")
    print(f"  improvement       : {learned - baseline:+6.2f}")

    # --- show the learned policy -----------------------------------------
    print("\nLearned greedy policy (state -> action):")
    for s in range(N_STATES):
        db, il = divmod(s, N_INTENSITY)
        a = int(np.argmax(agent.Q[s]))
        print(f"  dist={_DIST_NAMES[db]:8s} intensity={_INT_NAMES[il]:4s}"
              f"  ->  {ACTIONS[a]}")


if __name__ == "__main__":
    main()
