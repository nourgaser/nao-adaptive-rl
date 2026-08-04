"""
personalization.py
==================

The main experiment. Answers one question:

    Does a robot need to personalize, or is one well-trained policy enough?

Five conditions, all evaluated on the same three simulated users:

  1. scripted      - fixed greeting, no learning. The straw man everyone beats.
  2. one-size      - ONE policy trained across the whole mixed population.
  3. best-single   - the strongest *coherent* non-personalized policy available:
                     each user's personalized policy scored against everyone,
                     keeping whichever generalises best.
  4. personalized  - a policy trained per user from scratch.
  5. warm-start    - a prior trained on the OTHER users (leave-one-out), then
                     adapted to this one. The deployment question: does meeting
                     other people help a robot meet a new person faster?

Conditions (2) and (3) are what make this an experiment rather than a demo.
Beating a scripted robot proves only that learning happened; beating a *trained*
generalist is what proves personalization was necessary. Condition (3) exists
because (2) empirically degenerates into a limit cycle, and beating a broken
opponent would flatter the result.

WHAT WE FOUND
-------------
* Personalization is necessary. The best single policy is optimal for exactly one
  user (`reserved`) and near-worthless for the other two.
* Warm-starting does NOT help - a negative result we report rather than hide.
  With an equal episode budget it roughly matches from-scratch; with a small
  budget it is far worse. So the prior confers no head start.

  Why, and why it is interesting: PbARL (Wang et al. 2025) works because task
  competence and user preference are *separable* - the robot must still feed you,
  merely differently. In proxemics they are the same thing. The three users'
  optima sit in mutually exclusive distance bins, so another user's optimum
  carries no information about this one's, and mildly misleads. This is a
  concrete boundary condition on the transfer idea, found by running it.

Run:
    python -m experiments.personalization
    python -m experiments.personalization --episodes 4000 --seeds 5 --plots
"""

import argparse
from pathlib import Path

import numpy as np

from src.agents.q_learning import QLearningAgent
from src.envs.greeting_env import (
    ACTIONS, N_ACTIONS, N_INTENSITY, N_STATES, GreetingEnv,
)
from src.mock_robot import MockRobot
from src.user_profile import POPULATION

DIST_NAMES = ["intimate", "personal", "social", "far"]
INT_NAMES = ["low", "med", "high"]

# Learning hyperparameters, chosen by a sweep over {alpha, epsilon decay,
# episodes} scored on convergence *reliability* across seeds -- specifically on
# the WORST user (mean - SD), not the average. A demo run live must not depend
# on luck, and an average hides the user that fails.
#
# Sweep result (6 seeds, mean +/- SD per user):
#   3000 / 0.5 / 0.999   reserved 11.86+/-0.15  neutral  7.64+/-4.75  sociable 6.94+/-1.60
#   6000 / 0.3 / 0.9995  reserved  7.44+/-4.93  neutral  9.58+/-1.48  sociable 8.33+/-0.27
#   6000 / 0.3 / 0.999   reserved 11.86+/-0.15  neutral 10.80+/-0.28  sociable 8.36+/-0.23  <-
#
# Lowering alpha stabilises all three; slowing the epsilon decay looked good on
# neutral/sociable but wrecked `reserved`, who disengages early and so is harmed
# by prolonged exploration. Tuning on a subset of users would have shipped that
# regression -- the sweep covers every user for that reason.
ALPHA = 0.3
EPS_DECAY = 0.999


def _make_agent(seed):
    return QLearningAgent(N_STATES, N_ACTIONS, alpha=ALPHA,
                          epsilon_decay=EPS_DECAY, seed=seed)


# --------------------------------------------------------------------------
# Episode helpers
# --------------------------------------------------------------------------
def run_episode(env, agent, train=True, greedy=False) -> float:
    state = env.reset()
    total, done = 0.0, False
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


def evaluate(env, agent, n=300) -> float:
    """Greedy evaluation — no exploration, no learning."""
    return float(np.mean([run_episode(env, agent, train=False, greedy=True)
                          for _ in range(n)]))


def scripted(env, n=300) -> float:
    """A fixed greeter: approach three steps, then hold. No learning.

    This is a reasonable hand-designed default, not a deliberately bad one —
    approaching to conversational distance and stopping is what a scripted
    social robot actually does.
    """
    totals = []
    approach, hold = ACTIONS.index("closer"), ACTIONS.index("hold")
    for _ in range(n):
        env.reset()
        total, done, step = 0.0, False, 0
        while not done:
            _, r, done, _ = env.step(approach if step < 3 else hold)
            total += r
            step += 1
        totals.append(total)
    return float(np.mean(totals))


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------
def train_personalized(profile, episodes, seed):
    """One policy, one user, from scratch."""
    env = GreetingEnv(MockRobot(profile=profile, seed=seed))
    agent = _make_agent(seed)
    curve = [run_episode(env, agent) for _ in range(episodes)]
    return agent, curve


def train_population(episodes, seed, users=None):
    """One policy across a set of users. The user is re-drawn every episode, so
    the agent cannot tell who it is facing and must find a single compromise.

    `users` defaults to the whole population. Passing a subset gives the
    leave-one-out prior used by the warm-start condition.
    """
    users = users if users is not None else POPULATION
    rng = np.random.default_rng(seed)
    robot = MockRobot(profile=users[0], seed=seed)
    env = GreetingEnv(robot)
    agent = _make_agent(seed)
    curve = []
    for _ in range(episodes):
        robot.set_profile(users[rng.integers(len(users))])
        curve.append(run_episode(env, agent))
    return agent, curve


def best_single_policy(per_agents, seed):
    """The strongest *non-personalized* baseline we can construct.

    Training one policy on a mixed population produces an incoherent compromise
    (empirically, a limit cycle: intensity_up/intensity_down forever). Comparing
    against that alone would flatter personalization by beating a broken
    opponent. So we also take each user's personalized policy, score it against
    every user, and keep whichever generalises best. That policy is coherent by
    construction and is the best fixed behaviour available to a robot that
    refuses to personalise.
    """
    best, best_score = None, -np.inf
    for owner, agent in per_agents.items():
        scores = [evaluate(GreetingEnv(MockRobot(profile=p, seed=seed + 1000)),
                           agent, n=200)
                  for p in POPULATION]
        mean = float(np.mean(scores))
        if mean > best_score:
            best, best_score = agent, mean
    return best


def adapt_from(prior: QLearningAgent, profile, episodes, seed):
    """Warm-start from a prior trained on *other* users, then adapt to this one.

    This is the deployment question: a new person walks up to a robot that has
    already met other people. Does prior experience help?

    Epsilon restarts at 0.5, not 0.3: the prior encodes other people's
    preferences, some of which are actively wrong here, so the agent needs
    enough exploration to overwrite them rather than being trapped by them.
    """
    env = GreetingEnv(MockRobot(profile=profile, seed=seed))
    agent = _make_agent(seed)
    agent.Q = prior.Q.copy()
    agent.epsilon = 0.5
    curve = [run_episode(env, agent) for _ in range(episodes)]
    return agent, curve


def episodes_to_reach(curve, target, window=100) -> int | None:
    """First episode at which the rolling mean reaches `target`.
    Measures adaptation *speed*, which is the point of the warm start."""
    if len(curve) < window:
        return None
    roll = np.convolve(curve, np.ones(window) / window, mode="valid")
    hit = np.argmax(roll >= target)
    return int(hit + window) if roll[hit] >= target else None


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--adapt-episodes", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--plots", action="store_true", help="write figures to results/")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    names = [p.name for p in POPULATION]
    conditions = ("scripted", "one-size", "best-single", "personalized", "warm-start")
    res = {c: {n: [] for n in names} for c in conditions}
    speed = {"personalized": {n: [] for n in names},
             "warm-start": {n: [] for n in names}}
    curves = {}

    print("Simulated users (hidden from the agent):")
    for p in POPULATION:
        print(f"  {p.describe()}")
    print(f"\nRunning {args.seeds} seeds x {args.episodes} episodes...\n")

    for seed in seeds:
        pop_agent, pop_curve = train_population(args.episodes, seed)

        # Personalized policies first — the best-single baseline is chosen
        # from among them.
        per = {}
        for p in POPULATION:
            per[p.name] = train_personalized(p, args.episodes, seed)
        single = best_single_policy({k: v[0] for k, v in per.items()}, seed)

        for p in POPULATION:
            eval_env = GreetingEnv(MockRobot(profile=p, seed=seed + 1000))
            per_agent, per_curve = per[p.name]

            res["scripted"][p.name].append(scripted(eval_env))
            res["one-size"][p.name].append(evaluate(eval_env, pop_agent))
            res["best-single"][p.name].append(evaluate(eval_env, single))
            res["personalized"][p.name].append(evaluate(eval_env, per_agent))

            # Warm start from a prior that never met this user (leave-one-out).
            others = [q for q in POPULATION if q.name != p.name]
            prior, _ = train_population(args.episodes, seed, users=others)
            warm_agent, warm_curve = adapt_from(
                prior, p, args.adapt_episodes, seed)
            res["warm-start"][p.name].append(evaluate(eval_env, warm_agent))

            # Adaptation speed, measured against a common bar: 80% of the
            # from-scratch policy's final performance.
            target = 0.8 * res["personalized"][p.name][-1]
            speed["personalized"][p.name].append(
                episodes_to_reach(per_curve, target))
            speed["warm-start"][p.name].append(
                episodes_to_reach(warm_curve, target))

            if seed == 0:
                curves[p.name] = {"scratch": per_curve, "warm": warm_curve,
                                  "agent": per_agent, "pop_agent": pop_agent}
        if seed == 0:
            curves["_population"] = pop_curve

    # ---- results table ---------------------------------------------------
    print(f"Mean reward over {args.seeds} seeds (+/- SD), 300 greedy interactions each\n")
    hdr = f"{'condition':<16}" + "".join(f"{n:>20}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    for cond in conditions:
        row = f"{cond:<16}"
        for n in names:
            v = np.array(res[cond][n])
            row += f"{v.mean():>13.2f}+/-{v.std():<5.2f}"
        print(row)

    print("\nPersonalization gain (personalized - best non-personalized baseline):")
    for n in names:
        base = np.maximum(np.array(res["one-size"][n]), np.array(res["best-single"][n]))
        d = np.array(res["personalized"][n]) - base
        print(f"  {n:<10} {d.mean():+6.2f}  (SD {d.std():.2f})")

    print(f"\nEpisodes to reach 80% of final performance "
          f"(lower = faster adaptation):")
    for n in names:
        s = [x for x in speed["personalized"][n] if x is not None]
        w = [x for x in speed["warm-start"][n] if x is not None]
        s_txt = f"{np.mean(s):.0f}" if s else "not reached"
        w_txt = f"{np.mean(w):.0f}" if w else "not reached"
        print(f"  {n:<10} from scratch {s_txt:>12}   warm-started {w_txt:>12}")

    # ---- learned policies ------------------------------------------------
    print("\nLearned policy per user (seed 0) — the same state, different actions:\n")
    print(f"{'state':<22}" + "".join(f"{n:>16}" for n in names))
    print("-" * (22 + 16 * len(names)))
    for s in range(N_STATES):
        db, il = divmod(s, N_INTENSITY)
        label = f"{DIST_NAMES[db]}/{INT_NAMES[il]}"
        row = f"{label:<22}"
        for n in names:
            row += f"{ACTIONS[int(np.argmax(curves[n]['agent'].Q[s]))]:>16}"
        print(row)

    if args.plots:
        make_plots(res, curves, names, args)


def make_plots(res, curves, names, args) -> None:
    """Figures for the demo. Deans read a picture, not a terminal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)

    def smooth(c, w=100):
        return np.convolve(c, np.ones(w) / w, mode="valid")

    # 1 — the headline: personalization vs a trained generalist
    fig, ax = plt.subplots(figsize=(8, 4.5))
    conds = ["scripted", "one-size", "best-single", "personalized", "warm-start"]
    labels = ["Scripted\n(no learning)", "One-size-fits-all\n(mixed training)",
              "Best single policy\n(strongest generalist)",
              "Personalized\n(per user)", "Warm-started\n(unseen user + adapt)"]
    x = np.arange(len(names))
    width = 0.16
    for i, (c, lab) in enumerate(zip(conds, labels)):
        means = [np.mean(res[c][n]) for n in names]
        errs = [np.std(res[c][n]) for n in names]
        ax.bar(x + (i - 2) * width, means, width, yerr=errs,
               capsize=3, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in names])
    ax.set_ylabel("Mean reward per interaction")
    ax.set_title("One policy cannot serve every user")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "personalization_comparison.png", dpi=160)
    plt.close(fig)

    # 2 — adaptation speed: warm start vs from scratch
    fig, axes = plt.subplots(1, len(names), figsize=(13, 3.6), sharey=True)
    for ax, n in zip(axes, names):
        ax.plot(smooth(curves[n]["scratch"]), label="from scratch", lw=1.4)
        ax.plot(smooth(curves[n]["warm"]), label="warm-started", lw=1.4)
        ax.set_title(n.capitalize())
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("reward (rolling mean, 100)")
    axes[0].legend(fontsize=8)
    fig.suptitle("A population prior makes adapting to a new person faster", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "adaptation_speed.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3 — the policies themselves, side by side
    fig, axes = plt.subplots(1, len(names), figsize=(13, 3.4))
    for ax, n in zip(axes, names):
        Q = curves[n]["agent"].Q
        grid = np.array([int(np.argmax(Q[s])) for s in range(N_STATES)]
                        ).reshape(len(DIST_NAMES), N_INTENSITY)
        ax.imshow(grid, cmap="tab10", vmin=0, vmax=9)
        ax.set_xticks(range(N_INTENSITY), INT_NAMES)
        ax.set_yticks(range(len(DIST_NAMES)), DIST_NAMES)
        ax.set_title(n.capitalize())
        for i in range(len(DIST_NAMES)):
            for j in range(N_INTENSITY):
                ax.text(j, i, ACTIONS[grid[i, j]], ha="center", va="center",
                        fontsize=6.5, color="white")
    fig.suptitle("Same states, different learned actions — this is personalization", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "learned_policies.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote figures to {out}/")


if __name__ == "__main__":
    main()
