"""
yanshee_demo.py
===============

The personalization result, on a real robot.

The per-user policies are learned in simulation exactly as in
`personalization.py` (6000 episodes, seed 0, the same policies drawn in
results/learned_policies.png). This script then runs one of them on the Yanshee,
greedily and without further learning, through the unchanged `GreetingEnv`.
The robot is told which person it is greeting, the same assumption every
personalized condition in the study makes; it is never told what that person
prefers.

Why no learning on the robot: a policy needs about 2,300 interactions to reach
80% of its final performance (see the experiment's output). That is hours of a
person standing in front of a robot. Learning in simulation and deploying on
hardware is the standard way around this, and the robot code path is identical.

    # no robot: the fake YanAPI with the simulated person behind it
    python -m experiments.yanshee_demo --profile reserved --dry-run

    # the mismatch the study is about: a policy meant for someone else
    python -m experiments.yanshee_demo --profile sociable --person reserved --dry-run

    # the robot, with its sensor pack
    python -m experiments.yanshee_demo --profile reserved --ip 192.168.x.x

    # the robot without the sensor pack: you type zone and touch each step
    python -m experiments.yanshee_demo --profile reserved --ip 192.168.x.x --operator
"""

import argparse
import time

from src.envs.greeting_env import ACTIONS, N_INTENSITY, GreetingEnv
from src.user_profile import BY_NAME
from src.yanshee_robot import KeyboardOperator, YansheeRobot

from experiments.personalization import train_personalized

DIST_NAMES = ["intimate", "personal", "social", "far"]
INT_NAMES = ["low", "med", "high"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(BY_NAME), required=True,
                    help="whose learned policy the robot runs")
    ap.add_argument("--person", choices=sorted(BY_NAME),
                    help="dry run only: who is actually standing there (default: --profile)")
    ap.add_argument("--dry-run", action="store_true", help="fake YanAPI, no robot")
    ap.add_argument("--ip", default="127.0.0.1", help="the robot's IP address")
    ap.add_argument("--operator", action="store_true",
                    help="type distance zone and touch each step (no sensor pack)")
    ap.add_argument("--operator-touch", action="store_true",
                    help="infrared works but touch does not: type touch only")
    ap.add_argument("--scale", type=float, default=1 / 3,
                    help="physical metres per model metre (default 1/3: bins at 15/30/50 cm)")
    ap.add_argument("--walk-repeat", type=int, default=2,
                    help="walk cycles per closer/back; calibrate so one step is about 10 cm")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    profile = BY_NAME[args.profile]
    t0 = time.time()
    agent, _ = train_personalized(profile, args.episodes, args.seed)
    print(f"Learned the '{profile.name}' policy in simulation "
          f"({args.episodes} episodes, {time.time() - t0:.1f} s).")

    operator = None
    if args.operator or args.operator_touch:
        operator = KeyboardOperator(ask_distance=args.operator, ask_touch=True)

    if args.dry_run:
        from src.yanshee_fake import FakeYanAPI
        person = BY_NAME[args.person or args.profile]
        api = FakeYanAPI(person, distance_scale=args.scale, seed=args.seed)
        robot = YansheeRobot(api=api, operator=operator, distance_scale=args.scale,
                             settle_s=0, on_reset=api.reset, verbose=False)
        print(f"Dry run: fake Yanshee, simulated person '{person.name}'.")
    else:
        robot = YansheeRobot(ip=args.ip, operator=operator, distance_scale=args.scale,
                             walk_repeat=args.walk_repeat)
        print(f"Connected to Yanshee at {args.ip}.")

    env = GreetingEnv(robot, max_steps=args.steps)
    state = env.reset()
    total, done, step = 0.0, False, 0
    print(f"\n{'step':>4}  {'state':<16} {'action':<15} {'reward':>6}")
    while not done:
        d, i = divmod(state, N_INTENSITY)
        action = agent.act(state, greedy=True)
        state, reward, done, info = env.step(action)
        step += 1
        sig = info["signals"]
        note = "  person left" if not sig["present"] else ("  touch" if sig["touch"] else "")
        print(f"{step:>4}  {DIST_NAMES[d] + '/' + INT_NAMES[i]:<16} "
              f"{ACTIONS[action]:<15} {reward:>+6.2f}{note}")
        total += reward

    print(f"\nTotal reward: {total:+.2f} over {step} steps.")


if __name__ == "__main__":
    main()
