"""
yanshee_check.py
================

For the session with the robot, before the demo. Tests each YanAPI call the
demo depends on, one at a time, and prints the raw responses, so that problems
show up as a named failed step rather than halfway through the demo.

    python -m experiments.yanshee_check --ip 192.168.x.x
    python -m experiments.yanshee_check --ip 192.168.x.x --only sensors

Each step waits for [enter] so you can watch the robot. What to do with the
output is in docs/ROBOT_SESSION_CHECKLIST.md.
"""

import argparse
import time

from src.yanshee_robot import CALLS, GESTURE, VOLUME, YanAPIError, _first_number


def step(title: str) -> None:
    input(f"\n== {title} [enter] ")


def try_call(api, key: str, *args, **kwargs):
    name = CALLS[key]
    fn = getattr(api, name, None)
    if fn is None:
        print(f"   MISSING  YanAPI.{name} (CALLS['{key}'])")
        return None
    try:
        out = fn(*args, **kwargs)
        print(f"   ok       YanAPI.{name} -> {out}")
        return out
    except Exception as e:
        print(f"   FAILED   YanAPI.{name}: {e!r}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--only", choices=["names", "sensors", "motion", "voice"])
    ap.add_argument("--walk-repeat", type=int, default=2)
    args = ap.parse_args()

    import YanAPI as api   # copied from the robot; see the checklist
    run = lambda part: args.only in (None, part)

    if run("names"):
        step("1. function names")
        names = sorted(n for n in dir(api) if not n.startswith("_"))
        for key, name in CALLS.items():
            print(f"   {'ok     ' if name in names else 'MISSING'}  {key:9s} -> {name}")
        for word in ("sensor", "motion", "touch", "infrared", "tts", "volume"):
            hits = [n for n in names if word in n.lower()]
            print(f"   names containing '{word}': {hits}")

    try_call(api, "init", args.ip)

    if run("sensors"):
        step("2. infrared: hold a hand at ~15, ~30, ~50 cm, then take it away")
        for _ in range(12):
            out = try_call(api, "infrared")
            if out is not None:
                try:
                    print(f"            reading {_first_number(out)}")
                except YanAPIError as e:
                    print(f"            {e}")
            time.sleep(0.5)
        step("3. touch: touch the sensor during the next readings")
        for _ in range(10):
            try_call(api, "touch")
            time.sleep(0.5)

    if run("motion"):
        step(f"4. walk forward x{args.walk_repeat}: measure how far it goes")
        try_call(api, "motion", name="walk", direction="forward", repeat=args.walk_repeat)
        step(f"5. walk backward x{args.walk_repeat}")
        try_call(api, "motion", name="walk", direction="backward", repeat=args.walk_repeat)
        for level, gesture in GESTURE.items():
            if gesture:
                step(f"6. gesture for intensity {level}: '{gesture}'")
                try_call(api, "motion", name=gesture, repeat=1)

    if run("voice"):
        for level, vol in VOLUME.items():
            step(f"7. volume {vol} (intensity {level}) and a greeting")
            try_call(api, "volume", vol)
            try_call(api, "tts", "Hello, nice to see you.", True)

    print("\nDone. Fix anything MISSING or FAILED in CALLS / GESTURE in src/yanshee_robot.py.")


if __name__ == "__main__":
    main()
