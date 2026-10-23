"""
yanshee_robot.py
================

`RobotInterface` for the UBTECH Yanshee, the humanoid available for the demo
(the NAO is not). This file is the whole port: the environment, the agent and the
experiments are unchanged, which is the point of `robot_interface.py`.

Yanshee is driven from the laptop over Wi-Fi through UBTECH's Python SDK,
YanAPI. `YanAPI.py` is not on PyPI; copy it from the robot
(`/usr/lib/python3.5/YanAPI.py`) next to this project, or pass any object with
the same functions as `api=`.

How the greeting MDP maps onto Yanshee
--------------------------------------
  closer / back         walk forward / backward (`sync_play_motion("walk", ...)`)
  intensity_up / _down  speaker volume plus a gesture, then a short greeting
  distance              infrared distance sensor (sensor pack, 8-150 cm)
  touch (the reward)    touch sensor (sensor pack)
  present               the infrared reading: nobody in front reads as far away

Yanshee is a 37 cm desktop robot, so the demo happens on a table, not across a
room. `distance_scale` maps the physical reading onto the proxemic bins the
policies were learned in: with the default 1/3, the bins at 0.45 / 0.90 / 1.50 m
become 15 / 30 / 50 cm. The robot and the person are scaled down together, so
the learned policies apply unchanged.

If the sensor pack is not fitted, `operator=` supplies distance and touch
from the keyboard instead (see `KeyboardOperator`). The policy, the actions and
the robot's behaviour are the same either way; only the sensing is manual.

Function names
--------------
Every YanAPI call goes through `CALLS` below. Entries marked "verified" appear in
published YanAPI examples; the others are best guesses and must be checked on the
robot with `dir(YanAPI)` before the demo. Fix them here and nowhere else.
"""

from __future__ import annotations

import time
from typing import Callable

from .robot_interface import RobotInterface, Signals

# name used in this file  ->  YanAPI function name
CALLS = {
    "init": "yan_api_init",                 # verified
    "motion": "sync_play_motion",           # verified: (name=, direction=, repeat=)
    "tts": "sync_do_tts",                   # verified: (text, interrupt)
    "volume": "set_robot_volume",           # verified: (0-100)
    "infrared": "get_sensors_infrared_value",  # verified name; response shape unchecked
    "touch": "get_sensors_touch_value",     # UNVERIFIED
}

# Greeting intensity {0: low, 1: medium, 2: high} as volume, gesture and line.
# Gesture names are UNVERIFIED (the robot's built-in motion list varies by
# firmware); a gesture that fails is skipped, the volume and speech still happen.
VOLUME = {0: 35, 1: 65, 2: 100}
GESTURE = {0: None, 1: "raise", 2: "wave"}
GREETING = {0: "Hi.", 1: "Hello, nice to see you.", 2: "Hello! Great to see you!"}

# Physical distances (metres) the operator's keys stand for, before scaling.
# One value inside each proxemic bin: intimate, personal, social, far.
_BIN_KEYS = {"i": 0.30, "p": 0.65, "s": 1.20, "f": 1.80}

# Closest the robot will walk, in model metres (the simulation clamps at 0.20).
# Below this a "closer" is not executed: on a table it would walk into the person.
_MIN_APPROACH = 0.30


class YanAPIError(RuntimeError):
    pass


class KeyboardOperator:
    """Manual sensing, for when the sensor pack is missing.

    Each step it asks for whatever the robot cannot sense itself. Distance is
    given as a proxemic zone (i/p/s/f), already in model units, so no scaling
    applies; touch and leaving are single keys.
    """

    def __init__(self, ask_distance: bool = True, ask_touch: bool = True,
                 prompt: Callable[[str], str] = input):
        self.ask_distance = ask_distance
        self.ask_touch = ask_touch
        self._prompt = prompt

    def read(self) -> dict:
        parts = []
        if self.ask_distance:
            parts.append("zone i/p/s/f")
        if self.ask_touch:
            parts.append("t = touch")
        parts.append("l = person left")
        while True:
            keys = self._prompt(f"  [{', '.join(parts)}] > ").strip().lower()
            out = {"touch": "t" in keys, "left": "l" in keys}
            if self.ask_distance:
                zone = [k for k in keys if k in _BIN_KEYS]
                if not zone and not out["left"]:
                    print("    need a zone: i, p, s or f")
                    continue
                out["distance"] = _BIN_KEYS[zone[0]] if zone else _BIN_KEYS["f"]
            return out

    def wait(self, message: str) -> None:
        self._prompt(f"  {message} [enter] ")


class YansheeRobot(RobotInterface):
    def __init__(
        self,
        ip: str = "127.0.0.1",
        api=None,
        operator: KeyboardOperator | None = None,
        distance_scale: float = 1 / 3,
        absent_beyond: float = 1.2,   # physical metres; farther reads as "nobody there"
        walk_repeat: int = 2,         # walk cycles per closer/back step
        settle_s: float = 1.0,        # pause after a reset when no operator confirms it
        on_reset: Callable[[], None] | None = None,   # dry run: reset the fake world
        verbose: bool = True,
    ):
        if api is None:
            try:
                import YanAPI as api   # copied from the robot, see module docstring
            except ImportError as e:
                raise ImportError(
                    "YanAPI.py not found. Copy it from the robot "
                    "(/usr/lib/python3.5/YanAPI.py) into this directory.") from e
        self._api = api
        self._op = operator
        self._scale = distance_scale
        self._absent_beyond = absent_beyond
        self._walk_repeat = walk_repeat
        self._settle_s = settle_s
        self._on_reset = on_reset
        self._verbose = verbose
        self._intensity = 0
        self._left = False
        self._last_distance: float | None = None
        self._call("init", ip)

    # --- YanAPI plumbing --------------------------------------------------
    def _call(self, key: str, *args, **kwargs):
        name = CALLS[key]
        fn = getattr(self._api, name, None)
        if fn is None:
            raise YanAPIError(
                f"YanAPI has no '{name}' (used for {key}). Run dir(YanAPI) on the "
                f"robot and correct CALLS in src/yanshee_robot.py.")
        return fn(*args, **kwargs)

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"    yanshee: {msg}")

    # --- actuation --------------------------------------------------------
    def say(self, text: str) -> None:
        self._call("tts", text, True)

    def perform(self, action: str) -> None:
        if action == "closer":
            if self._last_distance is not None and self._last_distance < _MIN_APPROACH:
                self._log("closer: already at minimum distance, not walking")
                return
            self._call("motion", name="walk", direction="forward", repeat=self._walk_repeat)
        elif action == "back":
            self._call("motion", name="walk", direction="backward", repeat=self._walk_repeat)
        elif action in ("intensity_up", "intensity_down"):
            step = 1 if action == "intensity_up" else -1
            self._intensity = min(2, max(0, self._intensity + step))
            self._greet()
        elif action == "hold":
            pass
        else:
            raise ValueError(f"unknown action: {action}")
        self._log(action)

    def _greet(self) -> None:
        level = self._intensity
        self._call("volume", VOLUME[level])
        if GESTURE[level]:
            try:
                self._call("motion", name=GESTURE[level], repeat=1)
            except Exception as e:   # unknown gesture name: greet without it
                self._log(f"gesture '{GESTURE[level]}' failed ({e}); skipped")
        self.say(GREETING[level])

    # --- sensing ----------------------------------------------------------
    def sense(self) -> Signals:
        manual = self._op.read() if self._op else {}

        if "distance" in manual:
            distance = manual["distance"]      # operator zones are model units
            present = not manual["left"]
        else:
            physical = _first_number(self._call("infrared"))
            if physical > 3:                   # the sensor reports centimetres
                physical /= 100
            distance = physical / self._scale
            present = physical < self._absent_beyond and not manual.get("left", False)

        if self._op and self._op.ask_touch:
            touch = manual["touch"]
        else:
            touch = bool(_first_number(self._call("touch")))

        self._left = self._left or not present
        self._last_distance = distance
        self._log(f"distance {distance:.2f} (model m)  touch {touch}  present {present}")
        return Signals(distance=distance, touch=touch, present=not self._left)

    def get_intensity(self) -> int:
        return self._intensity

    # --- episode lifecycle ------------------------------------------------
    def reset_interaction(self) -> None:
        """Yanshee cannot re-home itself, so the operator places it back."""
        self._intensity = 0
        self._left = False
        self._last_distance = None
        self._call("volume", VOLUME[0])
        if self._on_reset:
            self._on_reset()
        if self._op:
            self._op.wait("place Yanshee at the start position, facing the person")
        else:
            time.sleep(self._settle_s)


def _first_number(response) -> float:
    """YanAPI answers with nested dicts ({'code': 0, 'data': {...}}); the exact
    shape per sensor is unchecked, so take the first number under 'data'."""
    if isinstance(response, dict) and "code" in response and response["code"] != 0:
        raise YanAPIError(f"robot returned an error: {response}")
    stack = [response.get("data", response) if isinstance(response, dict) else response]
    while stack:
        x = stack.pop(0)
        if isinstance(x, bool):
            return float(x)
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, dict):
            stack.extend(v for k, v in x.items() if k not in ("id", "timestamp"))
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
    raise YanAPIError(f"no numeric reading in response: {response}")
