"""
yanshee_fake.py
===============

A stand-in for the YanAPI module, so that `YansheeRobot` can be exercised
without the robot: in the tests, and in `experiments/yanshee_demo.py --dry-run`.

It answers the same calls as the real SDK, in the same response shape
(`{'code': 0, 'data': {...}}`), but the "world" behind them is the simulated
person from `MockRobot`: walking moves the robot through that person's comfort
zones, the volume sets the greeting intensity, and the infrared and touch
readings come from the person's hidden profile. Distances are reported in
physical centimetres, scaled down to a tabletop exactly as on the real robot.

What this proves is that the Yanshee port is wired correctly. It says nothing
about the real robot's sensor noise, gait or function names; that is what the
session with the robot is for.
"""

from __future__ import annotations

from .mock_robot import MockRobot
from .user_profile import UserProfile
from .yanshee_robot import VOLUME

_SENSOR_MAX_CM = 150.0   # infrared range of the sensor pack


class FakeYanAPI:
    def __init__(self, profile: UserProfile, distance_scale: float = 1 / 3,
                 seed: int | None = None):
        self.world = MockRobot(profile=profile, sensor_noise=0.0, seed=seed)
        self.scale = distance_scale
        self.calls: list[tuple] = []
        self._touch = False

    def yan_api_init(self, ip):
        self.calls.append(("init", ip))

    def sync_play_motion(self, name="", direction="", repeat=1, **_):
        self.calls.append(("motion", name, direction))
        if name == "walk":
            self.world.perform("closer" if direction == "forward" else "back")
        return {"code": 0, "data": {}}

    def set_robot_volume(self, value):
        self.calls.append(("volume", value))
        level = {v: k for k, v in VOLUME.items()}[value]
        while self.world.get_intensity() < level:
            self.world.perform("intensity_up")
        while self.world.get_intensity() > level:
            self.world.perform("intensity_down")
        return {"code": 0, "data": {}}

    def sync_do_tts(self, text, interrupt=True):
        self.calls.append(("tts", text))
        return {"code": 0, "data": {}}

    def get_sensors_infrared_value(self):
        # One reading of the person per step: the touch that follows belongs to it.
        sig = self.world.sense()
        self._touch = sig["touch"]
        cm = sig["distance"] * self.scale * 100 if sig["present"] else _SENSOR_MAX_CM
        return {"code": 0, "data": {"infrared": [{"id": 1, "value": cm}]}}

    def get_sensors_touch_value(self):
        return {"code": 0, "data": {"touch": [{"id": 1, "value": int(self._touch)}]}}

    def reset(self):
        """Start a new interaction with the same person."""
        self.world.reset_interaction()
