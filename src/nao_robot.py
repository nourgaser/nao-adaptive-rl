"""
nao_robot.py
============

The real-hardware implementation of `RobotInterface`, wrapping the NAOqi SDK.

It is intentionally left as a documented *stub*: the NAOqi Python SDK is not
installed in our prototyping environment (and the robot version on campus is
still to be confirmed). Every method shows exactly which NAOqi proxy/API call
would go there, so that swapping `MockRobot()` -> `NaoRobot(ip=...)` in an
experiment is a one-line change once hardware is available.

NAOqi reference (Aldebaran docs):
  ALMotion        - joint angles, stiffness, moveTo, posture
  ALMemory        - central blackboard; read tactile-sensor events
  ALSonar         - left/right ultrasonic distance
  ALTextToSpeech  - speech synthesis (.say, volume via setParameter)
  ALFaceDetection - presence / engagement
"""

from .robot_interface import RobotInterface, Signals

# Tactile-sensor memory keys (NAOqi). Reading these via ALMemory tells us
# whether the head was touched this step.
_HEAD_TACTILE_KEYS = (
    "FrontTactilTouched",
    "MiddleTactilTouched",
    "RearTactilTouched",
)


class NaoRobot(RobotInterface):
    def __init__(self, ip: str = "127.0.0.1", port: int = 9559):
        self._intensity = 0
        # --- connect to NAOqi (uncomment on a machine with the SDK) -------
        # import qi
        # self._session = qi.Session()
        # self._session.connect(f"tcp://{ip}:{port}")
        # self._motion = self._session.service("ALMotion")
        # self._memory = self._session.service("ALMemory")
        # self._sonar  = self._session.service("ALSonar")
        # self._tts    = self._session.service("ALTextToSpeech")
        # self._sonar.subscribe("nao_adaptive_rl")
        raise NotImplementedError(
            "NaoRobot is a stub. Install the NAOqi SDK and uncomment the "
            "connection code to run on a real or virtual NAO."
        )

    def say(self, text: str) -> None:
        # self._tts.say(text)
        ...

    def perform(self, action: str) -> None:
        # if action == "closer":  self._motion.moveTo(0.30, 0, 0)
        # elif action == "back":  self._motion.moveTo(-0.30, 0, 0)
        # elif action == "intensity_up":   self._intensity = min(2, self._intensity + 1)
        # elif action == "intensity_down": self._intensity = max(0, self._intensity - 1)
        # elif action == "hold":  pass
        ...

    def sense(self) -> Signals:
        # left  = self._memory.getData("Device/SubDeviceList/US/Left/Sensor/Value")
        # right = self._memory.getData("Device/SubDeviceList/US/Right/Sensor/Value")
        # distance = min(left, right)
        # touch = any(self._memory.getData(k) > 0.5 for k in _HEAD_TACTILE_KEYS)
        # present = self._face_present()   # via ALFaceDetection
        # return Signals(distance=distance, touch=touch, present=present)
        ...

    def get_intensity(self) -> int:
        return self._intensity

    def reset_interaction(self) -> None:
        # self._motion.moveTo(...)  # re-home to a fixed start pose
        self._intensity = 0
