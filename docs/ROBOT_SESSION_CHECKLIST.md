# Robot session checklist

For the session with the Yanshee before the demo. The code is written and
tested against a fake robot; this session only has to confirm what the fake
cannot: the robot's function names, its sensors, and how far it walks.
Budget about an hour. Bring the laptop with this repository and its `.venv`.

Every fix goes in one place: `CALLS`, `GESTURE` and `VOLUME` at the top of
`src/yanshee_robot.py`.

---

## 1. Connect (10 min)

- [ ] Robot charged, laptop and robot on the **same Wi-Fi**.
- [ ] Robot's IP address (Yanshee mobile app, or ask whoever manages it): `__________`
- [ ] Copy the SDK from the robot into the repository root (login from the
  Yanshee manual or the lab):

```bash
scp pi@<IP>:/usr/lib/python3.5/YanAPI.py .
ssh pi@<IP> 'find / -name YanAPI.py 2>/dev/null'   # if it is not there
```

- [ ] `python -c "import YanAPI"` works. If it complains about a missing
  module (usually `requests`), `uv pip install` it into `.venv`.

## 2. Function names (5 min)

```bash
python -m experiments.yanshee_check --ip <IP> --only names
```

- [ ] Every line in step 1 says `ok`. For any `MISSING`, find the real name in
  the "names containing" lists and put it in `CALLS`.
- [ ] The touch function is the one most likely to be different.

## 3. Sensors (15 min)

```bash
python -m experiments.yanshee_check --ip <IP> --only sensors
```

- [ ] **Is the sensor pack fitted?** If infrared and touch both fail, it is not:
  use `--operator` on the day and skip the rest of this step.
- [ ] Infrared: readings change as a hand moves to about 15, 30 and 50 cm.
  Note them: `15 cm -> ____   30 cm -> ____   50 cm -> ____   nobody -> ____`
- [ ] The readings are centimetres (the code divides by 100 when a reading is
  above 3). If they are something else, adjust `sense()`.
- [ ] "Nobody there" reads above 120 cm. If not, change `absent_beyond`.
- [ ] Touch: the value changes when touched. If touch fails but infrared
  works, use `--operator-touch` on the day.
- [ ] Where is the infrared sensor mounted, and at what height does it see a
  seated person? Decide where the person sits or holds a hand.

## 4. Movement and voice (15 min)

```bash
python -m experiments.yanshee_check --ip <IP> --only motion --walk-repeat 2
python -m experiments.yanshee_check --ip <IP> --only voice
```

- [ ] Walking forward and backward works and the robot stays upright.
- [ ] One "closer" should move about **10 cm** (0.30 m scaled by 1/3). Measured
  with `--walk-repeat 2`: `____ cm`. Choose the repeat count that gives about
  10 cm: `--walk-repeat ___`
- [ ] Gestures `raise` and `wave` exist. If not, pick two from the robot's motion
  list (small for medium, big for high) and put them in `GESTURE`.
- [ ] The three volumes are clearly different in the room. Adjust `VOLUME` if not.

## 5. Full runs (15 min)

Follow `DEMO_RUNBOOK_YANSHEE.md` exactly, with the flags decided above.

- [ ] Run 1 (reserved): stops in the 30-50 cm zone, quiet.
- [ ] Run 2 (sociable): walks up close, loud, and does not collide (it refuses
  to walk closer than about 10 cm).
- [ ] Run 3 (sociable policy, reserved person): the person leaves.
- [ ] Time each run: `____  ____  ____`. Lower `--steps` if too long.
- [ ] Commit the changes to `CALLS` / `GESTURE` / `VOLUME` and push.

## Write down for the demo day

```
IP: ____________   --walk-repeat ___   --scale 0.333
mode: sensors / --operator-touch / --operator
where the person sits: ______________________________
```
