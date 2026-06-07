# NAO Adaptive RL

Reinforcement learning for **simple adaptive robot behavior** on the NAO
humanoid. Course project for the *Interactive & Social AI for Humanoid Robots*
seminar (GIU, SS 2026) --- Topic 6 / Track 8: Learning, Personalization &
Adaptation in HRI.

**Authors:** Nour Gaser, Eman

---

## What this is (and what it is not yet)

We have **not** locked a single experiment. By design. The seminar asks us to
shape our own research question, so we want that decision to come out of the
discussion with the committee.

What we *have* done is build the **common plumbing** that every candidate
direction shares: a robot-agnostic RL loop, a Gym-style environment, a hardware
abstraction layer, and a runnable laptop simulation. So the openness is
**readiness, not indecision** --- we can commit to any of the directions below
(or a new one the committee suggests) and start fine-tuning immediately.

The one thing that *is* implemented end-to-end is an **illustrative** experiment
(adaptive greeting / social distance), used to prove the architecture works and
that tabular Q-learning converges from social signals alone.

## Candidate directions (to be decided at the seminar)

| Direction               | Adaptive behavior                  | Reward signal                | Baseline             |
|-------------------------|------------------------------------|------------------------------|----------------------|
| **A** Greeting/proxemics| approach distance + greeting intensity | head touch + sonar distance | fixed scripted greet |
| **B** Tutoring/pace     | speech rate, pauses, difficulty    | answer correctness + engagement | fixed-pace tutor  |
| **C** Engagement max.   | pick behavior from a repertoire    | change in gaze + proximity   | random / fixed rotation |

All three are the **same RL loop**; only the state features, actions, and reward
wiring differ. See [`docs/architecture.md`](docs/architecture.md).

## Quickstart

```bash
pip install -r requirements.txt
python -m experiments.train_greeting
```

Expected output: a learning curve, then the learned policy beating a scripted
baseline by a wide margin, e.g.

```
Evaluation (avg reward over 200 interactions):
  scripted baseline :   1.95
  learned policy    :  10.85
  improvement       :  +8.90

Learned greedy policy (state -> action):
  dist=intimate  intensity=...  ->  back        # learned: don't crowd
  dist=personal  intensity=med  ->  hold        # learned: stay in the comfortable zone
  dist=social    intensity=...  ->  closer      # learned: approach
  ...
```

The agent is told **nothing** about the person's comfort preference --- it
discovers "don't crowd, settle in the personal zone at medium intensity" purely
from the head-touch and presence signals.

## Repository layout

```
nao-adaptive-rl/
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   └── architecture.md          # the layered design + NAOqi mapping
├── src/
│   ├── robot_interface.py        # abstract contract (the key seam)
│   ├── mock_robot.py             # laptop impl + simulated person
│   ├── nao_robot.py              # real-NAO impl (NAOqi stub, documented)
│   ├── envs/
│   │   └── greeting_env.py       # Gym-style greeting/proxemics MDP
│   └── agents/
│       └── q_learning.py         # tabular Q-learning
└── experiments/
    └── train_greeting.py         # runs the full loop; reports convergence
```

## Going from laptop to robot

The RL code never imports a concrete robot. To run on a (virtual or real) NAO,
change one line in the experiment:

```python
# from src.mock_robot import MockRobot;  robot = MockRobot()
from src.nao_robot import NaoRobot
robot = NaoRobot(ip="<robot-ip>")        # everything else is unchanged
```

`NaoRobot` is currently a documented stub (the NAOqi SDK is not installed in our
prototyping environment, and the campus robot version is still to be confirmed).
Each method shows the exact NAOqi proxy call that belongs there.

## Roadmap

- [x] Robot-agnostic architecture + hardware abstraction layer
- [x] Tabular Q-learning converging on the illustrative greeting MDP (laptop)
- [x] Scripted baseline for comparison
- [ ] **Decide the experiment with the committee**
- [ ] Move to a virtual NAO (Choregraphe / Webots)
- [ ] Confirm campus NAO version + hardware access; fine-tune on the real robot
- [ ] Within-subject user study; combine objective metrics + Godspeed questionnaire

## Key references

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018.
- Mitsunaga et al., *Robot behavior adaptation for HRI based on policy gradient RL*, IROS 2005 / IEEE T-RO 2008.
- Patompak et al., *Learning Proxemics for Personalized Human–Robot Social Interaction*, Int. J. Social Robotics, 2020.
- Irfan et al., *Personalization in Long-Term HRI* (HRI 2019) and *LEAP-HRI* (HRI 2021 companion).
- Tozadore et al., *Towards Adaptation and Personalization in Task-Based HRI*, LARS/SBR 2018.
- Pollmann et al., *Entertainment vs. manipulation*, Technological Forecasting & Social Change, 2023.

## License

MIT --- see [LICENSE](LICENSE).
