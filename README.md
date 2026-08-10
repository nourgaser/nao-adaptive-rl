# NAO Adaptive RL

Reinforcement learning for **simple adaptive robot behavior** on the NAO
humanoid. Course project for the *Interactive & Social AI for Humanoid Robots*
seminar (GIU, SS 2026) --- Topic 6 / Track 8: Learning, Personalization &
Adaptation in HRI.

**Authors:** Nour Gaser, Eman Saleh

---

## What this is

A simulation study of **whether a robot should learn one policy for everyone or
one policy per person**, in the proxemics/greeting setting: how close to
approach, and how intense a greeting to give.

Three simulated users have different, hidden comfort preferences (reserved,
neutral, sociable). The robot is told nothing about them and must infer
behaviour from social signals alone — head touch and presence.

**The headline result.** The best possible *single* policy is optimal for one
user and worse than doing nothing for the other two. Personalising recovers
that, and the gap is large:

| condition | reserved | neutral | sociable |
|---|---|---|---|
| scripted baseline | −1.26 | 2.03 | −0.75 |
| one policy, trained on all users | 6.45 | −0.75 | −0.75 |
| best single policy | **11.81** | −0.75 | −0.75 |
| **personalised (one per user)** | **11.83** | **10.31** | **8.34** |

Mean reward per interaction, 8 seeds, 6000 episodes. Personalisation gains
**+11.06** and **+9.09** on the two users a single policy cannot serve.

**A negative result we report rather than hide.** Warm-starting a new user's
policy from other users' experience does **not** help — at equal training budget
it matches learning from scratch, and on a small budget it is worse. Our reading:
transfer works when task competence and preference are separable, and in
proxemics the task *is* the preference, so there is nothing to transfer. This is
the most interesting thing in the study to discuss.

Scoped out deliberately, and stated in the abstract rather than quietly dropped:
deep RL, real hardware, human participants, and subjective (Godspeed) measures.

## Quickstart

```bash
pip install -r requirements.txt
python -m experiments.personalization --plots     # the main result
python -m experiments.train_greeting              # the single-user warm-up
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
