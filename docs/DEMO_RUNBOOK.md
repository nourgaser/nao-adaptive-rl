# Demo runbook

For the live session. Written so the demo can be driven without thinking on the
spot, and so nothing shown depends on a number being regenerated live.

**Total time: ~6 minutes** of running, plus discussion.

---

## Before the room

```bash
cd seminar/project
python -m pytest tests -q          # 15 tests, ~1 s — proves it is not a mock
ls results/                        # three figures should already exist
```

If `results/` is empty, regenerate once **before** the session (it takes a few
minutes; do not do this live):

```bash
python -m experiments.personalization --plots
```

Have open in tabs: `results/personalization_comparison.png`,
`results/learned_policies.png`, `results/adaptation_speed.png`.

---

## The 30-second framing

> "A robot that greets everyone the same way will be wrong for most people.
> Some want you close, some want distance. We asked whether the robot can
> *learn each person's preference from their reactions alone* — no profile, no
> questionnaire — and whether that actually beats the best single policy you
> could possibly ship."

---

## Run 1 — the learned policy is sensible (~40 s)

```bash
python -m experiments.personalization --episodes 400 --seeds 2
```

Point at the policy table. The thing to say:

> "Same algorithm, three different people, three different learned policies.
> The reserved user's column says *back off*; the sociable user's says
> *get closer*. Nobody programmed that — it came from head-touch feedback."

This run is short on purpose: it demonstrates the mechanism, not the result.

---

## Run 2 — the headline (use the saved figure, do not re-run)

Show `results/personalization_comparison.png` and read the table from the
README. The line that matters:

> "The best possible *single* policy is optimal for the reserved user and
> **worse than doing nothing** for the other two. That is the case for
> personalisation — not that it is a bit better on average, but that a single
> policy actively fails most users."

**Anticipated question — "why is one-size-fits-all worse than the best single
policy?"**
Because training on a mixture of users gives contradictory feedback for the same
state, so it converges to a compromise that suits nobody. The "best single
policy" row is the strongest possible opponent: we picked the best per-user
policy and applied it to everyone. We beat that, not a strawman.

---

## Run 3 — the negative result (talk, no run)

Show `results/adaptation_speed.png`.

> "We expected warm-starting a new user from other users' experience to speed
> things up. It doesn't. At equal budget it matches from-scratch; on a small
> budget it's worse. We think that's because transfer needs task competence and
> preference to be *separable* — and in proxemics, the task IS the preference.
> There's nothing generic to carry over."

Say this deliberately. Reporting it is the point.

---

## If asked to prove it is not staged

```bash
python -m pytest tests -q -k regression   # incl. the one-sense-per-step test
git log --oneline | head
```

The environment enforces exactly one `sense()` call per step — an earlier bug
double-sampled the user's reaction and inflated results. The test exists so it
cannot come back.

---

## Known limits — say these before being asked

- **Simulation only.** No NAO hardware was available; the users are simulated
  with hand-specified comfort zones. So this measures whether the *learning*
  works, not whether real people behave like the model.
- **Tabular Q-learning**, small discrete state/action space. Deliberate: it is
  interpretable and converges on a laptop. Deep RL is future work.
- **Three users**, not a population. The claim is about the mechanism, not an
  effect size that generalises to real users.
- **No subjective measures.** Real HRI would need Godspeed or similar; reward is
  a proxy for comfort, not comfort itself.

---

## Fallback if nothing can be run

The three figures in `results/` plus the README table carry the whole story.
The abstract (`seminar/abstract/abstract.pdf`) is self-contained and states the
result, the negative result, and the scope limits.
