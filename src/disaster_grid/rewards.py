"""
src/disaster_grid/rewards.py
=============================
Three independent reward verifiers for the disaster_grid GRPO training loop.

Design philosophy — why three verifiers?
-----------------------------------------
A single monolithic reward function is trivially hackable: the agent learns
whatever shortcut maximises the scalar without learning the intended behaviour.
Splitting the reward across three *orthogonal* verifiers forces the agent to
satisfy all three constraints simultaneously, which is exponentially harder to
game than any one of them in isolation.

* **R1 (health)** — *did the city actually get better?*  Anchors the signal
  to observable world outcomes.  An agent cannot inflate R1 by taking busy
  actions that look purposeful but leave entropy unchecked.
* **R2 (efficiency)** — *was the improvement worth the energy cost?*  Prevents
  the "lucky repair" failure mode where the agent stumbles onto a broken sector
  after 30 wasted moves and still earns a high health delta.
* **R3 (format)** — *did the agent communicate in the expected schema?*  Acts
  as a hard prerequisite: an agent that cannot reliably emit valid JSON cannot
  earn positive rewards from R1 or R2, because its actions never execute.  The
  harsh ``-2.0`` penalty (larger in magnitude than any single-step R1 or R2
  gain) ensures that schema compliance is always the dominant priority during
  early GRPO training.

The three verifiers are kept as module-level functions rather than methods on a
class so that the GRPO trainer can compose or replace individual verifiers
without subclassing — a critical property for rapid hackathon iteration.

Reward scale reference (approximate single-step ranges)
---------------------------------------------------------
+-----------+----------------------------------------------+-------------------+
| Verifier  | Scenario                                     | Score             |
+===========+==============================================+===================+
| R1 health | Perfect repair, no entropy hits repaired sec | ≈ +1.0            |
|           | Entropy hits two unrepaired sectors          | ≈ -0.4            |
|           | Net-zero (repair cancelled by entropy)       |   0.0             |
+-----------+----------------------------------------------+-------------------+
| R2 eff.   | Repair after direct 1-move approach          | ≈ +1.47           |
|           | Repair after 10 wasted moves                 | ≈ +0.21           |
|           | Recharge / WAIT / error (no energy spent)    |   0.0             |
+-----------+----------------------------------------------+-------------------+
| R3 format | Valid JSON, valid ActionType                 |  +1.0             |
|           | Malformed JSON or hallucinated action key    |  -2.0             |
+-----------+----------------------------------------------+-------------------+
"""

from __future__ import annotations

from typing import Any

from .models import StepResult

# ── Verifier weights (hackathon tuning dials) ─────────────────────────────────
#
# Adjust these at the top of the file rather than hunting through the
# ``compute_reward`` body.  Standard starting point for GRPO on this task:
#
#   w_health     = 1.0  — primary objective signal
#   w_efficiency = 0.5  — secondary shaping signal (half weight to avoid
#                         over-penalising legitimate recharge trips)
#   w_format     = 1.0  — schema compliance; equal weight to health because
#                         an agent that cannot communicate cannot act
#
_W_HEALTH: float = 1.0
_W_EFFICIENCY: float = 0.5
_W_FORMAT: float = 1.0

# ── R2 scaling constant ───────────────────────────────────────────────────────
#
# ``city_health_after`` and ``city_health_before`` are *averages* over 25
# sectors (range 0–100).  Multiplying the positive delta by 25 converts the
# average improvement back into a total health-points figure, making the
# numerator and denominator of the efficiency ratio dimensionally consistent:
#   numerator   → total health points gained   (0–25 per REPAIR in isolation)
#   denominator → energy points spent          (1–100 per step)
#
_HEALTH_SCALE: float = 25.0


# ─────────────────────────────────────────────────────────────────────────────
# Verifier 1 — Objective (Health)
# ─────────────────────────────────────────────────────────────────────────────


def get_health_reward(step_result: StepResult) -> float:
    """
    Measure whether the city is objectively healthier after this step.

    The reward is the raw delta of average city health across all 25 sectors::

        R1 = city_health_after − city_health_before

    Range
    -----
    * Positive (≈ 0.0 to +1.0) when a REPAIR outweighs entropy.
    * Negative (≈ -0.4 per step) when the agent does nothing and entropy
      degrades two sectors by 5 points each (net average loss ≈ 0.4).
    * Near-zero when the agent's repair precisely cancels entropy damage.

    Why this alone is insufficient (reward-hacking risk)
    -----------------------------------------------------
    R1 does not care *how* the health gain was achieved.  A lucky agent could
    wander randomly, occasionally stumble onto a critical sector, issue a
    REPAIR, and earn a positive R1 without ever demonstrating strategic
    routing.  R2 closes this gap by penalising the energy cost of reaching
    the repaired sector.

    Parameters
    ----------
    step_result : StepResult
        The verifier receipt produced by ``CityGrid.step()``.

    Returns
    -------
    float
        Signed health delta.  Negative values are valid and expected on steps
        where the agent recharged, moved without repairing, or issued an
        invalid action.
    """
    return step_result.city_health_after - step_result.city_health_before


# ─────────────────────────────────────────────────────────────────────────────
# Verifier 2 — Strategy (Efficiency)
# ─────────────────────────────────────────────────────────────────────────────


def get_efficiency_reward(step_result: StepResult) -> float:
    """
    Measure whether the health improvement was worth its energy cost.

    The reward is the ratio of total health points gained to energy points
    spent::

        health_gained = max(0, city_health_after − city_health_before) × 25
        energy_spent  = energy_before − energy_after
        R2            = health_gained / energy_spent   (if energy_spent > 0)
                      = 0.0                            (otherwise)

    Why multiply the health delta by 25?
    -------------------------------------
    ``city_health_before`` and ``city_health_after`` are *averages* over 25
    sectors (domain [0, 100]).  Multiplying the positive delta by 25 converts
    the mean improvement back into a total health-points figure so the
    numerator and denominator are dimensionally comparable:

    * Numerator:   total health points added this step  (max ≈ 25 per REPAIR)
    * Denominator: energy points consumed this step     (1 for WAIT, 15 for
                   REPAIR, 2 per MOVE, etc.)

    Edge cases
    ----------
    ``energy_spent <= 0`` occurs on three legitimate step types:

    1. **RECHARGE at sector 12** – energy increased; ``energy_after >
       energy_before``.  No efficiency score is appropriate because the agent
       gained a resource rather than spending one.
    2. **Parse error** – no energy was deducted; the step was a no-op.
       Penalising efficiency here would double-penalise the agent on top of
       R3's format penalty.
    3. **Energy already at 0** – the episode is terminating; the ratio would
       be undefined.

    In all three cases the function returns ``0.0`` (neutral) rather than an
    error, keeping R3 as the sole penalty signal for errors and R1 as the
    health signal for recharge steps.

    Why this prevents the "lucky repair" failure mode
    --------------------------------------------------
    An agent that spends 20 energy units on ``MOVE`` actions to reach a
    single critical sector and then REPAIRs it earns roughly
    ``(+25 health) / (20 + 15 energy) ≈ 0.71`` in R2 — significantly lower
    than an agent that routes directly (2 moves + REPAIR):
    ``(+25 health) / (4 + 15 energy) ≈ 1.32``.  GRPO will push the policy
    toward the direct-routing strategy even if both agents produce the same
    R1 health delta.

    Parameters
    ----------
    step_result : StepResult
        The verifier receipt produced by ``CityGrid.step()``.

    Returns
    -------
    float
        Non-negative efficiency ratio, or ``0.0`` when no energy was spent.
    """
    health_gained: float = (
        max(0.0, step_result.city_health_after - step_result.city_health_before)
        * _HEALTH_SCALE
    )
    energy_spent: int = step_result.energy_before - step_result.energy_after

    if energy_spent <= 0:
        return 0.0

    # Prevent the agent from getting a massive score multiplier by repairing
    # with less than 15 energy remaining right before the episode terminates.
    if health_gained > 0 and energy_spent < 15:
        energy_spent = 15

    return health_gained / energy_spent

# ─────────────────────────────────────────────────────────────────────────────
# Verifier 3 — Guardrail (Format & Syntax)
# ─────────────────────────────────────────────────────────────────────────────


def get_format_reward(step_result: StepResult) -> float:
    """
    Enforce strict adherence to the ``AgentAction`` JSON schema.

    The binary reward is asymmetric by design::

        R3 = +1.0   if action_parsed is not None AND is_error is False
           = -2.0   otherwise

    Why ``-2.0`` rather than ``0.0`` for a failed parse?
    ------------------------------------------------------
    A neutral penalty (``0.0``) would allow the agent to achieve a positive
    total reward purely from occasional lucky health improvements (R1) while
    ignoring the schema entirely.  Setting the failure penalty to ``-2.0``
    (larger in magnitude than the maximum single-step R1 gain of ≈ +1.0)
    ensures that schema non-compliance *always* produces a negative total
    reward, even when entropy happens not to degrade any sector on that step.
    This makes format compliance the lexicographically dominant priority
    during early GRPO training, before the policy has learned to generate
    valid JSON reliably.

    What counts as a format failure?
    ---------------------------------
    * ``is_error=True``: the environment flagged this step as invalid.  This
      covers malformed JSON, unrecognised ``ActionType`` strings (e.g.
      ``"MOVE_NE"``), missing required fields (``action`` or ``reasoning``),
      and ``reasoning`` strings shorter than 10 characters.
    * ``action_parsed is None``: parsing raised an exception that set
      ``action_parsed`` to ``None`` in the ``StepResult``.  This is redundant
      with ``is_error`` for parse failures but provides a belt-and-suspenders
      check for any future code path that sets ``action_parsed=None`` without
      explicitly setting ``is_error=True``.

    Note: RECHARGE-outside-sector-0 sets ``is_error=True`` even though the
    JSON was syntactically valid.  This is intentional — the agent is
    penalised for spatial reasoning errors that produce semantically invalid
    actions, not just syntactic ones.

    Parameters
    ----------
    step_result : StepResult
        The verifier receipt produced by ``CityGrid.step()``.

    Returns
    -------
    float
        ``+1.0`` for schema-compliant steps, ``-2.0`` for any violation.
    """
    # Use local bindings so static analyzers always see a concrete symbol flow.
    is_error = step_result.is_error
    parsed_action = step_result.action_parsed
    if is_error or parsed_action is None:
        return -2.0
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Master verifier — Aggregation
# ─────────────────────────────────────────────────────────────────────────────


def compute_reward(info: dict[str, Any]) -> float:
    """
    Aggregate the three independent verifier scores into a single GRPO reward.

    This is the only function called by ``grpo_trainer.py``.  It extracts the
    ``StepResult`` receipt from the environment's ``info`` dict, calls the
    three verifiers, and returns the weighted sum::

        reward = w_health     × R1(step_result)
               + w_efficiency × R2(step_result)
               + w_format     × R3(step_result)

    With the default weights ``(1.0, 0.5, 1.0)`` the approximate reward range
    per step is roughly ``[−3.4, +3.7]``:

    * **Best case** (direct repair, valid schema): R1 ≈ +1.0, R2 ≈ +1.47,
      R3 = +1.0 → total ≈ **+3.24**.
    * **Worst case** (bad JSON, heavy entropy): R1 ≈ −0.4, R2 = 0.0,
      R3 = −2.0 → total ≈ **−2.4**.

    This spread gives GRPO a clear gradient signal to differentiate between
    rollouts — a signal-to-noise ratio that would collapse if the three
    verifiers were merged into a single hand-crafted formula.

    Robustness contract
    -------------------
    * If ``"step_result"`` is absent from ``info`` (e.g. the environment
      returned early due to an internal error), the function returns ``0.0``
      rather than raising — the GRPO trainer can safely continue sampling
      without crashing the batch.
    * ``StepResult(**info["step_result"])`` re-validates the receipt through
      Pydantic at aggregation time.  If the environment somehow produced an
      out-of-range value (e.g. ``energy_before=150``), this raises a
      ``ValidationError`` that surfaces immediately rather than corrupting the
      gradient computation silently.

    Tuning guidance
    ---------------
    Modify ``_W_HEALTH``, ``_W_EFFICIENCY``, and ``_W_FORMAT`` at the top of
    this module rather than editing this function body.  Suggested schedule for
    a 30-hour hackathon:

    * Hours  0–10: ``w_format=2.0`` — prioritise schema compliance above all.
    * Hours 10–20: ``w_format=1.0, w_health=1.0`` — balanced regime once the
      model reliably emits valid JSON.
    * Hours 20–30: ``w_efficiency=1.0`` — maximise strategic routing now that
      the policy is schema-stable.

    Parameters
    ----------
    info : dict[str, Any]
        The ``info`` dictionary returned by ``CityGrid.step()``.  Expected to
        contain a ``"step_result"`` key whose value is a ``StepResult``
        serialised by ``model_dump()``.

    Returns
    -------
    float
        Weighted sum of R1, R2, and R3.  Returns ``0.0`` if ``"step_result"``
        is absent from ``info``.
    """
    if "step_result" not in info:
        return 0.0

    step_result = StepResult(**info["step_result"])
    r1: float = get_health_reward(step_result)
    r2: float = get_efficiency_reward(step_result)
    r3: float = get_format_reward(step_result)

    return (_W_HEALTH * r1) + (_W_EFFICIENCY * r2) + (_W_FORMAT * r3)