"""
tests/test_env.py
==================
Production-ready pytest suite for the ``CityGrid`` OpenEnv environment.

Coverage
--------
* Environment initialisation invariants
* OpenEnv 5-tuple return contract
* Grid boundary physics (wall collision)
* Parse-error guardrail (malformed action handling)
* REPAIR mechanics (health gain + energy cost)
* RECHARGE mechanics (energy gain, location gating)
* Episode termination conditions
* Reward-verifier integration via ``compute_reward``

Entropy isolation strategy
---------------------------
Several tests assert exact health values after a ``step()`` call.  Because
``_apply_entropy`` randomly degrades two sectors per step, a naive test would
be non-deterministic.  Where exact post-step health matters, the test patches
``random.sample`` (via ``unittest.mock.patch``) to return a fixed pair of
sector indices that do *not* include the sector under observation.  This keeps
tests deterministic without coupling them to a specific RNG seed.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest  # pyright: ignore[reportMissingImports]

from src.disaster_grid.environment import CityGrid
from src.disaster_grid.models import ActionType
from src.disaster_grid.rewards import compute_reward

# ── Shared constants ──────────────────────────────────────────────────────────

_VALID_REASONING = "Analysing critical sectors and choosing the optimal action."
_CRISIS_HEALTH = 20          # value forced onto 5 sectors at reset
_NUM_CRISIS_SECTORS = 5      # how many sectors start in crisis


# ── Helpers ───────────────────────────────────────────────────────────────────

def _action(action_type: ActionType, reasoning: str = _VALID_REASONING) -> str:
    """Serialise an ``AgentAction``-compatible payload to a JSON string."""
    return json.dumps({"action": action_type.value, "reasoning": reasoning})


def _action_dict(action_type: ActionType, reasoning: str = _VALID_REASONING) -> dict:
    """Return an ``AgentAction``-compatible payload as a plain dict."""
    return {"action": action_type.value, "reasoning": reasoning}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def env() -> CityGrid:
    """
    Provide a freshly instantiated and reset ``CityGrid`` for each test.

    Using a fixture (rather than a module-level singleton) guarantees full
    state isolation: no test can accidentally mutate grid health, agent
    position, or step count in a way that affects a subsequent test.
    """
    city = CityGrid()
    city.reset(seed=0)
    return city


# ── Test 1: Initialisation invariants ────────────────────────────────────────


def test_initial_state(env: CityGrid) -> None:
    """
    Verify that ``reset()`` produces the exact starting conditions required
    by the disaster-grid specification.

    Checks
    ------
    * Agent starts at sector 12 (the recharge station).
    * Agent starts with full energy (100).
    * Step counter is at 0.
    * Grid contains exactly ``_NUM_CRISIS_SECTORS`` (5) sectors forced to
      health 20.
    * No sector starts at health 0 (the reset must not accidentally kill any
      sector beyond the forced crisis sectors).
    """
    assert env.agent_pos == 12, "Agent must start at sector 12 (recharge station)."
    assert env.agent_energy == 100, "Agent must start with full energy (100)."
    assert env.step_count == 0, "Step counter must be 0 immediately after reset."

    crisis_count = sum(1 for h in env.grid_health if h == _CRISIS_HEALTH)
    assert crisis_count == _NUM_CRISIS_SECTORS, (
        f"Exactly {_NUM_CRISIS_SECTORS} sectors must start at health "
        f"{_CRISIS_HEALTH}; found {crisis_count}."
    )

    assert len(env.grid_health) == 25, "Grid must contain exactly 25 sectors."
    assert all(h >= 0 for h in env.grid_health), "No sector may start below 0."
    # Sector 12 must never be forced into crisis (it is the recharge station).
    assert env.grid_health[12] > _CRISIS_HEALTH, (
        "Sector 12 (recharge station) must not start in crisis."
    )


# ── Test 2: OpenEnv return-tuple compliance ────────────────────────────────────


def test_step_returns_openenv_tuple(env: CityGrid) -> None:
    """
    Confirm that ``step()`` returns the canonical OpenEnv 5-tuple
    ``(obs, reward, terminated, truncated, info)`` and that ``info``
    contains the verifier receipt required by ``rewards.py``.

    The OpenEnv / Gymnasium spec mandates exactly five return values.
    Violating this contract silently breaks any trainer that unpacks the
    tuple with positional assignment.
    """
    result = env.step(_action(ActionType.MOVE_S))

    assert len(result) == 5, (
        f"step() must return exactly 5 values; got {len(result)}."
    )

    obs, reward, terminated, truncated, info = result

    # Observation must be a non-empty dict
    assert isinstance(obs, dict), "Observation must be a dict."
    assert obs, "Observation dict must not be empty."

    # Reward is always 0.0 from the environment (verifiers handle scoring)
    assert reward == 0.0, "Environment reward must always be 0.0."

    # Truncated is always False in this environment
    assert truncated is False, "truncated must always be False."

    # Info must carry the StepResult receipt
    assert "step_result" in info, (
        "info must contain 'step_result' for the reward verifiers."
    )
    step_result = info["step_result"]
    assert isinstance(step_result, dict), "'step_result' must be a dict."

    # Spot-check key fields expected by rewards.py
    for field in (
        "energy_before",
        "energy_after",
        "city_health_before",
        "city_health_after",
        "is_error",
        "action_attempted",
    ):
        assert field in step_result, (
            f"'step_result' is missing required field '{field}'."
        )


# ── Test 3: Boundary physics — wall collision ─────────────────────────────────


def test_wall_collision_north(env: CityGrid) -> None:
    """
    Verify that MOVE_N at the top wall keeps the agent in place but still
    deducts the movement energy cost.

    Behavioural contract
    --------------------
    * Agent is at sector 0 → (x=0, y=0), the topmost row.
    * MOVE_N would produce y=-1, which is off-grid.
    * The environment must clamp the position to 0 (no movement).
    * The energy cost of -2 is applied regardless of whether movement
      succeeded — this punishes the agent for poor pathfinding without
      requiring explicit boundary detection in the reward verifiers.
    """
    env.agent_pos = 0
    assert env.agent_pos == 0, "Pre-condition: agent must be at sector 0."
    energy_before = env.agent_energy  # 100

    env.step(_action(ActionType.MOVE_N))

    assert env.agent_pos == 0, (
        "Agent must remain at sector 0 after hitting the north wall."
    )
    assert env.agent_energy == energy_before - 2, (
        "Wall collision must still deduct 2 energy (movement penalty applies)."
    )


# ── Test 4: Format guardrail — malformed action ───────────────────────────────


def test_malformed_action_does_not_crash(env: CityGrid) -> None:
    """
    Verify that a schema-violating action payload is handled gracefully.

    Contract
    --------
    * The environment must not raise an exception.
    * ``is_error`` must be ``True`` in the StepResult receipt.
    * The action is treated as a no-op: *no energy is deducted* for the
      failed parse itself (energy cost on error is 0; the R3 format verifier
      applies the ``-2.0`` penalty through the reward signal instead).
    * ``action_parsed`` must be ``None`` (the payload could not be validated).

    Why no energy cost on error?
    ----------------------------
    Deducting energy for a parse error would conflate two independent signals:
    the R3 format penalty already carries the training gradient for schema
    non-compliance.  Adding an energy cost would create an unintended coupling
    between R2 (efficiency) and R3 (format) that makes reward attribution
    ambiguous during GRPO analysis.
    """
    energy_before = env.agent_energy

    _, _, _, _, info = env.step(json.dumps({"bad_key": "INVALID"}))

    step_result = info["step_result"]
    assert step_result["is_error"] is True, (
        "Malformed action must set is_error=True in the StepResult."
    )
    assert step_result["action_parsed"] is None, (
        "action_parsed must be None when the payload fails validation."
    )
    assert env.agent_energy == energy_before, (
        "No energy must be deducted for a parse error "
        "(R3 verifier carries the penalty signal)."
    )


# ── Test 5: REPAIR mechanics ──────────────────────────────────────────────────


def test_repair_increases_health_and_costs_energy(env: CityGrid) -> None:
    """
    Verify that REPAIR adds exactly 25 health to the current sector and
    deducts exactly 15 energy from the agent.

    Entropy isolation
    -----------------
    ``_apply_entropy`` would randomly degrade two sectors *after* the repair,
    potentially hitting sector 12 and producing a non-deterministic post-step
    health value.  This test patches ``random.sample`` to return sectors 1 and
    2 (neither of which is sector 12), ensuring a deterministic assertion
    without coupling to any specific RNG seed.
    """
    env.agent_pos = 12
    env.grid_health[12] = 20
    energy_before = env.agent_energy  # 100

    with patch(
        "src.disaster_grid.environment.random.sample",
        return_value=[1, 2],  # entropy hits sectors 1 and 2, not 12
    ):
        env.step(_action(ActionType.REPAIR))

    assert env.grid_health[12] == 45, (
        f"REPAIR must increase health from 20 to 45 (20 + 25); "
        f"got {env.grid_health[12]}."
    )
    assert env.agent_energy == energy_before - 15, (
        f"REPAIR must cost 15 energy; expected {energy_before - 15}, "
        f"got {env.agent_energy}."
    )


# ── Test 6: REPAIR health cap ─────────────────────────────────────────────────


def test_repair_caps_at_100(env: CityGrid) -> None:
    """
    Verify that REPAIR does not push sector health above the maximum of 100.

    A sector at health 90 should reach 100 (capped), not 115, after a REPAIR
    that nominally adds 25 points.
    """
    env.agent_pos = 5
    env.grid_health[5] = 90

    with patch(
        "src.disaster_grid.environment.random.sample",
        return_value=[1, 2],
    ):
        env.step(_action(ActionType.REPAIR))

    assert env.grid_health[5] == 100, (
        f"REPAIR must cap sector health at 100; got {env.grid_health[5]}."
    )


# ── Test 7: RECHARGE at sector 12 ─────────────────────────────────────────────


def test_recharge_at_station_increases_energy(env: CityGrid) -> None:
    """
    Verify that RECHARGE at sector 12 adds exactly 20 energy and does not
    set ``is_error``.

    The agent starts at sector 12 and at full energy (100), so the first
    RECHARGE would overflow.  We manually set energy to 70 to give a
    visible +20 delta without hitting the cap.
    """
    env.agent_pos = 12
    env.agent_energy = 70

    _, _, _, _, info = env.step(_action(ActionType.RECHARGE))

    assert env.agent_energy == 90, (
        f"RECHARGE at sector 12 must add 20 energy; expected 90, "
        f"got {env.agent_energy}."
    )
    assert info["step_result"]["is_error"] is False, (
        "RECHARGE at sector 12 must not set is_error."
    )


# ── Test 8: RECHARGE outside sector 12 ───────────────────────────────────────


def test_recharge_outside_station_is_penalised(env: CityGrid) -> None:
    """
    Verify that RECHARGE attempted outside sector 12 is flagged as an error
    and costs 1 energy (the wrong-location penalty).

    This test confirms that the spatial gating logic in ``step()`` correctly
    prevents free energy generation from arbitrary positions.
    """
    env.agent_pos = 0   # non-station sector (top-left)
    energy_before = env.agent_energy

    _, _, _, _, info = env.step(_action(ActionType.RECHARGE))

    assert info["step_result"]["is_error"] is True, (
        "RECHARGE outside sector 12 must set is_error=True."
    )
    assert env.agent_energy == energy_before - 1, (
        "RECHARGE outside sector 12 must deduct 1 energy (wrong-location penalty)."
    )


# ── Test 9: RECHARGE energy cap ───────────────────────────────────────────────


def test_recharge_caps_at_100(env: CityGrid) -> None:
    """
    Verify that RECHARGE does not push agent energy above 100.

    An agent at 95 energy should reach 100, not 115, after a valid RECHARGE.
    """
    env.agent_pos = 12
    env.agent_energy = 95

    env.step(_action(ActionType.RECHARGE))

    assert env.agent_energy == 100, (
        f"RECHARGE must cap agent energy at 100; got {env.agent_energy}."
    )


# ── Test 10: WAIT costs 1 energy ──────────────────────────────────────────────


def test_wait_costs_one_energy(env: CityGrid) -> None:
    """
    Verify that WAIT deducts exactly 1 energy per step.

    WAIT is a deliberate "do-nothing" penalty that discourages the agent from
    idling while entropy degrades the city.  The cost must be positive so that
    WAIT is never strictly dominant over REPAIR or MOVE.
    """
    energy_before = env.agent_energy

    env.step(_action(ActionType.WAIT))

    assert env.agent_energy == energy_before - 1, (
        f"WAIT must cost 1 energy; expected {energy_before - 1}, "
        f"got {env.agent_energy}."
    )


# ── Test 11: Step counter increments ──────────────────────────────────────────


def test_step_counter_increments(env: CityGrid) -> None:
    """
    Verify that ``step_count`` increments by exactly 1 per call to ``step()``,
    regardless of whether the action was valid or malformed.
    """
    assert env.step_count == 0

    env.step(_action(ActionType.WAIT))
    assert env.step_count == 1

    env.step(json.dumps({"bad_key": "INVALID"}))
    assert env.step_count == 2


# ── Test 12: Termination — energy depletion ────────────────────────────────────


def test_terminated_when_energy_reaches_zero(env: CityGrid) -> None:
    """
    Verify that the episode terminates (``terminated=True``) when agent energy
    hits 0.

    Energy is set to 2 so a single MOVE action (cost 2) drains it completely.
    A WAIT (cost 1) would leave 1 energy remaining, so MOVE is the safer
    choice here for a deterministic drain.
    """
    env.agent_pos = 5   # row 1 — can move north without any pre-checks
    env.agent_energy = 2

    _, _, terminated, _, _ = env.step(_action(ActionType.MOVE_N))

    assert terminated is True, (
        "Episode must terminate when agent energy reaches 0."
    )


# ── Test 13: Termination — step limit ─────────────────────────────────────────


def test_terminated_at_step_50(env: CityGrid) -> None:
    """
    Verify that the episode terminates at step 50 regardless of agent energy.

    The test fast-forwards ``step_count`` to 49 and executes one final WAIT.
    After that call ``step_count`` becomes 50 and ``terminated`` must be True.
    """
    env.step_count = 49
    env.agent_energy = 100  # plenty of energy — termination is time-based

    _, _, terminated, _, _ = env.step(_action(ActionType.WAIT))

    assert env.step_count == 50
    assert terminated is True, "Episode must terminate at step 50."


# ── Test 14: Observation schema fields ────────────────────────────────────────


def test_observation_contains_required_fields(env: CityGrid) -> None:
    """
    Verify that the observation dict returned by ``step()`` contains all
    fields required by ``GridObservation`` (and therefore by the LLM prompt
    template).

    Missing fields would cause silent KeyErrors deep inside the GRPO trainer's
    prompt assembly loop, making them hard to debug.
    """
    obs, *_ = env.step(_action(ActionType.WAIT))

    required_fields = {
        "step_number",
        "agent_position",
        "agent_energy",
        "current_sector_health",
        "critical_sectors",
        "average_city_health",
    }
    missing = required_fields - obs.keys()
    assert not missing, (
        f"Observation is missing required fields: {missing}"
    )


# ── Test 15: Entropy is applied every step ────────────────────────────────────


def test_entropy_degrades_two_sectors_per_step(env: CityGrid) -> None:
    """
    Verify that exactly two sectors lose health on each step due to entropy.

    The test patches ``random.sample`` to return a fixed pair of sectors
    (indices 3 and 4), sets those sectors to health 50 before the step, and
    asserts they are both at 45 afterwards.  All other sectors must be
    unchanged by entropy (they may be changed by REPAIR, but WAIT does not
    touch health).
    """
    # Force sectors 3 and 4 to a known health
    env.grid_health[3] = 50
    env.grid_health[4] = 50
    health_snapshot = env.grid_health.copy()

    with patch(
        "src.disaster_grid.environment.random.sample",
        return_value=[3, 4],
    ):
        env.step(_action(ActionType.WAIT))

    assert env.grid_health[3] == 45, (
        f"Entropy must reduce sector 3 health from 50 to 45; "
        f"got {env.grid_health[3]}."
    )
    assert env.grid_health[4] == 45, (
        f"Entropy must reduce sector 4 health from 50 to 45; "
        f"got {env.grid_health[4]}."
    )

    # All sectors other than 3 and 4 should be untouched by entropy and WAIT
    for i, (before, after) in enumerate(
        zip(health_snapshot, env.grid_health)
    ):
        if i not in (3, 4):
            assert after == before, (
                f"Sector {i} health changed unexpectedly: {before} → {after}."
            )


# ── Test 16: Entropy floor at 0 ───────────────────────────────────────────────


def test_entropy_does_not_push_health_below_zero(env: CityGrid) -> None:
    """
    Verify that entropy cannot reduce a sector's health below 0.

    A sector at health 3 subjected to a -5 entropy tick must settle at 0,
    not -2.
    """
    env.grid_health[10] = 3

    with patch(
        "src.disaster_grid.environment.random.sample",
        return_value=[10, 11],
    ):
        env.step(_action(ActionType.WAIT))

    assert env.grid_health[10] == 0, (
        f"Entropy must floor sector health at 0; got {env.grid_health[10]}."
    )


# ── Test 17: Reward verifier integration ──────────────────────────────────────


def test_compute_reward_positive_for_successful_repair(env: CityGrid) -> None:
    """
    Integration test confirming that ``compute_reward`` produces a positive
    aggregate reward for a strategically sound REPAIR action.

    A REPAIR on a low-health sector with no wasted movement should satisfy
    all three verifiers:
    * R1 (health): positive delta from the repair.
    * R2 (efficiency): positive ratio (health gained / energy spent).
    * R3 (format): +1.0 for valid JSON schema.
    """
    env.agent_pos = 3
    env.grid_health[3] = 10

    with patch(
        "src.disaster_grid.environment.random.sample",
        return_value=[1, 2],  # entropy avoids sector 3
    ):
        _, _, _, _, info = env.step(_action(ActionType.REPAIR))

    reward = compute_reward(info)
    assert reward > 0.0, (
        f"A successful repair on a critical sector must yield a positive "
        f"aggregate reward; got {reward:.4f}."
    )


def test_compute_reward_negative_for_malformed_action(env: CityGrid) -> None:
    """
    Integration test confirming that ``compute_reward`` produces a negative
    aggregate reward when the action payload is schema-invalid.

    R3 applies a ``-2.0`` format penalty, which must exceed any positive
    contribution from R1 or R2, guaranteeing that format non-compliance is
    always costly regardless of entropy luck.
    """
    _, _, _, _, info = env.step(json.dumps({"broken": True}))

    reward = compute_reward(info)
    assert reward < 0.0, (
        f"A malformed action must yield a negative aggregate reward; "
        f"got {reward:.4f}."
    )


def test_compute_reward_returns_zero_on_missing_step_result() -> None:
    """
    Verify that ``compute_reward`` returns 0.0 gracefully when ``info`` does
    not contain a ``'step_result'`` key.

    This guards against trainer crashes caused by environment misconfigurations
    or custom wrappers that strip the ``info`` dict.
    """
    reward = compute_reward({})
    assert reward == 0.0, (
        f"compute_reward must return 0.0 on empty info; got {reward}."
    )


# ── Test 18: reset() is idempotent ────────────────────────────────────────────


def test_reset_restores_full_state(env: CityGrid) -> None:
    """
    Verify that calling ``reset()`` after several steps fully restores the
    environment to a valid starting state.

    This catches bugs where reset forgets to clear ``step_count`` or leaves
    stale health values from the previous episode.
    """
    # Dirty the environment
    for _ in range(5):
        env.step(_action(ActionType.WAIT))

    assert env.step_count == 5

    # Reset and re-check invariants
    env.reset(seed=1)

    assert env.step_count == 0, "reset() must zero the step counter."
    assert env.agent_energy == 100, "reset() must restore full energy."
    assert env.agent_pos == 12, "reset() must return agent to sector 12."

    crisis_count = sum(1 for h in env.grid_health if h == _CRISIS_HEALTH)
    assert crisis_count == _NUM_CRISIS_SECTORS, (
        "reset() must re-seed exactly 5 crisis sectors."
    )


# ── Test 19: dict action input (non-string) ───────────────────────────────────


def test_step_accepts_dict_action(env: CityGrid) -> None:
    """
    Verify that ``step()`` accepts a plain Python dict as well as a JSON
    string, as required by synthetic data generators and unit test helpers
    that construct actions programmatically.
    """
    result = env.step(_action_dict(ActionType.WAIT))

    assert len(result) == 5, "step() must return 5-tuple for dict action input."
    _, _, _, _, info = result
    assert info["step_result"]["is_error"] is False, (
        "A valid dict action must not produce is_error=True."
    )