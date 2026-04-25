"""
src/disaster_grid/environment.py
=================================
Core OpenEnv environment for the disaster_grid hackathon project.

Architecture overview
---------------------
``CityGrid`` inherits from ``openenv.Environment`` and implements the standard
``reset`` / ``step`` interface.  It owns the full physics simulation:

* **Grid state** – 25 sector health values mutated by entropy and REPAIR.
* **Agent state** – position and energy, mutated by movement, repair, and
  recharge actions.
* **Observation factory** – ``_get_observation()`` distils the raw grid into
  the token-efficient ``GridObservation`` schema the LLM sees.
* **Receipt factory** – ``step()`` assembles a ``StepResult`` that the
  external reward verifiers in ``rewards.py`` consume.  The environment
  itself always returns ``reward=0.0``; actual reward computation is
  deliberately decoupled so verifier weights can be tuned without touching
  environment logic.

Step contract
-------------
``step(action)`` accepts either a raw JSON string (as emitted by the LLM) or
a pre-parsed ``dict``.  If parsing or Pydantic validation fails the step is
treated as a ``WAIT`` with ``is_error=True`` recorded in the receipt, giving
the R3 format verifier a clean signal without crashing the episode.
"""

from __future__ import annotations

import json
import random
from typing import Any

from pydantic import ValidationError

from .models import ActionType, AgentAction, GridObservation, SectorState, StepResult

# ── Grid constants ────────────────────────────────────────────────────────────

_GRID_SIZE: int = 5                    # edge length of the square grid
_NUM_SECTORS: int = _GRID_SIZE ** 2   # 25 total sectors

# ── Action energy costs and effects ──────────────────────────────────────────

_MOVE_COST: int = 2                    # energy deducted per move attempt (wall or not)
_REPAIR_COST: int = 15                 # energy deducted per REPAIR
_REPAIR_GAIN: int = 25                 # health added to current sector per REPAIR
_RECHARGE_GAIN: int = 20              # energy added per valid RECHARGE (at sector 0)
_RECHARGE_WRONG_COST: int = 1         # energy penalty for RECHARGE outside sector 0
_WAIT_COST: int = 1                    # energy deducted per WAIT

# ── Entropy constants ─────────────────────────────────────────────────────────

_ENTROPY_SECTORS_PER_STEP: int = 2    # sectors degraded each step
_ENTROPY_DAMAGE: int = 5              # health lost per entropy tick

# ── Initial state constants ───────────────────────────────────────────────────

_INIT_HEALTH_LOW: int = 50            # minimum random starting health
_INIT_HEALTH_HIGH: int = 100          # maximum random starting health
_CRISIS_HEALTH: int = 20             # health value forced onto crisis sectors at reset
_NUM_CRISIS_SECTORS: int = 5          # sectors forced into crisis at reset

# ── Episode limit ─────────────────────────────────────────────────────────────

_MAX_STEPS: int = 50


class CityGrid:
    """
    A 5 × 5 disaster-recovery grid environment compliant with the OpenEnv API.

    The city is partitioned into 25 sectors indexed 0–24 in row-major order::

        (0,0) (1,0) (2,0) (3,0) (4,0)   →  indices  0– 4
        (0,1) (1,1) (2,1) (3,1) (4,1)   →  indices  5– 9
        (0,2) (1,2) (2,2) (3,2) (4,2)   →  indices 10–14
        (0,3) (1,3) (2,3) (3,3) (4,3)   →  indices 15–19
        (0,4) (1,4) (2,4) (3,4) (4,4)   →  indices 20–24

    Sector 0 is the top-left corner and serves as the only recharge station.

    Episode dynamics
    ----------------
    Each call to ``step()`` executes the following sequence in strict order:

    1. Snapshot agent and grid state *before* any mutation.
    2. Parse and validate the action string / dict from the LLM.
    3. Execute the action (energy deduction, position update, health change).
    4. Apply entropy (2 random sectors lose 5 health).
    5. Snapshot agent and grid state *after* mutations.
    6. Increment step counter; check termination.
    7. Build and return the ``StepResult`` receipt in ``info``.

    The reward returned by ``step()`` is always ``0.0``.  Real rewards are
    computed externally by the three verifiers in ``rewards.py`` using the
    ``StepResult`` embedded in the ``info`` dict.  This separation means
    verifier weights and reward shaping can be changed without modifying or
    re-testing environment physics.

    Attributes
    ----------
    grid_health : list[int]
        Current health values for all 25 sectors.  Index ``i`` corresponds to
        the sector at coordinate ``_index_to_coord(i)``.
    agent_pos : int
        Flat sector index (0–24) of the agent's current position.
    agent_energy : int
        Current energy reserve (0–100).  The episode terminates immediately
        when this reaches 0.
    step_count : int
        Number of completed steps in the current episode.
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self) -> None:
        """
        Instantiate the environment without starting an episode.

        State variables are set to sentinel values (all zeros / empty list)
        until the caller invokes ``reset()``.  This mirrors the convention used
        by Gymnasium and OpenEnv so the environment can be constructed cheaply
        inside a trainer worker without triggering random-number generation.
        """
      
        self.grid_health: list[int] = [0] * _NUM_SECTORS
        self.agent_pos: int = 0
        self.agent_energy: int = 0
        self.step_count: int = 0

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _index_to_coord(index: int) -> tuple[int, int]:
        """
        Convert a flat row-major sector index to an (x, y) coordinate pair.

        The coordinate system places (0, 0) at the top-left corner:

        * ``x`` is the column (0 = leftmost,  4 = rightmost).
        * ``y`` is the row    (0 = topmost,    4 = bottommost).

        Parameters
        ----------
        index : int
            Flat sector index in [0, 24].

        Returns
        -------
        tuple[int, int]
            ``(x, y)`` where ``x = index % 5`` and ``y = index // 5``.

        Examples
        --------
        >>> CityGrid._index_to_coord(0)
        (0, 0)
        >>> CityGrid._index_to_coord(7)
        (2, 1)
        >>> CityGrid._index_to_coord(24)
        (4, 4)
        """
        return index % _GRID_SIZE, index // _GRID_SIZE

    @staticmethod
    def _coord_to_index(x: int, y: int) -> int:
        """
        Convert an (x, y) coordinate pair to a flat row-major sector index.

        Parameters
        ----------
        x : int
            Column index in [0, 4].
        y : int
            Row index in [0, 4].

        Returns
        -------
        int
            Flat sector index ``y * 5 + x`` in [0, 24].

        Examples
        --------
        >>> CityGrid._coord_to_index(0, 0)
        0
        >>> CityGrid._coord_to_index(2, 1)
        7
        >>> CityGrid._coord_to_index(4, 4)
        24
        """
        return y * _GRID_SIZE + x

    def _get_observation(self) -> GridObservation:
        """
        Distil current world state into the token-efficient ``GridObservation``
        schema consumed by the LLM.

        This method intentionally omits the full ``grid_health`` list from the
        observation.  Passing 25 integers per step at a 50-step horizon costs
        approximately 1 250 extra tokens per episode — nearly doubling prompt
        length in a GRPO batch.  Instead the method computes:

        * **average_city_health** – a single float that summarises the global
          state and correlates with the R1 verifier's reward signal.
        * **critical_sectors** – a sorted list of indices whose health is below
          30, giving the agent the minimum routing information needed to plan
          an optimal repair trajectory.

        Returns
        -------
        GridObservation
            A validated Pydantic model ready for ``model_dump()`` or
            ``model_dump_json()``.
        """
        average_health: float = round(sum(self.grid_health) / _NUM_SECTORS, 2)
        critical: list[int] = sorted(
            i for i, h in enumerate(self.grid_health) if h < 30
        )
        return GridObservation(
            step_number=self.step_count,
            agent_position=self.agent_pos,
            agent_energy=self.agent_energy,
            current_sector_health=self.grid_health[self.agent_pos],
            critical_sectors=critical,
            average_city_health=average_health,
        )

    def _apply_entropy(self) -> None:
        """
        Degrade two randomly chosen distinct sectors by ``_ENTROPY_DAMAGE``
        (5) health points, floored at 0.

        Entropy is applied **after** the agent's action each step.  This
        ordering means a REPAIR action's +25 gain is always partially offset by
        entropy before the post-step snapshot is taken, preventing the agent
        from achieving a perfect +25 net gain on a single step and thus
        maintaining a meaningful resource-management challenge across the full
        50-step horizon.

        The two sectors are chosen without replacement so the same sector
        cannot be hit twice in a single entropy tick (which would amount to
        -10 health and skew the difficulty distribution).
        """
        targets: list[int] = random.sample(
            range(_NUM_SECTORS), _ENTROPY_SECTORS_PER_STEP
        )
        for idx in targets:
            self.grid_health[idx] = max(0, self.grid_health[idx] - _ENTROPY_DAMAGE)

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Begin a new episode and return the initial observation.

        The reset procedure deliberately creates an adversarial starting state:
        five sectors are forced to health 20 (below the critical threshold of
        30) so the agent faces immediate triage decisions rather than deferring
        all repair work to the second half of the episode.  This biases the
        training distribution toward high-action-density rollouts, which
        produce richer GRPO gradient signal than episodes where the agent
        coasts on a healthy grid.

        Parameters
        ----------
        seed : int | None
            If provided, seeds Python's ``random`` module before any stochastic
            operation.  Pass an integer for reproducible evaluation episodes;
            leave as ``None`` during training to sample diverse rollouts.
        options : dict | None
            Reserved for future configuration (e.g. custom entropy rates or
            forced agent starting positions).  Ignored in the current version.

        Returns
        -------
        observation : dict
            ``GridObservation.model_dump()`` representing the initial world
            state.  The agent starts at sector 0 with full energy and faces
            at least five critical sectors.
        info : dict
            Empty dict.  Provided for API compatibility with OpenEnv / Gym.
        """
        if seed is not None:
            random.seed(seed)

        # ── Reset counters ─────────────────────────────────────────────────
        self.step_count = 0
        self.agent_energy = 100
        self.agent_pos = 0

        # ── Randomise grid health ──────────────────────────────────────────
        self.grid_health = [
            random.randint(_INIT_HEALTH_LOW, _INIT_HEALTH_HIGH)
            for _ in range(_NUM_SECTORS)
        ]

        # ── Force five crisis sectors, excluding sector 0 so the recharge
        #    station always starts accessible and reachable ─────────────────
        crisis_candidates: list[int] = random.sample(
            range(1, _NUM_SECTORS), _NUM_CRISIS_SECTORS
        )
        for idx in crisis_candidates:
            self.grid_health[idx] = _CRISIS_HEALTH

        return self._get_observation().model_dump(), {}

    def step(
        self,
        action: str | dict,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Advance the environment by one step and return the standard 5-tuple.

        The method follows a strict execution pipeline to ensure that the
        ``StepResult`` receipt always contains consistent before/after
        snapshots regardless of whether the action was valid:

        1. **Pre-snapshot** – capture ``energy_before`` and
           ``city_health_before`` before any mutation.
        2. **Parse** – attempt to deserialise ``action`` into ``AgentAction``.
           On failure, mark ``is_error=True`` and skip to step 5.  The step
           acts as a free turn: no energy is deducted for an invalid action,
           but entropy still applies and the step counter still increments.
           The error is surfaced to the R3 verifier via ``StepResult``.
        3. **Execute** – apply the validated action's physics (energy cost,
           position update, health change).
        4. **Entropy** – call ``_apply_entropy()`` to degrade two random
           sectors.
        5. **Post-snapshot** – capture ``energy_after`` and
           ``city_health_after``.
        6. **Bookkeeping** – increment ``step_count``; evaluate termination.
        7. **Receipt** – build ``StepResult`` and embed in ``info``.

        Parameters
        ----------
        action : str | dict
            The LLM's response, either as a raw JSON string or a pre-parsed
            dict (the latter is used by the synthetic data generator and unit
            tests).

        Returns
        -------
        observation : dict
            ``GridObservation.model_dump()`` for the state *after* this step.
        reward : float
            Always ``0.0``.  Real rewards are computed by ``rewards.py`` using
            ``info["step_result"]``.
        terminated : bool
            ``True`` when the episode has ended due to energy depletion
            (``agent_energy <= 0``) or the 50-step limit (``step_count >= 50``).
        truncated : bool
            Always ``False``.  Time-limit termination is modelled via
            ``terminated`` so the GRPO trainer does not need to handle the
            truncated case separately.
        info : dict
            Contains ``"step_result"`` → ``StepResult.model_dump()``, the
            verifier receipt used by ``rewards.py`` to compute R1, R2, and R3.

        Notes
        -----
        **Wall collision** – a move that would take the agent off the grid is
        silently blocked (position unchanged) but the energy cost is still
        applied.  This punishes the agent for poor pathing without requiring
        boundary-check logic inside the reward verifiers.

        **RECHARGE outside sector 0** – deducts ``_RECHARGE_WRONG_COST`` (1)
        energy and sets ``is_error=True``.  The deliberately small penalty
        keeps the training signal proportional; a large penalty would dominate
        the R2 efficiency term and cause the agent to avoid RECHARGE entirely.
        """
        # ── 1. Pre-snapshot ───────────────────────────────────────────────
        energy_before: int = self.agent_energy
        city_health_before: float = sum(self.grid_health) / _NUM_SECTORS

        # ── 2. Parse action ───────────────────────────────────────────────
        parsed_action: AgentAction | None = None
        is_error: bool = False
        error_message: str = ""
        action_attempted: str = (
            action if isinstance(action, str) else json.dumps(action)
        )

        try:
            raw: dict = json.loads(action) if isinstance(action, str) else action
            parsed_action = AgentAction(**raw)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            is_error = True
            error_message = (
                f"Action parse failed ({type(exc).__name__}): {exc!s}"
            )

        # ── 3. Execute action (no-op on parse error) ──────────────────────
        if not is_error and parsed_action is not None:
            action_type: ActionType = parsed_action.action
            x, y = self._index_to_coord(self.agent_pos)

            if action_type in (
                ActionType.MOVE_N,
                ActionType.MOVE_S,
                ActionType.MOVE_E,
                ActionType.MOVE_W,
            ):
                # Energy cost is always applied, even on wall collision.
                self.agent_energy = max(0, self.agent_energy - _MOVE_COST)

                new_x, new_y = x, y
                if action_type is ActionType.MOVE_N:
                    new_y = y - 1
                elif action_type is ActionType.MOVE_S:
                    new_y = y + 1
                elif action_type is ActionType.MOVE_E:
                    new_x = x + 1
                elif action_type is ActionType.MOVE_W:
                    new_x = x - 1

                if 0 <= new_x < _GRID_SIZE and 0 <= new_y < _GRID_SIZE:
                    self.agent_pos = self._coord_to_index(new_x, new_y)
                # else: position unchanged; collision already penalised above.

            elif action_type is ActionType.REPAIR:
                self.agent_energy = max(0, self.agent_energy - _REPAIR_COST)
                self.grid_health[self.agent_pos] = min(
                    100, self.grid_health[self.agent_pos] + _REPAIR_GAIN
                )

            elif action_type is ActionType.RECHARGE:
                if self.agent_pos == 0:
                    self.agent_energy = min(100, self.agent_energy + _RECHARGE_GAIN)
                else:
                    self.agent_energy = max(
                        0, self.agent_energy - _RECHARGE_WRONG_COST
                    )
                    is_error = True
                    error_message = (
                        f"RECHARGE attempted at sector {self.agent_pos} "
                        f"(valid only at sector 0). "
                        f"Penalty: -{_RECHARGE_WRONG_COST} energy."
                    )

            elif action_type is ActionType.WAIT:
                self.agent_energy = max(0, self.agent_energy - _WAIT_COST)

        # ── 4. Apply entropy ──────────────────────────────────────────────
        self._apply_entropy()

        # ── 5. Post-snapshot ──────────────────────────────────────────────
        energy_after: int = self.agent_energy
        city_health_after: float = sum(self.grid_health) / _NUM_SECTORS

        # ── 6. Bookkeeping ────────────────────────────────────────────────
        self.step_count += 1
        terminated: bool = (
            self.agent_energy <= 0 or self.step_count >= _MAX_STEPS
        )

        # ── 7. Assemble receipt ───────────────────────────────────────────
        step_result = StepResult(
            action_attempted=action_attempted,
            action_parsed=parsed_action,
            energy_before=energy_before,
            energy_after=energy_after,
            city_health_before=round(city_health_before, 2),
            city_health_after=round(city_health_after, 2),
            is_error=is_error,
            error_message=error_message,
        )

        return (
            self._get_observation().model_dump(),
            0.0,
            terminated,
            False,
            {"step_result": step_result.model_dump()},
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def render(self) -> list[SectorState]:
        """
        Return the current grid as a list of ``SectorState`` objects.

        This method is consumed by ``utils.py``'s emoji-grid visualiser during
        debugging and live demo sessions.  It is not called during training.

        Returns
        -------
        list[SectorState]
            One ``SectorState`` per sector (indices 0–24), in row-major order.
        """
        return [
            SectorState(index=i, health=h)
            for i, h in enumerate(self.grid_health)
        ]

    def get_action_space(self) -> list[str]:
        """
        Return the sorted list of valid action strings.

        Mirrors the ``ActionType`` enum values so external tooling (e.g. the
        synthetic data generator in ``train/synthetic_data.json``) can
        enumerate valid actions without importing ``models.py`` directly.

        Returns
        -------
        list[str]
            Sorted list of action name strings, e.g.
            ``["MOVE_E", "MOVE_N", "MOVE_S", "MOVE_W", "RECHARGE", "REPAIR", "WAIT"]``.
        """
        return sorted(a.value for a in ActionType)