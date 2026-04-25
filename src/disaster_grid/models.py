"""
src/disaster_grid/models.py
============================
Pydantic schemas and enumerations for the disaster_grid OpenEnv environment.

Design philosophy
-----------------
All data structures that cross a module boundary are defined here so that
`environment.py`, `rewards.py`, and `grpo_trainer.py` share a single source
of truth.  Pydantic is chosen over `dataclasses` for three reasons:

1. **FastAPI integration** – every model serialises to / deserialises from
   JSON with zero extra work, enabling a `/step` endpoint with no adapter
   layer.
2. **Runtime validation** – `Field` constraints (ge, le, min_length …) are
   enforced at construction time, so an out-of-range energy value raises a
   clear `ValidationError` rather than silently corrupting training data.
3. **LLM prompt hygiene** – `model.model_dump_json()` produces the exact JSON
   string we can embed in a system prompt, keeping the schema the LLM is
   trained on perfectly in sync with the schema the environment enforces.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    """
    The complete set of actions available to the disaster-recovery agent.

    Inheriting from ``str`` means Pydantic serialises these as plain strings
    (e.g. ``"MOVE_N"``), which is what the LLM will emit and what FastAPI
    will accept in a JSON body without any custom encoder.

    Members
    -------
    MOVE_N
        Move the agent one row upward  (row -= 1).
        Energy cost: -2.  Invalid if the agent is already on row 0.
    MOVE_S
        Move the agent one row downward (row += 1).
        Energy cost: -2.  Invalid if the agent is already on row 4.
    MOVE_E
        Move the agent one column rightward (col += 1).
        Energy cost: -2.  Invalid if the agent is already on col 4.
    MOVE_W
        Move the agent one column leftward  (col -= 1).
        Energy cost: -2.  Invalid if the agent is already on col 0.
    REPAIR
        Repair the sector the agent currently occupies.
        Energy cost: -15.  Sector health gain: +25 (capped at 100).
        Has no effect if sector health is already 100.
    RECHARGE
        Replenish the agent's energy reserve.
        Energy gain: +20 (capped at 100).
        **Only valid at sector index 12** (the recharge station at the center).
        Attempting RECHARGE elsewhere is treated as a no-op with an error flag.
    WAIT
        The agent stays in place and takes no action.
        No energy cost.  Entropy still applies.
        Useful when the agent is awaiting a recharge opportunity or has no
        repair targets within reach.
    """

    MOVE_N = "MOVE_N"
    MOVE_S = "MOVE_S"
    MOVE_E = "MOVE_E"
    MOVE_W = "MOVE_W"
    REPAIR = "REPAIR"
    RECHARGE = "RECHARGE"
    WAIT = "WAIT"


# ─────────────────────────────────────────────────────────────────────────────
# Primitive domain models
# ─────────────────────────────────────────────────────────────────────────────


class SectorState(BaseModel):
    """
    The state of a single cell in the 5 × 5 city grid.

    The grid uses row-major indexing: sector ``index = row * 5 + col``.
    Sector 0  is the top-left corner.
    Sector 24 is the bottom-right corner.

    ``SectorState`` objects are held in ``environment.py``'s internal grid
    list and are **not** passed directly to the LLM – they are too verbose
    for a 50-step prompt budget.  Only derived summaries (average health,
    critical indices) appear in ``GridObservation``.

    Fields
    ------
    index : int
        Flat grid index in [0, 24].  Immutable after construction.
    health : int
        Current health of the sector in [0, 100].
        0  → sector is completely degraded (city infrastructure lost).
        100 → sector is at full operational capacity.
        Mutated by the environment's entropy step and by REPAIR actions.
    """

    index: int = Field(
        ...,
        ge=0,
        le=24,
        description="Flat row-major index of this sector within the 5×5 grid.",
    )
    health: int = Field(
        ...,
        ge=0,
        le=100,
        description="Current sector health.  Degraded by entropy; restored by REPAIR.",
    )

    @computed_field  # type: ignore[misc]
    @property
    def is_critical(self) -> bool:
        """
        Return ``True`` when this sector requires urgent intervention.

        A sector is *critical* when its health drops below 30.  Critical
        sectors are surfaced to the LLM in ``GridObservation.critical_sectors``
        so the agent can prioritise routing decisions without scanning the
        full grid.

        The threshold (30) is a domain constant chosen so that a single
        entropy tick of -5 would push a critical sector to health 25, giving
        the agent at most ~5 additional steps before the sector reaches 0.
        """
        return self.health < 30

    @model_validator(mode="after")
    def _clamp_health(self) -> "SectorState":
        """Silently clamp health into [0, 100] after any mutation helper."""
        self.health = max(0, min(100, self.health))
        return self


# ─────────────────────────────────────────────────────────────────────────────
# LLM-facing observation schema
# ─────────────────────────────────────────────────────────────────────────────


class GridObservation(BaseModel):
    """
    The structured observation delivered to the LLM at the start of each step.

    **Token budget rationale** – a naïve approach would serialise all 25
    ``SectorState`` objects (~200 tokens per step × 50 steps = 10 000 tokens
    per episode).  Instead this model passes:

    * The agent's current coordinates and energy.
    * The health of only the sector the agent stands on (immediately
      actionable information).
    * The *average* city health (global signal for the R1 verifier proxy).
    * The flat indices of *critical* sectors (health < 30) so the agent can
      plan efficient routes without enumerating the whole grid.

    This reduces observation size to ~60 tokens per step while preserving all
    information required for rational decision-making.

    The environment serialises this as a JSON string and injects it into the
    LLM prompt as the ``<observation>`` block.

    Fields
    ------
    step_number : int
        Current step in [0, 50].  Lets the agent reason about remaining
        horizon (e.g. "I have 10 steps left, is a recharge trip worth it?").
    agent_position : int
        Flat index of the agent's current sector.  Combined with the 5-wide
        grid, the agent can derive (row, col) via ``divmod(position, 5)``.
    agent_energy : int
        Current energy in [0, 100].  The done condition fires at 0.
    current_sector_health : int
        Health of the sector at ``agent_position``.  Included so the agent
        can decide *immediately* whether to REPAIR here or move elsewhere.
    critical_sectors : list[int]
        Sorted list of sector indices whose health is below 30.  May be
        empty.  The agent should prefer routing to the nearest element of
        this list.  Presented in ascending index order so prompt content is
        deterministic across identical states (useful for reproducible evals).
    average_city_health : float
        Mean health across all 25 sectors, rounded to two decimal places.
        Acts as the agent's running score proxy.  A strategic agent should
        try to keep this above 70.0.
    """

    step_number: int = Field(
        ...,
        ge=0,
        le=50,
        description="Current environment step.  Episode ends at step 50.",
    )
    agent_position: int = Field(
        ...,
        ge=0,
        le=24,
        description="Flat index of the agent's current grid position.",
    )
    agent_energy: int = Field(
        ...,
        ge=0,
        le=100,
        description="Remaining energy units.  Episode ends immediately if this reaches 0.",
    )
    current_sector_health: int = Field(
        ...,
        ge=0,
        le=100,
        description="Health of the sector currently occupied by the agent.",
    )
    critical_sectors: list[int] = Field(
        default_factory=list,
        description=(
            "Sorted indices of sectors with health < 30.  "
            "Empty list means no sectors are currently critical."
        ),
    )
    average_city_health: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Mean health across all 25 sectors, rounded to 2 d.p.",
    )

    @model_validator(mode="after")
    def _validate_critical_sectors(self) -> "GridObservation":
        """Ensure all critical sector indices are within the valid grid range."""
        for idx in self.critical_sectors:
            if not (0 <= idx <= 24):
                raise ValueError(
                    f"critical_sectors contains out-of-range index {idx!r}. "
                    "All indices must be in [0, 24]."
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# LLM output schema (action payload)
# ─────────────────────────────────────────────────────────────────────────────


class AgentAction(BaseModel):
    """
    The structured JSON payload the LLM must emit in response to each
    ``GridObservation``.

    **Two-field design** – separating ``reasoning`` from ``action`` is a
    deliberate inductive bias:

    * ``reasoning`` is placed *first* in the schema so that the LLM is
      forced to produce a chain-of-thought (CoT) token sequence before
      committing to an ``action`` token.  Empirically, CoT prefix generation
      significantly improves action quality on multi-step planning tasks.
    * ``action`` is validated against ``ActionType`` at parse time, so any
      hallucinated action string (e.g. ``"MOVE_NE"``) raises a
      ``ValidationError`` that the environment catches and converts into a
      ``StepResult`` with ``is_error=True``.  This error signal flows through
      to the R3 (format) verifier.

    The GRPO trainer embeds the ``model_json_schema()`` of this class in the
    system prompt so the LLM always sees the exact schema it is being graded
    against.

    Fields
    ------
    reasoning : str
        A free-text explanation (minimum 10 characters) of *why* the agent
        chose this action given the current observation.  Enforcing a minimum
        length discourages degenerate single-token rationales.  The ``rewards``
        module does **not** score reasoning quality – it is included purely as
        a CoT generation mechanism during GRPO fine-tuning.
    action : ActionType
        The chosen action from the ``ActionType`` enum.  Must be one of the
        seven valid strings; anything else will fail Pydantic validation.
    """

    reasoning: str = Field(
        ...,
        min_length=10,
        description=(
            "Chain-of-thought rationale produced *before* the action token.  "
            "Forces the model to reason about position, energy, and critical "
            "sectors prior to committing to a move.  Not scored by verifiers."
        ),
    )
    action: ActionType = Field(
        ...,
        description=(
            "The discrete action to execute this step.  Must be a member of "
            "ActionType.  Invalid strings cause a ValidationError that is "
            "captured as is_error=True in StepResult and penalised by R3."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verifier receipt (internal – never seen by the LLM)
# ─────────────────────────────────────────────────────────────────────────────


class StepResult(BaseModel):
    """
    A complete record of everything that happened during one environment step.

    ``StepResult`` is the *contract* between ``environment.py`` and
    ``rewards.py``.  It is constructed by ``CityGrid.step()`` and passed
    directly to the three reward verifiers – neither module needs to
    re-execute environment logic to calculate its score.

    **Why a separate receipt model?**  Each verifier reads a different subset
    of fields:

    * R1 (health)       reads ``city_health_before`` / ``city_health_after``.
    * R2 (efficiency)   reads ``energy_before`` / ``energy_after`` plus the
                        city health delta.
    * R3 (format)       reads ``action_attempted`` and ``action_parsed``
                        (``None`` means parse failed → score 0.0).

    Passing a single rich object avoids fragile argument lists and keeps
    verifier signatures stable as the environment evolves.

    This model is **never serialised into the LLM prompt**.  It is used
    exclusively within the Python training loop.

    Fields
    ------
    action_attempted : str
        The raw string the LLM emitted (or the trainer injected for synthetic
        rollouts).  Preserved verbatim so R3 can detect subtle schema
        violations even when Pydantic parsing partially succeeds.
    action_parsed : AgentAction | None
        The validated ``AgentAction`` object if ``action_attempted`` was valid
        JSON conforming to the schema; ``None`` otherwise.  ``None`` is a
        direct penalty signal for R3.
    energy_before : int
        Agent energy at the *start* of this step, before the action was
        applied.  Used by R2 to compute energy expenditure.
    energy_after : int
        Agent energy at the *end* of this step, after the action (and any
        recharge) was applied.  R2 uses ``energy_before - energy_after`` as
        the cost denominator.
    city_health_before : float
        Average city health across all 25 sectors *before* this step's
        entropy tick and repair.  R1 and R2 use this as the baseline.
    city_health_after : float
        Average city health *after* entropy and the agent's action.
        ``city_health_after - city_health_before`` is the net health delta
        for this step; positive values indicate the agent added more value
        than entropy removed.
    is_error : bool
        ``True`` when the action could not be executed for any reason:
        invalid JSON, invalid ``ActionType``, illegal move (e.g. MOVE_N from
        row 0), or RECHARGE attempted outside sector 12.  An error step is a
        no-op for the environment but still incurs entropy.
    error_message : str
        Human-readable description of the error.  Empty string when
        ``is_error`` is ``False``.  Surfaced in training logs and unit tests
        to aid debugging without re-running the full episode.
    """

    action_attempted: str = Field(
        ...,
        description=(
            "Raw LLM output string, preserved verbatim for R3 format scoring "
            "and debugging.  Never parsed or executed after this field is set."
        ),
    )
    action_parsed: Optional[AgentAction] = Field(
        default=None,
        description=(
            "Validated AgentAction if action_attempted was schema-compliant; "
            "None if JSON parsing or Pydantic validation failed.  "
            "None is the primary R3 penalty signal."
        ),
    )
    energy_before: int = Field(
        ...,
        ge=0,
        le=100,
        description="Agent energy at step start, used as R2 cost baseline.",
    )
    energy_after: int = Field(
        ...,
        ge=0,
        le=100,
        description="Agent energy at step end, after action execution.",
    )
    city_health_before: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Mean sector health before this step's entropy + repair.",
    )
    city_health_after: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Mean sector health after this step's entropy + repair.",
    )
    is_error: bool = Field(
        default=False,
        description=(
            "True when the action was invalid or illegal.  "
            "The environment applies entropy but skips action execution."
        ),
    )
    error_message: str = Field(
        default="",
        description=(
            "Diagnostic message when is_error is True.  "
            "Empty string on successful steps."
        ),
    )

    @model_validator(mode="after")
    def _error_message_consistency(self) -> "StepResult":
        """
        Enforce that ``error_message`` is non-empty iff ``is_error`` is True.

        This prevents silent failures where an error condition is set but the
        message is accidentally cleared (or vice-versa), which would make
        training logs misleading.
        """
        if self.is_error and not self.error_message:
            raise ValueError(
                "is_error is True but error_message is empty.  "
                "Provide a diagnostic string so training logs are actionable."
            )
        if not self.is_error and self.error_message:
            raise ValueError(
                "error_message is set but is_error is False.  "
                "Either set is_error=True or clear error_message."
            )
        return self