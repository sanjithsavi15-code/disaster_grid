"""
src/disaster_grid/utils.py
===========================
Debug-focused terminal UI and emoji-grid visualiser for manual playtesting
of the ``CityGrid`` environment.

Running this module directly launches an interactive session::

    python -m src.disaster_grid.utils

Controls
--------
+-------+------------+-----------------------------------------+
| Key   | Action     | Notes                                   |
+=======+============+=========================================+
| w     | MOVE_N     | Move agent one row upward               |
| s     | MOVE_S     | Move agent one row downward             |
| d     | MOVE_E     | Move agent one column right             |
| a     | MOVE_W     | Move agent one column left              |
| r     | REPAIR     | Repair current sector (+25 health, -15) |
| c     | RECHARGE   | Recharge energy (+20, only at sector 12) |
| q     | WAIT       | Skip turn (-1 energy)                   |
| x     | Quit       | End the session immediately             |
+-------+------------+-----------------------------------------+

Debug receipt
-------------
After every step the terminal prints the raw ``StepResult`` fields so the
developer can immediately diagnose Pydantic validation failures, wall
collisions, or wrong-location RECHARGE attempts without reading log files.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .environment import CityGrid
from .models import ActionType

# ── Display constants ─────────────────────────────────────────────────────────

_EMOJI_AGENT: str = "🤖"
_EMOJI_BASE: str = "🏢"
_EMOJI_FIRE: str = "🔥"
_EMOJI_HEALTHY: str = "🟩"
_CRITICAL_THRESHOLD: int = 30

_SEPARATOR_THICK: str = "═" * 52
_SEPARATOR_THIN: str = "─" * 52

# ── Key → ActionType mapping ──────────────────────────────────────────────────

_KEY_MAP: dict[str, str] = {
    "w": ActionType.MOVE_N.value,
    "s": ActionType.MOVE_S.value,
    "d": ActionType.MOVE_E.value,
    "a": ActionType.MOVE_W.value,
    "r": ActionType.REPAIR.value,
    "c": ActionType.RECHARGE.value,
    "q": ActionType.WAIT.value,
}

_QUIT_KEY: str = "x"


# ── Terminal helpers ──────────────────────────────────────────────────────────


def _clear() -> None:
    """Clear the terminal screen on both Windows and POSIX systems."""
    os.system("cls" if os.name == "nt" else "clear")


def _header(text: str) -> str:
    """Return a centred header line padded to the separator width."""
    return text.center(52)


# ── Grid renderer ─────────────────────────────────────────────────────────────


def render_grid(env: CityGrid) -> str:
    """
    Render the 5 × 5 grid as a multi-line emoji string.

    Each cell is exactly one emoji wrapped in square brackets so columns
    align consistently across UTF-8 terminals.  The agent overrides all
    other cell states — if the agent is standing on the base or a fire
    sector the robot emoji takes precedence to avoid ambiguity during play.

    Parameters
    ----------
    env : CityGrid
        A live environment instance.  Reads ``env.agent_pos`` and
        ``env.grid_health`` directly.

    Returns
    -------
    str
        A five-line string ready for ``print()``.
    """
    rows: list[str] = []
    for row in range(5):
        cells: list[str] = []
        for col in range(5):
            idx = row * 5 + col
            if env.agent_pos == idx:
                emoji = _EMOJI_AGENT
            elif idx == 12:
                emoji = _EMOJI_BASE
            elif env.grid_health[idx] < _CRITICAL_THRESHOLD:
                emoji = _EMOJI_FIRE
            else:
                emoji = _EMOJI_HEALTHY
            cells.append(f"[{emoji}]")
        rows.append("  " + " ".join(cells))
    return "\n".join(rows)


# ── Health bar ────────────────────────────────────────────────────────────────


def _health_bar(value: float, width: int = 20) -> str:
    """
    Return a compact ASCII progress bar for a health / energy value.

    Parameters
    ----------
    value : float
        Current value in [0, 100].
    width : int
        Total number of bar characters (default 20).

    Returns
    -------
    str
        e.g. ``[████████████░░░░░░░░]  60.0``
    """
    filled = int(round((value / 100.0) * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {value:5.1f}"


# ── HUD panels ────────────────────────────────────────────────────────────────


def _render_hud(env: CityGrid, step_number: int) -> str:
    """Return the agent-status panel string."""
    x, y = env.agent_pos % 5, env.agent_pos // 5
    avg_health = sum(env.grid_health) / 25
    critical = sorted(i for i, h in enumerate(env.grid_health) if h < _CRITICAL_THRESHOLD)
    critical_str = str(critical) if critical else "none 🎉"

    lines = [
        _SEPARATOR_THICK,
        _header("🏙️  DISASTER RECOVERY GRID  🏙️"),
        _SEPARATOR_THICK,
        f"  Step       : {step_number:>3} / 50",
        f"  Position   : sector {env.agent_pos:>2}  (col={x}, row={y})",
        f"  Energy     : {_health_bar(env.agent_energy)}",
        f"  City Health: {_health_bar(avg_health)}",
        f"  🔥 Critical: {critical_str}",
        _SEPARATOR_THIN,
    ]
    return "\n".join(lines)


def _render_controls() -> str:
    """Return the controls reference panel string."""
    lines = [
        _SEPARATOR_THIN,
        _header("— CONTROLS —"),
        "  [W] Move North    [S] Move South",
        "  [A] Move West     [D] Move East",
        "  [R] Repair        [C] Recharge (sector 12 only)",
        "  [Q] Wait          [X] Quit",
        _SEPARATOR_THIN,
    ]
    return "\n".join(lines)


def _render_receipt(step_result: dict[str, Any]) -> str:
    """
    Format the ``StepResult`` debug receipt.

    All fields are printed explicitly so the developer can immediately see
    whether Pydantic parsed the action, whether an error was flagged, and
    the exact health delta — without needing to inspect raw dicts.

    Parameters
    ----------
    step_result : dict[str, Any]
        The ``info["step_result"]`` dict from the last ``env.step()`` call.

    Returns
    -------
    str
        Multi-line receipt string.
    """
    action_parsed = step_result.get("action_parsed")
    is_error = step_result.get("is_error", False)
    error_msg = step_result.get("error_message", "")
    health_before = step_result.get("city_health_before", 0.0)
    health_after = step_result.get("city_health_after", 0.0)
    health_delta = health_after - health_before
    energy_before = step_result.get("energy_before", 0)
    energy_after = step_result.get("energy_after", 0)
    energy_delta = energy_after - energy_before
    action_attempted = step_result.get("action_attempted", "—")

    # Colour-code the health delta with arrows
    if health_delta > 0:
        delta_str = f"▲ +{health_delta:.4f}"
    elif health_delta < 0:
        delta_str = f"▼  {health_delta:.4f}"
    else:
        delta_str = f"   {health_delta:.4f}"

    # Error status badge
    error_badge = "❌ YES" if is_error else "✅ NO"

    # Parsed action summary
    if action_parsed is None:
        parsed_str = "None  ← ⚠️  Pydantic validation FAILED"
    else:
        parsed_action_val = (
            action_parsed.get("action", "?")
            if isinstance(action_parsed, dict)
            else str(action_parsed)
        )
        parsed_reasoning = (
            action_parsed.get("reasoning", "")[:40]
            if isinstance(action_parsed, dict)
            else ""
        )
        parsed_str = f"{parsed_action_val!r}  (reasoning: {parsed_reasoning!r})"

    lines = [
        _SEPARATOR_THICK,
        _header("— STEP RECEIPT (StepResult) —"),
        _SEPARATOR_THIN,
        f"  Attempted   : {action_attempted}",
        f"  Parsed      : {parsed_str}",
        _SEPARATOR_THIN,
        f"  Error?      : {error_badge}",
        f"  Error Msg   : {error_msg if error_msg else '—'}",
        _SEPARATOR_THIN,
        f"  Energy      : {energy_before:>4}  →  {energy_after:>4}  "
        f"(delta: {energy_delta:+d})",
        f"  City Health : {health_before:>8.4f}  →  {health_after:>8.4f}  "
        f"(delta: {delta_str})",
        _SEPARATOR_THICK,
    ]
    return "\n".join(lines)


# ── Main playtest loop ────────────────────────────────────────────────────────


def play_manual(seed: int | None = None) -> None:
    """
    Launch an interactive manual playtest session in the terminal.

    The loop renders the full game state, waits for a keypress, constructs
    a valid ``AgentAction``-compatible payload, calls ``env.step()``, then
    prints the ``StepResult`` debug receipt before pausing so the developer
    can read the output before the screen clears.

    Parameters
    ----------
    seed : int | None
        Optional RNG seed passed to ``env.reset()``.  Set to a fixed integer
        for reproducible debug sessions.

    Payload construction note
    -------------------------
    The payload is built as a plain ``dict`` (not a JSON string) because
    ``CityGrid.step()`` accepts both formats.  Using a dict skips the
    ``json.dumps`` / ``json.loads`` round-trip and surfaces Pydantic
    validation errors more directly.  The ``reasoning`` value is a fixed
    string; the ``action`` value is the exact ``.value`` of the enum member,
    which is what ``AgentAction`` expects::

        action_payload = {
            "action": key_map[user_input],   # e.g. "MOVE_N"
            "reasoning": "Manual playtest",
        }
    """
    env = CityGrid()
    obs, _ = env.reset(seed=seed)

    # ── Welcome splash ────────────────────────────────────────────────────
    _clear()
    print(_SEPARATOR_THICK)
    print(_header("🚨  DISASTER GRID — MANUAL PLAYTEST  🚨"))
    print(_header("debug mode · all StepResult fields visible"))
    print(_SEPARATOR_THICK)
    print()
    print("  Legend:")
    print(f"    [{_EMOJI_AGENT}] Agent position")
    print(f"    [{_EMOJI_BASE}] Base / Recharge station (sector 12)")
    print(f"    [{_EMOJI_FIRE}] Critical sector (health < 30)")
    print(f"    [{_EMOJI_HEALTHY}] Healthy sector (health ≥ 30)")
    print()
    print("  Press Enter to begin...")
    input()

    last_receipt: dict[str, Any] = {}
    episode_over: bool = False

    while not episode_over:
        _clear()

        # ── Render HUD ────────────────────────────────────────────────────
        print(_render_hud(env, env.step_count))
        print()

        # ── Render grid ───────────────────────────────────────────────────
        print(render_grid(env))
        print()

        # ── Print last receipt (empty on first turn) ──────────────────────
        if last_receipt:
            print(_render_receipt(last_receipt))
        else:
            print(_SEPARATOR_THIN)
            print(_header("— no action taken yet —"))
            print(_SEPARATOR_THIN)

        print()
        print(_render_controls())
        print()

        # ── Prompt ────────────────────────────────────────────────────────
        raw = input("  Your move › ").strip().lower()

        if raw == _QUIT_KEY:
            print()
            print("  Exiting playtest session. Goodbye! 👋")
            sys.exit(0)

        if raw not in _KEY_MAP:
            print()
            print(f"  ⚠️  Unknown key {raw!r}. Valid keys: "
                  f"{sorted(_KEY_MAP)} or '{_QUIT_KEY}' to quit.")
            input("  Press Enter to continue...")
            continue

        # ── Build the exact payload the AgentAction schema expects ────────
        action_payload: dict[str, str] = {
            "action": _KEY_MAP[raw],
            "reasoning": "Manual playtest",
        }

        print()
        print(f"  Sending payload → {action_payload}")

        # ── Step the environment ──────────────────────────────────────────
        try:
            obs, reward, done, truncated, info = env.step(action_payload)
        except Exception as exc:  # noqa: BLE001
            # Surface any unexpected environment crash without losing the
            # session — developer can inspect the state and continue.
            print()
            print(f"  🚨 UNEXPECTED ENVIRONMENT ERROR: {type(exc).__name__}: {exc}")
            input("  Press Enter to continue...")
            continue

        # ── Extract StepResult receipt ────────────────────────────────────
        last_receipt = info.get("step_result", {})

        episode_over = bool(done or truncated)

    # ── Episode over screen ───────────────────────────────────────────────
    _clear()
    avg_health = sum(env.grid_health) / 25
    print(_SEPARATOR_THICK)
    print(_header("🏁  EPISODE COMPLETE  🏁"))
    print(_SEPARATOR_THICK)
    print(f"  Steps taken     : {env.step_count}")
    print(f"  Final energy    : {env.agent_energy}")
    print(f"  Final avg health: {avg_health:.2f} / 100.00")
    print()
    if avg_health >= 70:
        print(_header("✅ CITY SAVED — excellent management!"))
    elif avg_health >= 40:
        print(_header("⚠️  CITY DAMAGED — recovery possible."))
    else:
        print(_header("💀 CITY LOST — better luck next time."))
    print(_SEPARATOR_THICK)

    # ── Final receipt ─────────────────────────────────────────────────────
    if last_receipt:
        print()
        print(_render_receipt(last_receipt))

    print()
    input("  Press Enter to exit...")


if __name__ == "__main__":
    play_manual()