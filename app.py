"""
DISASTER GRID — Autonomous AI Emergency Manager
Streamlit dashboard — production-ready for Hugging Face Spaces
"""

import streamlit as st
import random
import time

# ── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="DISASTER GRID — Autonomous AI Emergency Manager",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Try real backend; graceful stub fallback ──────────────────────────────────
try:
    from src.disaster_grid.environment import CityGrid
    from src.disaster_grid.models import ActionType
    REAL_ENV = True
except ImportError:
    REAL_ENV = False

if not REAL_ENV:

    class ActionType:
        MOVE_N   = "MOVE_N"
        MOVE_S   = "MOVE_S"
        MOVE_E   = "MOVE_E"
        MOVE_W   = "MOVE_W"
        REPAIR   = "REPAIR"
        RECHARGE = "RECHARGE"
        WAIT     = "WAIT"

    class CityGrid:
        def __init__(self):
            self.grid_health  = [100] * 25
            self.agent_pos    = 0
            self.agent_energy = 100

        def reset(self, options=None):
            self.grid_health  = [random.randint(60, 100) for _ in range(25)]
            self.grid_health[0] = 100
            self.agent_pos    = 0
            self.agent_energy = 100
            if options and "target_crises" in options:
                for idx in options["target_crises"]:
                    if 0 <= idx < 25:
                        self.grid_health[idx] = random.randint(5, 25)

        def step(self, action: dict):
            act    = action.get("action", "WAIT")
            reward = -0.5
            # entropy: one random sector degrades
            decay_idx = random.randint(1, 24)
            self.grid_health[decay_idx] = max(
                0, self.grid_health[decay_idx] - random.randint(3, 8)
            )
            self.agent_energy = max(0, self.agent_energy - 2)

            row, col = divmod(self.agent_pos, 5)
            if   act == "MOVE_N" and row > 0: self.agent_pos -= 5
            elif act == "MOVE_S" and row < 4: self.agent_pos += 5
            elif act == "MOVE_W" and col > 0: self.agent_pos -= 1
            elif act == "MOVE_E" and col < 4: self.agent_pos += 1
            elif act == "REPAIR":
                if self.grid_health[self.agent_pos] < 30 and self.agent_energy >= 10:
                    self.grid_health[self.agent_pos] = min(
                        100, self.grid_health[self.agent_pos] + 30
                    )
                    self.agent_energy -= 10
                    reward += 20
            elif act == "RECHARGE":
                if self.agent_pos == 0:
                    self.agent_energy = min(100, self.agent_energy + 40)
                    reward += 5

            avg_hp = sum(self.grid_health) / 25
            done   = avg_hp < 20 or self.agent_energy <= 0
            info   = {"avg_health": avg_hp, "reward": reward}
            return {}, reward, done, False, info


# ── Heuristic agent policy ────────────────────────────────────────────────────
def pick_action(grid_health, agent_pos, agent_energy):
    """
    Lightweight rule-based fallback policy.
    Returns (action_string, reasoning_string) — identical shape to the LLM agent.
    """
    critical = [i for i, h in enumerate(grid_health) if h < 30]

    if agent_energy <= 20 and agent_pos == 0:
        return ActionType.RECHARGE, (
            f"Energy critically low ({agent_energy}%) — recharging at base."
        )

    if agent_energy <= 20:
        row, col = divmod(agent_pos, 5)
        act = ActionType.MOVE_N if row > 0 else ActionType.MOVE_W
        return act, f"Energy low ({agent_energy}%) — routing back to base for recharge."

    if agent_pos in critical:
        return ActionType.REPAIR, (
            f"Sector {agent_pos} is critical (HP={grid_health[agent_pos]}) — "
            f"initiating emergency repair."
        )

    if critical:
        target = critical[0]
        trow, tcol = divmod(target, 5)
        row,  col  = divmod(agent_pos, 5)
        if   trow < row:   act = ActionType.MOVE_N
        elif trow > row:   act = ActionType.MOVE_S
        elif tcol < col:   act = ActionType.MOVE_W
        else:              act = ActionType.MOVE_E
        return act, (
            f"Navigating toward critical sector {target} "
            f"(HP={grid_health[target]})."
        )

    worst = min(range(25), key=lambda i: grid_health[i])
    if worst == agent_pos:
        return ActionType.WAIT, "All sectors nominal — holding position, monitoring entropy."

    wrow, wcol = divmod(worst, 5)
    row,  col  = divmod(agent_pos, 5)
    if   wrow < row:   act = ActionType.MOVE_N
    elif wrow > row:   act = ActionType.MOVE_S
    elif wcol < col:   act = ActionType.MOVE_W
    else:              act = ActionType.MOVE_E
    return act, (
        f"Patrolling toward degraded sector {worst} "
        f"(HP={grid_health[worst]}) — entropy watch active."
    )


# ── Session-state initialisation ─────────────────────────────────────────────
def _init_state():
    if "env" not in st.session_state:
        env = CityGrid()
        env.reset()
        st.session_state.env = env

    e = st.session_state.env
    defaults = {
        "grid_health":    list(e.grid_health),
        "agent_pos":      e.agent_pos,
        "agent_energy":   e.agent_energy,
        "step_count":     0,
        "total_reward":   0.0,
        "repair_count":   0,
        "recharge_count": 0,
        "auto_running":   False,
        "game_over":      False,
        "log_lines": [
            "SYS › DISASTER GRID v1.0 — systems online.",
            "SYS › 25 sectors initialised — all readings nominal.",
            "SYS › Awaiting operator command...",
        ],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sync_from_env():
    e = st.session_state.env
    st.session_state.grid_health  = list(e.grid_health)
    st.session_state.agent_pos    = e.agent_pos
    st.session_state.agent_energy = e.agent_energy


def _add_log(msg: str, kind: str = "sys"):
    prefix = {
        "sys":  "SYS ›",
        "act":  "ACT ›",
        "warn": "⚠   ›",
        "crit": "🔥  ›",
        "done": "✗   ›",
    }
    st.session_state.log_lines.append(f"{prefix.get(kind, '   ›')} {msg}")
    if len(st.session_state.log_lines) > 100:
        st.session_state.log_lines = st.session_state.log_lines[-100:]


def _do_step():
    if st.session_state.game_over:
        return
    action, reasoning = pick_action(
        st.session_state.grid_health,
        st.session_state.agent_pos,
        st.session_state.agent_energy,
    )
    act_str = action.value if hasattr(action, "value") else str(action)
    _, reward, done, _, _ = st.session_state.env.step(
        {"action": act_str, "reasoning": reasoning}
    )
    _sync_from_env()
    st.session_state.step_count   += 1
    st.session_state.total_reward += float(reward)
    if act_str == "REPAIR":   st.session_state.repair_count   += 1
    if act_str == "RECHARGE": st.session_state.recharge_count += 1

    _add_log(f'"{reasoning}"', "act")
    _add_log(
        f"Action: {act_str}  |  Reward: {reward:+.1f}  "
        f"|  Step #{st.session_state.step_count}",
        "sys",
    )
    if done:
        st.session_state.game_over    = True
        st.session_state.auto_running = False
        _add_log("SIMULATION ENDED — city collapsed or energy depleted.", "done")


def _full_reset():
    st.session_state.auto_running   = False
    st.session_state.game_over      = False
    st.session_state.step_count     = 0
    st.session_state.total_reward   = 0.0
    st.session_state.repair_count   = 0
    st.session_state.recharge_count = 0
    st.session_state.log_lines      = [
        "SYS › City reset to baseline.",
        "SYS › All 25 sectors restored to nominal.",
        "SYS › Awaiting operator command...",
    ]
    st.session_state.env.reset()
    _sync_from_env()


# ── CSS injection ─────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600&display=swap');

/* ── Global dark background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMain"] {
    background-color: #070d1a !important;
    color: #c8e6e0 !important;
    font-family: 'Exo 2', sans-serif !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* kill default block padding */
.block-container {
    padding-top:    0.5rem  !important;
    padding-bottom: 0.3rem  !important;
    padding-left:   1.1rem  !important;
    padding-right:  1.1rem  !important;
    max-width: 100% !important;
}

/* hide sidebar */
section[data-testid="stSidebar"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #070d1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 99px; }

/* ── Divider ── */
hr { border-color: #1e3a5f !important; margin: 0.35rem 0 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0d1b2e !important;
    border:     1px solid #1e3a5f !important;
    border-radius: 6px !important;
    padding: 0.35rem 0.6rem !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Share Tech Mono', monospace !important;
    font-size:   0.62rem !important;
    color:       #4a8a9a !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size:   1.25rem !important;
    color:       #00ff9d !important;
    text-shadow: 0 0 12px #00ff9d60 !important;
}

/* ── Progress bars ── */
[data-testid="stProgressBar"] > div {
    background: #0d1b2e !important;
    border:     1px solid #1e3a5f !important;
    border-radius: 99px !important;
    height: 10px !important;
    padding: 0 !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #00ff9d, #00cc7a) !important;
    border-radius: 99px !important;
    height: 100% !important;
    transition: width 0.5s ease !important;
}

/* ── Buttons — base ── */
.stButton > button {
    font-family:    'Share Tech Mono', monospace !important;
    font-size:      0.73rem !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    background:     transparent !important;
    border:         1px solid #00ff9d !important;
    color:          #00ff9d !important;
    border-radius:  4px !important;
    padding:        0.42rem 0.75rem !important;
    width:          100% !important;
    transition:     all 0.15s !important;
    text-align:     left !important;
}
.stButton > button:hover {
    background:  rgba(0,255,157,0.09) !important;
    box-shadow:  0 0 14px rgba(0,255,157,0.35) !important;
}
.stButton > button:active { transform: scale(0.97) !important; }

/* variant wrappers */
.btn-danger .stButton > button {
    border-color: #ff4d4d !important;
    color:        #ff4d4d !important;
}
.btn-danger .stButton > button:hover {
    background: rgba(255,77,77,0.09) !important;
    box-shadow: 0 0 14px rgba(255,77,77,0.35) !important;
}
.btn-amber .stButton > button {
    border-color: #ffb800 !important;
    color:        #ffb800 !important;
}
.btn-amber .stButton > button:hover {
    background: rgba(255,184,0,0.09) !important;
    box-shadow: 0 0 14px rgba(255,184,0,0.35) !important;
}
.btn-slate .stButton > button {
    border-color: #4a7a8a !important;
    color:        #4a7a8a !important;
}

/* ── Panel chrome ── */
.dg-panel-title {
    font-family:    'Orbitron', sans-serif;
    font-size:      0.58rem;
    letter-spacing: 0.22em;
    color:          #00ff9d;
    text-transform: uppercase;
    padding:        0.45rem 0.6rem;
    background:     #0b1930;
    border-bottom:  1px solid #1e3a5f;
    border-radius:  6px 6px 0 0;
    margin-bottom:  0;
}

/* ── City grid ── */
.dg-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 5px;
    padding: 8px;
    background: #0b1527;
    border: 1px solid #1e3a5f;
    border-radius: 0 0 8px 8px;
}
.dg-cell {
    aspect-ratio: 1 / 1;
    border-radius: 6px;
    border: 1px solid #1a3050;
    background: #091220;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: clamp(1.2rem, 2.2vw, 2rem);
    position: relative;
    transition: background 0.3s, border-color 0.3s, box-shadow 0.3s;
    overflow: hidden;
}
.dg-cell .idx {
    position: absolute;
    top: 2px; left: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.42rem;
    color: #1e3a5f;
    line-height: 1;
    user-select: none;
}
.dg-cell .hp-bar {
    position: absolute;
    bottom: 0; left: 0;
    height: 3px;
    border-radius: 0 0 6px 6px;
    transition: width 0.4s, background 0.3s;
}
/* cell state classes */
.cell-agent {
    border-color: #00ff9d !important;
    background: #061a10 !important;
    box-shadow: inset 0 0 16px rgba(0,255,157,0.18) !important;
}
.cell-base {
    border-color: rgba(60,100,255,0.4) !important;
    background: #06091a !important;
}
.cell-crit {
    border-color: rgba(255,77,77,0.45) !important;
    background: #180606 !important;
    animation: fire-glow 1.4s ease-in-out infinite;
}
.cell-ok { /* default, nothing extra */ }

@keyframes fire-glow {
    0%,100% { box-shadow: inset 0 0 6px rgba(255,77,77,0.2); }
    50%      { box-shadow: inset 0 0 22px rgba(255,77,77,0.55); }
}

/* ── Agent log terminal ── */
.dg-log-wrap {
    background:    #050d14;
    border:        1px solid #1e3a5f;
    border-radius: 0 0 8px 8px;
}
.dg-log {
    font-family:  'Share Tech Mono', monospace;
    font-size:    0.72rem;
    line-height:  1.8;
    color:        #4aaa88;
    padding:      0.6rem 0.75rem;
    height:       clamp(200px, 38vh, 380px);
    overflow-y:   auto;
    white-space:  pre-wrap;
    word-break:   break-word;
}
.l-act  { color: #00ff9d; }
.l-warn { color: #ffb800; }
.l-crit { color: #ff4d4d; }
.l-done { color: #ff88ff; font-weight: bold; }
.l-sys  { color: #2a6676; }

/* ── Telemetry stat rows ── */
.dg-stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3px 8px;
    padding: 0.4rem 0.6rem 0.5rem;
    background: #0b1527;
    border: 1px solid #1e3a5f;
    border-radius: 0 0 8px 8px;
}
.dg-stat-row {
    font-family: 'Share Tech Mono', monospace;
    font-size:   0.68rem;
    color:       #3a7a8a;
    display:     flex;
    justify-content: space-between;
    gap: 4px;
    line-height: 1.9;
}
.dg-stat-val          { color: #00ff9d; font-weight: bold; }
.dg-stat-val.s-amber  { color: #ffb800 !important; }
.dg-stat-val.s-danger { color: #ff4d4d !important; }
.dg-stat-val.s-blue   { color: #5599ff !important; }

/* ── Legend ── */
.dg-legend {
    font-family: 'Share Tech Mono', monospace;
    font-size:   0.7rem;
    color:       #3a7a8a;
    line-height: 2.1;
    padding:     0.45rem 0.65rem 0.55rem;
    background:  #0b1527;
    border:      1px solid #1e3a5f;
    border-radius: 0 0 8px 8px;
}

/* ── Header ── */
.dg-header {
    display:     flex;
    align-items: center;
    gap:         1rem;
    padding:     0.2rem 0 0.45rem;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 0.5rem;
}
.dg-title {
    font-family:    'Orbitron', sans-serif;
    font-weight:    900;
    font-size:      clamp(0.95rem, 1.9vw, 1.4rem);
    color:          #00ff9d;
    text-shadow:    0 0 18px rgba(0,255,157,0.5);
    letter-spacing: 0.1em;
    white-space:    nowrap;
}
.dg-subtitle {
    font-family:    'Share Tech Mono', monospace;
    font-size:      0.62rem;
    color:          #2a7766;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-top:     2px;
}
.dg-badge {
    font-family:    'Share Tech Mono', monospace;
    font-size:      0.62rem;
    letter-spacing: 0.13em;
    padding:        0.18rem 0.55rem;
    border-radius:  3px;
    text-transform: uppercase;
    border:         1px solid;
    white-space:    nowrap;
}
.badge-standby  { color:#00ff9d; border-color:#00ff9d; background:rgba(0,255,157,0.08); }
.badge-running  { color:#ffb800; border-color:#ffb800; background:rgba(255,184,0,0.08);
                  animation: blink 1s steps(1) infinite; }
.badge-gameover { color:#ff4d4d; border-color:#ff4d4d; background:rgba(255,77,77,0.08); }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.35;} }

.dg-step-counter {
    margin-left: auto;
    font-family: 'Share Tech Mono', monospace;
    font-size:   0.62rem;
    color:       #2a6676;
    white-space: nowrap;
}
.dg-step-counter span {
    font-family: 'Orbitron', sans-serif;
    font-size:   0.9rem;
    color:       #00ff9d;
}

/* ── Section mini-label ── */
.sec-label {
    font-family:    'Share Tech Mono', monospace;
    font-size:      0.58rem;
    letter-spacing: 0.18em;
    color:          #1e4455;
    text-transform: uppercase;
    margin-bottom:  0.2rem;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ── HTML builders ─────────────────────────────────────────────────────────────
def _build_grid_html(grid_health: list, agent_pos: int) -> str:
    cells = []
    for i in range(25):
        hp       = grid_health[i]
        is_agent = i == agent_pos
        is_base  = i == 0
        is_crit  = hp < 30

        if is_agent:
            emoji = "🤖"; css_cls = "cell-agent"
        elif is_base:
            emoji = "🏢"; css_cls = "cell-base"
        elif is_crit:
            emoji = "🔥"; css_cls = "cell-crit"
        else:
            emoji = "🟩"; css_cls = "cell-ok"

        hp_color = (
            "#ff4d4d" if hp < 30 else
            "#ffb800" if hp < 60 else
            "#00ff9d"
        )
        cells.append(f"""
<div class="dg-cell {css_cls}" title="Sector {i} | HP: {hp}">
  <span class="idx">{i}</span>
  <span style="user-select:none">{emoji}</span>
  <div class="hp-bar" style="width:{hp}%;background:{hp_color};"></div>
</div>""")

    return '<div class="dg-grid">' + "".join(cells) + "</div>"


def _build_log_html(log_lines: list) -> str:
    lines_html = []
    for line in log_lines[-80:]:
        if line.startswith("ACT"):
            lines_html.append(f'<span class="l-act">{line}</span>')
        elif line.startswith("⚠"):
            lines_html.append(f'<span class="l-warn">{line}</span>')
        elif line.startswith("🔥"):
            lines_html.append(f'<span class="l-crit">{line}</span>')
        elif line.startswith("✗"):
            lines_html.append(f'<span class="l-done">{line}</span>')
        else:
            lines_html.append(f'<span class="l-sys">{line}</span>')

    inner = "<br/>".join(lines_html)
    # tiny inline JS to keep the log scrolled to bottom
    scroll = (
        "<script>(function(){var el=document.getElementById('dg-log');"
        "if(el)el.scrollTop=el.scrollHeight;})();</script>"
    )
    return (
        '<div class="dg-log-wrap">'
        f'<div class="dg-log" id="dg-log">{inner}</div>'
        "</div>" + scroll
    )


# ── Derived values ────────────────────────────────────────────────────────────
gh    = st.session_state.grid_health
nrg   = st.session_state.agent_energy
avg   = sum(gh) / 25
crit  = sum(1 for h in gh if h < 30)
steps = st.session_state.step_count
rew   = st.session_state.total_reward

# ── Header ───────────────────────────────────────────────────────────────────
if st.session_state.game_over:
    badge_cls, badge_txt = "badge-gameover", "GAME OVER"
elif st.session_state.auto_running:
    badge_cls, badge_txt = "badge-running",  "● RUNNING"
else:
    badge_cls, badge_txt = "badge-standby",  "STANDBY"

st.markdown(f"""
<div class="dg-header">
  <div>
    <div class="dg-title">⚡ DISASTER GRID</div>
    <div class="dg-subtitle">Autonomous AI Emergency Manager — GRPO / Llama-3</div>
  </div>
  <div class="dg-badge {badge_cls}">{badge_txt}</div>
  <div class="dg-step-counter">STEP&nbsp;<span>{steps:04d}</span></div>
</div>
""", unsafe_allow_html=True)

# ── Telemetry metrics row ─────────────────────────────────────────────────────
mc = st.columns(6, gap="small")
labels = ["Agent Energy", "City Avg HP", "Critical Sectors",
          "Total Reward",  "Repairs",    "Recharges"]
values = [f"{nrg}", f"{avg:.1f}", str(crit),
          f"{rew:+.1f}", str(st.session_state.repair_count),
          str(st.session_state.recharge_count)]
for col, lbl, val in zip(mc, labels, values):
    with col:
        st.metric(lbl, val)

# ── Progress bars ─────────────────────────────────────────────────────────────
pb1, pb2 = st.columns(2, gap="small")
with pb1:
    st.markdown('<div class="sec-label">Agent Energy</div>', unsafe_allow_html=True)
    st.progress(max(0, min(100, nrg)) / 100)
with pb2:
    st.markdown('<div class="sec-label">City Avg Health</div>', unsafe_allow_html=True)
    st.progress(max(0.0, min(100.0, avg)) / 100)

st.markdown("<hr/>", unsafe_allow_html=True)

# ── Main 3-column layout ──────────────────────────────────────────────────────
left_col, center_col, right_col = st.columns([2, 5, 3], gap="medium")

# ─────────────────────────────── LEFT PANEL ──────────────────────────────────
with left_col:

    # ── Command Interface ──
    st.markdown('<div class="dg-panel-title">◈ COMMAND INTERFACE</div>',
                unsafe_allow_html=True)
    st.markdown("")  # breathing room

    if st.session_state.auto_running:
        st.markdown('<div class="btn-amber">', unsafe_allow_html=True)
        if st.button("◼  Stop Agent", key="btn_stop"):
            st.session_state.auto_running = False
            _add_log("Agent halted by operator.", "sys")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        if not st.session_state.game_over:
            if st.button("▶  Run Autonomous Agent", key="btn_run"):
                st.session_state.auto_running = True
                _add_log("Autonomous agent activated — Llama-3 online.", "act")
                st.rerun()

    if not st.session_state.auto_running and not st.session_state.game_over:
        if st.button("⏭  Step Once", key="btn_step"):
            _do_step()
            st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
    if st.button("🌋  Inject Random Disaster", key="btn_dis"):
        crises = random.sample(range(1, 25), 4)
        st.session_state.env.reset(options={"target_crises": crises})
        
        # FIX: Tell Streamlit to sync with the newly updated physics engine!
        _sync_from_env() 
        _add_log(f"RANDOM DISASTER: Sectors {crises} have caught fire!", "crit")
        
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
    if st.button("🎯  Targeted Attack (Corners)", key="btn_corners"):
        corners = [4, 20, 24, 14]
        # FIX 2: Added 'options=' here so this button doesn't crash either!
        st.session_state.env.reset(options={"target_crises": corners})
        _sync_from_env()
        _add_log("TARGETED STRIKE: Corners [4, 14, 20, 24] hit!", "crit")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown('<div class="btn-slate">', unsafe_allow_html=True)
    if st.button("↺  Reset City", key="btn_reset"):
        _full_reset()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Legend ──
    st.markdown('<div class="dg-panel-title">◈ LEGEND</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="dg-legend">
🤖 &nbsp;Agent position<br/>
🏢 &nbsp;Base (Sector 0)<br/>
🔥 &nbsp;Critical sector (&lt;30 HP)<br/>
🟩 &nbsp;Healthy sector<br/>
<span style="color:#1e3a5f">▬</span> &nbsp;HP bar at cell base
</div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Session stats ──
    st.markdown('<div class="dg-panel-title">◈ SESSION STATS</div>', unsafe_allow_html=True)
    rew_cls  = "s-danger" if rew < 0 else "s-amber" if rew == 0 else ""
    crit_cls = "s-danger" if crit > 3 else "s-amber" if crit > 0 else ""
    st.markdown(f"""
<div class="dg-stat-grid">
  <div class="dg-stat-row">
    <span>Critical</span>
    <span class="dg-stat-val {crit_cls}">{crit}</span>
  </div>
  <div class="dg-stat-row">
    <span>Reward</span>
    <span class="dg-stat-val {rew_cls}">{rew:+.1f}</span>
  </div>
  <div class="dg-stat-row">
    <span>Repairs</span>
    <span class="dg-stat-val">{st.session_state.repair_count}</span>
  </div>
  <div class="dg-stat-row">
    <span>Recharges</span>
    <span class="dg-stat-val s-blue">{st.session_state.recharge_count}</span>
  </div>
  <div class="dg-stat-row">
    <span>Steps</span>
    <span class="dg-stat-val s-amber">{steps}</span>
  </div>
  <div class="dg-stat-row">
    <span>Avg HP</span>
    <span class="dg-stat-val">{avg:.1f}</span>
  </div>
</div>""", unsafe_allow_html=True)


# ──────────────────────────── CENTER PANEL (Grid) ────────────────────────────
with center_col:
    st.markdown('<div class="dg-panel-title">◈ CITY GRID — 5×5 OPERATIONAL THEATRE</div>',
                unsafe_allow_html=True)
    grid_html = _build_grid_html(st.session_state.grid_health, st.session_state.agent_pos)
    st.markdown(grid_html, unsafe_allow_html=True)


# ──────────────────────────── RIGHT PANEL (Log) ──────────────────────────────
with right_col:
    st.markdown('<div class="dg-panel-title">◈ AGENT BRAIN LOG</div>',
                unsafe_allow_html=True)
    log_html = _build_log_html(st.session_state.log_lines)
    st.markdown(log_html, unsafe_allow_html=True)


# ── Auto-run engine (at script bottom — runs ONE step then reruns) ────────────
if st.session_state.auto_running and not st.session_state.game_over:
    _do_step()
    time.sleep(0.38)   # ~2.6 steps/sec — adjust to taste
    st.rerun()
