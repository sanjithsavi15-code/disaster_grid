---
title: AI-Kinetic Disaster Recovery Grid
emoji: 🏙️
colorFrom: red
colorTo: yellow
sdk: docker
app_file: app.py
pinned: true
license: mit
short_description: Multi-Agent RL crisis-grid recovery on OpenEnv
tags:
  - reinforcement-learning
  - multi-agent
  - openenv
  - pytorch
  - llm-agents
  - hackathon
---

# 🏙️ AI-Kinetic Disaster Recovery Grid

> A Multi-Agent Reinforcement Learning environment built on **OpenEnv** for the **Meta PyTorch × Scaler OpenEnv Hackathon**.

---

## 🚨 Problem Statement

Modern Large Language Models excel at conversational tasks and static reasoning, but they fail catastrophically at **real-time resource management** and **long-horizon planning** during active crises.

When a disaster strikes — a solar flare, cyber-attack, or severe weather event — infrastructure grids suffer cascading failures. In these scenarios:

- **Information is sparse.** Communication lines sever, creating partial observability.
- **Resources are finite.** Repairing one sector may mean abandoning another.
- **Rewards are delayed.** Actions taken now (gathering tools) may not yield results until much later.

Traditional centralized AI models assume perfect information and unlimited bandwidth. Placed in a constrained, decaying environment, they collapse.

---

## 💡 The Proposed Solution

We built the **AI-Kinetic Disaster Recovery Grid** — a Multi-Agent Reinforcement Learning (MARL) environment. Instead of a single "Omni-Agent," we deployed a distributed team of specialized agents acting on a dynamic 5×5 city grid.

### Agent Roles

| Agent | Role |
|---|---|
| 🛠️ **Scavenger** | Navigates the grid to collect scarce battery packs and repair parts. |
| ⚙️ **Engineer** | Executes complex, multi-step repair sequences. |
| 📡 **Dispatcher** | Holds the global map and intelligently guides the field agents. |

---

## ⚙️ Workflow

### 1. Environment Simulation (OpenEnv)

The world is modeled as a 5×5 matrix using `openenv-core`. Each sector tracks internal state: **Health**, **Power**, and **Risk**.

![OpenEnv Local Development Configuration](./Cursor_iz08V4mFrU.png)
*VS Code showing `environment.py` defining the `CityGrid` class and episode dynamics.*

### 2. Training Scripts (GRPO + Unsloth)

We used **Group Relative Policy Optimization (GRPO)**, accelerated with **Unsloth** to manage memory constraints during training.

```python
from openenv import OpenEnvGrid
from trl import GRPOTrainer

env = OpenEnvGrid(size=5, agents=["scavenger", "engineer", "dispatcher"])

trainer = GRPOTrainer(
    model="meta-llama/Llama-3-8B",
    args=config,
    train_dataset=env.get_training_data(),
)
trainer.train()
```

---

## 🧪 Validation

### Local Validation & Execution

Before pushing to the Hugging Face Space, we validate agent logic and entropy sequences via a local CLI built on `openenv-core` for rapid debugging of the 5×5 grid state transitions.

<video controls src="./Vrxpe4DR6s.mp4" width="100%"></video>

▶ [Watch validation video](./Vrxpe4DR6s.mp4)

![Local Terminal Simulation Output](./warp_D3ZKOqiyfq.png)
*Local terminal simulator showing a completed 50-step sequence tracking City Health and Energy metrics.*

### Live Agent Simulation (Operational Theatre)

To visualize multi-agent orchestration, we built a web-based UI — the **Operational Theatre** — deployed on Hugging Face Spaces. It allows random anomaly injection while observing LLM-powered agents respond in real time.

<video controls src="./opera_Z2g5nn07sJ.mp4" width="100%"></video>

▶ [Watch operational theatre video](./opera_Z2g5nn07sJ.mp4)

*HF Space UI showing the autonomous agent navigating the grid with the Agent Brain Log streaming reasoning.*

---

## 🌍 Real-World Applications

- **Autonomous Smart Grid Management** — Dynamically rerouting power around physical failures to prevent cascading blackouts.
- **Search and Rescue Drone Swarms** — Coordinating ground and air drones in disaster zones with limited battery life.
- **Self-Healing Server Networks** — Autonomously rerouting global traffic and deploying patches during cyber-attacks.

---

## ⚖️ Advantages & Trade-offs

### ✅ Advantages

- **Task Specialization** — Specialized roles converge faster than a single omni-agent.
- **Fault Tolerance** — No single point of failure; the Dispatcher reroutes around incapacitated agents.
- **Realistic Observability** — Native "fog of war," forcing agents to communicate to gain a global view.

### ⚠️ Trade-offs

- **Credit Assignment** — Hard to reward early strategic moves whose payoff arrives many steps later.
- **Non-Stationarity** — Other agents act simultaneously, making convergence non-trivial.

---

## 📈 Scalability

The architecture scales via **Hierarchical Multi-Agent Systems (HMAS)**. In a city-wide deployment, this 5×5 grid represents a single *Neighborhood*. Each Neighborhood Dispatcher reports to a higher-level **Zone Commander**, allowing the system to manage thousands of nodes without overwhelming any individual agent's context window.

---

## 🚀 Run It Locally

```bash
git clone https://huggingface.co/spaces/<your-username>/disaster-grid
cd disaster-grid

python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\Activate.ps1 on Windows

pip install -e .

# Run the local CLI simulator
python -m src.disaster_grid.utils

# Or launch the operational theatre
python app.py
```

**Controls in CLI mode:** `W/A/S/D` to move · `R` to repair · `C` to recharge · `Q` to wait · `EXIT` to terminate.

---

## 🏁 Conclusion

The **AI-Kinetic Disaster Recovery Grid** demonstrates that LLMs can transition from *strategic chatting* to **strategic action**. By implementing multi-agent coordination inside a decaying environment, we are engineering the autonomous emergency responders of the future.

---

*Developed by **Sanjith P** and **Varsha D** for the Meta PyTorch × Scaler OpenEnv Hackathon.*
