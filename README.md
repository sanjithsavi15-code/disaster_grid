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

### 🤖 The Agent (Autonomous Emergency Manager)

The agent operates on a 5x5 grid (25 sectors) and must balance two critical metrics:
1. **City Health:** Sectors degrade over time due to simulated entropy. The agent must navigate to critical sectors (health < 30) and perform `REPAIR` actions.
2. **Energy:** Every action costs energy (`MOVE` costs 2, `REPAIR` costs 15). If energy hits 0, the agent dies. The agent must periodically return to the Base (Sector 12) to `RECHARGE`.

### 🧠 The Reward System (Orthogonal Verifiers)

To prevent reward hacking, we use three independent verifiers during GRPO training:
* **R1 (Health):** Did the average city health improve? (Objective)
* **R2 (Efficiency):** Was the health gained worth the energy spent? (Strategy)
* **R3 (Format):** Did the LLM output valid JSON matching the `AgentAction` schema? (Compliance)

---

## ⚙️ Workflow & Architecture

### 1. Environment Simulation (OpenEnv)
The world is modeled using `openenv-core`. The `CityGrid` class manages the physics: grid state, agent position, energy deduction, and random entropy generation. It provides a clean `reset()` and `step()` interface.

### 2. Training (GRPO + Unsloth)
We utilize **Group Relative Policy Optimization (GRPO)** via the `trl` library, accelerated with **Unsloth** for 4-bit quantization, allowing us to train Llama-3 models efficiently within constrained GPU memory.

### 3. Frontend Dashboard (Streamlit)
A production-ready Streamlit interface acts as the "Operational Theatre." It visualizes the grid state, agent telemetry, and the agent's "brain log" (reasoning output) in real-time.

---

## 🌍 Real-World Scalability

While currently a 5x5 simulation, this architecture scales to real-world disaster management:
* **Grid Expansion:** The mathematical model can map to larger geographic sectors using real GIS data.
* **Multi-Agent Orchestration:** The framework supports introducing specialized agents (e.g., drones for scouting, heavy machinery for repair) coordinating via shared observations.
* **Predictive Maintenance:** By training on historical degradation rates, the model shifts from reactive repair to proactive maintenance.

---

## 📈 Development Progress & Demos

Throughout the hackathon, we iterated from manual logic validation to full autonomous control. Below are recordings of our progress.

### Phase 1: Engine Validation (Manual CLI)
Before training the LLM, we built a terminal-based CLI to manually playtest the `CityGrid` physics. This ensured movement costs, boundary collisions, repair mechanics, and the three-part reward system functioned perfectly.

<video controls src="./video1.mp4" width="100%"></video>

### Phase 2: Autonomous Agent Integration (UI Dashboard)
We then connected our GRPO-trained Llama-3 agent to the Streamlit UI. While the agent successfully parses the environment and issues valid JSON commands (satisfying the R3 verifier), we are currently tuning the R1/R2 reward weights. As seen in the demo, the agent sometimes exhibits suboptimal routing or gets trapped in energy-depletion loops, highlighting the challenge of balancing long-horizon planning with immediate crisis response.

video2.mp4

---
*Developed by Sanjith P & Varsha D for the Meta PyTorch OpenEnv Hackathon.*
