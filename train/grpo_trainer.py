import os
import sys
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel, PatchFastRL

# ── 1. GRPO PATCHING & PATH INJECTION ────────────────────────────────────────
PatchFastRL("GRPO", FastLanguageModel)

# Ensure Python can find your 'src' directory
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from disaster_grid.environment import CityGrid
from disaster_grid.rewards import compute_reward
from disaster_grid.models import AgentAction

# ── 2. CONFIGURATION & MODELS ────────────────────────────────────────────────
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
SYSTEM_PROMPT = """
You are the Disaster Grid Autonomous Manager. 
Your goal is to maintain city health and manage your energy.
RULES:
1. Always respond in valid JSON format.
2. The grid is 5x5 (Indices 0-24). Base is at 0.
3. MOVE costs 2 energy. REPAIR costs 15 energy.
4. RECHARGE (+20 energy) ONLY works at index 0.
5. Prioritize REPAIRing critical sectors (health < 30) but return to base before energy hits 0.
"""

# ── 3. LOAD MODEL & TOKENIZER ────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 1024,
    load_in_4bit = True,
    trust_remote_code = True,
)

# ── 4. DATASET PREPARATION ───────────────────────────────────────────────────
def format_dataset(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]}
        ]
    }

dataset = load_dataset("json", data_files="train/synthetic_data.json", split="train")
dataset = dataset.map(format_dataset)

# ── 5. REWARD FUNCTION WRAPPER ───────────────────────────────────────────────
env = CityGrid()

def disaster_reward_func(prompts, completions, **kwargs):
    """
    Bridge between LLM outputs and your physics engine.
    Computes the weighted R1 (Health), R2 (Efficiency), and R3 (Format) rewards.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Reset environment for a fresh rollout
        env.reset()
        
        # Extract the assistant's response content
        content = completion[0]["content"] if isinstance(completion, list) else completion
        
        # Execute action in the environment (handles JSON parsing internally)
        obs, reward_env, done, truncated, info = env.step(content)
        
        # Calculate the aggregate reward from rewards.py logic
        score = compute_reward(info)
        rewards.append(score)
    return rewards

# ── 6. TRAINER INITIALIZATION ────────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir = "grpo_disaster_grid",
    learning_rate = 2e-5,
    logging_steps = 5,
    max_steps = 100,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 2,
    max_completion_length = 128,
)

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [disaster_reward_func],
    args = training_args,
    train_dataset = dataset,
)

# ── 7. EXECUTION & GRAPHING ──────────────────────────────────────────────────
print("🚀 Initiating GRPO Training Loop...")
try:
    trainer.train()
finally:
    # Always generate a graph, even if training is interrupted
    history = trainer.state.log_history
    steps = [log["step"] for log in history if "reward" in log]
    rewards = [log["reward"] for log in history if "reward" in log]

    if steps:
        plt.figure(figsize=(10, 5))
        plt.plot(steps, rewards, marker='o', color='#10b981', linewidth=2)
        plt.title('Disaster Grid Optimization - Learning Curve', color='white')
        plt.gca().set_facecolor('#1e1e1e')
        plt.gcf().patch.set_facecolor('#1e1e1e')
        plt.savefig('learning_curve.png', bbox_inches='tight', dpi=300)
        plt.show()
        print("✅ Training complete. Graph saved as 'learning_curve.png'.")

# ── 8. SAVE TRAINED BRAIN ────────────────────────────────────────────────────
model.save_pretrained("disaster_agent_final")
tokenizer.save_pretrained("disaster_agent_final")