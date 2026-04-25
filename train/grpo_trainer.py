from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import torch

# 1. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
    max_seq_length = 512,
    load_in_4bit = True,
    fast_inference = True,
)

# 2. Add LoRA Adapters for training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)

# 3. Load your freshly generated data
dataset = load_dataset("json", data_files="synthetic_data.json", split="train")

# 4. Define Reward Functions (The logic for your Disaster Grid)
def reward_reach_goal(compts, **kwargs):
    # Reward for including the correct target coordinates in the output
    rewards = [1.0 if "Final Move:" in c else 0.0 for c in compts]
    return rewards

# 5. Training Configuration
training_args = GRPOConfig(
    learning_rate = 5e-6,
    num_train_epochs = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    outputs_dir = "outputs",
    optim = "adamw_8bit",
)

# 6. Initialize Trainer
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_reach_goal],
    args = training_args,
    train_dataset = dataset,
)

# 7. Start Training
trainer.train()
model.save_pretrained_merged("disaster_model_final", tokenizer, save_method = "merged_16bit")