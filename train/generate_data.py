import json
import random

def generate_disaster_data(num_samples=100):
    dataset = []
    grid_size = 5
    
    for _ in range(num_samples):
        # Randomize positions
        start = (random.randint(0, 4), random.randint(0, 4))
        target = (random.randint(0, 4), random.randint(0, 4))
        while target == start:
            target = (random.randint(0, 4), random.randint(0, 4))
            
        obstacle = (random.randint(0, 4), random.randint(0, 4))
        while obstacle == start or obstacle == target:
            obstacle = (random.randint(0, 4), random.randint(0, 4))

        instruction = (
            f"You are a disaster response drone. Grid is {grid_size}x{grid_size}. "
            f"Start: {start}. Goal: {target}. Obstacle at {obstacle}. "
            "Output the move sequence (North, South, East, West) to reach the goal safely."
        )
        
        # Simple placeholder logic for 'ideal' output
        # In a real run, GRPO will learn to improve this
        dataset.append({
            "instruction": instruction,
            "input": "",
            "output": "Thinking: I must calculate the path... Final Move: [Sequence]"
        })

    with open("synthetic_data.json", "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"✅ Successfully generated {num_samples} samples in train/synthetic_data.json")

if _name_ == "_main_":
    generate_disaster_data(200) # Generates 200 scenarios
