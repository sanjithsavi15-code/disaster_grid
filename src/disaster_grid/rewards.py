from typing import Dict, Any
from .models import StepResult

def get_health_reward(step_result: StepResult) -> float:
    """
    R1: The Objective Reward. 
    Rewards the agent strictly for increasing the average health of the city.
    Penalizes it when health drops (due to entropy or ignoring fires).
    """
    delta = step_result.city_health_after - step_result.city_health_before
    # Delta is positive if they repaired, negative if entropy degraded the grid.
    return float(delta)

def get_efficiency_reward(step_result: StepResult) -> float:
    """
    R2: The Strategic Reward.
    Prevents the agent from wandering aimlessly. It measures how much 
    health was gained per unit of energy spent.
    """
    # Convert average city health back to total health points (x 25 sectors)
    health_gained = max(0.0, step_result.city_health_after - step_result.city_health_before) * 25.0
    energy_spent = step_result.energy_before - step_result.energy_after
    
    # If they spent no energy (or gained it via RECHARGE), efficiency is 0 for this specific step
    if energy_spent <= 0:
        return 0.0
        
    return float(health_gained / energy_spent)

def get_format_reward(step_result: StepResult) -> float:
    """
    R3: The Guardrail Reward.
    Crucial for LLMs. If it hallucinates a command or outputs bad JSON, 
    we hit it with a strict penalty so it learns the schema fast.
    """
    if step_result.is_error or step_result.action_parsed is None:
        return -2.0 # Harsh penalty for breaking the rules
    return 1.0      # Small constant reward for obeying the schema

def compute_reward(info: Dict[str, Any]) -> float:
    """
    The Master Verifier. 
    Extracts the 'Receipt' from the environment and calculates the final gradient.
    """
    if "step_result" not in info:
        return 0.0
        
    step_result = StepResult(**info["step_result"])
    
    # --- THE HACKATHON DIALS ---
    # Varsha: Tweak these weights during your 30-hour training!
    w_health = 1.0      # How much we care about saving the city
    w_efficiency = 0.5  # How much we care about saving energy
    w_format = 1.0      # How much we care about valid JSON
    
    r_health = get_health_reward(step_result)
    r_efficiency = get_efficiency_reward(step_result)
    r_format = get_format_reward(step_result)
    
    total_reward = (w_health * r_health) + (w_efficiency * r_efficiency) + (w_format * r_format)
    
    return float(total_reward)