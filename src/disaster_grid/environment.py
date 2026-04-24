import random
import json
from typing import Any, Tuple, Dict
from openenv.core import Environment
from pydantic import ValidationError
from .models import ActionType, DisasterObservation, AgentAction, StepResult

class CityGrid(Environment):
    """
    Disaster Grid Environment for OpenEnv.
    A 5x5 grid where an agent manages energy to repair decaying sectors.
    """
    def __init__(self):
        super().__init__()
        self.grid_health = [100] * 25
        self.agent_pos = 0
        self.agent_energy = 100
        self.step_count = 0

    def _index_to_coord(self, index: int) -> Tuple[int, int]:
        """Convert a 1D array index (0-24) to 2D (x,y) coordinates."""
        return index % 5, index // 5

    def _coord_to_index(self, x: int, y: int) -> int:
        """Convert 2D (x,y) coordinates back to 1D array index."""
        return y * 5 + x

    def _get_observation(self) -> DisasterObservation:
        """Calculates city state and returns a strict Pydantic observation."""
        critical_sectors = [i for i, h in enumerate(self.grid_health) if h < 30]
        avg_health = sum(self.grid_health) / 25.0
        
        return DisasterObservation(
            step_number=self.step_count,
            agent_position=self.agent_pos,
            agent_energy=self.agent_energy,
            current_sector_health=self.grid_health[self.agent_pos],
            critical_sectors=critical_sectors,
            average_city_health=avg_health
        )

    def _apply_entropy(self):
        """Randomly degrades 2 distinct sectors by 5 health points."""
        indices = random.sample(range(25), 2)
        for idx in indices:
            self.grid_health[idx] = max(0, self.grid_health[idx] - 5)

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[dict, dict]:
        """Resets the world to a new starting state with 5 random critical emergencies."""
        if seed is not None:
            random.seed(seed)
            
        self.step_count = 0
        self.agent_energy = 100
        self.agent_pos = 0
        
        # Initialize grid with random health between 50 and 100
        self.grid_health = [random.randint(50, 100) for _ in range(25)]
        
        # Force 5 random distinct sectors into "critical" mode (20 health)
        critical_indices = random.sample(range(25), 5)
        for idx in critical_indices:
            self.grid_health[idx] = 20
            
        return self._get_observation().model_dump(), {}

    def step(self, action: str | dict) -> Tuple[dict, float, bool, bool, dict]:
        """Executes the agent's action and updates the world physics."""
        # 1. Snapshot Before
        energy_before = self.agent_energy
        city_health_before = sum(self.grid_health) / 25.0
        
        is_error = False
        error_message = ""
        action_parsed = None
        action_attempted = str(action)
        
        # 2. Action Parsing & Validation
        try:
            if isinstance(action, str):
                action_dict = json.loads(action)
            else:
                action_dict = action
            action_parsed = AgentAction(**action_dict)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            is_error = True
            error_message = str(e)
            
        # 3. Execution Physics
        if not is_error and action_parsed:
            act = action_parsed.action
            x, y = self._index_to_coord(self.agent_pos)
            
            if act == ActionType.MOVE_N:
                if y > 0:
                    self.agent_energy -= 2
                    self.agent_pos = self._coord_to_index(x, y - 1)
                else:
                    is_error = True
                    error_message = "Illegal move: MOVE_N from top boundary (row 0)."
            elif act == ActionType.MOVE_S:
                if y < 4:
                    self.agent_energy -= 2
                    self.agent_pos = self._coord_to_index(x, y + 1)
                else:
                    is_error = True
                    error_message = "Illegal move: MOVE_S from bottom boundary (row 4)."
            elif act == ActionType.MOVE_E:
                if x < 4:
                    self.agent_energy -= 2
                    self.agent_pos = self._coord_to_index(x + 1, y)
                else:
                    is_error = True
                    error_message = "Illegal move: MOVE_E from right boundary (col 4)."
            elif act == ActionType.MOVE_W:
                if x > 0:
                    self.agent_energy -= 2
                    self.agent_pos = self._coord_to_index(x - 1, y)
                else:
                    is_error = True
                    error_message = "Illegal move: MOVE_W from left boundary (col 0)."
            elif act == ActionType.REPAIR:
                self.agent_energy -= 15
                self.grid_health[self.agent_pos] = min(100, self.grid_health[self.agent_pos] + 25)
            elif act == ActionType.RECHARGE:
                if self.agent_pos == 0:
                    self.agent_energy = min(100, self.agent_energy + 20)
                else:
                    is_error = True
                    error_message = "Illegal recharge: RECHARGE is only valid at sector 0."
                    self.agent_energy -= 1  # Wasted energy trying to recharge nowhere
            elif act == ActionType.WAIT:
                pass
        else:
            # The LLM output garbage JSON, so it wastes 1 energy standing confused
            self.agent_energy -= 1

        # Keep energy in schema-compatible bounds for StepResult validation.
        self.agent_energy = max(0, min(100, self.agent_energy))

        # 4. Entropy & Time Update
        self._apply_entropy()
        self.step_count += 1
        
        # 5. Snapshot After
        energy_after = self.agent_energy
        city_health_after = sum(self.grid_health) / 25.0
        
        # 6. Check Termination
        done = (self.agent_energy <= 0) or (self.step_count >= 50)
        
        # 7. Create the "Receipt" for the Reward Verifiers
        step_result = StepResult(
            action_attempted=action_attempted,
            action_parsed=action_parsed,
            energy_before=energy_before,
            energy_after=energy_after,
            city_health_before=city_health_before,
            city_health_after=city_health_after,
            is_error=is_error,
            error_message=error_message
        )
        
        # Return standard OpenEnv tuple (obs, reward, done, truncated, info)
        # Note: We return 0.0 for reward here, as Varsha's rewards.py handles it!
        return self._get_observation().model_dump(), 0.0, done, False, {"step_result": step_result.model_dump()}