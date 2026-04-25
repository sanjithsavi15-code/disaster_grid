import os
from .environment import CityGrid
from .models import ActionType

def play_manual():
    env = CityGrid()
    env.reset()
    
    key_map = {
        'w': ActionType.MOVE_N.value,
        's': ActionType.MOVE_S.value,
        'd': ActionType.MOVE_E.value,
        'a': ActionType.MOVE_W.value,
        'r': ActionType.REPAIR.value,
        'c': ActionType.RECHARGE.value,
        'q': ActionType.WAIT.value
    }
    
    done = False
    while not done:
        # Clear the terminal for a smooth "frame rate"
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"🌆 DISASTER GRID 🌆")
        print(f"Step: {env.step_count}/50 | 🔋 Energy: {env.agent_energy} | 🏥 City Health: {sum(env.grid_health)/25:.1f}")
        print("-" * 25)
        
        # Render the grid
        for y in range(5):
            row_str = ""
            for x in range(5):
                idx = y * 5 + x
                if env.agent_pos == idx:
                    row_str += "[🤖]" # Agent
                elif idx == 0: 
                    row_str += "[🏢]" # Base / Recharge Station
                elif env.grid_health[idx] < 30:
                    row_str += "[🔥]" # Critical Fire
                else:
                    row_str += "[🟩]" # Healthy Sector
            print(row_str)
            
        print("-" * 25)
        print("Controls: W/A/S/D (Move) | R (Repair) | C (Recharge at 🏢) | Q (Wait) | EXIT")
        
        move = input("Command: ").strip().lower()
        if move == 'exit':
            break
            
        if move in key_map:
            # We wrap the action in the exact JSON schema the LLM will use
            action_payload = {"action": key_map[move], "reasoning": "Manual human override."}
            obs, reward, done, _, info = env.step(action_payload)
            
            # Print the secret receipt to see if rewards.py will score it right
            step_result = info.get("step_result", {})
            if step_result.get("is_error"):
                print(f"⚠️ ERROR: {step_result.get('error_message')}")
        else:
            print("Invalid command. Press Enter to continue.")
            input()

    print("\n🚨 SIMULATION TERMINATED 🚨")
    print(f"Final City Health: {sum(env.grid_health)/25:.1f}/100")

if __name__ == "__main__":
    play_manual()