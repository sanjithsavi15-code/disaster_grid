import os
import json
from openai import OpenAI
from src.disaster_grid.environment import CityGrid
from src.disaster_grid.models import AgentAction, ActionType

def run_inference():
    print("Initializing Disaster Grid Environment...")
    env = CityGrid()
    
    # Initialize the API client
    # NOTE: If the hackathon requires a specific API (like Grok or TogetherAI), 
    # just change the base_url and model name below!
    client = OpenAI(
        api_key=os.environ.get("API_KEY", "your-api-key-here"),
        base_url="https://api.openai.com/v1" 
    )
    
    print("\n--- Starting Disaster Scenario ---")
    # Our environment returns a tuple: (observation, info) on reset
    obs, _ = env.reset()
    
    done = False
    
    while not done:
        print(f"\nTime Step: {env.step_count}/50 | Energy: {env.agent_energy}")
        
        # 1. Package the environment state into a prompt for the LLM
        prompt = f"""
        You are an Autonomous AI Emergency Manager.
        
        Current Environment State:
        {json.dumps(obs, indent=2)}
        
        Rules:
        - You are on a 5x5 grid (indices 0 to 24). You start at index 0.
        - Moving (MOVE_N, MOVE_S, MOVE_E, MOVE_W) costs 2 energy.
        - REPAIR costs 15 energy and adds 25 health to your current sector.
        - RECHARGE adds 20 energy, but ONLY works if you are at index 0 (Base).
        - Do not let your energy hit 0. Navigate to critical sectors and repair them.
        
        Determine the best action. You MUST respond with a perfectly formatted JSON object matching this schema:
        {{"action": "MOVE_N" | "MOVE_S" | "MOVE_E" | "MOVE_W" | "REPAIR" | "RECHARGE" | "WAIT", "reasoning": "<string explaining your strategy>"}}
        """
        
        try:
            # 2. Call the LLM
            response = client.chat.completions.create(
                model="gpt-4o", # Replace with "grok-beta" or your required model
                messages=[
                    {"role": "system", "content": "You are a JSON-only API. You only output raw, valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # 3. Parse the JSON response
            raw_response = response.choices[0].message.content
            action_data = json.loads(raw_response)
            
            # Validate it through our Pydantic model just to be safe
            action_parsed = AgentAction(**action_data)
            
            print(f"🤖 AI decided: {action_parsed.action.value}")
            print(f"   Reasoning: {action_parsed.reasoning}")
            
            # 4. Execute the action in the environment
            # Our env.step returns a 5-item tuple and handles the dict parsing internally
            obs, reward, done, truncated, info = env.step(action_data)
            
            # Print any errors from the environment engine (like wall bumps)
            step_result = info.get("step_result", {})
            if step_result.get("is_error"):
                print(f"⚠️  Engine Warning: {step_result.get('error_message')}")
            
        except Exception as e:
            print(f"❌ Error during LLM processing: {e}")
            print("Forcing a WAIT action to prevent the loop from crashing...")
            fallback_action = {"action": ActionType.WAIT.value, "reasoning": "Fallback due to error"}
            obs, reward, done, truncated, info = env.step(fallback_action)

    # 5. The episode is finished. Print the final summary!
    print("\n" + "="*40)
    print("🎉 EPISODE COMPLETE 🎉")
    print(f"Final City Health: {sum(env.grid_health)/25:.1f}/100")
    print(f"Final Energy:      {env.agent_energy}")
    print(f"Steps Taken:       {env.step_count}")
    print("="*40)

if __name__ == "__main__":
    run_inference()
