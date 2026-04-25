import os

project_name = "disaster_grid"

structure = [
    f"{project_name}/.env",
    f"{project_name}/Dockerfile",
    f"{project_name}/app.py",
    f"{project_name}/pyproject.toml",
    f"{project_name}/src/disaster_grid/__init__.py",
    f"{project_name}/src/disaster_grid/models.py",
    f"{project_name}/src/disaster_grid/environment.py",
    f"{project_name}/src/disaster_grid/rewards.py",
    f"{project_name}/src/disaster_grid/utils.py",
    f"{project_name}/train/grpo_trainer.py",
    f"{project_name}/train/synthetic_data.json",
    f"{project_name}/tests/test_env.py",
]

for path in structure:
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Create the empty file
    with open(path, 'w') as f:
        pass

print(f"✅ Successfully created {project_name} structure!")