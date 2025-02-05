import subprocess

def switch_conda_environment(env_name, python_executable, script=None, args=None):
    """
    Switches to a specific conda environment and optionally runs a script or command.
    Args:
        env_name (str): Name of the conda environment.
        python_executable (str): Path to the Python executable in the environment.
        script (str): Path to a script to execute in the environment (optional).
        args (list): Additional arguments for the script (optional).
    """
    try:
        # Prepare the command to run in the target environment
        cmd = [python_executable]
        if script:
            cmd.append(script)
        if args:
            cmd.extend(args)

        # Execute the command
        print(f"Switching to conda environment: {env_name}")
        result = subprocess.run(cmd, text=True, capture_output=True)
        print(f"Output from {env_name}:\n{result.stdout}")
        if result.stderr:
            print(f"Errors from {env_name}:\n{result.stderr}")

    except Exception as e:
        print(f"Failed to switch to conda environment {env_name}: {e}")

# Paths to Python executables for different environments
env1_python = "/home/mohak/anaconda3/envs/occ_main/bin/python"
env2_python = "/home/mohak/anaconda3/envs/iSLAM/bin/python"

# Dummy script content for testing
dummy_script = """
import sys
import torch
print(f'Torch version: {torch.__version__}')
print(f'Python executable: {sys.executable}')
print(f'Python version: {sys.version}')
print('Hello from this environment!')
"""

# Save the dummy script to a file
script_path = "dummy_script.py"
with open(script_path, "w") as f:
    f.write(dummy_script)

# Test switching and running the script in two environments
switch_conda_environment("occ_main", env1_python, script=script_path)
switch_conda_environment("iSLAM", env2_python, script=script_path)

# Clean up the dummy script
import os
os.remove(script_path)
