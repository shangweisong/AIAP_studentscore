import yaml
import os
import sys
from pathlib import Path
import subprocess



def load_config():
    """
    Load configuration from the specified YAML file.
    """
    config_path = os.path.join(os.path.dirname(__file__),"config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def install_requirements(requirements_file="requirements.txt"):
    """
    Install packages listed in the requirements.txt file.
    """
    req_path = Path(requirements_file)
    if req_path.is_file():
        print(f"Installing dependencies from {requirements_file}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
            sys.exit(1)
    else:
        print(f"Requirements file '{requirements_file}' not found. Skipping installation.")

def setup_project():
    config = load_config()
    print("Configuration loaded successfully.")
    # Additional setup actions if needed
    print("Setup complete. You can now run the project.")

    # Install required packages
    install_requirements()

    print("Setup complete. You can now run the project.")

if __name__ == "__main__":
    setup_project()