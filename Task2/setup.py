import yaml
import os
import sys
from pathlib import Path
import subprocess

config_path = "Task2/config.yaml"

def load_config():
    with open(config_path,"r") as file:
        return yaml.safe_load(file) 
