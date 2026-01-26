import json
import torch
import os 
import numpy as np
import yaml
from typing import Any, Dict
from pathlib import Path
from utils.exception import CustomException
import sys
from utils.logger import logging
# ----------------------------------

def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its content as a dictionary."""
    try:
        if not os.path.exists(os.path.dirname(file_path)):
            raise FileNotFoundError(f"Directory does not exist: {os.path.dirname(file_path)}")
        with open(file_path, 'r') as json_file:
            data = json.load(json_file) 
        logging.info(f"JSON file loaded successfully from {file_path}")
        return data
    except Exception as e:
        raise CustomException(e, sys)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save a dictionary to a JSON file."""
    try:
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            logging.info(f"Created directory for JSON file at {os.path.dirname(file_path)}")
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        logging.info(f"JSON file saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def save_model(model: torch.nn.Module, file_path: str) -> None:
    """Save a PyTorch model to the specified file path."""
    try:
        if not file_path.endswith('.pt') and not file_path.endswith('.pth'):
            raise ValueError("File path must end with .pt or .pth extension")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            logging.info(f"Created directory for model at {os.path.dirname(file_path)}")
        torch.save(model.state_dict(), file_path)
        logging.info(f"Model saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_model(model: torch.nn.Module, file_path: str, device: str = "cpu") -> torch.nn.Module:
    '''Load a PyTorch model from the specified file path.'''
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model found at {file_path}")
        model.load_state_dict(torch.load(file_path, map_location=device))
        logging.info(f"Model weight loaded successfully from {file_path}")
        return model
    except Exception as e:
        raise CustomException(e, sys)

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML file and return its content as a dictionary."""
    try:
        if not os.path.exists(os.path.dirname(file_path)):
            raise FileNotFoundError(f"Directory does not exist: {os.path.dirname(file_path)}")
        with open(file_path, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
        logging.info(f"YAML file loaded successfully from {file_path}")
        return data
    except Exception as e:
        raise CustomException(e, sys)

def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """Save a dictionary to a YAML file."""
    try:
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            logging.info(f"Created directory for YAML file at {os.path.dirname(file_path)}")
        with open(file_path, 'w') as yaml_file:
            yaml.safe_dump(data, yaml_file)
        logging.info(f"YAML file saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_bin(file_path: str, dtype: Any = np.uint16) -> np.ndarray:
    """Load a binary file and return its content as a NumPy array."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No binary file found at {file_path}")
        data = np.fromfile(file_path, dtype=dtype)
        logging.info(f"Binary file loaded successfully from {file_path}")
        return data
    except Exception as e:
        raise CustomException(e, sys)

def save_bin(data: np.ndarray, file_path: str) -> None:
    """Save a NumPy array to a binary file."""
    try:
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            logging.info(f"Created directory for binary file at {os.path.dirname(file_path)}")
        if isinstance(data, list):
            data = np.array(data, dtype=np.uint16)
        data.tofile(file_path)
        logging.info(f"Binary file saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)