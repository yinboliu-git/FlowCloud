"""
Configuration management for dynamic point cloud modeling.

This module provides utilities for loading and accessing configuration
parameters from YAML files.
"""

import yaml
import torch
import os
from pathlib import Path


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config):
    """
    Get device from config with automatic fallback to CPU.

    Args:
        config: Configuration dictionary

    Returns:
        torch.device object
    """
    device_str = config.get('training', {}).get('device', 'cuda:0')
    if torch.cuda.is_available():
        return torch.device(device_str)
    else:
        print(f"CUDA not available, using CPU instead of {device_str}")
        return torch.device('cpu')


def get_data_path(config, dataset_name):
    """
    Get full data path for a dataset.

    Args:
        config: Configuration dictionary
        dataset_name: Name of dataset (e.g., 'simulation_2d')

    Returns:
        Full path to dataset file
    """
    predata_dir = config['data']['predata_dir']
    data_file = config['datasets'][dataset_name]['data_file']
    return os.path.join(predata_dir, data_file)


def get_save_dir(config, dataset_name):
    """
    Get save directory for a dataset.

    Args:
        config: Configuration dictionary
        dataset_name: Name of dataset

    Returns:
        Path to save directory
    """
    save_dir = config['data']['save_dir']
    return os.path.join(save_dir, dataset_name)


def get_epochs(config, dataset_name):
    """
    Get number of training epochs for a dataset.

    Args:
        config: Configuration dictionary
        dataset_name: Name of dataset

    Returns:
        Number of epochs
    """
    dataset_config = config['datasets'].get(dataset_name, {})
    return dataset_config.get('epochs', config['training']['epochs'])
