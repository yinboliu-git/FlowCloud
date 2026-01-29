"""
Training script for 3D Human dataset.

This script trains and evaluates the TransformerLatentODE model on Human
3D spatial-temporal point cloud data with gene expression.
"""

import torch
import os
import joblib
import pandas as pd
import numpy as np
import argparse

from flowcloud_model import (
    TransformerLatentODE, load_processed_data, split_data,
    train_dynamic_point_cloud_model, predict_and_format_structured_output,
    evaluate_test_set_structured, evaluate_interpolation_structured,
    evaluate_extrapolation_structured, save_evaluation_results,
    evaluate_population_mae, _calculate_all_metrics, evaluate_spatial_reconstruction,
    evaluate_cell_type_accuracy, evaluate_gene_scc, evaluate_and_visualize_model
)

from eval_model import (
    load_processed_data, split_data,
    evaluate_interpolation_structured,
    evaluate_extrapolation_structured
)

from config import load_config, get_device, get_data_path, get_save_dir, get_epochs


def main_run_pipeline(config_path='config.yaml', dataset_name='human_3d'):
    """
    Main training and evaluation pipeline.

    Args:
        config_path: Path to configuration YAML file
        dataset_name: Name of dataset in config file
    """
    # Load configuration
    config = load_config(config_path)
    device = get_device(config)
    epochs = get_epochs(config, dataset_name)
    data_path = get_data_path(config, dataset_name)
    save_dir = get_save_dir(config, dataset_name)

    print(f"Using device: {device}")
    print(f"Training for {epochs} epochs")

    # --- 1. Data Preparation ---
    print("\n--- 1. Loading and Splitting Data ---")

    try:
        print(f"Attempting to load data from {data_path}...")

        # Load preprocessed data
        data_list, type_list, gene_list_raw, gene_list_processed, time_steps, \
            cell_type_to_int, int_to_cell_type = load_processed_data(data_path)

        loaded_data_all = {
            'data_list': data_list,
            'type_list': type_list,
            'gene_list_raw': gene_list_raw,
            'gene_list_processed': gene_list_processed,
            'time_steps': time_steps,
            'cell_type_to_int': cell_type_to_int,
            'int_to_cell_type': int_to_cell_type,
        }

    except ImportError:
        print("Error: Cannot import necessary functions from flowcloud_model. "
              "Please ensure flowcloud_model.py is available.")
        return
    except FileNotFoundError:
        print(f"Error: Cannot find data file: {data_path}. Please check the path.")
        return

    # Split data into train and test sets
    train_data, test_data = split_data(loaded_data_all)

    # --- 2. Train Model ---
    print("\n--- 2. Training Model ---")

    # Extract training parameters
    if train_data['int_to_cell_type'] is None:
        num_cell_types = None
        num_genes = None
    else:
        num_cell_types = len(train_data['int_to_cell_type'])
        num_genes = train_data['gene_list_processed'][0].shape[1]

    # Define model save path
    model_save_path = os.path.join(save_dir, 'best_flowcloud_model.pt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model, train_features_list, _ = train_dynamic_point_cloud_model(
        data_list=train_data['data_list'],
        time_steps=train_data['time_steps'],
        type_list=train_data['type_list'],
        gene_list_processed=train_data['gene_list_processed'],
        num_cell_types=num_cell_types,
        num_genes=num_genes,
        epochs=epochs,
        device=device,
        Exist_use=False,
        model_save_name=model_save_path
    )

    print("\n--- Training Completed Successfully ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate model on Human 3D dataset'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='human_3d',
                       help='Dataset name in config file')

    args = parser.parse_args()
    main_run_pipeline(args.config, args.dataset)
