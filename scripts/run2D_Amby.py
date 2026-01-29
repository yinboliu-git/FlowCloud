"""
Training script for 2D Ambystoma dataset.

This script trains and evaluates the TransformerLatentODE model on Ambystoma
2D spatial-temporal point cloud data with gene expression.
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


def main_run_pipeline(config_path='config.yaml', dataset_name='ambystoma_2d'):
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
        Exist_use=True,
        model_save_name=model_save_path
    )

    # --- 3. Predict and Evaluate on Test Set ---
    print("\n--- 3. Predicting on Test Set and Evaluating ---")

    # Convert time_steps to consistent format
    if isinstance(test_data['time_steps'], torch.Tensor):
        test_data['time_steps'] = test_data['time_steps'].cpu().numpy()
    if isinstance(train_data['time_steps'], torch.Tensor):
        train_data['time_steps'] = train_data['time_steps'].cpu().numpy()

    test_time_steps_tensor = torch.from_numpy(test_data['time_steps']).float().to(device)
    train_time_steps_tensor = torch.from_numpy(train_data['time_steps']).float()

    # Generate predictions
    pred_data_dict = predict_and_format_structured_output(
        model,
        train_features_list,
        test_time_steps_tensor,
        device=device
    )

    # Evaluate interpolation
    df_interp = evaluate_interpolation_structured(
        true_data_dict=test_data,
        pred_data_dict=pred_data_dict,
        train_time_steps=train_time_steps_tensor,
        device=device
    )
    print("\n--- Interpolation Evaluation Results ---")
    print(df_interp)

    # Evaluate extrapolation
    df_extrap = evaluate_extrapolation_structured(
        true_data_dict=test_data,
        pred_data_dict=pred_data_dict,
        train_time_steps=train_time_steps_tensor,
        device=device
    )
    print("\n--- Extrapolation Evaluation Results ---")
    print(df_extrap)

    # Combine evaluation results
    df_results = pd.concat([df_interp, df_extrap]).sort_index()

    # --- 4. Save Results ---

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save evaluation metrics (all test points)
    csv_path_all = os.path.join(save_dir, 'test_evaluation_results_all.csv')
    df_results.to_csv(csv_path_all)
    print(f"Evaluation results (all test points) saved to: {csv_path_all}")

    # Save evaluation metrics (interpolation)
    csv_interp_path = os.path.join(save_dir, 'test_evaluation_results_interp.csv')
    df_interp.to_csv(csv_interp_path)
    print(f"Evaluation results (interpolation) saved to: {csv_interp_path}")

    # Save evaluation metrics (extrapolation)
    if not df_extrap.empty:
        csv_extrap_path = os.path.join(save_dir, 'test_evaluation_results_extrap.csv')
        df_extrap.to_csv(csv_extrap_path)
        print(f"Evaluation results (extrapolation) saved to: {csv_extrap_path}")

    # Save test data (ground truth)
    true_data_path = os.path.join(save_dir, 'test_true_data.pkl')
    # Convert tensors to numpy arrays
    for key in ['data_list', 'type_list', 'gene_list_raw', 'gene_list_processed']:
        if isinstance(test_data[key], list):
            test_data[key] = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t
                             for t in test_data[key]]

    joblib.dump(test_data, true_data_path)
    print(f"Ground truth test data saved to: {true_data_path}")

    # Save prediction results
    pred_data_path = os.path.join(save_dir, 'test_prediction_data.pkl')
    joblib.dump(pred_data_dict, pred_data_path)
    print(f"Model predictions saved to: {pred_data_path}")

    # --- 5. Generate Dynamic Visualization ---
    print("\n--- 5. Generating Dynamic Visualization ---")

    # Prepare visualization parameters
    gene_names = [f"Gene_{i+1}" for i in range(num_genes)] if num_genes > 0 else None

    # Generate visualization using all data
    evaluate_and_visualize_model(
        model=model,
        fixed_sampled_features_list=train_features_list,
        data_list=loaded_data_all['data_list'],
        time_steps=loaded_data_all['time_steps'],
        type_list=loaded_data_all['type_list'],
        gene_list_processed=loaded_data_all['gene_list_processed'],
        int_to_cell_type=loaded_data_all['int_to_cell_type'],
        gene_names=gene_names,
        device=device,
        output_save_path="full_trajectory_outputs.pt",
        extrapolation_factor=0.2,  # Extrapolate 20% beyond training range
        num_steps=50
    )
    print("\n--- Pipeline Completed Successfully ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate model on Ambystoma 2D dataset'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='ambystoma_2d',
                       help='Dataset name in config file')

    args = parser.parse_args()
    main_run_pipeline(args.config, args.dataset)
