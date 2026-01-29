# Dynamic Spatial-Temporal Point Cloud Modeling with Gene Expression

A PyTorch-based deep learning framework for modeling dynamic spatial-temporal point clouds with cell type and gene expression prediction using a Transformer-based Latent ODE architecture.

## Overview

This repository implements a continuous-time trajectory modeling system for spatial-temporal biological data. The model combines Neural ODEs with Transformer architectures to predict:
- Cell spatial trajectories over time
- Cell type classifications
- Gene expression patterns
- Population dynamics

## Features

- **Continuous-time modeling**: Uses Neural ODEs (via `torchdiffeq`) for smooth trajectory prediction
- **Multi-task learning**: Jointly predicts spatial coordinates, cell types, and gene expression
- **Hierarchical point cloud encoding**: Multi-scale feature extraction from spatial data
- **Interpolation and extrapolation**: Predicts both within and beyond training time ranges
- **Support for 2D and 3D data**: Handles both planar and volumetric spatial datasets

## Model Architecture

### TransformerLatentODE

The core model consists of:

1. **Hierarchical Point Cloud Encoder**: Extracts multi-scale features from spatial coordinates
2. **Time Encoder**: Encodes temporal information
3. **Latent Velocity Network**: Neural ODE function modeling continuous dynamics in latent space
4. **Joint Decoder**: Unified decoder for spatial coordinates, cell types, and gene expression

The model uses optimal transport-based losses to ensure trajectory consistency and smooth temporal evolution.

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Data Format

Preprocessed data files (`.pt` format) should contain:
- `data_list`: List of spatial coordinates per timepoint (N×2 or N×3 tensors)
- `type_list`: Cell type labels per timepoint
- `gene_list_raw`: Raw gene expression matrices
- `gene_list_processed`: Preprocessed gene expression
- `time_steps`: Array of time values
- `cell_type_to_int`, `int_to_cell_type`: Cell type mappings

Place data files in the `./predata/` directory.

### Configuration

Edit `config.yaml` to set:
- Training parameters (epochs, device, learning rate)
- Data paths
- Dataset-specific configurations

Example:
```yaml
training:
  epochs: 20
  device: "cuda:0"

data:
  predata_dir: "./predata"
  save_dir: "./save_data"

datasets:
  simulation_2d:
    data_file: "Simulation_2D.pt"
    epochs: 2000
```

### Training

Train on different datasets using the provided scripts:

**2D Datasets:**
```bash
# Simulated 2D data
python run2D_sim.py --config config.yaml --dataset simulation_2d

# Ambystoma 2D data
python run2D_Amby.py --config config.yaml --dataset ambystoma_2d
```

**3D Datasets:**
```bash
# Drosophila embryo 3D data
python run3D_Dros.py --config config.yaml --dataset drosophila_3d

# Human 3D data
python run3D_Human.py --config config.yaml --dataset human_3d
```

### Output

Training produces:
- `best_flowcloud_model.pt`: Saved model checkpoint
- `test_evaluation_results_all.csv`: Complete evaluation metrics
- `test_evaluation_results_interp.csv`: Interpolation metrics
- `test_evaluation_results_extrap.csv`: Extrapolation metrics
- `test_true_data.pkl`: Ground truth test data
- `test_prediction_data.pkl`: Model predictions
- Visualization GIFs showing trajectory evolution

## Evaluation Metrics

The model is evaluated on:
- **Spatial Reconstruction**: Chamfer distance between predicted and true coordinates
- **Cell Type Accuracy**: Classification accuracy for cell types
- **Gene Expression Correlation**: Spearman correlation for gene expression
- **Population MAE**: Mean absolute error in cell count prediction

## Key Components

### Core Files

- `flowcloud_model.py`: Main model implementation, training, and evaluation
- `eval_model.py`: Evaluation utilities
- `VAE_utils.py`: VAE utilities and PID controller for training
- `config.py`: Configuration management
- `run*.py`: Dataset-specific training scripts

### Main Functions

- `train_dynamic_point_cloud_model()`: Main training loop
- `predict_and_format_structured_output()`: Generate predictions
- `evaluate_interpolation_structured()`: Evaluate interpolation performance
- `evaluate_extrapolation_structured()`: Evaluate extrapolation performance
- `evaluate_and_visualize_model()`: Generate trajectory visualizations

## Model Details

### Loss Functions

The model uses a multi-objective loss combining:
- Spatial reconstruction loss (Chamfer distance)
- Cell type classification loss (focal loss)
- Gene expression correlation loss (Spearman correlation)
- Trajectory consistency loss (optimal transport-based)
- Existence prediction loss (for varying cell populations)

### ODE Integration

- Uses `torchdiffeq.odeint_adjoint` for memory-efficient backpropagation
- Automatic mixed precision for GPU efficiency
- Configurable tolerances (RTOL=1e-4, ATOL=1e-5)

## Datasets

The framework supports:
- **Simulation_2D**: Simulated 2D spatial-temporal data
- **Ambystoma_2D**: Ambystoma developmental data (2D)
- **DrosophilaEmbryo_3D**: Drosophila embryo development (3D)
- **Human_3D**: Human tissue spatial-temporal data (3D)

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added upon publication]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository.
