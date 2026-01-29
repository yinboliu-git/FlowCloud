"""
FlowCloud: Dynamic Spatial-Temporal Point Cloud Modeling

A PyTorch-based framework for modeling dynamic point clouds with cell type
and gene expression prediction using Transformer-based Latent ODE architecture.
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

# Import main components for easy access
from flowcloud.models.transformer_ode import TransformerLatentODE
from flowcloud.training.trainer import train_dynamic_point_cloud_model

__all__ = [
    'TransformerLatentODE',
    'train_dynamic_point_cloud_model',
]
