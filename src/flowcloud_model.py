import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from torchdiffeq import odeint_adjoint as odeint
import imageio
import os
import shutil
import math
import torch.nn.functional as F
from geomloss import SamplesLoss
from scipy.stats import spearmanr
import ot # <--- Ensure this import is at the top of the file
import torch
from torchcde import cdeint, linear_interpolation_coeffs, LinearInterpolation
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
torch.cuda.amp.autocast(enabled=True)  # Enable automatic mixed precision
from preprocess import geneSelection
from losses import (
    chamfer_distance_with_indices,
    compute_repulsion_loss,
    compute_repulsion_loss_v2,
    compute_focal_loss,
    trajectory_consistency_loss,
    fgw_trajectory_consistency_loss
)
# --- Improvement 2: ODE Solver Tolerances ---
RTOL = 1e-4
ATOL = 1e-5

import torch
import numpy as np

# --- Simulation data hyperparameters ---
NUM_POINTS_GENERATE = 1500 # Target number of generated points "center value"

# ==============================================================================
# ✅ New 2D simulation function (with 5 linearly varying genes)
# ==============================================================================
# --- Simulation data hyperparameters ---
import torch
import numpy as np
# (Ensure torch and numpy are imported)

# --- Simulation data hyperparameters ---
NUM_POINTS_GENERATE = 1500 # This value now serves as the peak of the point count function

def _generate_data_for_time(t, num_time_steps, base_vertices, x_shift_factor):
    """
    (✅ Helper function - no changes)
    Contains the internal logic of the original loop body for generating data at a single time point t.
    """

    # --- Geometric shape evolution ---
    # (The (num_time_steps - 1) here ensures correct extrapolation at t=10.5)
    triangle_scale = 1.0 + 1.5 * np.sin(np.pi * (t - 1) / (num_time_steps - 1))
    hole_radius = 0.1 + (0.9 * (t - 1) / (num_time_steps - 1))

    # --- Calculate target point count ---
    # (Note: at t=10.5, (t - mid_point)^2 continues to increase, point count decreases quadratically)
    mid_point = (num_time_steps + 1) / 2.0
    peak_count = NUM_POINTS_GENERATE
    start_end_ratio = 0.4
    a = (start_end_ratio * peak_count - peak_count) / (1 - mid_point)**2
    target_count = int(a * (t - mid_point)**2 + peak_count)
    # Ensure point count is not negative during extrapolation
    target_count = max(0, target_count)

    # --- Generate and filter candidate points ---
    num_candidates = target_count * 5
    if num_candidates <= 0: # <= 0 captures the case where target_count is 0
         final_points = np.empty((0, 2))
    else:
        r1, r2 = np.random.rand(num_candidates, 1), np.random.rand(num_candidates, 1)
        sqrt_r1 = np.sqrt(r1)
        candidate_points = ((1 - sqrt_r1) * base_vertices[0] + 
                            (sqrt_r1 * (1 - r2)) * base_vertices[1] + 
                            (sqrt_r1 * r2) * base_vertices[2])
        scaled_points = candidate_points * triangle_scale
        dist_from_center = np.linalg.norm(scaled_points, axis=1)
        accepted_mask = dist_from_center > hole_radius
        valid_points = scaled_points[accepted_mask]

        # --- Sample to target count ---
        if len(valid_points) > 0:
            final_target_count = min(len(valid_points), target_count)
            indices = np.random.choice(len(valid_points), final_target_count, replace=False)
            final_points = valid_points[indices]
        else:
            final_points = valid_points

    # --- Apply global translation ---
    current_x_shift = x_shift_factor * (t - 1)
    final_points_shifted = final_points.copy()
    if final_points_shifted.shape[0] > 0:
        final_points_shifted[:, 0] += current_x_shift

    # --- Assign cell types ---
    v0, v1, _ = base_vertices * triangle_scale
    radius_type0 = 0.5 + 2.0 * (t - 1) / (num_time_steps - 1)
    radius_type1 = 2.5 - 2.0 * (t - 1) / (num_time_steps - 1)
    dist_to_v0 = np.linalg.norm(final_points - v0, axis=1)
    dist_to_v1 = np.linalg.norm(final_points - v1, axis=1)
    
    final_types = np.full(final_points.shape[0], 2)
    final_types[dist_to_v1 < radius_type1] = 1
    final_types[dist_to_v0 < radius_type0] = 0

    # --- Simulate gene expression ---
    num_final_points = final_points.shape[0]
    gene_expressions = np.zeros((num_final_points, 5), dtype=np.int32)

    if num_final_points > 0:
        x, y = final_points[:, 0], final_points[:, 1]
        t_norm = (t - 1) / max(1, num_time_steps - 1) # At t=10.5, t_norm > 1.0 (correct extrapolation)

        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8); rate_g1 = 10.0 * x_norm
        gene_expressions[:, 0] = np.random.poisson(rate_g1)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8); rate_g2 = 10.0 * y_norm
        gene_expressions[:, 1] = np.random.poisson(rate_g2)
        dist_center = np.linalg.norm(final_points, axis=1)
        dist_norm = (dist_center - dist_center.min()) / (dist_center.max() - dist_center.min() + 1e-8); rate_g3 = 10.0 * dist_norm
        gene_expressions[:, 2] = np.random.poisson(rate_g3)
        rate_g4 = np.zeros(num_final_points); rate_g4[final_types == 0] = 12.0; rate_g4[final_types != 0] = 1.0
        gene_expressions[:, 3] = np.random.poisson(rate_g4)
        rate_g5 = np.zeros(num_final_points); rate_g5[final_types == 1] = 15.0 * t_norm # G5 will continue to increase at extrapolation points
        gene_expressions[:, 4] = np.random.poisson(np.maximum(0, rate_g5))

    # --- Add noise and store ---
    noisy_points = final_points_shifted + np.random.normal(0, 0.05, final_points_shifted.shape)
    
    point_groups_t = torch.tensor(noisy_points, dtype=torch.float32)
    type_groups_t = torch.tensor(final_types, dtype=torch.long)
    gene_groups_t = torch.tensor(gene_expressions, dtype=torch.float32)
    
    return (point_groups_t, type_groups_t, gene_groups_t, t)


def generate_spatial_data_with_genes(num_time_steps=10):
    """
    (✅ Modified - generates training set, interpolation test set, and extrapolation test set)
    Generates a time series of 2D point clouds with interleaved training points (t=1, 2, ...) and test points (t=1.5, ..., 10.5).
    """
    print(f"Generating simulated data with deterministic count change and translation...")
    print(f"  - Mode: Generating {num_time_steps} 'training' points (1..10) and {num_time_steps} 'test' points (1.5..10.5).")

    # --- 1. Basic setup ---
    base_vertices = np.array([[0.0, 2.0], [-1.732, -1.0], [1.732, -1.0]])
    int_to_cell_type = {0: 'Type_0', 1: 'Type_1', 2: 'Type_2'}
    x_shift_factor = 2.0 

    all_data_tuples = []

    # --- 2. Loop 1: Generate training set data (maintain random state) ---
    print("  - Generating 'training' data points (t=1, 2, 3, ...)")
    train_time_steps_np = np.arange(1, num_time_steps + 1) # [1, 2, ..., 10]

    for t in train_time_steps_np:
        data_tuple = _generate_data_for_time(t, num_time_steps, base_vertices, x_shift_factor)
        all_data_tuples.append(data_tuple)

    # --- 3. (✅ Core modification) Loop 2: Generate test set data (including 10.5) ---
    print("  - Generating 'test' data points (t=1.5, 2.5, ..., 10.5)")

    # Change np.arange endpoint from num_time_steps + 0.5 to num_time_steps + 1.0
    # np.arange(1.5, 11.0, 1.0) -> [1.5, 2.5, ..., 9.5, 10.5]
    test_time_steps_np = np.arange(1.5, num_time_steps + 1.0, 1.0)

    for t in test_time_steps_np:
        data_tuple = _generate_data_for_time(t, num_time_steps, base_vertices, x_shift_factor)
        all_data_tuples.append(data_tuple)

    # --- 4. Sort and unpack (no changes) ---
    print(f"  - Sorting all {len(all_data_tuples)} data points by time...")

    all_data_tuples.sort(key=lambda x: x[3])

    point_groups = list(map(lambda x: x[0], all_data_tuples))
    type_groups  = list(map(lambda x: x[1], all_data_tuples))
    gene_groups  = list(map(lambda x: x[2], all_data_tuples))
    all_times_np = np.array(list(map(lambda x: x[3], all_data_tuples)))

    print("Data generation complete.")
    print(f"Total of {len(point_groups)} time points generated.")
    print("Number of generated points per time point:", [p.shape[0] for p in point_groups])
    print("Generated time points:", all_times_np)

    # --- 5. Return 5 values (no changes) ---
    return point_groups, type_groups, gene_groups, torch.tensor(all_times_np, dtype=torch.float32), int_to_cell_type

import torch
import numpy as np

import torch
import numpy as np

def generate_3d_ellipsoid_data_dynamic_types(num_time_steps=10, num_points_peak=4096):
    """
    Generates a time series of 3D point clouds with dynamically changing cell type regions.
    - Shape: An ellipsoid.
    - Hollowing: A fixed-size cube removes some points from the ellipsoid.
    - Dynamics: The ellipsoid grows then shrinks, and translates along the X-axis.
    - Cell types: Two types (left and right hemispheres), with boundary moving left and right over time.
    """
    print(f"Generating 3D simulated data (Ellipsoid with dynamic types)...")

    point_groups, type_groups, gene_groups = [], [], []
    time_steps_np = np.arange(1, num_time_steps + 1)

    ellipsoid_radii = np.array([2.5, 1.5, 1.5])
    hole_center = np.array([1.0, 0.5, 0.5])
    hole_size = np.array([1.5, 1.5, 1.5])
    x_shift_factor = 3.0

    for t in time_steps_np:
        # --- Dynamic changes ---
        # 1. Ellipsoid size
        scale_factor = 1.0 + 0.5 * np.sin(np.pi * (t - 1) / (num_time_steps - 1))
        current_radii = ellipsoid_radii * scale_factor

        # 2. Total cell count
        mid_point = (num_time_steps + 1) / 2.0
        start_end_ratio = 0.4
        a = (start_end_ratio * num_points_peak - num_points_peak) / (1 - mid_point)**2
        target_count = int(a * (t - mid_point)**2 + num_points_peak)

        # --- ✅ Core modification: Dynamic cell type boundary ---
        # Boundary line moves back and forth between -0.8 and +0.8
        boundary_shift_amplitude = 0.8
        boundary_x = boundary_shift_amplitude * np.sin(2 * np.pi * (t - 1) / (num_time_steps - 1))

        # --- Generate point cloud (same as previous version) ---
        num_candidates = target_count * 8
        if num_candidates <= 0:
            final_points = np.empty((0, 3))
        else:
            vec = np.random.randn(num_candidates, 3)
            vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
            u = np.random.rand(num_candidates, 1)
            candidate_points = vec * (u ** (1./3.))
            scaled_points = candidate_points * current_radii
            
            min_bound = hole_center - hole_size / 2
            max_bound = hole_center + hole_size / 2
            is_inside_hole = np.all((scaled_points > min_bound) & (scaled_points < max_bound), axis=1)
            valid_points = scaled_points[~is_inside_hole]
            
            if len(valid_points) > 0:
                final_target_count = min(len(valid_points), target_count)
                indices = np.random.choice(len(valid_points), final_target_count, replace=False)
                final_points = valid_points[indices]
            else:
                final_points = np.empty((0, 3))

        # --- Apply global translation ---
        current_x_shift = x_shift_factor * (t - 1)
        final_points_shifted = final_points.copy()
        if final_points_shifted.shape[0] > 0:
            final_points_shifted[:, 0] += current_x_shift

        # --- Assign cell types (based on dynamic boundary) ---
        num_final_points = final_points.shape[0]
        final_types = np.zeros(num_final_points, dtype=np.int64)
        if num_final_points > 0:
            # Use time-varying boundary_x as the dividing line
            final_types[final_points[:, 0] > boundary_x] = 1

        # --- Simulate gene expression (based on pre-translation coordinates) ---
        gene_expressions = np.zeros((num_final_points, 5), dtype=np.float32)
        if num_final_points > 0:
            x, y, z = final_points[:, 0], final_points[:, 1], final_points[:, 2]
            gene_expressions[:, 0] = np.random.poisson(np.maximum(0, 5 * (x / current_radii[0])))
            dist_from_center = np.linalg.norm(final_points, axis=1)
            gene_expressions[:, 1] = np.random.poisson(8 * np.exp(-dist_from_center**2))
            rate_g3 = np.zeros(num_final_points); rate_g3[final_types == 0] = 12.0; rate_g3[final_types != 0] = 1.0
            gene_expressions[:, 2] = np.random.poisson(rate_g3)
            gene_expressions[:, 3] = np.random.poisson(np.maximum(0, 5 * (z - z.min()) / (z.max() - z.min() + 1e-8)))
            gene_expressions[:, 4] = np.random.poisson(2, num_final_points)

        # --- Add noise and store ---
        noisy_points = final_points_shifted + np.random.normal(0, 0.05, final_points_shifted.shape)

        point_groups.append(torch.tensor(noisy_points, dtype=torch.float32))
        type_groups.append(torch.tensor(final_types, dtype=torch.long))
        gene_groups.append(torch.tensor(gene_expressions, dtype=torch.float32))

    print("3D data generation with dynamic types complete.")
    print("Generated points per time step:", [p.shape[0] for p in point_groups])
    return point_groups, type_groups, gene_groups, torch.tensor(time_steps_np, dtype=torch.float32)
# ==============================================================================
# Module 1: (New) Farthest Point Sampling (FPS) Function
# ==============================================================================
# ==============================================================================
# Module 1: Helper Functions (FPS, etc.)
# ==============================================================================
def farthest_point_sample(xyz, npoint):
    """
    (No changes) Uses iterative farthest point sampling to select a subset from point cloud.
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    (Optimized version) Selects points from point cloud based on indices.
    This version is more concise, efficient, and less error-prone.

    Args:
        points (torch.Tensor): Input point cloud features, shape [B, N, C].
        idx (torch.Tensor):    Indices of points to sample, shape [B, S].

    Returns:
        torch.Tensor: Sampled points, shape [B, S, C].
    """
    B, _, C = points.shape
    S = idx.shape[1]

    # Create batch indices [0, 1, ..., B-1] and reshape to [B, 1] for broadcasting
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).unsqueeze(1)
    new_points = points[batch_indices, idx, :]

    return new_points

def group_points(points, idx):
    """
    (No changes) Groups points based on center point indices and their neighbor indices.
    """
    B, N, C = points.shape
    B, S, K = idx.shape
    grouped_points = index_points(points, idx.view(B, -1))
    grouped_points = grouped_points.view(B, S, K, C)
    return grouped_points



# --- 2. Upgraded Model Architecture ---

# class PointCloudEncoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_dim):
#         super(PointCloudEncoder, self).__init__()
#         self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
#     def forward(self, p_with_features):
#         features = self.net(p_with_features); return torch.max(features, dim=0)[0]

class LocalTransformerEncoder(nn.Module):
    # (This submodule unchanged)
    def __init__(self, in_channels, out_channels, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(in_channels, out_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels, nhead=nhead, dim_feedforward=out_channels*2,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, grouped_features):
        B, S, K, C = grouped_features.shape
        grouped_features = grouped_features.view(B * S, K, C)
        embedded = self.embedding(grouped_features)
        transformed = self.transformer(embedded)
        group_feature = torch.max(transformed, dim=1)[0]
        group_feature = group_feature.view(B, S, -1)
        return group_feature

class HierarchicalFeatureLayer(nn.Module):
    # Key optimizations and dimension-adaptive modifications
    def __init__(self, npoint, k, in_channels, out_channels, spatial_dim):
        super().__init__()
        self.npoint = npoint
        self.k = k
        # Input dimension now dynamically determined by spatial_dim
        self.local_encoder = LocalTransformerEncoder(in_channels + spatial_dim, out_channels)

    def forward(self, xyz, features):
        # xyz: [B, N, D], D can be 2 or 3
        # features: [B, N, C_in]

        # 1. Farthest Point Sampling (FPS)
        fps_idx = farthest_point_sample(xyz, self.npoint) # [B, npoint]
        new_xyz = index_points(xyz, fps_idx) # [B, npoint, D]

        # 2. Neighborhood grouping (key performance optimization)
        # Directly compute distance from center points to all points, avoiding global KNN
        dist_matrix = torch.cdist(new_xyz, xyz) # [B, npoint, N]
        # Find indices of k nearest points for each center point
        knn_idx = dist_matrix.topk(self.k, dim=-1, largest=False).indices # [B, npoint, k]

        # Group based on indices
        grouped_xyz = group_points(xyz, knn_idx) # [B, npoint, k, D]
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2) # Centering

        if features is not None:
            grouped_features = group_points(features, knn_idx)
            local_input_features = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)
        else:
            local_input_features = grouped_xyz_norm

        # 3. Local Transformer feature learning
        new_features = self.local_encoder(local_input_features) # [B, npoint, C_out]

        return new_xyz, new_features


# ==============================================================================
# (Modified) Final version of hierarchical point cloud encoder
# ==============================================================================
# class HierarchicalPointCloudEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim, spatial_dim):
#         super().__init__()
#         self.spatial_dim = spatial_dim
        
#         # --- Hierarchical Feature Abstraction Module (Unchanged) ---
#         # First layer: from N points -> 512 points, feature dimension 3+C_in -> 128
#         self.sa1 = HierarchicalFeatureLayer(npoint=512, k=32, in_channels=input_dim - spatial_dim, out_channels=128, spatial_dim=spatial_dim)
#         # Second layer: from 512 points -> 128 points, feature dimension 128 -> 256
#         self.sa2 = HierarchicalFeatureLayer(npoint=128, k=32, in_channels=128, out_channels=256, spatial_dim=spatial_dim)

#         # ✅ --- New: Transformer layer for feature aggregation ---
#         # 1. Create a learnable global feature token (Class Token)
#         #    Its dimension needs to match the output feature dimension of sa2 (256)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, 256))

#         # 2. Create a standard Transformer encoder layer
#         aggr_layer = nn.TransformerEncoderLayer(
#             d_model=256,         # Input dimension
#             nhead=4,             # Number of heads
#             dim_feedforward=512, # Feedforward network dimension
#             batch_first=True,
#             dropout=0.0
#         )

#         # 3. Instantiate aggregation Transformer (can use 1-2 layers)
#         self.aggregation_transformer = nn.TransformerEncoder(aggr_layer, num_layers=1)
#         # -----------------------------------------

#         # --- Output MLP (Unchanged) ---
#         self.output_mlp = nn.Sequential(
#             nn.Linear(256, hidden_dim), # Input dimension remains 256
#             nn.ReLU(),
#             nn.Linear(hidden_dim, latent_dim)
#         )

#     def forward(self, p_with_features):
#         B = 1
#         p_with_features = p_with_features.unsqueeze(0)

#         xyz = p_with_features[..., :self.spatial_dim]
#         features = p_with_features[..., self.spatial_dim:] if p_with_features.shape[-1] > self.spatial_dim else None
#         # --- Through hierarchical structure (Unchanged) ---
#         l1_xyz, l1_features = self.sa1(xyz, features)
#         l2_xyz, l2_features = self.sa2(l1_xyz, l1_features) # l2_features shape: [B, 128, 256]
#         cls_tokens = self.cls_token.expand(B, -1, -1) # Replicate cls_token B times
#         x = torch.cat((cls_tokens, l2_features), dim=1)
#         x = self.aggregation_transformer(x)
#         global_feature = x[:, 0]
#         latent_vector = self.output_mlp(global_feature)
#         return latent_vector.squeeze(0)

class HierarchicalPointCloudEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, spatial_dim):
        super().__init__()
        self.spatial_dim = spatial_dim
        
        # --- Hierarchical Feature Abstraction Modules (Unchanged) ---
        self.sa1 = HierarchicalFeatureLayer(npoint=512, k=32, in_channels=input_dim - spatial_dim, out_channels=128, spatial_dim=spatial_dim)
        self.sa2 = HierarchicalFeatureLayer(npoint=128, k=32, in_channels=128, out_channels=256, spatial_dim=spatial_dim)

        # --- ✅ Modified: Transformer for Feature Aggregation (No Class Token) ---
        # 1. We no longer need the self.cls_token
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 256))

        # 2. The aggregation transformer remains the same. It will now process
        #    only the point features.
        aggr_layer = nn.TransformerEncoderLayer(
            d_model=256,         
            nhead=4,             
            dim_feedforward=512, 
            batch_first=True,
            dropout=0.0
        )
        self.aggregation_transformer = nn.TransformerEncoder(aggr_layer, num_layers=2)
        # ----------------------------------------------------------------

        # --- Output MLP (Unchanged) ---
        self.output_mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, p_with_features):
        B = 1 
        p_with_features = p_with_features.unsqueeze(0)
        
        xyz = p_with_features[..., :self.spatial_dim]
        features = p_with_features[..., self.spatial_dim:] if p_with_features.shape[-1] > self.spatial_dim else None

        # --- Pass through hierarchical layers (Unchanged) ---
        l1_xyz, l1_features = self.sa1(xyz, features)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features) # l2_features shape: [B, 128, 256]
        contextual_features = self.aggregation_transformer(l2_features)
        # 2. Apply mean pooling to the output features to get the global feature vector
        global_feature = contextual_features.mean(dim=1) # Shape: [B, 256]
        latent_vector = self.output_mlp(global_feature)
        
        return latent_vector.squeeze(0)

import torch
import torch.nn as nn
from einops import rearrange

# --- Helper functions (FPS, index_points, group_points) remain the same ---

# ==============================================================================
# Module 1: (New) Point Cloud Swin Transformer Block
# ==============================================================================
class PointSwinTransformerBlock(nn.Module):
    def __init__(self, npoint1, k1, in_channels1, out_channels1, 
                 npoint2, k2, in_channels2, out_channels2, spatial_dim):
        super().__init__()
        self.hfl1 = HierarchicalFeatureLayer(npoint1, k1, in_channels1, out_channels1, spatial_dim)
        self.hfl2 = HierarchicalFeatureLayer(npoint2, k2, in_channels2, out_channels2, spatial_dim)

    def forward(self, xyz, features):
        xyz1, features1 = self.hfl1(xyz, features)
        xyz2, features2 = self.hfl2(xyz1, features1)
        return xyz2, features2

class SwinHierarchicalPointCloudEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, spatial_dim):
        super().__init__()
        self.spatial_dim = spatial_dim

        self.block1 = PointSwinTransformerBlock(
            npoint1=1024, k1=32, in_channels1=input_dim - spatial_dim, out_channels1=64,
            npoint2=256, k2=32, in_channels2=64, out_channels2=128,
            spatial_dim=spatial_dim
        )

        self.block2 = PointSwinTransformerBlock(
            npoint1=64, k1=32, in_channels1=128, out_channels1=256,
            npoint2=16, k2=16, in_channels2=256, out_channels2=512,
            spatial_dim=spatial_dim
        )
        
        # Final global feature aggregation layer (we still use max pooling because it is simple and effective)
        # Input dimension is 512 from block2 output
        self.output_mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, p_with_features):
        B = 1
        # p_with_features = p_with_features.unsqueeze(0)
        
        xyz = p_with_features[..., :self.spatial_dim]
        features = p_with_features[..., self.spatial_dim:] if p_with_features.shape[-1] > self.spatial_dim else None

        # --- Pass through hierarchical pyramid structure ---
        # Each layer uses all point information from the previous layer for grouping and feature extraction
        l1_xyz, l1_features = self.block1(xyz, features)
        l2_xyz, l2_features = self.block2(l1_xyz, l1_features)
        
        # --- Final global feature aggregation ---
        # Apply max pooling to the 16 highly abstract feature points output from the last layer
        global_feature = torch.max(l2_features, dim=1)[0]

        latent_vector = self.output_mlp(global_feature)
        
        return latent_vector.squeeze(0)

## Currently not used
class PointCloudEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, nhead=4, num_layers=3):
        """
        A Transformer-based point cloud encoder.
        Args:
            input_dim (int): Input feature dimension for each point (e.g., 2 for xy + 16 for type_embedding)
            model_dim (int): Internal working dimension of Transformer
            latent_dim (int): Dimension of final output global feature vector
            nhead (int): Number of attention heads
            num_layers (int): Number of Transformer encoder layers
        """
        super(PointCloudEncoder, self).__init__()
        
        # 1. Input embedding layer: map each point to Transformer working dimension
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # 2. Define a standard Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4, # Usually 4 times d_model
            batch_first=True # Very important: make input shape [batch, num_points, features]
        )
        
        # 3. Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output layer: pool all point features and map to final latent_dim
        # Here we use simple average pooling, then pass through an MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # hidden_dim comes from your hyperparameters
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, p_with_features):
        """
        Args:
            p_with_features (torch.Tensor): Input point cloud features, shape [num_points, input_dim]
        
        Return:
            torch.Tensor: A single global feature vector, shape [latent_dim]
        """
        # Transformer needs batch dimension, so we add it
        # [num_points, input_dim] -> [1, num_points, input_dim]
        # p_with_features = p_with_features.unsqueeze(0)
        
        # 1. Input embedding
        # [1, num_points, input_dim] -> [1, num_points, model_dim]
        embedded_points = self.embedding(p_with_features)
        
        # 2. Pass through Transformer encoder
        # Input and output shapes are both [1, num_points, model_dim]
        contextual_features = self.transformer_encoder(embedded_points)
        
        # 3. Pooling
        # Average all point features to get a global feature
        # [1, num_points, model_dim] -> [1, model_dim]
        global_feature = contextual_features.mean(dim=1)
        
        # 4. Pass through output MLP
        # [1, model_dim] -> [1, latent_dim]
        latent_vector = self.output_mlp(global_feature)
        
        # Remove batch dimension before returning
        return latent_vector.squeeze(0)
    
class TimeEncoder(nn.Module):
    def __init__(self, out_features):
        super(TimeEncoder, self).__init__()
        if out_features % 2 != 0:
            raise ValueError("out_features must be an even number.")
        self.out_features = out_features
        # Use the same log-linear frequency division as PositionalEncoding
        # div_term shape is [out_features / 2]
        div_term = torch.exp(torch.arange(0, out_features, 2).float() * (-math.log(10000.0) / out_features))
        self.register_buffer('div_term', div_term)
        
    def forward(self, t):
        """
        Args:
            t (torch.Tensor): a scalar tensor representing time.
        Return:
            torch.Tensor: time features of shape [1, out_features]
        """
        # t * self.div_term shapeis [out_features / 2]
        periodic_fns_arg = t * self.div_term
        # a[::2] isstep sizeas2slice of，i.e. take 0, 2, 4...
        # a[1::2] isfromindex1start，step sizeas2slice of，i.e. take 1, 3, 5...
        out = torch.zeros(self.out_features, device=t.device)
        out[0::2] = torch.sin(periodic_fns_arg)
        out[1::2] = torch.cos(periodic_fns_arg)
        return out # Return shape [1, out_features]

## Differential equation
class LatentVelocityNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim, time_encoder_features=32):
        super(LatentVelocityNet, self).__init__()
        
        self.time_encoder  = nn.Sequential(
                    nn.Linear(1, time_encoder_features),
                    nn.ReLU(),
                    nn.Linear(time_encoder_features, time_encoder_features)  # Add one linear mapping layer
                )
        
        # Input dimension updated to: latent dimension + time feature dimension
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_encoder_features, hidden_dim), 
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, t, z):
        # z shape is [latent_dim]
        # t is a scalar
        
        # 1. Encode time t into feature vector
        # time_features shape is [time_encoder_features]
        t_1d = t.view(1)  # 0D -> 1D (e.g., tensor(2.0) -> tensor([2.0]))
        
        # 1. Encode time features (output dimension:[time_encoder_features]）
        time_features = self.time_encoder(t_1d)  # Input 1D, output 1D
        
        # 2. Concatenate state and time features
        zt = torch.cat([z, time_features], dim=0)
        
        # 3. (Corrected) Only return dz/dt
        return self.net(zt) # [latent_dim]
    
## Not used
class CDEFunc(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Compared to ODE, CDE functions usually require stronger expressiveness
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # Output dimension is latent_dim * latent_dim, for subsequent interaction with control signal
            nn.Linear(hidden_dim, latent_dim * latent_dim)
        )

    def forward(self, t, z):
        # z shape is [batch, latent_dim]
        # t is a scalar
        
        # Reshape network output into a matrix [batch, latent_dim, latent_dim]
        # This matrix represents how the system responds to control signal changes
        return self.net(z).view(-1, self.latent_dim, self.latent_dim)

## Decoder
class JointDecoderMLP(nn.Module):
    def __init__(self, num_points, hidden_dim, latent_dim, spatial_dim=2, num_cell_types=None, num_gene=None, nhead=4, num_layers=2, Exist_use=True):
        """
        Args:
            num_points (int): Maximum number of points to generate (N_max)。
            num_cell_types (int): Total number of cell types。
            ...
        """
        super(JointDecoderMLP, self).__init__()
        self.num_points = num_points
        self.model_dim = hidden_dim

        # 1. Learnable query vectors (canvas)
        self.queries = nn.Parameter(torch.randn(num_points, hidden_dim))

        # 2. Project latent variable z to Transformer working dimension
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # 3. Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.0
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- 4. Upgrade output heads from nn.Linear to MLP ---
        mlp_hidden_dim = hidden_dim // 2
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, spatial_dim)
        )
        self.num_cell_types = num_cell_types
        if num_cell_types is not None:
            self.type_head = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, num_cell_types),
                # nn.Sigmoid()
            )
        self.num_gene = num_gene
        if num_gene is not None:
            self.gene_head = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, num_gene)
            )
        
        # New: Existence prediction head
        self.Exist_use = Exist_use
        if Exist_use:
            self.existence_head = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, 1)
            )

    def forward(self, z_trajectory):
        batch_size = z_trajectory.shape[0]
        memory = self.latent_proj(z_trajectory).unsqueeze(1)
        tgt = self.queries.unsqueeze(0).repeat(batch_size, 1, 1)
        
        refined_features = self.transformer_decoder(tgt=tgt, memory=memory)
        
        pos = self.shape_head(refined_features)
        if self.num_cell_types is not None:
            type_logits = self.type_head(refined_features)
        else:
            type_logits = None
            
        if self.num_gene is not None:
            gene = self.gene_head(refined_features)
        else:
            gene = None
        # Get existence logits and reshape to [Batch, Num_Points]
        if self.Exist_use:
            existence_logits = self.existence_head(refined_features).squeeze(-1)
        else:
            existence_logits = None
        # Return three tensors
        return pos, type_logits, gene, existence_logits
    
# --- Improvement 3: Transformer Encoder replacing GRU ---
# --- MODIFIED: PositionalEncoding ---
## This is actually a time encoder
class timeEncoding(nn.Module):
    """
    (Corrected) Standard time series positional encoder.
    It adds temporal information to the Transformer input sequence (representing different time points).
    This implementation is independent of spatial dimension (2D or 3D).
    """
    def __init__(self, d_model, max_len=100): # Removed confusing spatial_dim parameter
        super(timeEncoding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model ({d_model}) must be even to allocate sin/cos pairs.")
            
        position = torch.arange(max_len).unsqueeze(1)
        # Use standard div_term calculation method
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # pe shape: (max_len, 1, d_model)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence, shape: (seq_len, batch_size, d_model)
        """
        # x shape: (seq_len, batch, embedding_dim)
        # Point cloud features + time features
        return x + self.pe[:x.size(0)]

class TransformerLatentODE(nn.Module):
    """(Modified) Main model using a Transformer to encode the sequence."""
    
    def __init__(self, num_points, spatial_dim, latent_dim, hidden_dim, transformer_ff_dim, 
                 num_cell_types=None, num_gene=None, Exist_use=True, max_seq_len=100, 
                 encoder_model='Swin'):
        '''
        encoder_model in ['Trans', 'Swin']
        '''
        super(TransformerLatentODE, self).__init__()
        self.latent_dim = latent_dim
        input_dim = spatial_dim
        if not num_cell_types is None:
            input_dim += num_cell_types
        if not num_gene is None:
            input_dim += num_gene
            
        # --- ✅ Module 1: Replace Encoder instantiation ---
        # Replace SwinHierarchicalPointCloudEncoder with PointCloudEncoder
        # and remove the 'spatial_dim' parameter specific to Swin...Encoder
        print(f"Initializing simple PointCloudEncoder (InputDim: {input_dim}, HiddenDim: {hidden_dim}, LatentDim: {latent_dim})")
        if encoder_model == 'Trans':
            print('Starting Point Transformer encoder.')
            self.point_encoder = PointCloudEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim
                # Note: PointCloudEncoder uses nhead=4, num_layers=2 by default
                # If needed, you can explicitly pass them here, e.g., nhead=..., num_layers=...
            )
        elif encoder_model == 'Swin':
            print('Starting Point Swin Transformer encoder.')
            self.point_encoder = SwinHierarchicalPointCloudEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                spatial_dim=spatial_dim # <--- Key addition
            )
        else:
            raise ValueError("encoder_model in ['Trans', 'Swin']")

        # --- End of modification ---time
        
        self.time_encoder = timeEncoding(latent_dim, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=4, dim_feedforward=transformer_ff_dim, batch_first=False, dropout=0.0)
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.latent_to_z0 = nn.Linear(latent_dim, 2 * latent_dim)
        
        # (Keep your regularization modifications to LatentVelocityNet)
        self.latent_ode_func = LatentVelocityNet(latent_dim, hidden_dim)
        
        self.decoder = JointDecoderMLP(num_points, hidden_dim, latent_dim,spatial_dim=spatial_dim, num_cell_types=num_cell_types, num_gene=num_gene, Exist_use=Exist_use)
        self.num_cell_types=num_cell_types
        self.num_gene = num_gene
        
    def forward(self, data_with_features_list, time_steps):
        
        # ... (Code from fixed_features_tensor to z0 remains unchanged) ...
        fixed_features_tensor = torch.stack([p for p in data_with_features_list])
        latent_z_seq = self.point_encoder(fixed_features_tensor) # <--- Replace with this line
        latent_z_seq = latent_z_seq.unsqueeze(1)
        
        latent_z_seq_pos = self.time_encoder(latent_z_seq)
        transformer_out = self.sequence_encoder(latent_z_seq_pos)
        
        context_vector = transformer_out.mean(dim=0)
        
        z0_params = self.latent_to_z0(context_vector)
        z0_mu, z0_log_var = z0_params[:, :self.latent_dim], z0_params[:, self.latent_dim:]
        
        epsilon = torch.randn_like(z0_mu)
        z0 = torch.exp(0.5 * z0_log_var) * epsilon
        z0 = z0.squeeze(0) 
        latent_trajectory = odeint(
            self.latent_ode_func, 
            z0,
            time_steps, 
            rtol=RTOL, 
            atol=ATOL
        )
        recons_pos, recons_type_logits, recons_gene, recons_existence_logits = self.decoder(latent_trajectory)
        # (✅ Fix: Return latent_trajectory itself, not kinetic_reg_loss)
        return recons_pos, recons_type_logits, recons_gene, recons_existence_logits, z0_mu, z0_log_var, latent_trajectory


import torch.nn.functional as F

import torch
import torch.nn.functional as F

import torch
import torch

# --- Helper functions (extracted from your code and kept unchanged) ---
def clamp_P(P, k_beta=0.1, is_1d=False):
    """
    A helper function to truncate probability matrix P to enhance sparsity.
    """
    if is_1d:
        top_k_indices = torch.topk(P, k=int(P.shape[0] * k_beta), largest=True).indices
    else:
        top_k_indices = torch.topk(P, k=int(P.shape[1] * k_beta), largest=True, dim=1).indices

    mask = torch.zeros_like(P, dtype=torch.bool)
    if is_1d:
        mask[top_k_indices] = True
    else:
        mask.scatter_(1, top_k_indices, True)

    P_clamped = torch.where(mask, P, torch.tensor(0.0, device=P.device))
    return P_clamped

# --- Core wrapper function ---
def iter_calculate_correspondence_P(
    pos_A,
    pos_B,
    feat_A=None,
    feat_B=None,
    alpha=0.5,
    max_iter=20,
    tol=1e-5,
    device='cpu'
    ):
    """
    (Wrapped ICP+Feature version)
    Computes the correspondence probability matrix P between two point clouds based on the user-provided iterative alignment algorithm.
    This function considers both spatial positions and molecular features.

    Args:
        pos_A (torch.Tensor): Spatial coordinates of the first point cloud [N_A, D]
        pos_B (torch.Tensor): Spatial coordinates of the second point cloud [N_B, D]
        feat_A (torch.Tensor, optional): Molecular features of the first point cloud [N_A, F_A]
        feat_B (torch.Tensor, optional): Molecular features of the second point cloud [N_B, F_B]
        alpha (float): Weight for fusing spatial and feature similarity.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence threshold.

    Returns:
        torch.Tensor: Final correspondence probability matrix P, shape [N_A, N_B].
    """
    with torch.no_grad():
        D = pos_A.shape[1]
        
        # --- 1. Compute fixed molecular feature similarity (exp_P) ---
        has_features = feat_A is not None and feat_B is not None
        if has_features:
            dist_feat = torch.cdist(feat_A, feat_B, p=2.0)
            sigma_feat = torch.median(dist_feat) / 2.0
            exp_P = torch.exp(-dist_feat ** 2 / (2 * (sigma_feat ** 2) + 1e-8))
        else:
            # If no features, alpha auto-adjusts to 0, only consider spatial
            alpha = 0.0
            exp_P = torch.ones(pos_A.shape[0], pos_B.shape[0], device=device) # Placeholder with all-ones matrix

        # --- 2. Iterative alignment ---
        R = torch.eye(D, dtype=pos_A.dtype, device=device)
        t = torch.zeros(D, dtype=pos_A.dtype, device=device)

        # Dynamically adjust alpha
        current_alpha = alpha

        for _ in range(max_iter):
            prev_R, prev_t = R.clone(), t.clone()

            # a. Transform A
            pos_A_transformed = pos_A @ R.T + t

            # b. Compute spatial similarity (spatial_P)
            dist_space = torch.cdist(pos_A_transformed, pos_B, p=2.0)
            sigma_space = torch.median(dist_space) / 2.0
            spatial_P = torch.exp(-dist_space ** 2 / (2 * (sigma_space ** 2) + 1e-8))

            # c. Fuse to get combined probability P
            P = (1 - current_alpha) * spatial_P + current_alpha * exp_P

            # d. SVD to solve for new R and t
            total_weight = P.sum()
            weights_B = P.sum(dim=0)
            mu_XB = (pos_B.T @ weights_B) / total_weight
            weights_A = P.sum(dim=1)
            mu_XA = (pos_A.T @ weights_A) / total_weight

            A_svd = (pos_B - mu_XB).T @ (P.T @ (pos_A - mu_XA))
            U, _, Vh = torch.linalg.svd(A_svd)
            V = Vh.T
            C = torch.eye(D, dtype=U.dtype, device=device)
            if torch.det(U @ V.T) < 0:
                C[-1, -1] = -1.0
            R = U @ C @ V.T
            t = mu_XB - R @ mu_XA
            # e. Decay alpha
            current_alpha *= 0.9
            # f. Check convergence
            if (torch.norm(R - prev_R) < tol) and (torch.norm(t - prev_t) < tol):
                break
        # --- 3. Use final R and t to compute final joint probability ---
        pos_A_final = pos_A @ R.T + t
        dist_space_final = torch.cdist(pos_A_final, pos_B, p=2.0)
        sigma_space_final = torch.median(dist_space_final) / 2.0
        spatial_P_final = torch.exp(-dist_space_final ** 2 / (2 * (sigma_space_final ** 2) + 1e-8))    
        final_P = (1 - current_alpha) * spatial_P_final + current_alpha * exp_P
        return final_P
    
    
# --- Core wrapper function (non-iterative version) ---
def calculate_correspondence_P(
    pos_A,
    pos_B,
    feat_A=None,
    feat_B=None,
    alpha=0.5,
    device='cpu'
    ):
    """
    (Non-iterative version)
    Computes the correspondence probability matrix P between two point clouds
    through one-step weighted fusion of spatial similarity (spatial_P) and molecular feature similarity (exp_P).

    Args:
        pos_A (torch.Tensor): Spatial coordinates of the first point cloud [N_A, D]
        pos_B (torch.Tensor): Spatial coordinates of the second point cloud [N_B, D]
        feat_A (torch.Tensor, optional): Molecular features of the first point cloud [N_A, F_A]
        feat_B (torch.Tensor, optional): Molecular features of the second point cloud [N_B, F_B]
        alpha (float): Weight for fusing spatial and feature similarity.

    Returns:
        torch.Tensor: Final correspondence probability matrix P, shape [N_A, N_B].
    """
    with torch.no_grad():
        # --- 1. Compute spatial similarity (spatial_P) ---
        dist_space = torch.cdist(pos_A, pos_B, p=2.0)
        # Use median to dynamically determine a robust sigma, avoiding influence from outliers
        sigma_space = torch.median(dist_space) / 2.0
        spatial_P = torch.exp(-dist_space.pow(2) / (2 * sigma_space.pow(2) + 1e-8))

        # --- 2. Compute molecular feature similarity (exp_P) ---
        has_features = feat_A is not None and feat_B is not None
        if has_features:
            dist_feat = torch.cdist(feat_A, feat_B, p=2.0)
            sigma_feat = torch.median(dist_feat) / 2.0
            exp_P = torch.exp(-dist_feat.pow(2) / (2 * sigma_feat.pow(2) + 1e-8))
            effective_alpha = alpha
        else:
            # If no features, alpha is ineffective, only use spatial similarity
            exp_P = 0 # Set to 0, so it disappears during weighting
            effective_alpha = 0.0

        # --- 3. Weighted fusion according to formula ---
        P = (1 - effective_alpha) * spatial_P.to(device) + effective_alpha * exp_P.to(device)
        
        return P


import torch
import numpy as np
import pandas as pd
import torch
import numpy as np
import pandas as pd
from scipy import sparse
# (Ensure geneSelection function is defined in your script)
# (Ensure Set is imported, e.g., from typing import Set)
import torch
import torch.nn.functional as F
from scipy import sparse # Still needed for initial check in prepare_real_data

# (Your other imports remain unchanged)
# Not used
def geneSelection_torch(data, threshold=0, atleast=10,
                        yoffset=.02, xoffset=5, decay=1.5, n=None,
                        verbose=1):
    """
    (GPU accelerated version)
    Gene selection by mean-variance relationship, implemented in PyTorch.
    Assumes data is a DENSE torch.Tensor on the target device.
    """
    
    # Ensure data is torch.float32 for nanmean and other operations
    data = data.float()
    
    # 1. Calculate zeroRate and num_expressing_cells
    # (In PyTorch, (data > threshold) is treated as 1.0 and 0.0)
    num_expressing_cells = (data > threshold).sum(dim=0)
    zeroRate = 1.0 - num_expressing_cells / data.shape[0]

    # 2. Calculate meanExpr
    mask = data > threshold
    
    # Create a tensor with same shape as data, filled with nan
    logs = torch.full_like(data, float('nan'))
    
    # Only calculate log2 where mask is True
    # torch.log2(data[mask]) returns a 1D tensor, we need to put it back into logs
    logs[mask] = torch.log2(data[mask])
    
    # nanmean calculated along dim=0 (cells)
    meanExpr = torch.nanmean(logs, dim=0)

    # 3. Filter lowDetection
    lowDetection = num_expressing_cells < atleast
    zeroRate[lowDetection] = float('nan')
    meanExpr[lowDetection] = float('nan')
            
    # 4. Find n
    if n is not None:
        up = 10.0
        low = 0.0
        for t in range(100): # Binary search
            nonan = ~torch.isnan(zeroRate)
            selected = torch.zeros_like(zeroRate, dtype=torch.bool, device=data.device)
            
            # Core selection logic (in PyTorch)
            selected[nonan] = zeroRate[nonan] > torch.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            
            num_selected = torch.sum(selected)
            if num_selected == n:
                break
            elif num_selected < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        if verbose > 0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~torch.isnan(zeroRate)
        selected = torch.zeros_like(zeroRate, dtype=torch.bool, device=data.device)
        selected[nonan] = zeroRate[nonan] > torch.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
    
    # 5. Return boolean mask (on GPU)
    # (We removed plotting logic in this version because it requires .cpu() and slows down)
    return selected

# (Ensure Set is imported, e.g., from typing import Set)


def prepare_real_data(adata_list, time_list, device='cpu'): # Add device parameter
    """
    (Modified - follows user HVG intersection scheme and runs on GPU)
    Convert list of AnnData objects to input data structure required by model.

    Args:
        adata_list (list): List of AnnData objects arranged in chronological order.
        time_list (list): List containing numerical values for each time point.
        device (torch.device): Device for computation (e.g. 'cuda:0')

    Returns:
        tuple: ..as before)
    """
    print("Preparing real data for model (GPU accelerated version)...")

    if not adata_list:
        raise ValueError("adata_list cannot be empty.")

    # --- 1. Data extraction & move to GPU ---
    print(f"Step 1: Extract all count matrices and move to {device}...")
    num_genes = adata_list[0].n_vars
    
    all_counts_gpu = []
    for i, adata in enumerate(adata_list):
        data_t = adata.layers['matrix']
        # Check if sparse, if so, convert to dense array
        if sparse.issparse(data_t):
            data_t_dense = data_t.toarray()
        else:
            data_t_dense = data_t
        # Convert to PyTorch tensor and move to GPU
        all_counts_gpu.append(torch.tensor(data_t_dense, device=device))

    print(f"  - Converted {len(all_counts_gpu)} matrices to dense tensors on GPU.")

    # --- 2.ction (executed on GPU) ---
    
    N_HVG_PER_SLICE = 2000  # 1. First find 1000 for each time point
    N_GENES_TO_SELECT = 10  # 4. Final target is 10
    
    print(f"Step 2.1: [GPU] Finding Top {N_HVG_PER_SLICE} HVGs...")
    
    all_hvg_masks_gpu = []
    for i, data_t_gpu in enumerate(all_counts_gpu):
        print(f"  - Processing time point {i+1}/{len(all_counts_gpu)}...")
        # Call geneSelection_torch
        mask_t_gpu = geneSelection_torch(data=data_t_gpu, n=N_HVG_PER_SLICE, verbose=0) 
        all_hvg_masks_gpu.append(mask_t_gpu)
        print(f"    ... Found {torch.sum(mask_t_gpu).item()} HVGs.")

    print(f"Step 2.2: [GPU] Calculating {len(all_hvg_masks_gpu)} HVG set intersections...")
    
    # Use torch.logical_and.reduce (PyTorch 1.7+)
    # If your PyTorch version is older, you may need to manually loop
    try:
        intersection_mask_gpu = torch.logical_and.reduce(*all_hvg_masks_gpu)
    except AttributeError:
        # Fallback for older PyTorch versions
        print("  - (Using fallback logical_and loop...)")
        intersection_mask_gpu = all_hvg_masks_gpu[0]
        for i in range(1, len(all_hvg_masks_gpu)):
            intersection_mask_gpu = torch.logical_and(intersection_mask_gpu, all_hvg_masks_gpu[i])
            
    num_intersection_genes = torch.sum(intersection_mask_gpu).item()
    print(f"  - Intersection contains {num_intersection_genes} genes.")

    if num_intersection_genes == 0:
        raise ValueError("HVG gene intersection is empty! Try smaller N_HVG_PER_SLICE (e.g., 2000) or check your data.")

    # --- Check if intersection is sufficient ---
    if num_intersection_genes < N_GENES_TO_SELECT:
        print(f"Warning: Number of genes in intersection ({num_intersection_genes}) is less than final target ({N_GENES_TO_SELECT})。")
        print(f"Steps 2.3 & 2.4: Will use all {num_intersection_genes} intersection genes.")
        selected_gene_mask_gpu = intersection_mask_gpu # Final mask is this intersection

    else:
        print(f"Step 2.3: [GPU] From {num_intersection_genes} intersection genes, merging and finding Top {N_GENES_TO_SELECT} genes...")
        
        # Create aggregated matrix (on GPU)
        combined_matrix_gpu = torch.cat(all_counts_gpu, dim=0)
        print(f"  - Created aggregated matrix on GPU, shape: {combined_matrix_gpu.shape}")
        
        # Filter aggregated matrix (on GPU)
        combined_matrix_filtered_gpu = combined_matrix_gpu[:, intersection_mask_gpu]
        print(f"  - Created filtered aggregated matrix on GPU, shape: {combined_matrix_filtered_gpu.shape}")

        # Run geneSelection_torch on this *filtered* GPU matrix
        local_mask_gpu = geneSelection_torch(data=combined_matrix_filtered_gpu, n=N_GENES_TO_SELECT, verbose=1)
        
        print(f"Step 2.4: [GPU] Mapping {N_GENES_TO_SELECT} locally selected genes back to global indices...")
        
        # Create a final mask of all False
        selected_gene_mask_gpu = torch.zeros(num_genes, dtype=torch.bool, device=device)
        
        # Put local_mask values back to positions where intersection_mask is True
        selected_gene_mask_gpu[intersection_mask_gpu] = local_mask_gpu

    # --- Gene selection complete ---
    
    num_final_genes = torch.sum(selected_gene_mask_gpu).item()
    print(f"Gene selection complete. Finally selected {num_final_genes} genes.")
    
    # Step 2.5: Move final mask back to CPU (NumPy) for AnnData slicing
    selected_gene_mask = selected_gene_mask_gpu.cpu().numpy()


    # --- 3. Process cell types (unchanged) ---
    all_cell_types = set()
    for adata in adata_list:
        all_cell_types.update(adata.obs['cell_type'].unique())
    
    sorted_cell_types = sorted(list(all_cell_types))
    cell_type_to_int = {cell_type: i for i, cell_type in enumerate(sorted_cell_types)}
    int_to_cell_type = {i: cell_type for i, cell_type in enumerate(sorted_cell_types)}
    num_unique_cell_types = len(sorted_cell_types)
    print(f"Found a total of {num_unique_cell_types} unique cell types across all datasets.")

    # --- 4. Process each AnnData object (loop body unchanged) ---
    data_list = []
    type_list = []
    gene_list = []
    for i, adata in enumerate(adata_list):
        # Extract spatial coordinates
        coords = adata.obsm['spatial']
        data_list.append(torch.tensor(coords, dtype=torch.float32))
        
        # Extract and map cell types
        types_str = adata.obs['cell_type']
        types_int = types_str.map(cell_type_to_int).values
        type_list.append(torch.tensor(types_int, dtype=torch.long))
        
        # --- Modified part: Only extract finally selected genes ---
        # This line now uses our newly computed selected_gene_mask from GPU (NumPy)
        top_counts = adata.layers['matrix'][:, selected_gene_mask]
        
        if sparse.issparse(top_counts):
            top_counts_dense = top_counts.toarray()
        else:
            top_counts_dense = top_counts
            
        gene_list.append(torch.tensor(top_counts_dense, dtype=torch.float32))
        
        print(f"Processed time point {time_list[i]}: {coords.shape[0]} cells, extracted Top {num_final_genes} gene data.")

    # --- 5. Spatial coordinate normalization (unchanged) ---
    scaled_data_list = []
    target_min = -10.0
    target_max = 10.0
    for i, d in enumerate(data_list):
        if d.numel() > 0:
            local_min = d.min()
            local_max = d.max()
            d_normalized = (d - local_min) / (local_max - local_min + 1e-8)
            d_scaled = d_normalized * (target_max - target_min) + target_min
            scaled_data_list.append(d_scaled)
        else:
            scaled_data_list.append(d)
    
    data_list = scaled_data_list
    
    # --- 6. Process time steps (unchanged) ---
    time_steps = torch.tensor(time_list, dtype=torch.float32)
    
    print("Real data preparation complete.")
    
    return data_list, type_list, gene_list, time_steps, cell_type_to_int, int_to_cell_type
import torch
import numpy as np
import os # (Just for code completeness, this function is not directly used)

def load_processed_data(path_or_data):
    """
    (Modified)
    Load preprocessed data dictionary from .pt file, or directly use a loaded dictionary.

    Args:
        path_or_data (str or dict): 
            - (str): Path to .pt file to load.
            - (dict): An already loaded, format-compliant data dictionary.

    Returns:
        tuple: Contains the following items:
            (data_list, type_list, gene_list_raw, gene_list_processed, 
             time_steps, cell_type_to_int, int_to_cell_type)
            
            If loading/input fails, all items will be None.
            If a key is missing from file, that item will be None.
    """
    print(f"Loading and unpacking data...")
    
    loaded_data = None
    
    # --- 1. (New) Check input type ---
    if isinstance(path_or_data, str):
        # --- A. Input is file path (str) ---
        file_path = path_or_data
        print(f"  - Input is a path. Trying to load from {file_path} loading...")
        try:
            loaded_data = torch.load(file_path, map_location='cpu')
            print("  - Data loaded from file.")
        except FileNotFoundError:
            print(f"  - Error: File not found '{file_path}'。")
            return None, None, None, None, None, None, None # Return Nones
        except Exception as e:
            print(f"  - Error loading file: {e}")
            return None, None, None, None, None, None, None # Return Nones
            
    elif isinstance(path_or_data, dict):
        # --- B. Input is dictionary (dict) ---
        print("  - Input is a dictionary. Using data directly...")
        loaded_data = path_or_data
        
    else:
        # --- C. Input type error ---
        print(f"  - Error: Input must be file path (str) or data dictionary (dict), but received {type(path_or_data)}。")
        return None, None, None, None, None, None, None # Return Nones

    # --- 2. Safely recover your variables from dictionary (unchanged) ---
    print("  - Unpacking dictionary keys...")
    data_list = loaded_data.get('data_list', None)
    type_list = loaded_data.get('type_list', None)
    gene_list = loaded_data.get('gene_list_raw', None)
    gene_list_processed = loaded_data.get('gene_list_processed', None)
    time_steps = loaded_data.get('time_steps', None)
    cell_type_to_int = loaded_data.get('cell_type_to_int', None)
    int_to_cell_type = loaded_data.get('int_to_cell_type', None)

    # --- 3. Validate data (unchanged) ---
    print("\n--- Data Validation ---")
    if time_steps is not None:
        print(f"  - Number of loaded time points: {len(time_steps)}")
    if data_list is not None and len(data_list) > 0:
        print(f"  - Coordinate shape at first time point: {data_list[0].shape}")
    if gene_list is not None and len(gene_list) > 0:
        print(f"  - Raw gene data shape at first time point: {gene_list[0].shape}")
    if gene_list_processed is not None and len(gene_list_processed) > 0:
        print(f"  - Processed gene data shape at first time point: {gene_list_processed[0].shape}")
    if int_to_cell_type is not None:
         print(f"  - Cell type mapping (example): {list(int_to_cell_type.items())[:3]}")

    # --- 4. Return unpacked tuple (unchanged) ---
    return (data_list, type_list, gene_list, gene_list_processed, 
            time_steps, cell_type_to_int, int_to_cell_type)
    
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

def preprocess_gene_data(gene_data_list):
    """
    Perform log1p transformation and Z-score normalization on gene data list.
    This is the standard preprocessing pipeline before inputting gene data to neural network.

    Args:
        gene_data_list (list): List containing raw gene count tensors.

    Returns:
        list: List of preprocessed (log1p + normalized) gene data tensors.
    """
    if not gene_data_list or not any(g.numel() > 0 for g in gene_data_list):
        print("Gene data list is empty or contains empty tensors, skipping preprocessing.")
        return []

    print("Preprocessing gene data (log1p + scaling)...")
    
    # --- 1. Apply log1p to data at each time point ---
    # First convert PyTorch tensors to Numpy arrays for computation
    log_transformed_data = [np.log1p(g.cpu().numpy()) for g in gene_data_list]

    # --- 2. Fit global Scaler ---
    # Merge data from all time points to calculate global mean and std, ensuring transformation consistency
    try:
        # Only merge non-empty arrays
        combined_data = np.vstack([g for g in log_transformed_data if g.shape[0] > 0])
        if combined_data.shape[0] == 0:
            print("Warning: No data points found to fit scaler. Returning log-transformed data only.")
            return [torch.tensor(g, dtype=torch.float32) for g in log_transformed_data]

        scaler = StandardScaler()
        scaler.fit(combined_data) # Fit on all data

        # --- 3. Transform data ---
        # Use same scaler to transform data at each time point
        scaled_data_list = [
            torch.tensor(scaler.transform(g), dtype=torch.float32) if g.shape[0] > 0 else torch.empty(0, g.shape[1], dtype=torch.float32)
            for g in log_transformed_data
        ]
        
        print("Gene data preprocessing complete.")
        return scaled_data_list

    except ValueError as e:
        # If all time points have no cells, vstack will fail, handle here
        print(f"Error during gene preprocessing: {e}. Returning log-transformed data only.")
        return [torch.tensor(g, dtype=torch.float32) for g in log_transformed_data]

def preprocess_data_list(data_list):
    """
    Perform Z-score normalization on spatial coordinate data in data_list.
    Normalize using global mean and std from all time point data.

    Args:
        data_list (list): List containing spatial coordinate torch.Tensors.
                          Each Tensor shape should be [N_points, D_spatial]。

    Returns:
        list: List of Z-score normalized spatial coordinate data tensors.
    """
    if not data_list or not any(d.numel() > 0 for d in data_list):
        print("Data list is empty or contains empty tensors, skipping normalization.")
        return data_list

    print("Preprocessing spatial data (Z-score scaling)...")

    # --- 1. Merge all data to calculate global mean and std ---
    try:
        # Only merge non-empty tensors
        non_empty_tensors = [d for d in data_list if d.shape[0] > 0]
        if not non_empty_tensors:
            print("Warning: No data points found to calculate mean/std. Returning original data.")
            return data_list
            
        combined_data = torch.cat(non_empty_tensors, dim=0)

        # --- 2. Calculate global mean and std ---
        # Calculate mean and std on each spatial dimension (dim=1)
        # mean and std shapes will be [D_spatial]
        mean = torch.mean(combined_data, dim=0)
        std = torch.std(combined_data, dim=0)
        
        # Add small epsilon to prevent division by zero
        std = std + 1e-8 
        
        print(f"  Calculated global mean: {mean.cpu().numpy()}")
        print(f"  Calculated global std:  {std.cpu().numpy()}")

        # --- 3. Apply normalization ---
        # Use calculated global mean and std to normalize data at each time point
        scaled_data_list = []
        for d in data_list:
            if d.shape[0] > 0:
                # Ensure mean and std are on same device as d
                mean_d = mean.to(d.device)
                std_d = std.to(d.device)
                # Apply Z-score: (x - mean) / std
                scaled_d = (d - mean_d) / std_d
                scaled_data_list.append(scaled_d)
            else:
                # Keep empty tensors
                scaled_data_list.append(d)
        
        print("Spatial data Z-score scaling complete.")
        return scaled_data_list

    except Exception as e:
        # Catch potential errors
        print(f"Error during spatial data preprocessing: {e}. Returning original data.")
        return data_list
    
# ==============================================================================
# ✅ NEW: MODULAR VISUALIZATION FUNCTION
# ==============================================================================
def visualize_trajectory(outputs, t_eval_smooth, data_list, int_to_cell_type, output_gif_path, exist_threshold=0.4):
    """
    Creates and saves a GIF of cell type evolution, with corrected colormap and tensor handling.
    (Revised: Fixed Key Error: 0, ensuring legend uses actually existing cell type IDs)
    """
    print("Stitching frames for cell types into GIF...")
    smooth_pos, smooth_logits, _, smooth_exist, *_ = outputs
    
    if int_to_cell_type is None:
        print("Warning: int_to_cell_type is missing. Skipping cell type visualization.")
        return
        
    num_cell_types = len(int_to_cell_type)
    
    gif_dir = "gif_frames_types"
    if os.path.exists(gif_dir): shutil.rmtree(gif_dir)
    os.makedirs(gif_dir)

    all_points = np.concatenate([p.cpu().numpy() for p in data_list if p.shape[0] > 0], axis=0)
    x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
    y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1
    
    # ✅ FIX 1: Use the modern, correct Matplotlib API to get the colormap.
    cmap = plt.colormaps['viridis']

    # --- Legend/colormap fix ---
    # Ensure colormap range is from 0 to num_cell_types - 1, as required by scatter plot
    cell_type_ids = sorted(int_to_cell_type.keys())
    num_unique_types = len(cell_type_ids)

    # Manually get colors for legend, ensuring colors match type_t indices [0, 1, 2...]
    legend_colors = cmap(np.linspace(0, 1, num_unique_types))
    
    filenames = []
    for i in range(len(t_eval_smooth)):
        if smooth_exist is not None:
            keep_mask = torch.sigmoid(smooth_exist[i]) > exist_threshold
        else:
            keep_mask = torch.ones(smooth_pos[i].shape[0], dtype=torch.bool)

        # ✅ FIX 2: Use detach().cpu()
        p_t = smooth_pos[i][keep_mask].detach().cpu()
        
        if p_t.shape[0] == 0: continue

        fig, ax = plt.subplots(figsize=(10, 8))
        
        if smooth_logits is not None:
            # ✅ FIX 2: Use detach().argmax().cpu()
            type_t = smooth_logits[i][keep_mask].detach().argmax(dim=-1).cpu()
            
            # The scatter function will now work correctly.
            scatter = ax.scatter(p_t[:, 0], p_t[:, 1], s=10, alpha=0.9, c=type_t, cmap=cmap, vmin=0, vmax=num_cell_types - 1)
            
            # Iterate over actual ID keys and use enumerate index to match colors
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=int_to_cell_type[j], 
                                          markerfacecolor=legend_colors[idx], markersize=10) 
                               for idx, j in enumerate(cell_type_ids)]
            ax.legend(handles=legend_elements, loc='upper right')
        else:
            scatter = ax.scatter(p_t[:, 0], p_t[:, 1], s=10, alpha=0.9)

        num_predicted_points = p_t.shape[0]
        ax.set_title(f"Predicted Evolution | Time = {t_eval_smooth[i].item():.2f} | Cell Count = {num_predicted_points}", fontsize=14)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', 'box'); ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        filename = f"{gif_dir}/frame_{i:03d}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close(fig)

    with imageio.get_writer(output_gif_path, mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            writer.append_data(imageio.imread(filename))
    shutil.rmtree(gif_dir)
    print(f"Cell type animation successfully saved to {output_gif_path}")
    
    

def visualize_gene_trajectory(outputs, t_eval_smooth, data_list, gene_index, gene_name, output_gif_path, exist_threshold=0.4):
    """
    Creates and saves a GIF of gene expression, with corrected tensor handling.
    """
    print(f"Stitching frames for gene '{gene_name}' into GIF...")
    smooth_pos, _, recons_gene, smooth_exist, *_ = outputs
    
    if recons_gene is None:
        print(f"No gene predictions found. Skipping visualization for {gene_name}.")
        return

    gif_dir = f"gif_frames_gene_{gene_name}"
    if os.path.exists(gif_dir): shutil.rmtree(gif_dir)
    os.makedirs(gif_dir)

    all_points = np.concatenate([p.cpu().numpy() for p in data_list if p.shape[0] > 0], axis=0)
    x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
    y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1
    
    # ✅ FIX 2: Add .detach() here.
    all_gene_values = [g[:, gene_index].detach().cpu().numpy() for g in recons_gene]
    v_min = np.min([v.min() for v in all_gene_values if v.size > 0])
    v_max = np.max([v.max() for v in all_gene_values if v.size > 0])
    
    filenames = []
    for i in range(len(t_eval_smooth)):
        if smooth_exist is not None:
            keep_mask = torch.sigmoid(smooth_exist[i]) > exist_threshold
        else:
            keep_mask = torch.ones(smooth_pos[i].shape[0], dtype=torch.bool)

        # ✅ FIX 2: Add .detach() here.
        p_t = smooth_pos[i][keep_mask].detach().cpu()
        
        if p_t.shape[0] == 0: continue

        # ✅ FIX 2: Add .detach() here.
        gene_t = recons_gene[i][keep_mask, gene_index].detach().cpu()
        t = t_eval_smooth[i].item()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(p_t[:, 0], p_t[:, 1], s=10, alpha=0.9, c=gene_t, cmap=plt.colormaps['viridis'], vmin=v_min, vmax=v_max)
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(f"Log-Normalized Expression")
        
        num_predicted_points = p_t.shape[0]
        ax.set_title(f"Predicted Gene Expression | Gene: {gene_name}\nTime = {t:.2f} | Cell Count = {num_predicted_points}", fontsize=14)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', 'box'); ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        filename = f"{gif_dir}/frame_{i:03d}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close(fig)

    with imageio.get_writer(output_gif_path, mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            writer.append_data(imageio.imread(filename))
    shutil.rmtree(gif_dir)
    print(f"Gene animation successfully saved to {output_gif_path}")
    
from mpl_toolkits.mplot3d import Axes3D # Import 3D plotting tools

def visualize_3d_trajectory(outputs, t_eval_smooth, data_list, int_to_cell_type, output_gif_path, exist_threshold=0.4):
    """
    Creates and saves a 3D GIF animation of cell type evolution.
    (Revised: Fixed Key Error: 0, ensuring legend uses actually existing cell type IDs)
    """
    print("Stitching 3D frames for cell types into GIF...")
    smooth_pos, smooth_logits, _, smooth_exist, *_ = outputs
    
    if int_to_cell_type is None:
        print("Warning: int_to_cell_type is missing. Skipping cell type visualization.")
        return
        
    num_cell_types = len(int_to_cell_type)
    
    gif_dir = "gif_frames_types_3d"
    if os.path.exists(gif_dir): shutil.rmtree(gif_dir)
    os.makedirs(gif_dir)

    all_points = np.concatenate([p.detach().cpu().numpy() for p in data_list if p.shape[0] > 0], axis=0)
    x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
    y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1
    # Assume 3D, need z axis
    spatial_dim = all_points.shape[1]
    z_min, z_max = (all_points[:, 2].min() - 1, all_points[:, 2].max() + 1) if spatial_dim >= 3 else (-1, 1)

    cmap = plt.colormaps['viridis'] # Use colormaps instead of plt.cm.get_cmap

    # --- Legend/colormap fix ---
    cell_type_ids = sorted(int_to_cell_type.keys())
    num_unique_types = len(cell_type_ids)
    legend_colors = cmap(np.linspace(0, 1, num_unique_types))
    
    filenames = []
    for i in range(len(t_eval_smooth)):
        keep_mask = torch.sigmoid(smooth_exist[i]) > exist_threshold
        p_t = smooth_pos[i][keep_mask].cpu().detach().numpy()
        if p_t.shape[0] == 0: continue

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        type_t = smooth_logits[i][keep_mask].argmax(dim=-1).cpu().detach().numpy()
        scatter = ax.scatter(p_t[:, 0], p_t[:, 1], p_t[:, 2], s=10, alpha=0.9, c=type_t, cmap=cmap, vmin=0, vmax=num_cell_types - 1)
        
        # Iterate over actual ID keys
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=int_to_cell_type[j],
                                     markerfacecolor=legend_colors[idx], markersize=10)
                           for idx, j in enumerate(cell_type_ids)]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title(f"Predicted 3D Evolution | Time = {t_eval_smooth[i].item():.2f} | Cell Count = {p_t.shape[0]}", fontsize=14)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        # Set a fixed viewing angle to prevent GIF jittering
        ax.view_init(elev=20., azim=120)

        filename = f"{gif_dir}/frame_{i:03d}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close(fig)

    with imageio.get_writer(output_gif_path, mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            writer.append_data(imageio.imread(filename))
    shutil.rmtree(gif_dir)
    print(f"3D Cell type animation successfully saved to {output_gif_path}")


def visualize_3d_gene_trajectory(outputs, t_eval_smooth, data_list, gene_index, gene_name, output_gif_path, exist_threshold=0.4):
    """ Creates and saves a 3D GIF animation of single gene expression evolution. """
    print(f"Stitching 3D frames for gene '{gene_name}' into GIF...")
    smooth_pos, _, recons_gene, smooth_exist, *_ = outputs
    if recons_gene is None: return

    gif_dir = f"gif_frames_gene_3d_{gene_name}"
    if os.path.exists(gif_dir): shutil.rmtree(gif_dir)
    os.makedirs(gif_dir)

    all_points = np.concatenate([p.detach().cpu().numpy() for p in data_list if p.shape[0] > 0], axis=0)
    x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
    y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1
    z_min, z_max = all_points[:, 2].min() - 1, all_points[:, 2].max() + 1
    
    all_gene_values = [torch.relu(recons_gene[i][:, gene_index]).detach().cpu().numpy() for i in range(len(t_eval_smooth))]
    v_min = np.min([v.min() for v in all_gene_values if v.size > 0])
    v_max = np.max([v.max() for v in all_gene_values if v.size > 0])
    
    filenames = []
    for i in range(len(t_eval_smooth)):
        keep_mask = torch.sigmoid(smooth_exist[i]) > exist_threshold
        p_t = smooth_pos[i][keep_mask].detach().cpu()
        if p_t.shape[0] == 0: continue

        gene_t = recons_gene[i][keep_mask, gene_index].detach().cpu()
        t = t_eval_smooth[i].item()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(p_t[:, 0], p_t[:, 1], p_t[:, 2], s=10, alpha=0.9, c=gene_t, cmap='viridis', vmin=v_min, vmax=v_max)
        
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label(f"Log-Normalized Expression")
        
        ax.set_title(f"Predicted 3D Gene Expression | Gene: {gene_name}\nTime = {t:.2f} | Cell Count = {p_t.shape[0]}", fontsize=14)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=20., azim=120)

        filename = f"{gif_dir}/frame_{i:03d}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close(fig)

    with imageio.get_writer(output_gif_path, mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            writer.append_data(imageio.imread(filename))
    shutil.rmtree(gif_dir)
    print(f"3D Gene animation successfully saved to {output_gif_path}")

def evaluate_spatial_reconstruction(true_pos_list, recon_pos, true_num_points_list, device):
    """
    (✅ Fixed version: Added 'device' parameter)
    Evaluates spatial position reconstruction similarity using optimal transport (Sinkhorn distance).
    """
    ot_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="tensorized")
    total_ot_distance = 0.0
    num_timesteps = len(true_pos_list)
    
    for i in range(num_timesteps):
        true_n = true_num_points_list[i]
        if true_n == 0:
            continue
            
        true_points = true_pos_list[i][:true_n].to(device) # <-- Bug fix
        recon_points = recon_pos[i, :true_n]

        ot_distance = ot_loss_fn(true_points, recon_points).item()
        total_ot_distance += ot_distance

    return total_ot_distance / num_timesteps if num_timesteps > 0 else 0.0


def evaluate_cell_type_accuracy(true_pos_list, true_type_list, recon_pos, recon_type_logits, true_num_points_list, device):
    """
    (✅ Fixed version: Added 'device' parameter)
    Evaluates cell type classification accuracy.
    """
    total_accuracy = 0.0
    num_timesteps = len(true_pos_list)
    
    for i in range(num_timesteps):
        true_n = true_num_points_list[i]
        if true_n == 0:
            continue

        true_points = true_pos_list[i][:true_n].to(device) # <-- Bug fix
        true_types = true_type_list[i][:true_n].to(device) # <-- Bug fix
        
        recon_points = recon_pos[i, :true_n]
        recon_types = torch.argmax(recon_type_logits[i, :true_n], dim=-1)

        dist_matrix = torch.cdist(recon_points, true_points)
        nearest_true_indices = torch.argmin(dist_matrix, dim=1)
        aligned_true_types = true_types[nearest_true_indices]
        
        correct_matches = (recon_types == aligned_true_types).float().sum()
        accuracy = correct_matches / true_n
        total_accuracy += accuracy.item()
        
    return total_accuracy / num_timesteps if num_timesteps > 0 else 0.0


def evaluate_gene_scc(true_pos_list, true_gene_list, recon_pos, recon_gene, true_num_points_list, device):
    """
    (✅ Fixed version: Added 'device' parameter)
    Evaluates gene expression similarity using Spearman Correlation Coefficient (SCC).
    """
    if not true_gene_list or recon_gene is None:
        return 0.0 # Return a float
    
    recon_gene = torch.relu(recon_gene)
    num_timesteps = len(true_pos_list)
    
    if num_timesteps == 0 or len(true_gene_list) == 0:
        return 0.0
        
    num_genes = true_gene_list[0].shape[1]
    gene_scc_scores = [[] for _ in range(num_genes)]

    for i in range(num_timesteps):
        true_n = true_num_points_list[i]
        if true_n < 2:
            continue

        true_points = true_pos_list[i][:true_n].to(device) # <-- Bug fix
        true_genes = true_gene_list[i][:true_n].to(device) # <-- Bug fix
        
        recon_points = recon_pos[i, :true_n]
        recon_genes = recon_gene[i, :true_n]

        dist_matrix = torch.cdist(recon_points, true_points)
        nearest_true_indices = torch.argmin(dist_matrix, dim=1)
        aligned_true_genes = true_genes[nearest_true_indices]

        for g_idx in range(num_genes):
            recon_g_np = recon_genes[:, g_idx].detach().cpu().numpy()
            aligned_true_g_np = aligned_true_genes[:, g_idx].detach().cpu().numpy()
            
            scc, _ = spearmanr(recon_g_np, aligned_true_g_np)
            
            if not np.isnan(scc):
                gene_scc_scores[g_idx].append(scc)

    all_scores = [score for scores in gene_scc_scores for score in scores]
    overall_avg_scc = np.mean(all_scores) if all_scores else 0.0
    
    return overall_avg_scc


import ot # <--- Ensure this import is at the top of the file
import torch
import torch.nn.functional as F


import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Simulation data hyperparameters ---
NUM_POINTS_GENERATE = 1500 # This value now serves as the peak of the point count function
def generate_spatial_data_with_genes(num_time_steps=10):
    """
    (✅ Fixed version: Added int_to_cell_type definition and return)
    Generates a time series of 2D point clouds with the following features:
    1. Total cell count follows a deterministic function that increases then decreases.
    2. The entire point cloud translates along the positive x-axis over time.
    """
    print(f"Generating simulated data with deterministic count change and translation...")

    # --- 1. Basic geometry and time setup ---
    base_vertices = np.array([[0.0, 2.0], [-1.732, -1.0], [1.732, -1.0]])
    point_groups, type_groups, gene_groups = [], [], []
    time_steps_np = np.arange(1, num_time_steps + 1)

    # --- (New) Define global translation parameters ---
    x_shift_factor = 2.0  # Translation distance along x-axis per time step

    # --- (✅ New) Define cell type mapping ---
    # This must match the type assignment logic (0, 1, 2) in step 6 below
    int_to_cell_type = {0: 'Type_0', 1: 'Type_1', 2: 'Type_2'}

    for t in time_steps_np:
        # --- severalwhatshapeevolution (remains unchanged) ---
        triangle_scale = 1.0 + 1.5 * np.sin(np.pi * (t - 1) / (num_time_steps - 1))
        hole_radius = 0.1 + (0.9 * (t - 1) / (num_time_steps - 1))
        
        # --- 2. computegoalpoint count (remains unchanged) ---
        mid_point = (num_time_steps + 1) / 2.0
        peak_count = NUM_POINTS_GENERATE
        start_end_ratio = 0.4 
        a = (start_end_ratio * peak_count - peak_count) / (1 - mid_point)**2
        target_count = int(a * (t - mid_point)**2 + peak_count)
        
        # --- 3. generateandfiltercandidatepoint (remains unchanged) ---
        num_candidates = target_count * 5
        if num_candidates == 0: 
             final_points = np.empty((0, 2))
        else:
            r1, r2 = np.random.rand(num_candidates, 1), np.random.rand(num_candidates, 1)
            sqrt_r1 = np.sqrt(r1)
            candidate_points = ((1 - sqrt_r1) * base_vertices[0] + 
                                (sqrt_r1 * (1 - r2)) * base_vertices[1] + 
                                (sqrt_r1 * r2) * base_vertices[2])
            scaled_points = candidate_points * triangle_scale
            dist_from_center = np.linalg.norm(scaled_points, axis=1)
            accepted_mask = dist_from_center > hole_radius
            valid_points = scaled_points[accepted_mask]
            
            # --- 4. samplingtogoalquantity (remains unchanged) ---
            if len(valid_points) > 0:
                final_target_count = min(len(valid_points), target_count)
                indices = np.random.choice(len(valid_points), final_target_count, replace=False)
                final_points = valid_points[indices]
            else:
                final_points = valid_points

        # --- 5. applicationglobaltranslate (remains unchanged) ---
        current_x_shift = x_shift_factor * (t - 1)
        final_points_shifted = final_points.copy()
        if final_points_shifted.shape[0] > 0:
            final_points_shifted[:, 0] += current_x_shift

        # --- 6. allocationcell type (remains unchanged) ---
        v0, v1, _ = base_vertices * triangle_scale
        radius_type0 = 0.5 + 2.0 * (t - 1) / (num_time_steps - 1)
        radius_type1 = 2.5 - 2.0 * (t - 1) / (num_time_steps - 1)
        dist_to_v0 = np.linalg.norm(final_points - v0, axis=1)
        dist_to_v1 = np.linalg.norm(final_points - v1, axis=1)
        
        final_types = np.full(final_points.shape[0], 2) # defaulttype 2
        final_types[dist_to_v1 < radius_type1] = 1      # type 1
        final_types[dist_to_v0 < radius_type0] = 0      # type 0

        # --- 7. simulategeneexpression (useyouontimeneedclearpattern) ---
        num_final_points = final_points.shape[0]
        gene_expressions = np.zeros((num_final_points, 5), dtype=np.int32)
        
        if num_final_points > 0:
            x, y = final_points[:, 0], final_points[:, 1] 
            t_norm = (t - 1) / max(1, num_time_steps - 1)
            
            # gene 1: X-axis gradient
            x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
            rate_g1 = 10.0 * x_norm
            gene_expressions[:, 0] = np.random.poisson(rate_g1)

            # gene 2: Y-axis gradient
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
            rate_g2 = 10.0 * y_norm
            gene_expressions[:, 1] = np.random.poisson(rate_g2)

            # gene 3: pathtowardgradient
            dist_center = np.linalg.norm(final_points, axis=1)
            dist_norm = (dist_center - dist_center.min()) / (dist_center.max() - dist_center.min() + 1e-8)
            rate_g3 = 10.0 * dist_norm
            gene_expressions[:, 2] = np.random.poisson(rate_g3)

            # gene 4: cell type 0 (Type_0) marker
            rate_g4 = np.zeros(num_final_points)
            rate_g4[final_types == 0] = 12.0 
            rate_g4[final_types != 0] = 1.0  
            gene_expressions[:, 3] = np.random.poisson(rate_g4)
            
            # gene 5: cell type 1 (Type_1) dynamicmarker
            rate_g5 = np.zeros(num_final_points)
            rate_g5[final_types == 1] = 15.0 * t_norm 
            gene_expressions[:, 4] = np.random.poisson(np.maximum(0, rate_g5)) 
            
        # --- 8. addnoiseandstoragealldata (remains unchanged) ---
        noisy_points = final_points_shifted + np.random.normal(0, 0.05, final_points_shifted.shape)
        
        point_groups.append(torch.tensor(noisy_points, dtype=torch.float32))
        type_groups.append(torch.tensor(final_types, dtype=torch.long))
        gene_groups.append(torch.tensor(gene_expressions, dtype=torch.float32))

    print("Data generation complete.")
    print("Generated points per time step:", [p.shape[0] for p in point_groups])
    
    # --- (✅ correction) return 5 values ---
    return point_groups, type_groups, gene_groups, torch.tensor(time_steps_np, dtype=torch.float32), int_to_cell_type

    # --- Helper function for pre-computation ---
# def precompute_fgw_transport_plans(pos_list, type_list_one_hot=None, gene_list=None, alpha=0.999, device='cpu'):
#         """
#         Pre-computes the FGW transport plans between all adjacent time points.
#         This is a costly operation that should only be run ONCE before the training loop.
#         """
#         print("--- Pre-computing ground truth transport plans (FGW)... This may take a while. ---")
#         transport_plans = []
#         num_time_pairs = len(pos_list) - 1
#         def standardize_cloud(cloud):
#             if cloud.shape[0] == 0:
#                 return cloud
#             mean = torch.mean(cloud, dim=0)
#             std = torch.std(cloud, dim=0)
#             return (cloud - mean) / (std + 1e-8) # epsilonprevent division by zero

#         # --- step1: infunctionbeginningforallpointcloudperformonetimeindependentnormalization ---
#         # (avoidinloopinrepeatedcompute)
#         pos_list = [standardize_cloud(p) for p in pos_list]

#         with torch.no_grad():
#             for i in range(num_time_pairs):
#                 true_A_pos = pos_list[i].to(device)
#                 true_B_pos = pos_list[i+1].to(device)
                
#                 if true_A_pos.shape[0] < 2 or true_B_pos.shape[0] < 2:
#                     print(f"Skipping plan for time pair {i} -> {i+1} due to insufficient points.")
#                     transport_plans.append(None) # Add a placeholder for pairs with too few points
#                     continue

#                 # Prepare molecular features
#                 feature_parts_A, feature_parts_B = [], []
#                 if type_list_one_hot is not None and len(type_list_one_hot) > i+1:
#                     feature_parts_A.append(type_list_one_hot[i].to(device))
#                     feature_parts_B.append(type_list_one_hot[i+1].to(device))
#                 if gene_list is not None and len(gene_list) > i+1:
#                     feature_parts_A.append(gene_list[i].to(device))
#                     feature_parts_B.append(gene_list[i+1].to(device))

#                 has_molecular_features = len(feature_parts_A) > 0
#                 if has_molecular_features:
#                     features_A = torch.cat(feature_parts_A, dim=1)
#                     features_B = torch.cat(feature_parts_B, dim=1)
#                     effective_alpha = alpha
#                 else:
#                     features_A = torch.zeros(true_A_pos.shape[0], 1, device=device)
#                     features_B = torch.zeros(true_B_pos.shape[0], 1, device=device)
#                     effective_alpha = 1.0

#                 # Calculate cost matrices
#                 C_molecular = ot.dist(features_A, features_B)
#                 C1_spatial = ot.dist(true_A_pos, true_A_pos)
#                 C2_spatial = ot.dist(true_B_pos, true_B_pos)
                
#                 # Prepare marginal distributions
#                 p = ot.unif(true_A_pos.shape[0], type_as=true_A_pos)
#                 q = ot.unif(true_B_pos.shape[0], type_as=true_A_pos)

#                 # Solve FGW Problem
#                 P, _ = ot.gromov.fused_gromov_wasserstein(
#                     M=C_molecular, C1=C1_spatial, C2=C2_spatial,
#                     p=p, q=q, alpha=effective_alpha, log=True
#                 )
                
#                 transport_plans.append(P.to(device))
#                 print(f"Computed and stored transport plan for time pair {i} -> {i+1}")

#         print("--- FGW pre-computation complete. ---")
#         return transport_plans

# computefgwtransport
def precompute_fgw_transport_plans(pos_list, alpha=0.5, device='cpu'):
    """
    Pre-computes the Fused Gromov-Wasserstein transport plans.
    By using alpha < 1, this version balances structural matching (GW) with
    absolute spatial positioning (Wasserstein) to avoid symmetry issues.
    """
    print("\n--- Step 2: Pre-computing ground truth transport plans (Fused Gromov-Wasserstein)... ---")
    transport_plans = []
    num_time_pairs = len(pos_list) - 1
    def standardize_cloud(cloud):
        if cloud.shape[0] == 0:
            return cloud
        mean = torch.mean(cloud, dim=0)
        std = torch.std(cloud, dim=0)
        return (cloud - mean) / (std + 1e-8) # epsilonprevent division by zero

    pos_list = [standardize_cloud(p) for p in pos_list]
    
    with torch.no_grad():
        for i in range(num_time_pairs):
            print(f"  Calculating plan for time pair {i+1} -> {i+2}...")
            true_A_pos = pos_list[i].to(device)
            true_B_pos = pos_list[i+1].to(device)

            if true_A_pos.shape[0] < 2 or true_B_pos.shape[0] < 2:
                print(f"    - Skipped due to insufficient points.")
                transport_plans.append(None)
                continue

            # Convert point clouds to numpy for ot.dist
            true_A_pos_np = true_A_pos.cpu().numpy()
            true_B_pos_np = true_B_pos.cpu().numpy()

            # ✅ KEY CHANGE 1: Calculate the cost matrix M between the two clouds' absolute positions.
            # This matrix represents the "fuel cost" for the Wasserstein part of FGW.
            M_spatial_np = ot.dist(true_A_pos_np, true_B_pos_np)
            
            # The C1 and C2 matrices represent the internal geometry for the Gromov part.
            C1_internal_np = ot.dist(true_A_pos_np, true_A_pos_np)
            C2_internal_np = ot.dist(true_B_pos_np, true_B_pos_np)

            # Prepare marginal distributions as NumPy arrays
            p_np = ot.unif(true_A_pos.shape[0])
            q_np = ot.unif(true_B_pos.shape[0])

            # ✅ KEY CHANGE 2: Call fused_gromov_wasserstein with the spatial cost matrix M
            # and an alpha < 1 to balance structure and position.
            P_np, _ = ot.gromov.fused_gromov_wasserstein(
                M=M_spatial_np, C1=C1_internal_np, C2=C2_internal_np,
                p=p_np, q=q_np, alpha=alpha, log=True
            )
            
            # Convert the final plan back to a tensor
            P = torch.from_numpy(P_np).to(dtype=true_A_pos.dtype, device=device)
            transport_plans.append(P)
            print(f"    - Computed plan with alpha={alpha}. Done.")

    print("--- FGW pre-computation complete. ---")
    return transport_plans

import torch
# note：thisnewversionnotagainneed 'ot' (Python Optimal Transport) library
# itcompletelyallbased on PyTorch comecomputenearest neighbortransportplan
def precompute_nn_transport_plans(pos_list, device='cpu'):
    """
    Pre-computes the transport plans based on Nearest Neighbor (NN) matching.

    This is a simpler, deterministic alternative to (Fused) Gromov-Wasserstein.
    For each point in cloud A, it finds the single closest point in cloud B
    and creates a "hard" transport plan (a sparse matrix with one '1' per row).
    """
    print("\n--- Step 2: Pre-computing ground truth transport plans (Nearest Neighbor)... ---")
    transport_plans = []
    num_time_pairs = len(pos_list) - 1

    def standardize_cloud(cloud):
        if cloud.shape[0] == 0:
            return cloud
        mean = torch.mean(cloud, dim=0)
        std = torch.std(cloud, dim=0)
        return (cloud - mean) / (std + 1e-8) # epsilonprevent division by zero

    # standardizationallpointcloud
    pos_list = [standardize_cloud(p) for p in pos_list]
    
    with torch.no_grad():
        for i in range(num_time_pairs):
            print(f"  Calculating plan for time pair {i+1} -> {i+2}...")
            true_A_pos = pos_list[i].to(device)
            true_B_pos = pos_list[i+1].to(device)

            N, M = true_A_pos.shape[0], true_B_pos.shape[0]

            if N < 1 or M < 1: # as long ashaveapointcloudasemptythenjump
                print(f"      - Skipped due to insufficient points (A: {N}, B: {M}).")
                transport_plans.append(None)
                continue

            # ✅ key changes 1: directlyin Torch incompute A and B betweenL2distancematrix
            # M_spatial[i, j] = A[i] and B[j] betweendistance
            M_spatial = torch.cdist(true_A_pos, true_B_pos)
            
            # ✅ key changes 2: findtonearest neighbor
            # for M_spatial eachoneline (Aineachpoint) findmin valueindex (Binpoint)
            # thiswillreturna [N] shape tensor
            nearest_neighbor_indices = torch.argmin(M_spatial, dim=1)

            # ✅ key changes 3: construct“hard”transportplan P
            # P isa (N, M)  0 matrix
            P = torch.zeros(N, M, dtype=true_A_pos.dtype, device=device)
            
            # P[i, j] = 1.0 if B[j] is A[i] nearest neighbor
            # weuse scatter_ comeheighteneffectinfingerlocalizationplacefillenter 1.0
            # .unsqueeze(1) isasletindexdimension (N, 1) and P (N, M) match
            P.scatter_(1, nearest_neighbor_indices.unsqueeze(1), 1.0)
            
            transport_plans.append(P)
            print(f"      - Computed plan (Nearest Neighbor). Done.")

    print("--- NN pre-computation complete. ---")
    return transport_plans

import torch
import torch.nn.functional as F
import time
from geomloss import SamplesLoss
from torchdiffeq import odeint_adjoint as odeint

#trainingcode
def train_dynamic_point_cloud_model(
    # --- Data ---
    data_list,
    time_steps,
    type_list=None,
    gene_list_processed=None,
    precomputed_plans=None,
    
    # --- Model & Data Parameters ---
    num_sampled_points=None,
    num_cell_types=None,
    num_genes=None,
    
    # --- Training Hyperparameters ---
    epochs=10000,
    beta_max=1.0,
    beta_anneal_epochs=1000,
    
    # --- Loss Weights ---
    alpha_recon_loss=5.0,
    gamma_type_loss=1.0,
    gamma_gene_loss=1.0,
    gamma_existence_loss=1.0,
    gamma_trajectory_loss=1.0,
    lambda_kinetic_reg=1e-7, # ✅ 1. addmovepowerregularizationitemweight
    focal_loss_gamma=2.0,
    # -- model --
    latent_dim = 256,
    hidden_dim = 256,
    transformer_ff_dim=512,
    # -- opt --
    learning_rate=1e-4,
    weight_decay=1e-7,
    max_seq_len = 100,
    
    Exist_use = True,
    model_save_name='./save_data/model.pt', # <--- addedmodelsavepathparameter
    # --- Utilities ---
    device='cpu'
):
    """
    (alreadymodify) training TransformerLatentODE model。
    """
    assert len(data_list) == len(time_steps)
    # --- 1. Data Preparation (Padding & Feature Concatenation) ---
    print("\n--- Preparing Fixed Training Data (Padding ONCE) ---")
    fixed_sampled_pos_list, fixed_sampled_types_one_hot_list = [], []
    fixed_sampled_gene_list, fixed_sampled_features_list = [], []
    true_num_points_list = []
    MAX_data_number = max([data.shape[0] for data in data_list]) 
    MIN_data_number = min([data.shape[0] for data in data_list]) 
    spatial_dim = max([data.shape[1] for data in data_list]) 
    num_sampled_points = num_sampled_points if num_sampled_points is not None and num_sampled_points > MAX_data_number  else MAX_data_number
    if not Exist_use:
        num_sampled_points = num_sampled_points if num_sampled_points is not None and num_sampled_points < MIN_data_number  else MIN_data_number
    
    print(f"num_sampled_points: {num_sampled_points}")
    model = TransformerLatentODE(
        num_points=num_sampled_points, spatial_dim=spatial_dim, latent_dim=latent_dim,
        hidden_dim=hidden_dim, transformer_ff_dim=transformer_ff_dim,
        num_cell_types=num_cell_types,
        num_gene=num_genes,
        Exist_use=Exist_use,max_seq_len=max_seq_len
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # try:
    #     model.load_state_dict(torch.load(model_save_name, map_location=device))
    #     print(f"✅ Successfully loaded model from {model_save_name}")
    # except Exception as e:
    #     print(f"❌ Failed to load model: {e} ")
    #     exit(1)
    # (dataprepareitspartremains unchanged)
    for i in range(len(data_list)):
        p_full = data_list[i]
        t_full = type_list[i] if type_list else None
        g_full = gene_list_processed[i] if gene_list_processed and i < len(gene_list_processed) else None
        
        num_points_in_snapshot = p_full.shape[0]
        true_num_points_list.append(num_points_in_snapshot)
        
        p_padded, t_padded, g_padded = p_full, t_full, g_full

        padding_size = num_sampled_points - p_full.shape[0]
        if padding_size > 0:
            p_padding = torch.zeros(padding_size, p_padded.shape[1], device='cpu')
            p_padded = torch.cat([p_padded, p_padding], dim=0)
            if t_padded is not None:
                t_padding = torch.zeros(padding_size, dtype=torch.long, device='cpu')
                t_padded = torch.cat([t_padded, t_padding], dim=0)
            if g_padded is not None:
                g_padding = torch.zeros(padding_size, g_padded.shape[1], dtype=torch.float32, device='cpu')
                g_padded = torch.cat([g_padded, g_padding], dim=0)

        f_padded = p_padded.clone()
        if t_padded is not None and num_cell_types is not None:
            t_one_hot = F.one_hot(t_padded, num_classes=num_cell_types).float()
            fixed_sampled_types_one_hot_list.append(t_one_hot)
            f_padded = torch.cat([f_padded, t_one_hot], dim=1)
        if g_padded is not None:
            fixed_sampled_gene_list.append(g_padded)
            f_padded = torch.cat([f_padded, g_padded], dim=1)
            
        fixed_sampled_pos_list.append(p_padded)
        fixed_sampled_features_list.append(f_padded.to(device))
    
    # --- 2. Initialize Loss Functions ---
    ot_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="tensorized")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    alpha_weights = None
    # --- (✅ correction) 3. compute "graduallycutslice" (Per-Slice)  Alpha weight ---
    # based onyousuggest，weaseachtime pointcomputeindependentweight
    alpha_weights_list = []
    if type_list is not None and num_cell_types > 0:
        print("Calculating per-slice alpha weights...")
        for i, labels_at_t in enumerate(type_list):
            try:
                # checkwhenbeforetime pointisnohavecell
                if labels_at_t.numel() == 0:
                    alpha_weights_list.append(None) # nocell，weightas None
                    continue

                class_counts = torch.bincount(labels_at_t, minlength=num_cell_types)
                total_samples = class_counts.sum()

                # checkisnohave > 0 sample
                if total_samples == 0:
                    alpha_weights_list.append(None)
                    continue

                # computewhenbeforecutsliceweight
                weights_at_t = total_samples / (class_counts + 1e-6)
                weights_at_t = weights_at_t / weights_at_t.sum() * num_cell_types
                weights_at_t = torch.clamp(weights_at_t, min=0.5, max=5.0)
                alpha_weights_list.append(weights_at_t.to(device)) # addtolist
                
            except Exception as e:
                print(f"Warning: Could not compute alpha weights for slice {i} ({e}).")
                alpha_weights_list.append(None)
        print("Per-slice alpha weight calculation complete.")
    else:
         # ifnotypedata，createa Nones list
        alpha_weights_list = [None] * len(data_list)
    # --- 3. Training Loop ---
    print("\n--- Starting Training ---")
    start_time = time.time()
    loss_ot_temp = 1.0
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # ✅ 2. solvepackage 7 returnvalue
        recon_pos, recon_type_logits, recons_gene, recon_existence_logits, z0_mu, z0_log_var, latent_trajectory = model(fixed_sampled_features_list, time_steps.to(device))        
        # --- Loss Calculation ---
        # (lossinternalcomputeremains unchanged)
        total_recon_loss, loss_ot, total_type_loss, total_gene_loss, total_existence_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        num_timesteps = recon_pos.shape[0]
        for i in range(num_timesteps):
            true_n = true_num_points_list[i]
            existence_target = torch.zeros(num_sampled_points, device=device); existence_target[:true_n] = 1.0
            if Exist_use:
                total_existence_loss += bce_loss_fn(recon_existence_logits[i], existence_target)
                
            if true_n > 0:
                pred_pos_subset = recon_pos[i, :true_n]
                true_pos_subset = fixed_sampled_pos_list[i][:true_n].to(device)
                
                loss_ot += ot_loss_fn(pred_pos_subset, true_pos_subset)
                loss_recon_i, indices_p1_to_p2, indices_p2_to_p1 = chamfer_distance_with_indices(pred_pos_subset.unsqueeze(0), true_pos_subset.unsqueeze(0))
                total_recon_loss += loss_recon_i
                if loss_ot_temp<10.0:
                    if recon_type_logits is not None:
                        true_types_one_hot_subset = fixed_sampled_types_one_hot_list[i][:true_n].to(device)
                        pred_logits_subset = recon_type_logits[i, :true_n]
                        true_labels_subset = torch.argmax(true_types_one_hot_subset, dim=-1) # [true_n]

                        idx_p1_to_p2 = indices_p1_to_p2.squeeze(0) # [true_n]
                        idx_p2_to_p1 = indices_p2_to_p1.squeeze(0) # [true_n]

                        aligned_true_labels = torch.gather(true_labels_subset, 0, idx_p1_to_p2)
                        aligned_pred_logits = torch.gather(
                            pred_logits_subset, 0, 
                            idx_p2_to_p1.unsqueeze(-1).expand(-1, num_cell_types)
                        )
                        current_alpha_weights = alpha_weights_list[i]
                        loss_p1_to_p2 = compute_focal_loss(
                            pred_logits_subset, 
                            aligned_true_labels, 
                            gamma=focal_loss_gamma,
                            alpha_weights=current_alpha_weights # <--- usecutsliceiweight
                        )
                        loss_p2_to_p1 = compute_focal_loss(
                            aligned_pred_logits, 
                            true_labels_subset,
                            gamma=focal_loss_gamma,
                            alpha_weights=current_alpha_weights # <--- usecutsliceiweight
                        )
                        total_type_loss += (loss_p1_to_p2 + loss_p2_to_p1)
                        # print(loss_p2_to_p1)
                        
                    if recons_gene is not None:
                        true_gene_subset = fixed_sampled_gene_list[i][:true_n].to(device)
                        pred_gene_subset = recons_gene[i, :true_n]
                        idx_p1_to_p2 = indices_p1_to_p2.squeeze(0) # mapping: pred -> true
                        idx_p2_to_p1 = indices_p2_to_p1.squeeze(0) # mapping: true -> pred
                        gather_index_for_true = idx_p1_to_p2.unsqueeze(-1).expand(-1, num_genes)
                        gather_index_for_pred = idx_p2_to_p1.unsqueeze(-1).expand(-1, num_genes)
                        aligned_true_gene = torch.gather(
                            true_gene_subset,  # fromtruegenein
                            dim=0,             # alongpointdimension
                            index=gather_index_for_true # use (pred -> true) index
                        )
                        aligned_pred_gene = torch.gather(
                            pred_gene_subset,  # frompredictiongenein
                            dim=0,             # alongpointdimension
                            index=gather_index_for_pred # use (true -> pred) index
                        )
                        loss_p1_to_p2 = F.mse_loss(pred_gene_subset, aligned_true_gene)
                        loss_p2_to_p1 = F.mse_loss(aligned_pred_gene, true_gene_subset)
                        total_gene_loss += (loss_p1_to_p2 + loss_p2_to_p1)
        
        # (lossmeancomputeremains unchanged)
        recon_loss = total_recon_loss / num_timesteps if num_timesteps > 0 else 0.0
        loss_ot = loss_ot / num_timesteps if num_timesteps > 0 else 0.0
        type_loss = total_type_loss / num_timesteps if isinstance(total_type_loss, torch.Tensor) and num_timesteps > 0 else 0.0
        gene_loss = total_gene_loss / num_timesteps if isinstance(total_gene_loss, torch.Tensor) and num_timesteps > 0 else 0.0
        existence_loss = total_existence_loss / num_timesteps if num_timesteps > 0 else 0.0
        loss_ot_temp = loss_ot
        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + z0_log_var - z0_mu.pow(2) - z0_log_var.exp())
        
        # Trajectory Loss
        traj_loss = 0.0
        if Exist_use:
            if precomputed_plans is not None:
                traj_loss = fgw_trajectory_consistency_loss(
                    fixed_sampled_pos_list, recon_pos, time_steps,
                    precomputed_plans, true_num_points_list, device=device
                )
            else:
                traj_loss = trajectory_consistency_loss(fixed_sampled_pos_list, recon_pos, time_steps, true_num_points_list)
            
        # KL Annealing
        current_beta = beta_max * min(1.0, (epoch + 1) / beta_anneal_epochs)
        if len(time_steps) > 3:
            dt = (time_steps[1:] - time_steps[:-1]).to(device).unsqueeze(-1)
            # computespeed v = (z[t+1] - z[t]) / dt
            velocities = (latent_trajectory[1:] - latent_trajectory[:-1]) / dt
            # computeaddspeed a = (v[t+1] - v[t])
            # weonlypenaltyspeed *change*，makeitssmooth
            accelerations = velocities[1:] - velocities[:-1]
            # movecanregularizationloss = mean(a^2)
            kinetic_reg_loss = torch.mean(accelerations.pow(2))
        else:
            kinetic_reg_loss = 0.0
        # ✅ 3. finaladdloss：add kinetic_reg_loss
        loss = (alpha_recon_loss * recon_loss) + \
               (alpha_recon_loss * loss_ot) + \
               (gamma_type_loss * type_loss if isinstance(type_loss, torch.Tensor) else 0.0) + \
               (gamma_gene_loss * gene_loss if isinstance(gene_loss, torch.Tensor) else 0.0) + \
               (current_beta * kl_loss) + \
               (gamma_existence_loss * existence_loss) + \
               (gamma_trajectory_loss * traj_loss) + \
               (lambda_kinetic_reg * kinetic_reg_loss) # <--- addedlossitem
        
        if (epoch + 1) % 10 == 0:
            # ✅ 4. updatehitprintlog
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, OT: {loss_ot.item():.4f}, "
                  f"Type: {type_loss.item() if isinstance(type_loss, torch.Tensor) else 0.0:.4f}, "
                  f"Gene: {gene_loss.item() if isinstance(gene_loss, torch.Tensor) else 0.0:.4f}, "
                  f"Exist: {existence_loss.item() if isinstance(existence_loss, torch.Tensor) else 0.0:.4f},"
                  f"Traj: {traj_loss.item() if isinstance(traj_loss, torch.Tensor) else 0.0:.4f}, "
                  f"Kinetic: {kinetic_reg_loss.item() if isinstance(kinetic_reg_loss, torch.Tensor) else 0.0:.4f}") # <--- addedlog

        if model_save_name and loss.item() < best_loss:
            best_loss = loss.item()
            os.makedirs(os.path.dirname(model_save_name) or '.', exist_ok=True)
            torch.save(model.state_dict(), model_save_name)

        loss.backward()
        optimizer.step()
        # scheduler.step()

    # if model_save_name:
    #     try:
    #         model.load_state_dict(torch.load(model_save_name, map_location=device))
    #         print(f"\n✅ Loaded best model from {model_save_name} for final evaluation.")
    #     except:
    #         print(f"\nWarning: Could not load model from {model_save_name}. Using last trained weights.")

    recon_pos_list = []
    recon_pos_list2 = []
    recon_pos_list3 = []
    true_pos_list = []
    
    # ✅ 5. updatefunctionlasttail model() calltosolvepackage 7 values
    recon_pos_old, *_, kinetic_loss_1 = model(fixed_sampled_features_list, time_steps.to(device))
    with torch.no_grad():
            recon_pos_old2, *_, kinetic_loss_2 = model(fixed_sampled_features_list, time_steps.to(device))
    model.eval()
    recon_pos, *_, kinetic_loss_3 = model(fixed_sampled_features_list, time_steps.to(device))
    
    for pos in recon_pos_old:
        recon_pos_list2.append(pos.detach().cpu().numpy())
    for pos in recon_pos_old2:
        recon_pos_list3.append(pos.detach().cpu().numpy())      
    for pos in recon_pos:
        recon_pos_list.append(pos.detach().cpu().numpy())
    for pos in fixed_sampled_pos_list:
        true_pos_list.append(pos.detach().cpu().numpy())
    print(f"\nTraining complete. Time elapsed: {(time.time() - start_time):.2f} seconds")
    return model, fixed_sampled_features_list, {'recon3': recon_pos_list3,'recon2': recon_pos_list2, 'recon': recon_pos_list, 'true': true_pos_list}


def save_evaluation_results(eval_outputs, time_steps, save_path="evaluation_outputs.pt"):
    """
    willmodelevaluationoutputandforshouldtime pointsavetofile。

    Args:
        eval_outputs (tuple): containsmodelinoriginalbegintime pointonoutputtuple 
                              (eval_pos, eval_logits, eval_gene, ...)。
        time_steps (torch.Tensor): forshouldoriginalbegintime point。
        save_path (str): savefilepath。
    """
    print(f"\n--- Saving evaluation outputs to {save_path} ---")
    
    # fromoutputtupleinsolvepackage
    eval_pos, eval_logits, eval_gene, Exit_eval, *_ = eval_outputs
    
    # assquaresubsequentuse，willdataturnshifttoCPUandconvertasNumPyarray
    # ifhoperetainasPyTorchtensor，canomit .cpu().numpy()
    results_dict = {
        'time_steps': time_steps.detach().cpu().numpy(),
        'positions': eval_pos.detach().cpu().numpy(),
        'type_logits': eval_logits.detach().cpu().numpy() if eval_logits is not None else None,
        'gene_expressions': eval_gene.detach().cpu().numpy() if eval_gene is not None else None,
        'Exit': Exit_eval.detach().cpu().numpy() if Exit_eval is not None else None,
    }
    
    try:
        # use torch.save saveneatdictionary
        torch.save(results_dict, save_path)
        print(f"✅ Successfully saved results.")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def evaluate_and_visualize_model(
    # --- Model & Data ---
    model,
    fixed_sampled_features_list,
    data_list,
    time_steps,
    # --- Optional Data for Metrics ---
    type_list=None,
    gene_list_processed=None,
    # --- Mappings & Parameters ---
    int_to_cell_type=None,
    gene_names=None,
    device='cpu',
    root_dir='./',
    output_save_path="evaluation_outputs.pt", # allowcustomsavepath
    extrapolation_factor=0.1, # 0.0 means no extrapolation, 0.5 means extrapolate 50% beyond the last time point
    num_steps=100, 
):
    """
    Performs evaluation and generates visualizations for the trained point cloud trajectory model.

    Args:
        model (nn.Module): The trained TransformerLatentODE model.
        fixed_sampled_features_list (list): The list of padded feature tensors used for model input.
        time_steps (torch.Tensor): The tensor of original time steps.
        data_list (list): The list of original, unpadded point cloud tensors.
        type_list (list, optional): The list of original integer-encoded cell type tensors.
        gene_list (list, optional): The list of original, raw gene data tensors.
        gene_list_processed (list, optional): The list of preprocessed gene data tensors.
        int_to_cell_type (dict, optional): Mapping from integer to cell type string for visualization.
        num_genes (int): The number of gene features.
        evaluate_spatial_reconstruction (function): Function to calculate spatial reconstruction metric.
        evaluate_cell_type_accuracy (function): Function to calculate cell type accuracy.
        evaluate_gene_scc (function): Function to calculate gene expression correlation.
        visualize_3d_trajectory (function): Function to generate the 3D cell type GIF.
        visualize_3d_gene_trajectory (function): Function to generate the 3D gene expression GIF.
        device (torch.device): The device to run inference on.
    """
    print("\n--- Generating Smooth Interpolation Animation and Evaluating ---")        
# ensure root_dir savein
    os.makedirs(root_dir, exist_ok=True)
    
    with torch.enable_grad():
        if gene_list_processed is not None:
            num_genes = gene_list_processed[0].shape[1] if gene_list_processed and len(gene_list_processed) > 0 else 0
        model.eval() 
        
        # --- Create Smooth Interpolation ---
        min_time = time_steps.min()
        max_time = time_steps.max()
        time_range = max_time - min_time
        
        extrapolated_max_time = max_time + time_range * extrapolation_factor
        final_num_steps = int(num_steps * (1 + extrapolation_factor))

        t_eval_smooth = torch.linspace(min_time, extrapolated_max_time, final_num_steps).to(device)
        
        inferred_outputs = model(fixed_sampled_features_list, t_eval_smooth)
        
        # 【modification point 1】: use os.path.join construct .pt filecompletelyneatsavepath
        final_pt_save_path = os.path.join(root_dir, output_save_path)
        print(f"Saving evaluation outputs to: {final_pt_save_path}")
        if save_evaluation_results:
            save_evaluation_results(inferred_outputs, t_eval_smooth, save_path=final_pt_save_path)
        
        # --- Evaluate at Original Time Points ---
        eval_outputs = model(fixed_sampled_features_list, time_steps.to(device))
        eval_pos, eval_logits, eval_gene, *_ = eval_outputs
        
        original_true_num_points = [p.shape[0] for p in data_list]
        
        # --- Calculate and Print Metrics ---
        if evaluate_spatial_reconstruction:
            ot_distance = evaluate_spatial_reconstruction(data_list, eval_pos, original_true_num_points, device)
            print(f"Average Sinkhorn Distance (Spatial Reconstruction): {ot_distance:.4f}")

        if type_list and int_to_cell_type and evaluate_cell_type_accuracy:
            accuracy = evaluate_cell_type_accuracy(data_list, type_list, eval_pos, eval_logits, original_true_num_points, device)
            print(f"Cell Type Accuracy (Nearest Neighbor): {accuracy:.4f}")

        if gene_list_processed and evaluate_gene_scc:
            overall_scc = evaluate_gene_scc(data_list, gene_list_processed, eval_pos, eval_gene, original_true_num_points, device)
            print(f"Overall Gene SCC (Spearman Correlation): {overall_scc:.4f}")
            
        # --- Generate Visualizations ---
        spatial_dim = 0
        for p in data_list:
            if p.shape[0] > 0:
                spatial_dim = p.shape[1]
                break
        
        if spatial_dim == 2:
            print("\nData is 2D. Generating 2D visualizations...")
            if int_to_cell_type and visualize_trajectory:
                # 【modification point 2】: construct 2D cell type GIF path
                gif_path_2d_cell = os.path.join(root_dir, "data_2d_cell_type_evolution.gif")
                print(f"Saving 2D cell type GIF to: {gif_path_2d_cell}")
                visualize_trajectory(
                    inferred_outputs, t_eval_smooth, data_list, int_to_cell_type, 
                    gif_path_2d_cell
                )
            if gene_list_processed is not None and num_genes > 0 and visualize_gene_trajectory:
                if gene_names is None:
                    gene_names = [f"Gene_{i+1}" for i in range(num_genes)]
                
                # 【modification point 3】: construct 2D gene GIF path
                gene_gif_filename = f"data_2d_gene_{gene_names[0]}_evolution.gif"
                gif_path_2d_gene = os.path.join(root_dir, gene_gif_filename)
                print(f"Saving 2D gene GIF to: {gif_path_2d_gene}")
                visualize_gene_trajectory(
                    inferred_outputs, t_eval_smooth, data_list, 
                    gene_index=0, gene_name=gene_names[0], 
                    output_gif_path=gif_path_2d_gene
                )
        elif spatial_dim == 3:
            print("\nData is 3D. Generating 3D visualizations...")
            if int_to_cell_type and visualize_3d_trajectory:
                # 【modification point 4】: construct 3D cell type GIF path
                gif_path_3d_cell = os.path.join(root_dir, "data_3d_cell_type_evolution.gif")
                print(f"Saving 3D cell type GIF to: {gif_path_3d_cell}")
                visualize_3d_trajectory(
                    inferred_outputs, t_eval_smooth, data_list, int_to_cell_type, 
                    gif_path_3d_cell
                )
            if gene_list_processed is not None and num_genes > 0 and visualize_3d_gene_trajectory:
                if gene_names is None:
                    gene_names = [f"Gene_{i+1}" for i in range(num_genes)]
                
                # 【modification point 5】: construct 3D gene GIF path
                gene_gif_filename_3d = f"data_3d_gene_{gene_names[0]}_evolution.gif"
                gif_path_3d_gene = os.path.join(root_dir, gene_gif_filename_3d)
                print(f"Saving 3D gene GIF to: {gif_path_3d_gene}")
                visualize_3d_gene_trajectory(
                    inferred_outputs, t_eval_smooth, data_list, 
                    gene_index=0, gene_name=gene_names[0], 
                    output_gif_path=gif_path_3d_gene
                )
        else:
            print(f"Warning: Unsupported spatial dimension ({spatial_dim}). Skipping GIF generation.")


def evaluate_on_test_set(model, train_features_list, test_data, device, int_to_cell_type_map):
    """
    (✅ newfunction)
    usetraininggoodmodel，inindependenttestsetoncomputeandhitprintevaluationindicator。
    """
    print("\n--- [step 4] currently Test Set onevaluationmodel ---")
    
    # --- 1. gettestsetdata ---
    test_time_steps = test_data['time_steps'].to(device)
    test_data_list = test_data['data_list']
    test_type_list = test_data['type_list']
    test_gene_list_processed = test_data['gene_list_processed']
    
    if test_time_steps is None or len(test_time_steps) == 0:
        print("notestdatacanprovideevaluation。")
        return

    # --- 2. getpredictionresult ---
    model.eval()
    
    # train_features_list isfrom train_dynamic_point_cloud_model return，
    # itrepresentmodelused forcoding z0 trainingdata。
    # weuse *training*  z0 and *test* time pointcomegenerateprediction
    with torch.no_grad():
         (eval_pos, eval_logits, eval_gene, 
          _, _, _, _) = model(train_features_list, test_time_steps)
    
    print("  - alreadyastesttime pointgeneratemodelprediction。")

    # --- 3. prepareevaluation ---
    original_true_num_points = [p.shape[0] for p in test_data_list]
    
    # --- 4. computeindicator (useweonsurfacedefinitioncorrectionfunction) ---
    ot_distance = evaluate_spatial_reconstruction(
        test_data_list, eval_pos, original_true_num_points, device=device
    )
    print(f"  - [test set metrics] average Sinkhorn distance: {ot_distance:.4f}")

    if test_type_list and int_to_cell_type_map:
        accuracy = evaluate_cell_type_accuracy(
            test_data_list, test_type_list, eval_pos, eval_logits, 
            original_true_num_points, device=device
        )
        print(f"  - [test set metrics] cell typeaccuracy: {accuracy:.4f}")

    if test_gene_list_processed:
        overall_scc = evaluate_gene_scc(
            test_data_list, test_gene_list_processed, eval_pos, eval_gene, 
            original_true_num_points, device=device
        )
        print(f"  - [test set metrics] 总bodygene SCC: {overall_scc:.4f}")
    
    print("--- testsetevaluationcompleted ---")

import pandas as pd
# --- addedevaluationindicatorfunction：populationaverageabsoluteforerror (P-MAE) ---
def evaluate_population_mae(true_num_points_list, recon_existence_logits, exist_threshold=0.4):
    """
    (alreadyrevised) computenormalizationpopulationaverageabsoluteforerror (Normalized P-MAE, nP-MAE)。
    衡quantitymodelpredictionpoint count（throughsavein logits）withtruepoint countbetweenforabsoluteforerror。

    publicstyle: nP-MAE = (1/T) * sum(|N_pred - N_true| / N_true)

    Args:
        true_num_points_list (list): eachtime pointtruepoint countlist。
        recon_existence_logits (torch.Tensor): modelpredictionsavein logits，shape [T, N_max]。
        exist_threshold (float): saveinpredictionthreshold。

    Returns:
        float: normalizationaverageabsoluteforerror nP-MAE。
    """
    total_relative_mae = 0.0
    valid_timesteps_count = 0
    if not true_num_points_list or recon_existence_logits is None:
        return 0.0
    for i in range(len(true_num_points_list)):
        true_n = true_num_points_list[i]
        # iftruepoint countas0，thenjumpshouldtime point（nomethodcomputeforerror）
        if true_n == 0:
            continue
        # through Sigmoid andthresholdcomputepredictionpoint count
        pred_existence_prob = torch.sigmoid(recon_existence_logits[i])
        pred_n = (pred_existence_prob > exist_threshold).sum().item()
        # computeforabsoluteforerror
        relative_mae = abs(pred_n - true_n) / true_n
        total_relative_mae += relative_mae
        valid_timesteps_count += 1
    # nP-MAE isallvalidtime pointaverageforabsoluteforerror
    return total_relative_mae / valid_timesteps_count if valid_timesteps_count > 0 else 0.0

# --- structureevaluationcoreauxiliaryfunction ---
def _calculate_all_metrics(
    i,
    true_pos_list, 
    true_type_list, 
    true_gene_list_processed,
    recon_pos, 
    recon_type_logits, 
    recon_gene, 
    recon_existence_logits,
    true_num_points_list, 
    device,
    ot_loss_fn,
    exist_threshold=0.4
):
    """
    computesingletime point i allevaluationindicator。thisisainternalauxiliaryfunction。
    """
    results = {
        'CD_Loss': float('nan'),
        'OT_Loss': float('nan'),
        'CTA': float('nan'),
        'Feature_SCC': float('nan'),
        'nP_MAE': float('nan'),
    }
    
    true_n = true_num_points_list[i]
    if true_n == 0:
        return results

    # --- 1. getsetdataandshiftmovetodevice ---
    true_points = true_pos_list[i][:true_n].to(device)
    recon_points = recon_pos[i, :true_n]
    true_centroid = torch.mean(true_points, dim=0, keepdim=True)
    recon_centroid = torch.mean(recon_points, dim=0, keepdim=True)
    
    # 2. frompointcloudinsubtractgoheart
    true_points = true_points - true_centroid
    recon_points = recon_points - recon_centroid
    # --- 2. severalwhatkeeptruedegree (CD, OT) ---
    # CD Loss
    loss_cd_i, _, _ = chamfer_distance_with_indices(recon_points.unsqueeze(0), true_points.unsqueeze(0))
    results['CD_Loss'] = loss_cd_i.item()
    
    # OT Loss (Sinkhorn Distance)
    ot_distance = ot_loss_fn(recon_points, true_points).item()
    results['OT_Loss'] = ot_distance

    # --- 3. attributekeeptruedegree (CTA, Gene SCC) ---
    # use NN matchperformforneat
    dist_matrix = torch.cdist(recon_points, true_points)
    nearest_true_indices = torch.argmin(dist_matrix, dim=1) # [pred_n] -> [true_n] indices

    # CTA
    if recon_type_logits is not None and true_type_list is not None and true_type_list[i] is not None:
        true_types = true_type_list[i][:true_n].to(device)
        recon_types = torch.argmax(recon_type_logits[i, :true_n], dim=-1)
        
        aligned_true_types = true_types[nearest_true_indices]
        correct_matches = (recon_types == aligned_true_types).float().sum()
        accuracy = correct_matches / true_n
        results['CTA'] = accuracy.item()
    
    # Gene SCC
    if recon_gene is not None and true_gene_list_processed is not None and true_gene_list_processed[i] is not None:
        true_genes = true_gene_list_processed[i][:true_n].to(device)
        recon_genes = torch.relu(recon_gene[i, :true_n])
        
        aligned_true_genes = true_genes[nearest_true_indices]
        
        num_genes = true_genes.shape[1]
        gene_scc_scores = []
        for g_idx in range(num_genes):
            recon_g_np = recon_genes[:, g_idx].detach().cpu().numpy()
            aligned_true_g_np = aligned_true_genes[:, g_idx].detach().cpu().numpy()
            
            scc, _ = spearmanr(recon_g_np, aligned_true_g_np)
            if not np.isnan(scc):
                gene_scc_scores.append(scc)

        results['Feature_SCC'] = np.mean(gene_scc_scores) if gene_scc_scores else float('nan')

    # --- 4. populationdynamic (P-MAE) ---
    if recon_existence_logits is not None:
        pred_existence_prob = torch.sigmoid(recon_existence_logits[i])
        pred_n = (pred_existence_prob > exist_threshold).sum().item()
        results['nP_MAE'] = abs(pred_n - true_n)/true_n
    return results

# --- addedevaluationfunction (used forstructureoutput) ---
def _to_tensor(data, device):
    """
    auxiliaryfunction：
    安allwill None、np.ndarray or torch.Tensor 
    convertasinfingerdeviceon torch.Tensor。
    """
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        # ifalreadyistensor，onlyneedshiftmovetocorrectdevice
        return data.to(device)
    if isinstance(data, np.ndarray):
        # ifis Numpy array，thenfrom numpy convert
        return torch.from_numpy(data).to(device)
    
    # 作as备选solution，tryconvertitsothercaniterationobject（such aslist）
    try:
        return torch.tensor(data).to(device)
    except Exception as e:
        print(f"warning：nomethodwilldataconvertastensor: {e}")
        return None
    
def evaluate_test_set_structured(
    true_data_dict,             # truevalueLoaded_dataformat: dict, keys: ['data_list', 'type_list', 'gene_list_processed', 'time_steps']
    pred_data_dict,             # predictionvalueLoaded_dataformat: dict, keys: ['positions', 'type_logits', 'gene_expressions', 'time_steps', 'Exit']
    train_time_steps,           # trainingsettime point (used forinterpolation/extrapolationbreak)
    device='cpu'
):
    """
    computegivetruedataandpredictiondatadictionaryonallevaluationindicator，andreturnstructure DataFrame。

    Args:
        true_data_dict (dict): truevaluedictionary (contains list of Tensors and time_steps Tensor)。
        pred_data_dict (dict): predictionvaluedictionary (contains Tensor and time_steps Tensor)。
        train_time_steps (torch.Tensor): trainingsetintime point。
        device (str): computedevice。

    Returns:
        pd.DataFrame: totime pointasindex，eachitemindicatorascolumn DataFrame。
    """
    print("\n--- Starting Structured Evaluation ---")

    # --- 1. dataprepare ---
    true_time_steps_np = true_data_dict['time_steps'] # NumPy array
    pred_time_steps_np = pred_data_dict['time_steps'] # NumPy array
    
    if not np.allclose(true_time_steps_np, pred_time_steps_np):
        print("Error: True and Predicted time steps do not match. Aborting evaluation.")
        return pd.DataFrame()
        
    time_steps_eval = torch.from_numpy(true_time_steps_np).to(device)
    num_timesteps = len(time_steps_eval)

    # willtruedatalistintensorshiftmovetodeviceon (ifitnotin话)
    true_pos_list = [p.to(device) for p in true_data_dict.get('data_list', [None] * num_timesteps)]
    true_type_list = [t.to(device) for t in true_data_dict.get('type_list', [None] * num_timesteps)]
    true_gene_list_processed = [g.to(device) for g in true_data_dict.get('gene_list_processed', [None] * num_timesteps)]
    
    # frompredictiondictionaryinrecoverytensor
    recon_pos = _to_tensor(pred_data_dict.get('positions'), device)
    recon_type_logits = _to_tensor(pred_data_dict.get('type_logits'), device)
    recon_gene = _to_tensor(pred_data_dict.get('gene_expressions'), device)
    recon_existence_logits = _to_tensor(pred_data_dict.get('Exit'), device)
    
    true_num_points_list = [p.shape[0] for p in true_pos_list]

    # --- 2. corecorrection：gettrainingsettimerange ---
    # assume train_time_steps isa torch.Tensor (such asin run.py inprepare)
    if isinstance(train_time_steps, np.ndarray):
        train_time_steps = torch.from_numpy(train_time_steps).float()
        
    min_train_time = train_time_steps.min().item()
    max_train_time = train_time_steps.max().item()

    # --- 3. evaluationloop ---
    ot_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="auto")
    all_results = []
    
    with torch.no_grad():
        for i in range(num_timesteps):
            time_point = time_steps_eval[i].item()
            
            # ✅ correctionafterinterpolationbreaklogic：time pointfallin [min_train_time, max_train_time] areabetweeninside
            is_interpolation = (time_point >= min_train_time) and (time_point <= max_train_time)
            
            # computeallindicator
            # assume _calculate_all_metrics alreadybydefinitioninfileitsotherpart
            metrics = _calculate_all_metrics(
                i,
                true_pos_list, 
                true_type_list, 
                true_gene_list_processed,
                recon_pos, 
                recon_type_logits, 
                recon_gene, 
                recon_existence_logits,
                true_num_points_list, 
                device,
                ot_loss_fn,
            )
            
            # addtime pointandclassificationinformation
            metrics['Time'] = time_point
            metrics['Is_Interpolation'] = is_interpolation
            
            all_results.append(metrics)

    # --- 4. construct DataFrame ---
    df = pd.DataFrame(all_results).set_index('Time')
    print("--- Structured Evaluation Complete ---")
    return df

# belowfunctioncall evaluate_test_set_structured，故remains unchanged

def evaluate_interpolation_structured(true_data_dict, pred_data_dict, train_time_steps, device='cpu'):
    """
    inpredictionvaluein，for**interpolationtime point** (即intrainingsettimerangeinside) performstructureevaluation。
    """
    df_all = evaluate_test_set_structured(true_data_dict, pred_data_dict, train_time_steps, device)
    df_interp = df_all[df_all['Is_Interpolation'] == True].drop(columns=['Is_Interpolation'])
    print(f"Interpolation Evaluation Done. Results for {len(df_interp)} time points.")
    return df_interp


def evaluate_extrapolation_structured(true_data_dict, pred_data_dict, train_time_steps, device='cpu'):
    """
    inpredictionvaluein，for**extrapolationtime point** (即intrainingsettimerangeoutside) performstructureevaluation。
    """
    df_all = evaluate_test_set_structured(true_data_dict, pred_data_dict, train_time_steps, device)
    df_extrap = df_all[df_all['Is_Interpolation'] == False].drop(columns=['Is_Interpolation'])
    print(f"Extrapolation Evaluation Done. Results for {len(df_extrap)} time points.")
    return df_extrap
  
def split_data(loaded_data_all):
    """
    (Modified)
    willhandwrongdatadictionary [1.0, 1.5, 2.0, ...] splitdivideastrainingsetandtestset dictionary。

    Args:
        loaded_data_all (dict): acontainsalltime point、alreadyloaddatadictionary。

    Returns:
        tuple: (train_data, test_data)
            - train_data (dict): containstrainingpoint (0, 2, 4, ...) datadictionary。
            - test_data (dict): containstestpoint (1, 3, 5, ...) datadictionary。
    """
    print("currentlywilldatadictionarysplitdivideastrainingsetandtestset...")
    
    # --- 1. solvepackageinputdictionary ---
    data_list_all = loaded_data_all.get('data_list', None)
    type_list_all = loaded_data_all.get('type_list', None)
    gene_list_all = loaded_data_all.get('gene_list_raw', None)
    gene_list_proc_all = loaded_data_all.get('gene_list_processed', None)
    time_steps_all = loaded_data_all.get('time_steps', None)
    
    # (✅ added) extractionnotimesequencedata
    cell_type_to_int = loaded_data_all.get('cell_type_to_int', None)
    int_to_cell_type = loaded_data_all.get('int_to_cell_type', None)

    if time_steps_all is None:
        print("  - error: datadictionaryinno 'time_steps'。nomethodsplitdivide。")
        return None, None
        
    # --- 2. splitdividelogic (nochange) ---
    num_all_points = len(time_steps_all)
    train_indices = list(range(0, num_all_points, 2))
    test_indices = list(range(1, num_all_points, 2))
    
    def _split_list(input_list, indices):
        if input_list is None:
            return None
        return [input_list[i] for i in indices]

    # --- 3. (✅ modify) createtrainingset dictionary ---
    train_data = {
        "data_list": _split_list(data_list_all, train_indices),
        "type_list": _split_list(type_list_all, train_indices),
        "gene_list_raw": _split_list(gene_list_all, train_indices),
        "gene_list_processed": _split_list(gene_list_proc_all, train_indices),
        "time_steps": time_steps_all[train_indices],
        
        # (✅ added) willdataaddreturndictionary
        "cell_type_to_int": cell_type_to_int,
        "int_to_cell_type": int_to_cell_type
    }
    
    # --- 4. (✅ modify) createtestset dictionary ---
    test_data = {
        "data_list": _split_list(data_list_all, test_indices),
        "type_list": _split_list(type_list_all, test_indices),
        "gene_list_raw": _split_list(gene_list_all, test_indices),
        "gene_list_processed": _split_list(gene_list_proc_all, test_indices),
        "time_steps": time_steps_all[test_indices],
        "cell_type_to_int": cell_type_to_int,
        "int_to_cell_type": int_to_cell_type
    }
    
    print(f"  - trainingset: {len(train_data['time_steps'])} time point (t={train_data['time_steps'].numpy()})")
    print(f"  - testset: {len(test_data['time_steps'])} time point (t={test_data['time_steps'].numpy()})")
    
    return train_data, test_data


# --- addedpredictionandformatfunction ---
import torch
import numpy as np

def predict_and_format_structured_output(
    model, 
    train_features_list, 
    target_time_steps,
    device='cpu'
):
    """
    calltraininggoodmodelperformprediction，andwillresultformatas pred_data_dict structure。

    Args:
        model (nn.Module): traininggood TransformerLatentODE model。
        train_features_list (list): trainingsetinfixed（alreadypadding）featuretensorlist，used forcoding z0。
        target_time_steps (torch.Tensor): goaltime pointtensor，shape [T_target]。
        device (str): runpushreasondevice。

    Returns:
        dict: structurepredictionresultdictionary，keys match evaluate_test_set_structured  pred_data_dict format。
    """
    model.eval()
    
    # ensuretime pointincorrectdeviceon
    target_time_steps = target_time_steps.to(device)
    
    # --- 1. executeprediction ---
    with torch.no_grad():
        # model output: (eval_pos, eval_logits, eval_gene, Exit_eval, z0_mu, z0_log_var, latent_trajectory)
        outputs = model(train_features_list, target_time_steps)
        eval_pos, eval_logits, eval_gene, Exit_eval, _, _, _ = outputs
    
    # --- 2. formatas NumPy array (necessarystep) ---
    # willalloutputturnshiftto CPU andconvertas NumPy array
    results_dict = {
        'time_steps': target_time_steps.detach().cpu().numpy(),
        'positions': eval_pos.detach().cpu().numpy(),
        'type_logits': eval_logits.detach().cpu().numpy() if eval_logits is not None else None,
        'gene_expressions': eval_gene.detach().cpu().numpy() if eval_gene is not None else None,
        'Exit': Exit_eval.detach().cpu().numpy() if Exit_eval is not None else None,
    }
    
    return results_dict

def prep_data_for_eval(data_dict, device='cpu'):
    """
    (✅ added)
    will joblib loaddictionary（containsnumpyarray/list）convertreturnevaluationfunctionneed
    dictionary（containsPyTorchtensor/tensorlist）。
    """
    prepped_dict = {}
    for key, value in data_dict.items():
        if key in ['data_list', 'type_list', 'gene_list_raw', 'gene_list_processed']:
            # 'true_data' inkeyislist (list of numpy arrays)
            if isinstance(value, list):
                # willlistineach numpy arrayconvertas tensor
                prepped_dict[key] = [
                    torch.from_numpy(arr).to(device) if isinstance(arr, np.ndarray) 
                    else torch.empty(0).to(device) # handleemptylistitem
                    for arr in value
                ]
            else:
                prepped_dict[key] = value
        
        elif key in ['positions', 'type_logits', 'gene_expressions', 'Exit']:
            # 'pred_data' inkeyissingle numpy array (T, N, C)
            # weneedheavynameittomatch 'true_data' structure
            if key == 'positions':
                prepped_dict['recon_pos'] = torch.from_numpy(value).to(device)
            if key == 'type_logits':
                prepped_dict['recon_type_logits'] = torch.from_numpy(value).to(device)
            if key == 'gene_expressions':
                prepped_dict['recon_gene'] = torch.from_numpy(value).to(device)
            if key == 'Exit':
                prepped_dict['recon_existence_logits'] = torch.from_numpy(value).to(device)
        
        elif key == 'time_steps' and isinstance(value, np.ndarray):
             prepped_dict[key] = torch.from_numpy(value).float().to(device)
        
        else:
            # repeatallitsotherkey (such as 'int_to_cell_type', 'true_num_points_list')
            prepped_dict[key] = value
            
    # ensure time_steps inlayer，evaluationfunctionneedit
    if 'time_steps' not in prepped_dict and 'time_steps' in data_dict:
         prepped_dict['time_steps'] = torch.from_numpy(data_dict['time_steps']).float().to(device)

    return prepped_dict

import joblib
if __name__ == "__main__":
    # --- 1. configurationandset ---
    EPOCHS = 1000
    # ✅ coremodify：we3Ddataonly2cell type
    NUM_CELL_TYPES_SIMULATED = 2
    NUM_gene = 5
    int_to_cell_type = {0: 'Left Hemisphere', 1: 'Right Hemisphere'}
    # --- 0. Configuration & Seeding ---
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # --- 2. generate3Dsimulatedata ---
    # print("--- Generating 3D Simulated Data ---")
    # ✅ coremodify：callnew3Ddatageneratefunction
    # data_list, type_list, gene_list, time_steps = generate_3d_ellipsoid_data_dynamic_types(num_time_steps=10)
    from preprocess import AnnDataProcessor, Config
    # AnnData = AnnDataProcessor()
    # dir = '/home/nas2/biod/chensishuo/axolotl_align2/'
    # fns = [
    #     '2dpi.h5ad', '5dpi.h5ad', '10dpi.h5ad',
    #      '15dpi.h5ad', '20dpi.h5ad', '30dpi.h5ad', '60dpi.h5ad',
    # ]
    # time_list = [2, 5, 10,
    #               15, 20, 30, 60
    #              ]
    # adata_list = AnnData.load_adata_list(adata_dir=dir, file_names=fns, cell_type='Annotation')
    # adata_list = AnnData.normalize_spatial(adata_list,
    #                                        center_to_origin=True,
    #                                        scaler_spatial_global=True,
    #                                        range_aixs_change=False,
    #                                        spatial_dim=2)
    # data_list, type_list, gene_list, time_steps, cell_type_to_int, int_to_cell_type = prepare_real_data(adata_list, time_list)
    # # gene_list_processed = preprocess_gene_data(gene_list)
    # # ✅ **revised 1: enablegenedatapreprocessing**
    # # originalbegin gene_list isoriginalbeginnumber，needperform log1p andstandardizationhandleaftertalentcaninputmodel
    # gene_list_processed = preprocess_gene_data(gene_list)
    # print('genehandlegood')
    # # data_list = preprocess_data_list(data_list)
    # print('positionplacehandlegood')

    # # ✅ **revised 2: dynamicgetmodelparameter**
    # # frompreprocessingafterdataindynamicgetcell typeandgenequantity，而notisusehardcodingsimulatedatavalue
    # num_cell_types = len(cell_type_to_int)
    # # gene_list_processed mayasempty，needallcheck
    
    data_list, type_list, gene_list, gene_list_processed, time_steps, cell_type_to_int, int_to_cell_type = load_processed_data("Ambystoma_2D.pt")
    num_genes = gene_list_processed[0].shape[1] if gene_list_processed and len(gene_list_processed) > 0 else 0
    num_cell_types = len(cell_type_to_int)

    print(f"dynamicgettoparameter: num_cell_types={num_cell_types}, num_genes={num_genes}")

    # --- 3. trainingmodel ---
    # ✅ **revised 3: update train_dynamic_point_cloud_model call**
    # spreadenterhandlegood type_list, gene_list_processed, toanddynamicget num_cell_types and num_genes
    # precomputed_plans = precompute_nn_transport_plans(
    #     pos_list=data_list,       # integertypelist (in CPU on)
    #     device=device
    # )
    
    model, feature, pos = train_dynamic_point_cloud_model(
        data_list=data_list, 
        time_steps=time_steps, 
        type_list=type_list,
        # precomputed_plans=precomputed_plans,
        gene_list_processed=gene_list_processed,
        num_cell_types=num_cell_types,
        num_genes=num_genes,
        epochs=EPOCHS, 
        device=device, 
        Exist_use=True
    )
    
    # (✅ modifyfilenameto防覆cover)
    joblib.dump(pos, './pos_real13.pkl')
    
    # --- 4. evaluationandvisualizationmodel (remains unchanged) ---
    evaluate_and_visualize_model(
        model=model, 
        fixed_sampled_features_list=feature, 
        data_list=data_list, 
        time_steps=time_steps,
        type_list=type_list,
        gene_list_processed=gene_list_processed, # <--- spreadenterhandleaftergenedataused forevaluation
        int_to_cell_type=int_to_cell_type,     # <--- spreadentermappingtoingraphexampleindisplaycorrectcellnamename
        device=device,
        # (✅ modifyfilenameto防覆cover)
        output_save_path="evaluation_outputs_sim_2d_real13.pt"
    )