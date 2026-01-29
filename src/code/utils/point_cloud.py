"""
Point cloud utility functions.

This module provides basic operations for point cloud processing including
farthest point sampling, point indexing, and grouping operations.
"""

import torch
import torch.nn.functional as F


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
