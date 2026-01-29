import torch
import torch.nn.functional as F
import numpy as np


def chamfer_distance_with_indices(p1, p2):
    """
    Calculates the Chamfer distance and returns nearest neighbor indices.

    Args:
        p1 (torch.Tensor): The first point cloud, shape (batch, num_points_1, 2).
        p2 (torch.Tensor): The second point cloud, shape (batch, num_points_2, 2).

    Returns:
        torch.Tensor: Chamfer distance.
        torch.Tensor: Indices of nearest neighbors from p1 to p2.
        torch.Tensor: Indices of nearest neighbors from p2 to p1.
    """
    dist_matrix = torch.cdist(p1, p2, p=2)
    device = p1.device
    min_dist_p1_to_p2, min_idx_p1_to_p2 = torch.min(dist_matrix, dim=2)
    min_dist_p2_to_p1, min_idx_p2_to_p1 = torch.min(dist_matrix, dim=1)
    beta = 0.5
    chamfer_loss = torch.mean(min_dist_p1_to_p2) * beta + torch.mean(min_dist_p2_to_p1) * (1-beta)
    return chamfer_loss*2, min_idx_p1_to_p2.to(device), min_idx_p2_to_p1.to(device)


def compute_repulsion_loss(points, k=5, radius=0.07):
    """
    Compute repulsion loss for point cloud (uniformity loss from Farthest Point Sampling paper)

    Args:
        points (torch.Tensor): Predicted point cloud, shape [B, N, C]
        k (int): Find k nearest neighbors
        radius (float): Expected minimum radius
    """
    B, N, C = points.shape
    if N < k + 1:
        return torch.tensor(0.0, device=points.device)

    dist_matrix = torch.cdist(points, points)
    knn_dists, _ = torch.topk(dist_matrix, k + 1, dim=-1, largest=False)
    knn_dists = knn_dists[..., 1:]
    knn_dists = torch.clamp(knn_dists, max=radius)
    repulsion_loss = torch.mean((radius - knn_dists) * (knn_dists < radius).float())

    return repulsion_loss


def compute_repulsion_loss_v2(points, k=5):
    """
    A simpler repulsion loss that penalizes the reciprocal of distance
    """
    B, N, C = points.shape
    if N < k + 1:
        return torch.tensor(0.0, device=points.device)

    dist_matrix = torch.cdist(points, points)
    knn_dists, _ = torch.topk(dist_matrix, k + 1, dim=-1, largest=False)
    knn_dists = knn_dists[..., 1:]
    loss_repulsion = torch.mean(1.0 / (knn_dists + 1e-8))

    return loss_repulsion


def compute_focal_loss(logits, labels, gamma=2.0, alpha_weights=None):
    """
    Computes multi-class Focal Loss.

    Args:
        logits (torch.Tensor): Model's raw logits output, shape [N, C].
        labels (torch.Tensor): Ground truth labels (integers), shape [N].
        gamma (float): Focusing parameter (gamma >= 0). When set to 0, equivalent to Cross-Entropy.
        alpha_weights (torch.Tensor, optional): Class balancing weights, shape [C].

    Returns:
        torch.Tensor: Average Focal Loss for the batch.
    """
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    p = F.softmax(logits, dim=-1)
    p_t = p.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    modulating_factor = (1 - p_t).pow(gamma)

    if alpha_weights is not None:
        if alpha_weights.device != logits.device:
            alpha_weights = alpha_weights.to(logits.device)
        alpha_t = alpha_weights.gather(dim=0, index=labels)
        focal_loss = alpha_t * modulating_factor * ce_loss
    else:
        focal_loss = modulating_factor * ce_loss

    return focal_loss.mean()


def trajectory_consistency_loss(
    true_pos,
    recon_pos,
    time_steps,
    true_num_points_list
):
    """
    Computes trajectory consistency loss, correctly handling variable point counts and padded data.
    """
    num_time_pairs = len(time_steps) - 1
    if num_time_pairs <= 0 or len(true_pos) == 0:
        device = recon_pos.device
        return torch.tensor(0.0, device=device)

    true_pos_new = []
    recon_pos_new = []

    for i in range(len(true_pos)):
        true_n = true_num_points_list[i]
        pos_true_slice = true_pos[i][:true_n].to(recon_pos.device)
        pos_recon_slice = recon_pos[i, :true_n]

        if true_n > 1:
            mean_true = torch.mean(pos_true_slice, dim=0, keepdim=True)
            std_true = torch.std(pos_true_slice, dim=0, keepdim=True) + 1e-8
            true_pos_new.append((pos_true_slice - mean_true) / std_true)

            mean_recon = torch.mean(pos_recon_slice, dim=0, keepdim=True)
            std_recon = torch.std(pos_recon_slice, dim=0, keepdim=True) + 1e-8
            recon_pos_new.append((pos_recon_slice - mean_recon) / std_recon)
        else:
            true_pos_new.append(pos_true_slice)
            recon_pos_new.append(pos_recon_slice)

    loss = 0.0

    for i in range(num_time_pairs):
        dt = time_steps[i+1] - time_steps[i]
        dt = max(dt, 1e-8)

        true_A, true_B = true_pos_new[i], true_pos_new[i+1]
        recon_A, recon_B = recon_pos_new[i], recon_pos_new[i+1]

        if true_A.shape[0] < 2 or true_B.shape[0] < 2 or recon_A.shape[0] < 2 or recon_B.shape[0] < 2:
            continue

        dist_true_AB = torch.cdist(true_A, true_B)
        _, nn_indices_true_AtoB = torch.min(dist_true_AB, dim=1)
        true_B_matched = true_B[nn_indices_true_AtoB]
        empirical_velocity_AB = (true_B_matched - true_A) / dt

        dist_model_AB = torch.cdist(recon_A, recon_B)
        _, nn_indices_model_AtoB = torch.min(dist_model_AB, dim=1)
        recon_B_matched = recon_B[nn_indices_model_AtoB]
        model_velocity_AB = (recon_B_matched - recon_A) / dt

        dist_A_clouds = torch.cdist(true_A, recon_A)

        _, nn_t2r_AB = torch.min(dist_A_clouds, dim=1)
        aligned_model_velocity_AB = model_velocity_AB[nn_t2r_AB]
        loss_t2r_AB = F.mse_loss(empirical_velocity_AB, aligned_model_velocity_AB)

        _, nn_r2t_AB = torch.min(dist_A_clouds, dim=0)
        aligned_empirical_velocity_AB = empirical_velocity_AB[nn_r2t_AB]
        loss_r2t_AB = F.mse_loss(model_velocity_AB, aligned_empirical_velocity_AB)

        loss_AB = (loss_t2r_AB + loss_r2t_AB)

        dist_true_BA = torch.cdist(true_B, true_A)
        _, nn_indices_true_BtoA = torch.min(dist_true_BA, dim=1)
        true_A_matched = true_A[nn_indices_true_BtoA]
        empirical_velocity_BA = (true_A_matched - true_B) / dt

        dist_model_BA = torch.cdist(recon_B, recon_A)
        _, nn_indices_model_BtoA = torch.min(dist_model_BA, dim=1)
        recon_A_matched = recon_A[nn_indices_model_BtoA]
        model_velocity_BA = (recon_A_matched - recon_B) / dt

        dist_B_clouds = torch.cdist(true_B, recon_B)

        _, nn_t2r_BA = torch.min(dist_B_clouds, dim=1)
        aligned_model_velocity_BA = model_velocity_BA[nn_t2r_BA]
        loss_t2r_BA = F.mse_loss(empirical_velocity_BA, aligned_model_velocity_BA)

        _, nn_r2t_BA = torch.min(dist_B_clouds, dim=0)
        aligned_empirical_velocity_BA = empirical_velocity_BA[nn_r2t_BA]
        loss_r2t_BA = F.mse_loss(model_velocity_BA, aligned_empirical_velocity_BA)

        loss_BA = (loss_t2r_BA + loss_r2t_BA)
        loss += (loss_AB + loss_BA) / 2.0

    if num_time_pairs > 0:
        return loss / num_time_pairs * 10.0
    else:
        return torch.tensor(0.0, device=recon_pos.device)


def fgw_trajectory_consistency_loss(
    true_pos_list,
    recon_pos,
    time_steps,
    precomputed_transport_plans,
    true_num_points_list,
    device='cpu'
    ):
    """
    Uses true (unpadded) point counts to correctly slice all tensors, computing symmetric FGW trajectory loss.
    """
    loss = 0.0
    num_time_pairs = len(time_steps) - 1
    if num_time_pairs <= 0:
        return torch.tensor(0.0, device=recon_pos.device)

    for i in range(num_time_pairs):
        true_n_A = true_num_points_list[i]
        true_n_B = true_num_points_list[i+1]

        P_tensor_full = precomputed_transport_plans[i]

        if P_tensor_full is None or P_tensor_full.shape[0] != true_n_A or P_tensor_full.shape[1] != true_n_B:
            continue

        dt = time_steps[i+1] - time_steps[i]

        true_A_pos = true_pos_list[i][:true_n_A].to(device)
        true_B_pos = true_pos_list[i+1][:true_n_B].to(device)

        recon_A = recon_pos[i, :true_n_A].to(device)
        recon_B = recon_pos[i+1, :true_n_B].to(device)

        P_tensor_AB = P_tensor_full
        T_AB = P_tensor_AB / P_tensor_AB.sum(dim=1, keepdim=True).clamp(min=1e-8)

        with torch.no_grad():
            _, nn_indices_true_AtoB = torch.max(T_AB, dim=1)
            expected_B_pos = true_B_pos[nn_indices_true_AtoB]
            empirical_velocity_AB = (expected_B_pos - true_A_pos) / dt

        model_velocity_AB = (recon_pos[i+1, :true_n_A] - recon_A) / dt

        dist_A_clouds = torch.cdist(true_A_pos, recon_A)
        _, nn_t2r_AB = torch.min(dist_A_clouds, dim=1)
        aligned_model_velocity_AB = model_velocity_AB[nn_t2r_AB]
        loss_t2r_AB = F.mse_loss(empirical_velocity_AB, aligned_model_velocity_AB)

        _, nn_r2t_AB = torch.min(dist_A_clouds, dim=0)
        aligned_empirical_velocity_AB = empirical_velocity_AB[nn_r2t_AB]
        loss_r2t_AB = F.mse_loss(model_velocity_AB, aligned_empirical_velocity_AB)

        loss_AB = (loss_t2r_AB + loss_r2t_AB)

        P_tensor_BA = P_tensor_full.T
        T_BA = P_tensor_BA / P_tensor_BA.sum(dim=1, keepdim=True).clamp(min=1e-8)

        with torch.no_grad():
            _, nn_indices_true_BtoA = torch.max(T_BA, dim=1)
            expected_A_pos = true_A_pos[nn_indices_true_BtoA]
            empirical_velocity_BA = (expected_A_pos - true_B_pos) / dt

        model_velocity_BA = (recon_pos[i, :true_n_B] - recon_B) / dt

        dist_B_clouds = torch.cdist(true_B_pos, recon_B)
        _, nn_t2r_BA = torch.min(dist_B_clouds, dim=1)
        aligned_model_velocity_BA = model_velocity_BA[nn_t2r_BA]
        loss_t2r_BA = F.mse_loss(empirical_velocity_BA, aligned_model_velocity_BA)

        _, nn_r2t_BA = torch.min(dist_B_clouds, dim=0)
        aligned_empirical_velocity_BA = empirical_velocity_BA[nn_r2t_BA]
        loss_r2t_BA = F.mse_loss(model_velocity_BA, aligned_empirical_velocity_BA)

        loss_BA = (loss_t2r_BA + loss_r2t_BA)
        loss += (loss_AB + loss_BA) / 2.0

    return loss / (num_time_pairs * 10) if num_time_pairs > 0 else torch.tensor(0.0, device=recon_pos.device)
