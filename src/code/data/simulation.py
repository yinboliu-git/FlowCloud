"""
Simulation data generation functions for FlowCloud model.
Extracted from flowcloud_model.py for modularity.
"""

import torch
import numpy as np

# --- Simulation data hyperparameters ---
NUM_POINTS_GENERATE = 1500  # Target number of generated points "center value"


def _generate_data_for_time(t, num_time_steps, base_vertices, x_shift_factor):
    """
    Helper function for generating data at a single time point t.
    Contains the internal logic of the original loop body.
    """
    # --- Geometric shape evolution ---
    triangle_scale = 1.0 + 1.5 * np.sin(np.pi * (t - 1) / (num_time_steps - 1))
    hole_radius = 0.1 + (0.9 * (t - 1) / (num_time_steps - 1))

    # --- Calculate target point count ---
    mid_point = (num_time_steps + 1) / 2.0
    peak_count = NUM_POINTS_GENERATE
    start_end_ratio = 0.4
    a = (start_end_ratio * peak_count - peak_count) / (1 - mid_point)**2
    target_count = int(a * (t - mid_point)**2 + peak_count)
    target_count = max(0, target_count)

    # --- Generate and filter candidate points ---
    num_candidates = target_count * 5
    if num_candidates <= 0:
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
        t_norm = (t - 1) / max(1, num_time_steps - 1)

        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        rate_g1 = 10.0 * x_norm
        gene_expressions[:, 0] = np.random.poisson(rate_g1)

        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        rate_g2 = 10.0 * y_norm
        gene_expressions[:, 1] = np.random.poisson(rate_g2)

        dist_center = np.linalg.norm(final_points, axis=1)
        dist_norm = (dist_center - dist_center.min()) / (dist_center.max() - dist_center.min() + 1e-8)
        rate_g3 = 10.0 * dist_norm
        gene_expressions[:, 2] = np.random.poisson(rate_g3)

        rate_g4 = np.zeros(num_final_points)
        rate_g4[final_types == 0] = 12.0
        rate_g4[final_types != 0] = 1.0
        gene_expressions[:, 3] = np.random.poisson(rate_g4)

        rate_g5 = np.zeros(num_final_points)
        rate_g5[final_types == 1] = 15.0 * t_norm
        gene_expressions[:, 4] = np.random.poisson(np.maximum(0, rate_g5))

    # --- Add noise and store ---
    noisy_points = final_points_shifted + np.random.normal(0, 0.05, final_points_shifted.shape)

    point_groups_t = torch.tensor(noisy_points, dtype=torch.float32)
    type_groups_t = torch.tensor(final_types, dtype=torch.long)
    gene_groups_t = torch.tensor(gene_expressions, dtype=torch.float32)

    return (point_groups_t, type_groups_t, gene_groups_t, t)


def generate_spatial_data_with_genes(num_time_steps=10):
    """
    Generates a time series of 2D point clouds with interleaved training points (t=1, 2, ...)
    and test points (t=1.5, ..., 10.5).

    Returns:
        point_groups: List of point cloud tensors
        type_groups: List of cell type tensors
        gene_groups: List of gene expression tensors
        time_steps: Tensor of time values
        int_to_cell_type: Dict mapping int to cell type name
    """
    print(f"Generating simulated data with deterministic count change and translation...")
    print(f"  - Mode: Generating {num_time_steps} 'training' points (1..10) and {num_time_steps} 'test' points (1.5..10.5).")

    # --- 1. Basic setup ---
    base_vertices = np.array([[0.0, 2.0], [-1.732, -1.0], [1.732, -1.0]])
    int_to_cell_type = {0: 'Type_0', 1: 'Type_1', 2: 'Type_2'}
    x_shift_factor = 2.0

    all_data_tuples = []

    # --- 2. Loop 1: Generate training set data ---
    print("  - Generating 'training' data points (t=1, 2, 3, ...)")
    train_time_steps_np = np.arange(1, num_time_steps + 1)

    for t in train_time_steps_np:
        data_tuple = _generate_data_for_time(t, num_time_steps, base_vertices, x_shift_factor)
        all_data_tuples.append(data_tuple)

    # --- 3. Loop 2: Generate test set data (including 10.5) ---
    print("  - Generating 'test' data points (t=1.5, 2.5, ..., 10.5)")
    test_time_steps_np = np.arange(1.5, num_time_steps + 1.0, 1.0)

    for t in test_time_steps_np:
        data_tuple = _generate_data_for_time(t, num_time_steps, base_vertices, x_shift_factor)
        all_data_tuples.append(data_tuple)

    # --- 4. Sort and unpack ---
    print(f"  - Sorting all {len(all_data_tuples)} data points by time...")

    all_data_tuples.sort(key=lambda x: x[3])

    point_groups = list(map(lambda x: x[0], all_data_tuples))
    type_groups = list(map(lambda x: x[1], all_data_tuples))
    gene_groups = list(map(lambda x: x[2], all_data_tuples))
    all_times_np = np.array(list(map(lambda x: x[3], all_data_tuples)))

    print("Data generation complete.")
    print(f"Total of {len(point_groups)} time points generated.")
    print("Number of generated points per time point:", [p.shape[0] for p in point_groups])
    print("Generated time points:", all_times_np)

    return point_groups, type_groups, gene_groups, torch.tensor(all_times_np, dtype=torch.float32), int_to_cell_type


def generate_3d_ellipsoid_data_dynamic_types(num_time_steps=10, num_points_peak=4096):
    """
    Generates a time series of 3D point clouds with dynamically changing cell type regions.
    - Shape: An ellipsoid.
    - Hollowing: A fixed-size cube removes some points from the ellipsoid.
    - Dynamics: The ellipsoid grows then shrinks, and translates along the X-axis.
    - Cell types: Two types (left and right hemispheres), with boundary moving left and right over time.

    Returns:
        point_groups: List of point cloud tensors
        type_groups: List of cell type tensors
        gene_groups: List of gene expression tensors
        time_steps: Tensor of time values
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

        # --- Dynamic cell type boundary ---
        boundary_shift_amplitude = 0.8
        boundary_x = boundary_shift_amplitude * np.sin(2 * np.pi * (t - 1) / (num_time_steps - 1))

        # --- Generate point cloud ---
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
            final_types[final_points[:, 0] > boundary_x] = 1

        # --- Simulate gene expression (based on pre-translation coordinates) ---
        gene_expressions = np.zeros((num_final_points, 5), dtype=np.float32)
        if num_final_points > 0:
            x, y, z = final_points[:, 0], final_points[:, 1], final_points[:, 2]
            gene_expressions[:, 0] = np.random.poisson(np.maximum(0, 5 * (x / current_radii[0])))
            dist_from_center = np.linalg.norm(final_points, axis=1)
            gene_expressions[:, 1] = np.random.poisson(8 * np.exp(-dist_from_center**2))
            rate_g3 = np.zeros(num_final_points)
            rate_g3[final_types == 0] = 12.0
            rate_g3[final_types != 0] = 1.0
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
