# eval_model.py
# Minimal function set extracted from flowclowd_model.py for evaluation
# (✅ Refactored: Improved robustness for Tensor and ndarray inputs)

import torch
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from geomloss import SamplesLoss
import torch.nn.functional as F  # <--- ✅ 1. Add this line
# ==============================================================================
# 1. Data loading and splitting functions (from flowclowd_model.py)
# ==============================================================================

import torch
import numpy as np # Ensure numpy is imported

def load_processed_data(path_or_data, generate_missing_features: bool = False):
    """
    (✅ Modified)
    Load preprocessed data dictionary from .pt file, or directly use an already loaded dictionary.
    Can now generate missing features on request.

    Args:
        path_or_data (str or dict):
            - (str): Path to the .pt file to load.
            - (dict): An already loaded data dictionary in the expected format.
        generate_missing_features (bool, optional):
            If True, will generate mock data for missing type_list (random 0/1)
            and gene_lists (using coordinates). Defaults to False.

    Returns:
        tuple: Contains the following items:
            (data_list, type_list, gene_list_raw, gene_list_processed,
             time_steps, cell_type_to_int, int_to_cell_type)

            If loading/input fails, all items will be None.
            If a key is missing from the file (and generation not requested), that item will be None.
    """
    print(f"Loading and unpacking data...")

    # --- (New) Internal helper function for normalizing coordinates ---
    def _normalize_pc(pc_list):
        """Normalize coordinate list using mean-centering"""
        normalized_list = []
        if pc_list is None:
            return None
        for pc in pc_list:
            if pc is None or pc.shape[0] == 0:
                normalized_list.append(pc)
                continue
            # Use mean-centering
            try:
                centroid = torch.mean(pc, dim=0, keepdim=True)
                normalized_list.append(pc - centroid)
            except Exception as e:
                print(f"  - Warning: Error normalizing point cloud: {e}")
                normalized_list.append(pc) # Return original point cloud on error
        return normalized_list

    # --- 1. Load data (no change) ---
    loaded_data = None

    if isinstance(path_or_data, str):
        file_path = path_or_data
        print(f"  - Input is a path. Attempting to load from {file_path}...")
        try:
            loaded_data = torch.load(file_path, map_location='cpu')
            print("  - Data loaded from file.")
        except FileNotFoundError:
            print(f"  - Error: File not found '{file_path}'.")
            return None, None, None, None, None, None, None
        except Exception as e:
            print(f"  - Error: Error loading file: {e}")
            return None, None, None, None, None, None, None

    elif isinstance(path_or_data, dict):
        print("  - Input is a dictionary. Using data directly...")
        loaded_data = path_or_data

    else:
        print(f"  - Error: Input must be a file path (str) or data dictionary (dict), but received {type(path_or_data)}.")
        return None, None, None, None, None, None, None

    # --- 2. Safely recover your variables from dictionary (no change) ---
    print("  - Unpacking dictionary keys...")
    data_list = loaded_data.get('data_list', None)
    type_list = loaded_data.get('type_list', None)
    gene_list_raw = loaded_data.get('gene_list_raw', None)
    gene_list_processed = loaded_data.get('gene_list_processed', None)
    time_steps = loaded_data.get('time_steps', None)
    cell_type_to_int = loaded_data.get('cell_type_to_int', None)
    int_to_cell_type = loaded_data.get('int_to_cell_type', None)

    # --- 3. (✅ New) Generate missing features on demand ---
    if generate_missing_features:
        print("\n--- Checking and generating missing features ---")

        # Check data_list, which is the basis for all generation
        if data_list is None:
            print("  - Warning: 'data_list' is None. Cannot generate any missing features.")
        else:

            # --- Condition 1: type_list is None ---
            if type_list is None:
                print("  - Generating (Mock) type_list (random 0s and 1s)...")
                type_list = []
                for pc in data_list:
                    if pc is None or pc.shape[0] == 0:
                        type_list.append(torch.tensor([], dtype=torch.long))
                    else:
                        num_points = pc.shape[0]
                        mock_types = torch.randint(0, 2, (num_points,), dtype=torch.long)
                        type_list.append(mock_types)

                # Also generate type mappings (if they don't exist either)
                if cell_type_to_int is None:
                    cell_type_to_int = {'Mock_Type_0': 0, 'Mock_Type_1': 1}
                    print("  - Generating (Mock) cell_type_to_int.")
                if int_to_cell_type is None:
                    int_to_cell_type = {0: 'Mock_Type_0', 1: 'Mock_Type_1'}
                    print("  - Generating (Mock) int_to_cell_type.")

            # --- Condition 2: gene_list_raw is None ---
            if gene_list_raw is None:
                print("  - Using (Mock) coordinates (data_list) as gene_list_raw...")
                gene_list_raw = data_list

            # --- Condition 3: gene_list_processed is None ---
            if gene_list_processed is None:
                print("  - Using (Mock) normalized coordinates as gene_list_processed...")
                gene_list_processed = _normalize_pc(data_list)

    # --- 4. Data validation (no change, number changed from 3 to 4) ---
    print("\n--- Data Validation ---")
    if time_steps is not None:
        print(f"  - Number of loaded time points: {len(time_steps)}")
    if data_list is not None and len(data_list) > 0 and data_list[0] is not None:
        print(f"  - Coordinate shape at first time point: {data_list[0].shape}")
    if gene_list_raw is not None and len(gene_list_raw) > 0 and gene_list_raw[0] is not None:
        print(f"  - Raw gene data shape at first time point: {gene_list_raw[0].shape}")
    if gene_list_processed is not None and len(gene_list_processed) > 0 and gene_list_processed[0] is not None:
        print(f"  - Processed gene data shape at first time point: {gene_list_processed[0].shape}")
    if int_to_cell_type is not None:
         print(f"  - Cell type mapping (example): {list(int_to_cell_type.items())[:3]}")

    # --- 5. Return unpacked tuple (no change, number changed from 4 to 5) ---
    return (data_list, type_list, gene_list_raw, gene_list_processed, 
            time_steps, cell_type_to_int, int_to_cell_type)

def split_data(loaded_data_all):
    """
    (✅ Modified)
    Split interleaved data dictionary [1.0, 1.5, 2.0, ...] into train and test set dictionaries.

    Args:
        loaded_data_all (dict): A loaded data dictionary containing all time points.

    Returns:
        tuple: (train_data, test_data)
            - train_data (dict): Data dictionary containing training points (0, 2, 4, ...).
            - test_data (dict): Data dictionary containing test points (1, 3, 5, ...).
    """
    print("Splitting data dictionary into train and test sets...")

    # --- 1. Unpack input dictionary ---
    data_list_all = loaded_data_all.get('data_list', None)
    type_list_all = loaded_data_all.get('type_list', None)
    gene_list_all = loaded_data_all.get('gene_list_raw', None)
    gene_list_proc_all = loaded_data_all.get('gene_list_processed', None)
    time_steps_all = loaded_data_all.get('time_steps', None)

    # (✅ New) Extract non-time-series metadata
    cell_type_to_int = loaded_data_all.get('cell_type_to_int', None)
    int_to_cell_type = loaded_data_all.get('int_to_cell_type', None)

    if time_steps_all is None:
        print("  - Error: No 'time_steps' in data dictionary. Cannot split.")
        return None, None

    # --- 2. Split logic (no change) ---
    num_all_points = len(time_steps_all)
    train_indices = list(range(0, num_all_points, 2))
    test_indices = list(range(1, num_all_points, 2))

    def _split_list(input_list, indices):
        if input_list is None:
            return None
        return [input_list[i] for i in indices]

    # --- 3. (✅ Modified) Create training set dictionary ---
    train_data = {
        "data_list": _split_list(data_list_all, train_indices),
        "type_list": _split_list(type_list_all, train_indices),
        "gene_list_raw": _split_list(gene_list_all, train_indices),
        "gene_list_processed": _split_list(gene_list_proc_all, train_indices),
        # (✅ Fixed) Ensure time_steps indexing is safe
        "time_steps": time_steps_all[train_indices] if hasattr(time_steps_all, '__getitem__') else None,

        # (✅ New) Add metadata back to dictionary
        "cell_type_to_int": cell_type_to_int,
        "int_to_cell_type": int_to_cell_type
    }

    # --- 4. (✅ Modified) Create test set dictionary ---
    test_data = {
        "data_list": _split_list(data_list_all, test_indices),
        "type_list": _split_list(type_list_all, test_indices),
        "gene_list_raw": _split_list(gene_list_all, test_indices),
        "gene_list_processed": _split_list(gene_list_proc_all, test_indices),
        "time_steps": time_steps_all[test_indices] if hasattr(time_steps_all, '__getitem__') else None,
        "cell_type_to_int": cell_type_to_int,
        "int_to_cell_type": int_to_cell_type
    }

    # --- 5. (✅ Fixed) Safe printing ---
    # Use _safe_to_numpy helper function (defined below) for safe printing
    ts_train_np = _safe_to_numpy(train_data['time_steps'])
    ts_test_np = _safe_to_numpy(test_data['time_steps'])

    print(f"  - Training set: {len(train_data['data_list'])} time points (t={ts_train_np})")
    print(f"  - Test set: {len(test_data['data_list'])} time points (t={ts_test_np})")
    
    return train_data, test_data


# ==============================================================================
# 2. Structured evaluation functions (from flowclowd_model.py)
# ==============================================================================

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
    # Calculate pairwise distances
    dist_matrix = torch.cdist(p1, p2, p=2)
    device = p1.device
    # p1 -> p2 (distance and index)
    min_dist_p1_to_p2, min_idx_p1_to_p2 = torch.min(dist_matrix, dim=2)

    # p2 -> p1 (distance and index)
    min_dist_p2_to_p1, min_idx_p2_to_p1 = torch.min(dist_matrix, dim=1)
    beta = 0.1
    # Chamfer distance is the sum of the mean of these two distances
    chamfer_loss = torch.mean(min_dist_p1_to_p2) * beta + torch.mean(min_dist_p2_to_p1) * (2-beta)

    return chamfer_loss, min_idx_p1_to_p2.to(device), min_idx_p2_to_p1.to(device)


# --- (✅ New) Robust type conversion helper functions ---

def _to_tensor(data, device):
    """
    Helper function: (no change)
    Safely convert None, np.ndarray, or torch.Tensor
    to torch.Tensor on the specified device.
    """
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        # If already a tensor, just move to correct device
        return data.to(device)
    if isinstance(data, np.ndarray):
        # If Numpy array, convert from numpy
        return torch.from_numpy(data).to(device)

    # As fallback, try converting other iterables (like lists)
    try:
        return torch.tensor(data).to(device)
    except Exception as e:
        print(f"Warning: Unable to convert data to tensor: {e}")
        return None

def _safe_to_numpy(data):
    """
    (✅ New)
    Helper function:
    Safely convert None, torch.Tensor, or np.ndarray
    to np.ndarray on CPU for comparison or printing.
    """
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    # Fallback for lists or other iterables
    try:
        return np.array(data)
    except Exception as e:
        print(f"Warning: Unable to convert data to numpy array: {e}")
        return None

def _get_list_as_tensors(data_list, num_timesteps, device):
    """
    (✅ New)
    Helper function:
    Get a list (possibly Tensors, ndarrays, or Nones),
    and return a list where each element is ensured to be on the specified device via _to_tensor.
    """
    if data_list is None:
        return [None] * num_timesteps
    # Use _to_tensor to process each item in the list
    return [_to_tensor(item, device) for item in data_list]

# --- (End of helper functions) ---


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
    Calculate all evaluation metrics for a single time point i. This is an internal helper function.
    (✅ Modified: Removed redundant .to(device) calls)
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
        # (✅ Fixed) If true point count is 0, check predicted point count
        if recon_existence_logits is not None:
             pred_existence_prob = torch.sigmoid(recon_existence_logits[i])
             pred_n = (pred_existence_prob > exist_threshold).sum().item()
             if pred_n == 0:
                 results['nP_MAE'] = 0.0 # Prediction correct
        return results # Otherwise return NaNs

    # --- 1. Get subset data ---
    # (✅ Fixed) Assume these tensors are already placed on 'device' in evaluate_test_set_structured
    true_points = true_pos_list[i][:true_n]

    # Predicted point cloud may have N_max points, we also only take first true_n points for comparison
    recon_points = recon_pos[i, :true_n]

    true_centroid = torch.mean(true_points, dim=0, keepdim=True)
    recon_centroid = torch.mean(recon_points, dim=0, keepdim=True)

    # 2. Subtract centroid from point clouds
    true_points = true_points - true_centroid
    recon_points = recon_points - recon_centroid

    # --- 2. Geometric fidelity (CD, OT) ---
    # CD Loss
    loss_cd_i, _, _ = chamfer_distance_with_indices(recon_points.unsqueeze(0), true_points.unsqueeze(0))
    results['CD_Loss'] = loss_cd_i.item()
    
    # OT Loss (Sinkhorn Distance)
    ot_distance = ot_loss_fn(recon_points, true_points).item()
    results['OT_Loss'] = ot_distance

    # --- 3. Attribute fidelity (CTA, Gene SCC) ---
    # Use NN matching for alignment
    dist_matrix = torch.cdist(recon_points, true_points)
    nearest_true_indices = torch.argmin(dist_matrix, dim=1) # [pred_n] -> [true_n] indices

    # CTA
    if recon_type_logits is not None and true_type_list is not None and true_type_list[i] is not None:
        # (✅ Fixed) Remove .to(device)
        true_types = true_type_list[i][:true_n]
        recon_types = torch.argmax(recon_type_logits[i, :true_n], dim=-1)

        aligned_true_types = true_types[nearest_true_indices]
        correct_matches = (recon_types == aligned_true_types).float().sum()
        accuracy = correct_matches / true_n
        results['CTA'] = accuracy.item()

    # Gene SCC
    if recon_gene is not None and true_gene_list_processed is not None and true_gene_list_processed[i] is not None:
        # (✅ Fixed) Remove .to(device)
        true_genes = true_gene_list_processed[i][:true_n]
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

    # --- 4. Population dynamics (P-MAE) ---
    if recon_existence_logits is not None:
        pred_existence_prob = torch.sigmoid(recon_existence_logits[i])
        pred_n = (pred_existence_prob > exist_threshold).sum().item()
        results['nP_MAE'] = abs(pred_n - true_n)/true_n
    
    return results


def evaluate_test_set_structured(
    true_data_dict,          # True value Loaded_data format: dict, keys: ['data_list', 'type_list', 'gene_list_processed', 'time_steps']
    pred_data_dict,          # Predicted value Loaded_data format: dict, keys: ['positions', 'type_logits', 'gene_expressions', 'time_steps', 'Exit']
    train_time_steps,        # Training set time points (for interpolation/extrapolation determination)
    device='cpu'
):
    """
    (✅ Core modification)
    Calculate all evaluation metrics on given true and predicted data dictionaries, and return structured DataFrame.
    Can now robustly handle Tensor or ndarray inputs.
    """
    print("\n--- Starting Structured Evaluation ---")

    # --- 1. Data preparation (✅ Fixed: Use helper functions for robustness) ---

    # (✅ Fixed) Use _safe_to_numpy for comparison
    true_ts_np = _safe_to_numpy(true_data_dict.get('time_steps'))
    pred_ts_np = _safe_to_numpy(pred_data_dict.get('time_steps'))

    if true_ts_np is None or pred_ts_np is None:
        print("Error: True or Predicted time steps are None. Aborting evaluation.")
        return pd.DataFrame()
        
    if not np.allclose(true_ts_np, pred_ts_np):
        print("Error: True and Predicted time steps do not match. Aborting evaluation.")
        return pd.DataFrame()

    # (✅ Fixed) Use _to_tensor to replace torch.from_numpy (fixes TypeError)
    time_steps_eval = _to_tensor(true_ts_np, device)
    num_timesteps = len(time_steps_eval)

    # (✅ Fixed) Use _get_list_as_tensors to ensure list data is on 'device'
    true_pos_list = _get_list_as_tensors(true_data_dict.get('data_list'), num_timesteps, device)
    true_type_list = _get_list_as_tensors(true_data_dict.get('type_list'), num_timesteps, device)
    true_gene_list_processed = _get_list_as_tensors(true_data_dict.get('gene_list_processed'), num_timesteps, device)

    # Recover tensors from prediction dictionary
    recon_pos = _to_tensor(pred_data_dict.get('positions'), device)
    recon_type_logits = _to_tensor(pred_data_dict.get('type_logits'), device)
    recon_gene = _to_tensor(pred_data_dict.get('gene_expressions'), device)
    recon_existence_logits = _to_tensor(pred_data_dict.get('Exit'), device)

    # # Check if predicted point cloud dimension is 2D
    # if recon_pos is not None and recon_pos.dim() == 3 and recon_pos.shape[2] > 2:
    #     print(f"  Warning: Predicted positions are 3D ({recon_pos.shape[2]}D), will only take first 2 dimensions (X, Y) during evaluation.")
    #     recon_pos = recon_pos[:, :, :2]

    # (✅ Fixed) Calculate point count after checking None
    true_num_points_list = [p.shape[0] if p is not None else 0 for p in true_pos_list]

    # --- 2. Core fix: Get training set time range ---
    # (✅ Fixed) Use _to_tensor to ensure train_time_steps is a tensor
    train_time_steps_tensor = _to_tensor(train_time_steps, 'cpu').float()

    min_train_time = train_time_steps_tensor.min().item()
    max_train_time = train_time_steps_tensor.max().item()

    # --- 3. Evaluation loop ---
    ot_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="tensorized")
    all_results = []
    
    with torch.no_grad():
        for i in range(num_timesteps):
            time_point = time_steps_eval[i].item()

            # ✅ Fixed interpolation determination logic: time point falls within [min_train_time, max_train_time] interval
            is_interpolation = (time_point >= min_train_time) and (time_point <= max_train_time)

            # Calculate all metrics
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

            # Add time point and classification information
            metrics['Time'] = time_point
            metrics['Is_Interpolation'] = is_interpolation

            all_results.append(metrics)

    # --- 4. Build DataFrame ---
    df = pd.DataFrame(all_results).set_index('Time')
    print("--- Structured Evaluation Complete ---")
    return df


def evaluate_interpolation_structured(true_data_dict, pred_data_dict, train_time_steps, device='cpu'):
    """
    Perform structured evaluation on **interpolation time points** (i.e., within training set time range) in predictions.
    """
    df_all = evaluate_test_set_structured(true_data_dict, pred_data_dict, train_time_steps, device)
    if df_all.empty:
        return pd.DataFrame()
    df_interp = df_all[df_all['Is_Interpolation'] == True].drop(columns=['Is_Interpolation'])
    print(f"Interpolation Evaluation Done. Results for {len(df_interp)} time points.")
    return df_interp


def evaluate_extrapolation_structured(true_data_dict, pred_data_dict, train_time_steps, device='cpu'):
    """
    Perform structured evaluation on **extrapolation time points** (i.e., outside training set time range) in predictions.
    """
    df_all = evaluate_test_set_structured(true_data_dict, pred_data_dict, train_time_steps, device)
    if df_all.empty:
        return pd.DataFrame()
    df_extrap = df_all[df_all['Is_Interpolation'] == False].drop(columns=['Is_Interpolation'])
    print(f"Extrapolation Evaluation Done. Results for {len(df_extrap)} time points.")
    return df_extrap