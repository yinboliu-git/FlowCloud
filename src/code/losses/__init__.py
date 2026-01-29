"""Loss functions for FlowCloud."""

from flowcloud.losses.spatial import (
    chamfer_distance_with_indices,
    compute_repulsion_loss,
    compute_repulsion_loss_v2,
)

from flowcloud.losses.trajectory import (
    trajectory_consistency_loss,
    fgw_trajectory_consistency_loss,
    clamp_P,
    iter_calculate_correspondence_P,
    calculate_correspondence_P,
)

from flowcloud.losses.classification import (
    compute_focal_loss,
)

__all__ = [
    # Spatial losses
    'chamfer_distance_with_indices',
    'compute_repulsion_loss',
    'compute_repulsion_loss_v2',
    # Trajectory losses
    'trajectory_consistency_loss',
    'fgw_trajectory_consistency_loss',
    'clamp_P',
    'iter_calculate_correspondence_P',
    'calculate_correspondence_P',
    # Classification losses
    'compute_focal_loss',
]
