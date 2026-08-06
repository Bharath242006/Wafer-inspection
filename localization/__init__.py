"""
localization package initialization.
"""

from localization.candidate_generation import (
    extract_local_peaks,
    generate_candidate_pool_multi,
    rank_top500_candidates,
)
from localization.coarse_localization import locate_global_coarse
from localization.fine_localization import refine_subpixel_peak
from localization.inference import WaferLocalizerInference
from localization.visualization import draw_prediction_visualization

from localization.coordinate_aware_ranker import (
    CoordinateAwareRanker,
    load_coordinate_aware_ranker,
    compute_coordinate_ranker_scores,
)
from localization.context_aware_ranker_v2 import (
    ContextAwareRankerV2,
    load_context_aware_ranker_v2,
    compute_context_aware_v2_scores,
)

__all__ = [
    "extract_local_peaks",
    "generate_candidate_pool_multi",
    "rank_top500_candidates",
    "locate_global_coarse",
    "refine_subpixel_peak",
    "WaferLocalizerInference",
    "draw_prediction_visualization",
    "CoordinateAwareRanker",
    "load_coordinate_aware_ranker",
    "compute_coordinate_ranker_scores",
    "ContextAwareRankerV2",
    "load_context_aware_ranker_v2",
    "compute_context_aware_v2_scores",
]


