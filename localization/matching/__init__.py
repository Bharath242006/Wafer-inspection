"""
localization matching subpackage initialization.
"""

from localization.matching.fft_matching import fft_phase_correlation_score
from localization.matching.graph_matching import compute_graph_similarity
from localization.matching.template_matching import compute_local_variance_map, zmuv_ncc
from localization.matching.attention_matching import compute_attention_weighted_matching
from localization.matching.similarity import normalize_zscore_tanh

__all__ = [
    "fft_phase_correlation_score",
    "compute_graph_similarity",
    "compute_local_variance_map",
    "zmuv_ncc",
    "compute_attention_weighted_matching",
    "normalize_zscore_tanh",
]
