"""Single project-owned training, loss and evaluation implementation."""

from .evaluator import EvaluationResult, evaluate
from .losses import masked_score_aligned_hybrid
from .metrics import compute_metrics
from .trainer import TrainResult, Trainer

__all__ = [
    "EvaluationResult",
    "TrainResult",
    "Trainer",
    "compute_metrics",
    "evaluate",
    "masked_score_aligned_hybrid",
]
