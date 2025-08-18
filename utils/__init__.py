"""
Utilities package for MentorPDM.

This package contains utility functions for metrics, losses, and other helper functions.
"""

from .metrics import (
    evaluate_model,
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    compute_per_class_metrics,
    classification_report_dict
)

from .losses import (
    ContrastiveLoss,
    CurriculumLoss,
    TrendedLineLoss,
    ModalityDiversityLoss,
    MultiModalContrastiveLoss,
    TripletLoss,
    FocalLoss,
    RMSELoss,
    WeightedMSELoss,
    create_loss_function
)

__all__ = [
    # Metrics
    "evaluate_model",
    "calculate_metrics", 
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "compute_per_class_metrics",
    "classification_report_dict",
    # Losses
    "ContrastiveLoss",
    "CurriculumLoss",
    "TrendedLineLoss",
    "ModalityDiversityLoss", 
    "MultiModalContrastiveLoss",
    "TripletLoss",
    "FocalLoss",
    "RMSELoss",
    "WeightedMSELoss",
    "create_loss_function"
]