"""
MentorPDM: Multi-modal Graph Neural Network for Predictive Maintenance

Official implementation for the KDD paper on multi-modal graph neural networks
for industrial equipment fault diagnosis and predictive maintenance.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

from . import data
from . import models
from . import training
from . import utils

__all__ = ["data", "models", "training", "utils"]