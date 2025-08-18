"""
Loss functions for MentorPDM model.

This module contains custom loss functions including contrastive loss,
curriculum learning losses, and specialized losses for multi-modal
predictive maintenance tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative representations in embedding space.
    
    This loss encourages similar samples to be close together and dissimilar samples
    to be far apart in the learned embedding space.
    
    Args:
        margin (float): Margin for negative pairs in contrastive loss
    """
    
    def __init__(self, margin: float = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two embeddings.
        
        Args:
            output1: First embedding tensor
            output2: Second embedding tensor  
            label: Binary labels (0 for similar pairs, 1 for dissimilar pairs)
            
        Returns:
            torch.Tensor: Contrastive loss value
        """
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


class TripletLoss(nn.Module):
    """
    Triplet loss for learning embeddings with anchor, positive, and negative examples.
    
    Args:
        margin (float): Margin for triplet loss
        p (int): Power for distance calculation (default: 2 for Euclidean)
    """
    
    def __init__(self, margin: float = 1.0, p: int = 2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.p = p
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings
            
        Returns:
            torch.Tensor: Triplet loss value
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=self.p)
        neg_dist = F.pairwise_distance(anchor, negative, p=self.p)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class TrendedLineLoss(nn.Module):
    """
    Custom loss that penalizes abrupt changes in predictions to encourage
    smooth temporal transitions in time series predictions.
    
    Args:
        lambda_penalty (float): Weight for the smoothness penalty term
    """
    
    def __init__(self, lambda_penalty: float = 0.1):
        super(TrendedLineLoss, self).__init__()
        self.lambda_penalty = lambda_penalty
    
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute trended line loss with smoothness penalty.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            torch.Tensor: Combined MSE and smoothness penalty loss
        """
        # Calculate the original mean squared error loss
        mse_loss = F.mse_loss(y_true, y_pred, reduction='none').sum(dim=0)

        # Calculate the differences between consecutive predicted values
        if y_pred.size(0) > 1:
            differences = torch.abs(y_pred[1:] - y_pred[:-1])
            # Sum the absolute differences and scale by the penalty hyperparameter
            penalty = self.lambda_penalty * differences.sum(dim=0)
        else:
            penalty = torch.tensor(0.0, device=y_pred.device)

        # Calculate the total loss
        total_loss = mse_loss + penalty
        return total_loss.mean()


class RMSELoss(nn.Module):
    """Root Mean Square Error loss."""
    
    def __init__(self):
        super(RMSELoss, self).__init__()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSE loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            torch.Tensor: RMSE loss value
        """
        mse = torch.mean((y_pred - y_true) ** 2)
        rmse = torch.sqrt(mse + 1e-8)  # Add epsilon for numerical stability
        return rmse


class ModalityDiversityLoss(nn.Module):
    """
    Loss that encourages diversity among different modality representations
    by penalizing similarity between modalities.
    
    Args:
        num_modalities (int): Number of modalities
        p (int): Power for distance calculation (default: 2 for Euclidean)
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, num_modalities: int, p: int = 2, reduction: str = 'mean'):
        super(ModalityDiversityLoss, self).__init__()
        self.num_modalities = num_modalities
        self.p = p
        self.reduction = reduction
    
    def forward(self, representations: list) -> torch.Tensor:
        """
        Compute modality diversity loss.
        
        Args:
            representations: List of modality representations
            
        Returns:
            torch.Tensor: Diversity loss encouraging modality differences
        """
        if len(representations) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} representations, got {len(representations)}")
        
        loss = 0.0
        num_pairs = 0
        
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                # Calculate negative distance to encourage diversity
                distance = torch.norm(representations[i] - representations[j], p=self.p)
                loss += 1.0 / (distance + 1e-8)  # Inverse distance to encourage diversity
                num_pairs += 1
        
        if self.reduction == 'mean':
            return loss / num_pairs
        elif self.reduction == 'sum':
            return loss
        else:
            return loss


class CurriculumLoss(nn.Module):
    """
    Curriculum learning loss that adaptively weights samples based on their difficulty.
    Easier samples (lower loss) get higher weights early in training.
    
    Args:
        base_loss_fn: Base loss function to apply curriculum learning to
        lambda_1 (float): Curriculum parameter 1
        lambda_2 (float): Curriculum parameter 2
        regularize_weights (bool): Whether to apply regularization to curriculum weights
    """
    
    def __init__(self, base_loss_fn: nn.Module, lambda_1: float = 1.0, 
                 lambda_2: float = 1.0, regularize_weights: bool = True):
        super(CurriculumLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.regularize_weights = regularize_weights
    
    def curriculum_weights(self, individual_losses: torch.Tensor) -> torch.Tensor:
        """
        Compute curriculum weights based on individual sample losses.
        
        Args:
            individual_losses: Loss for each sample in the batch
            
        Returns:
            torch.Tensor: Curriculum weights for each sample
        """
        if self.lambda_2 == 0:
            return (individual_losses <= self.lambda_1).float()
        else:
            # Smooth curriculum weight calculation
            find_max = torch.max(
                torch.stack([
                    torch.zeros_like(individual_losses), 
                    1 - (individual_losses - self.lambda_1) / self.lambda_2
                ]), 
                dim=0
            ).values
            weights = torch.min(
                torch.stack([torch.ones_like(individual_losses), find_max]), 
                dim=0
            ).values
            return weights
    
    def curriculum_regularizer(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply regularization to curriculum weights.
        
        Args:
            weights: Curriculum weights
            
        Returns:
            torch.Tensor: Regularization term
        """
        return 0.5 * self.lambda_2 * weights ** 2 - (self.lambda_1 + self.lambda_2) * weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute curriculum-weighted loss.
        
        Args:
            predictions: Model predictions
            targets: Target labels/values
            
        Returns:
            torch.Tensor: Curriculum-weighted loss
        """
        # Calculate individual losses
        if hasattr(self.base_loss_fn, 'reduction'):
            original_reduction = self.base_loss_fn.reduction
            self.base_loss_fn.reduction = 'none'
            individual_losses = self.base_loss_fn(predictions, targets)
            self.base_loss_fn.reduction = original_reduction
        else:
            # For custom loss functions that don't have reduction parameter
            individual_losses = self.base_loss_fn(predictions, targets)
            if individual_losses.dim() == 0:
                # If scalar loss, expand to batch size
                individual_losses = individual_losses.expand(predictions.size(0))
        
        # Compute curriculum weights
        sample_weights = self.curriculum_weights(individual_losses)
        
        # Apply curriculum weighting
        weighted_loss = torch.mean(sample_weights * individual_losses)
        
        # Add regularization if enabled
        if self.regularize_weights:
            regularized_weights = self.curriculum_regularizer(sample_weights)
            weighted_loss += torch.mean(regularized_weights)
        
        return weighted_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by down-weighting easy examples.
    
    Args:
        alpha (float or Tensor): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha: Union[float, torch.Tensor] = 1.0, 
                 gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Target labels
            
        Returns:
            torch.Tensor: Focal loss value
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiModalContrastiveLoss(nn.Module):
    """
    Multi-modal contrastive loss that encourages alignment between different modalities
    while maintaining discriminative power within each modality.
    
    Args:
        temperature (float): Temperature parameter for contrastive learning
        modal_weight (float): Weight for inter-modal contrastive loss
    """
    
    def __init__(self, temperature: float = 0.07, modal_weight: float = 1.0):
        super(MultiModalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.modal_weight = modal_weight
    
    def forward(self, modal_embeddings: list, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-modal contrastive loss.
        
        Args:
            modal_embeddings: List of embeddings from different modalities
            labels: Class labels for contrastive learning
            
        Returns:
            torch.Tensor: Multi-modal contrastive loss
        """
        total_loss = 0.0
        num_modalities = len(modal_embeddings)
        
        # Intra-modal contrastive loss
        for embeddings in modal_embeddings:
            loss = self._compute_contrastive_loss(embeddings, labels)
            total_loss += loss
        
        # Inter-modal alignment loss
        if num_modalities > 1:
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    alignment_loss = self._compute_alignment_loss(
                        modal_embeddings[i], modal_embeddings[j], labels
                    )
                    total_loss += self.modal_weight * alignment_loss
        
        return total_loss / (num_modalities + num_modalities * (num_modalities - 1) / 2)
    
    def _compute_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for a single modality."""
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-similarity
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mean log-likelihood of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        
        loss = -mean_log_prob_pos.mean()
        return loss
    
    def _compute_alignment_loss(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                              labels: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss between two modalities."""
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)
        
        # Compute cross-modal similarity
        similarity = torch.sum(embeddings1 * embeddings2, dim=1) / self.temperature
        
        # Encourage alignment of same-class samples
        alignment_loss = -similarity.mean()
        
        return alignment_loss


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss with per-sample weighting.
    
    Args:
        weights (Tensor, optional): Per-sample weights
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            sample_weights: Optional per-sample weights
            
        Returns:
            torch.Tensor: Weighted MSE loss
        """
        mse = (predictions - targets) ** 2
        
        # Apply weights
        weights = sample_weights if sample_weights is not None else self.weights
        if weights is not None:
            mse = mse * weights.view(-1, 1) if mse.dim() > 1 else mse * weights
        
        if self.reduction == 'mean':
            return mse.mean()
        elif self.reduction == 'sum':
            return mse.sum()
        else:
            return mse


# Factory functions for easy loss creation
def create_contrastive_loss(margin: float = 2.0) -> ContrastiveLoss:
    """Create contrastive loss with specified margin."""
    return ContrastiveLoss(margin=margin)


def create_curriculum_loss(base_loss: str = 'ce', lambda_1: float = 1.0, 
                          lambda_2: float = 1.0, **kwargs) -> CurriculumLoss:
    """
    Create curriculum loss with specified base loss function.
    
    Args:
        base_loss: Base loss type ('ce', 'mse', 'focal')
        lambda_1: Curriculum parameter 1
        lambda_2: Curriculum parameter 2
        **kwargs: Additional arguments for base loss
        
    Returns:
        CurriculumLoss: Configured curriculum loss
    """
    if base_loss == 'ce':
        base_loss_fn = nn.CrossEntropyLoss(**kwargs)
    elif base_loss == 'mse':
        base_loss_fn = nn.MSELoss(**kwargs)
    elif base_loss == 'focal':
        base_loss_fn = FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported base loss type: {base_loss}")
    
    return CurriculumLoss(base_loss_fn, lambda_1, lambda_2)


def create_multimodal_loss(temperature: float = 0.07, modal_weight: float = 1.0) -> MultiModalContrastiveLoss:
    """Create multi-modal contrastive loss."""
    return MultiModalContrastiveLoss(temperature=temperature, modal_weight=modal_weight)


# Utility functions for backward compatibility with notebook code
def contrastive_loss_function(output1: torch.Tensor, output2: torch.Tensor, 
                            label: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """Backward compatibility function for contrastive loss calculation."""
    loss_fn = ContrastiveLoss(margin=margin)
    return loss_fn(output1, output2, label)


def trended_line_loss_pytorch(y_true: torch.Tensor, y_pred: torch.Tensor, 
                             lambda_penalty: float = 0.1) -> torch.Tensor:
    """Backward compatibility function for trended line loss calculation."""
    loss_fn = TrendedLineLoss(lambda_penalty=lambda_penalty)
    return loss_fn(y_true, y_pred)


def rmse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Backward compatibility function for RMSE loss calculation."""
    loss_fn = RMSELoss()
    return loss_fn(y_pred, y_true)


def difference_loss(representations: list, num_modality: int) -> torch.Tensor:
    """Backward compatibility function for modality difference loss calculation."""
    loss_fn = ModalityDiversityLoss(num_modality)
    return loss_fn(representations)