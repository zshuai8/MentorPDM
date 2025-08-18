"""
MentorPDM: Multi-head Multi-view Graph Neural Network for Predictive Maintenance

This module contains the core MentorPDM model and associated loss functions for 
predictive maintenance using multi-modal sensor data with graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.utils.data import Dataset
import random


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning discriminative representations."""
    
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with positive and negative pairs."""
    
    def __init__(self, data_len, positive_margin_range, negative_margin_range):
        self.data_len = data_len
        self.positive_margin_range = positive_margin_range
        self.negative_margin_range = negative_margin_range

    def __len__(self):
        return self.data_len  # Assumes all views have the same length

    def __getitem__(self, index):
        # For anchor, just return the index as is
        anchor_index = index
        
        # Get a positive example index
        positive_margin = random.randint(*self.positive_margin_range)
        positive_index = index + positive_margin
        if positive_index >= self.data_len:
            positive_index = index - positive_margin

        # Get a negative example index
        negative_margin = random.randint(*self.negative_margin_range)
        negative_index = index + negative_margin
        if negative_index >= self.data_len:
            negative_index = index - negative_margin

        return anchor_index, positive_index, negative_index


class MentorPDM(torch.nn.Module):
    """
    MentorPDM: Multi-head Multi-view Graph Neural Network for Predictive Maintenance
    
    This model processes multiple modalities of sensor data using graph neural networks
    with attention mechanisms and contrastive learning for robust feature extraction.
    
    Args:
        input_dims (list): List of input dimensions for each modality
        hidden_dim (int): Hidden dimension size
        output_dim (int): Output dimension size
        num_modality (int): Number of modalities/views
        num_heads (int): Number of attention heads
    """
    
    def __init__(self, input_dims, hidden_dim, output_dim, num_modality, num_heads):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_modality = num_modality
        self.num_heads = num_heads
        
        # Learnable weights for modality fusion
        self.view_weights = nn.Parameter(torch.ones(self.num_modality, dtype=torch.float)/self.num_modality)
        
        # Linear projection layers for each modality and head
        self.load = torch.nn.ModuleList([
            torch.nn.ModuleList([
                nn.Linear(input_dims[i], hidden_dim)
                for j in range(num_heads)
            ])
            for i in range(num_modality)
        ])
        
        # Graph convolution layers for each modality and head
        self.gcn_convs = torch.nn.ModuleList([
            torch.nn.ModuleList([
                SAGEConv(input_dims[i], hidden_dim)
                for j in range(num_heads)
            ])
            for i in range(num_modality)
        ])
        
    def node_loss(self, pred, labels):
        """Standard cross-entropy loss for node classification."""
        loss = F.cross_entropy(pred, labels)
        return loss
    
    def trended_line_loss_pytorch(self, y_true, y_pred, lambda_penalty):
        """
        Custom loss function that penalizes abrupt changes in predictions,
        encouraging smooth temporal transitions.
        """
        # Calculate the original mean squared error loss
        mse_loss = F.mse_loss(y_true, y_pred, reduction='none').sum(dim=0)

        # Calculate the differences between consecutive predicted values
        differences = torch.abs(y_pred[1:] - y_pred[:-1])

        # Sum the absolute differences and scale by the penalty hyperparameter
        penalty = lambda_penalty * differences.sum(dim=0)

        # Calculate the total loss
        total_loss = mse_loss + penalty

        return total_loss
    
    def curriculum_regularizer(self, weight, lambda_1=1, lambda_2=1):
        """Regularization term for curriculum learning."""
        return 1/2*lambda_2 *weight**2 - (lambda_1 + lambda_2)*weight

    def rmse_loss(self, y_pred, y_true):
        """Root Mean Square Error loss."""
        mse = torch.mean((y_pred - y_true) ** 2)
        rmse = torch.sqrt(mse)
        return rmse
    
    def difference_loss(self, representations):
        """
        Encourages diversity among different modality representations
        by penalizing similarity between modalities.
        """
        loss = 0.0
        for i in range(self.num_modality):
            for j in range(i + 1, self.num_modality):
                loss += torch.norm(representations[i] - representations[j], p=2)
        return loss
    
    def forward(self, x_views, edge_indices, use_cl=True, pretrain=True):
        """
        Forward pass of the MentorPDM model.
        
        Args:
            x_views (list): List of input tensors, one for each modality
            edge_indices (list): List of edge indices for each modality
            use_cl (bool): Whether to use contrastive learning (currently unused)
            pretrain (bool): Whether to return pretraining loss
            
        Returns:
            If pretrain=True: (summed_hidden_views, pretrain_loss)
            If pretrain=False: summed_hidden_views
        """
        # Apply GCN layers to each modality for each head
        hidden_views = []
        weighted_hidden_views = []
        
        for i in range(self.num_modality):
            hidden_heads = []
            for j in range(self.num_heads):
                # Apply graph convolution
                hidden_head = self.gcn_convs[i][j](x_views[i], edge_indices[i])
                hidden_head = F.relu(hidden_head)
                hidden_heads.append(hidden_head)
            
            # Get modality-specific weight
            weights = self.view_weights[i]
            
            # Concatenate heads for this modality
            hidden_heads = torch.cat(hidden_heads, dim=-1)
            hidden_views.append(hidden_heads)
            weighted_hidden_views.append(hidden_heads * weights)
            
        # Stack and sum weighted representations
        stacked_hidden_views = torch.stack(weighted_hidden_views)
        summed_hidden_views = torch.sum(stacked_hidden_views, dim=0)
        
        if pretrain:
            # Calculate pretraining loss to encourage modality diversity
            pretrain_loss = self.difference_loss(hidden_views)
            return summed_hidden_views, pretrain_loss
        
        return summed_hidden_views


class MentorPDMClassifier(nn.Module):
    """
    Classifier wrapper for MentorPDM with curriculum learning capabilities.
    
    This class wraps the MentorPDM model and adds a classification head
    with curriculum learning for improved training dynamics.
    """
    
    def __init__(self, gnn_model, output_dim):
        super(MentorPDMClassifier, self).__init__()
        # Use provided model or create a new one
        if isinstance(gnn_model, MentorPDM):
            self.gnn_model = gnn_model
        else:
            # Create a default model configuration
            self.gnn_model = MentorPDM(
                input_dims=[128], 
                hidden_dim=128, 
                output_dim=3, 
                num_modality=1, 
                num_heads=4
            )
        
        # Calculate output dimension from GNN model
        out_dim = self.gnn_model.hidden_dim * self.gnn_model.num_heads
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        
    def curriculum(self, loss, lambda_1=1, lambda_2=1):
        """
        Curriculum learning weight calculation.
        
        Assigns higher weights to easier samples (lower loss) early in training,
        gradually incorporating harder samples.
        """
        if lambda_2 == 0:
            return loss <= lambda_1
        else:
            find_max = torch.max(
                torch.stack([torch.zeros_like(loss), 1 - (loss - lambda_1)/lambda_2]), 
                dim=0
            ).values
            find_min = torch.min(
                torch.stack([torch.ones_like(loss), find_max]), 
                dim=0
            ).values
            return find_min
    
    def curriculum_regularizer(self, weights, lambda_1=1, lambda_2=1):
        """Regularization term for curriculum weights."""
        regularized_weights = 1/2 * lambda_2 * weights ** 2 - (lambda_1 + lambda_2) * weights
        return regularized_weights

    def forward(self, x_views, edge_indices, target, lambda_1=1, lambda_2=1):
        """
        Forward pass with curriculum learning.
        
        Args:
            x_views (list): Input features for each modality
            edge_indices (list): Edge indices for each modality
            target (torch.Tensor): Target labels
            lambda_1 (float): Curriculum parameter 1
            lambda_2 (float): Curriculum parameter 2
            
        Returns:
            tuple: (predictions, weighted_loss)
        """
        # Get embeddings from the GNN model
        embeddings = self.gnn_model(x_views, edge_indices, pretrain=False)
        
        # Get predictions from classifier
        output = self.classifier(embeddings).squeeze()
        
        # Calculate individual losses for curriculum learning
        ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
        individual_losses = ce_loss_fn(output, target.long())
        
        # Apply curriculum learning
        sample_weights = self.curriculum(individual_losses, lambda_1, lambda_2)
        regularized_weights = self.curriculum_regularizer(sample_weights, lambda_1, lambda_2)
        
        # Calculate final weighted loss
        weighted_loss = torch.mean(sample_weights * individual_losses) + torch.mean(regularized_weights)
        
        return output, weighted_loss


def normalize_tensor_along_first_dim(tensor):
    """
    Normalize tensor along the first dimension (batch dimension).
    
    Args:
        tensor (torch.Tensor): Input tensor to normalize
        
    Returns:
        torch.Tensor: Normalized tensor
    """
    # Compute the mean and standard deviation along the first dimension
    mean = torch.mean(tensor, dim=0, keepdim=True)
    std = torch.std(tensor, dim=0, keepdim=True)
    
    # Normalize the tensor
    normalized_tensor = (tensor - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
    
    return normalized_tensor


# Factory function for easy model creation
def create_mentorpdm_model(input_dims, hidden_dim=128, output_dim=3, num_modality=1, num_heads=4):
    """
    Factory function to create a MentorPDM model with standard configuration.
    
    Args:
        input_dims (list): Input dimensions for each modality
        hidden_dim (int): Hidden dimension size
        output_dim (int): Output dimension size  
        num_modality (int): Number of modalities
        num_heads (int): Number of attention heads
        
    Returns:
        MentorPDM: Configured model instance
    """
    return MentorPDM(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_modality=num_modality,
        num_heads=num_heads
    )


def create_mentorpdm_classifier(gnn_model=None, output_dim=3, input_dims=None, **kwargs):
    """
    Factory function to create a MentorPDM classifier.
    
    Args:
        gnn_model (MentorPDM, optional): Pre-trained GNN model
        output_dim (int): Number of output classes
        input_dims (list, optional): Input dimensions if creating new model
        **kwargs: Additional arguments for model creation
        
    Returns:
        MentorPDMClassifier: Configured classifier instance
    """
    if gnn_model is None and input_dims is not None:
        gnn_model = create_mentorpdm_model(input_dims, **kwargs)
    
    return MentorPDMClassifier(gnn_model, output_dim)