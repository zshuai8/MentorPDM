"""
Evaluation metrics and utilities for MentorPDM model.

This module provides comprehensive evaluation metrics including accuracy, precision,
recall, F1-score, confusion matrix, and classification reports for multi-class
classification tasks in predictive maintenance.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd


def calculate_accuracy(y_true: Union[np.ndarray, torch.Tensor], 
                      y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        float: Accuracy score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return accuracy_score(y_true, y_pred)


def calculate_precision(y_true: Union[np.ndarray, torch.Tensor], 
                       y_pred: Union[np.ndarray, torch.Tensor],
                       average: str = 'weighted') -> float:
    """
    Calculate precision score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', or None)
        
    Returns:
        float: Precision score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def calculate_recall(y_true: Union[np.ndarray, torch.Tensor], 
                    y_pred: Union[np.ndarray, torch.Tensor],
                    average: str = 'weighted') -> float:
    """
    Calculate recall score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', or None)
        
    Returns:
        float: Recall score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def calculate_f1_score(y_true: Union[np.ndarray, torch.Tensor], 
                      y_pred: Union[np.ndarray, torch.Tensor],
                      average: str = 'weighted') -> float:
    """
    Calculate F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', or None)
        
    Returns:
        float: F1 score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_confusion_matrix(y_true: Union[np.ndarray, torch.Tensor], 
                           y_pred: Union[np.ndarray, torch.Tensor],
                           labels: Optional[List] = None) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of label names
        
    Returns:
        np.ndarray: Confusion matrix
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return confusion_matrix(y_true, y_pred, labels=labels)


def generate_classification_report(y_true: Union[np.ndarray, torch.Tensor], 
                                 y_pred: Union[np.ndarray, torch.Tensor],
                                 target_names: Optional[List[str]] = None,
                                 output_dict: bool = False) -> Union[str, Dict]:
    """
    Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names
        output_dict: If True, return as dictionary instead of string
        
    Returns:
        Union[str, Dict]: Classification report
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return classification_report(
        y_true, y_pred, 
        target_names=target_names, 
        output_dict=output_dict,
        zero_division=0
    )


def compute_metrics(y_true: Union[np.ndarray, torch.Tensor], 
                   y_pred: Union[np.ndarray, torch.Tensor],
                   y_scores: Optional[Union[np.ndarray, torch.Tensor]] = None,
                   class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Optional prediction scores/probabilities for AUC calculation
        class_names: Optional list of class names
        
    Returns:
        Dict[str, float]: Dictionary containing all computed metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_scores is not None and isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = calculate_accuracy(y_true, y_pred)
    
    # Precision, Recall, F1 with different averaging methods
    for avg_method in ['micro', 'macro', 'weighted']:
        metrics[f'precision_{avg_method}'] = calculate_precision(y_true, y_pred, average=avg_method)
        metrics[f'recall_{avg_method}'] = calculate_recall(y_true, y_pred, average=avg_method)
        metrics[f'f1_{avg_method}'] = calculate_f1_score(y_true, y_pred, average=avg_method)
    
    # Per-class metrics
    precision_per_class = calculate_precision(y_true, y_pred, average=None)
    recall_per_class = calculate_recall(y_true, y_pred, average=None)
    f1_per_class = calculate_f1_score(y_true, y_pred, average=None)
    
    n_classes = len(np.unique(y_true))
    for i in range(n_classes):
        class_name = class_names[i] if class_names else f'class_{i}'
        metrics[f'precision_{class_name}'] = precision_per_class[i] if i < len(precision_per_class) else 0.0
        metrics[f'recall_{class_name}'] = recall_per_class[i] if i < len(recall_per_class) else 0.0
        metrics[f'f1_{class_name}'] = f1_per_class[i] if i < len(f1_per_class) else 0.0
    
    # AUC metrics if scores are provided
    if y_scores is not None:
        try:
            # For multi-class, compute AUC for each class vs rest
            if n_classes > 2:
                y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
                
                # Ensure y_scores has the right shape
                if y_scores.ndim == 1 or y_scores.shape[1] == 1:
                    # Binary classification case or single output
                    if n_classes == 2:
                        metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
                else:
                    # Multi-class case
                    try:
                        metrics['auc_roc_macro'] = roc_auc_score(
                            y_true_binarized, y_scores, average='macro', multi_class='ovr'
                        )
                        metrics['auc_roc_weighted'] = roc_auc_score(
                            y_true_binarized, y_scores, average='weighted', multi_class='ovr'
                        )
                    except ValueError:
                        # Fallback if issues with multi-class AUC
                        pass
            else:
                # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
                
        except (ValueError, IndexError) as e:
            # Skip AUC if calculation fails
            print(f"Warning: Could not calculate AUC metrics: {e}")
    
    return metrics


def compute_modality_specific_metrics(y_true_dict: Dict[str, Union[np.ndarray, torch.Tensor]], 
                                    y_pred_dict: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each modality separately.
    
    Args:
        y_true_dict: Dictionary mapping modality names to true labels
        y_pred_dict: Dictionary mapping modality names to predicted labels
        
    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with metrics for each modality
    """
    modality_metrics = {}
    
    for modality in y_true_dict.keys():
        if modality in y_pred_dict:
            modality_metrics[modality] = compute_metrics(
                y_true_dict[modality], 
                y_pred_dict[modality]
            )
    
    return modality_metrics


def plot_confusion_matrix(y_true: Union[np.ndarray, torch.Tensor], 
                         y_pred: Union[np.ndarray, torch.Tensor],
                         class_names: Optional[List[str]] = None,
                         normalize: bool = False,
                         title: str = 'Confusion Matrix',
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curves(y_true: Union[np.ndarray, torch.Tensor], 
                   y_scores: Union[np.ndarray, torch.Tensor],
                   class_names: Optional[List[str]] = None,
                   title: str = 'ROC Curves',
                   figsize: Tuple[int, int] = (10, 8),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores/probabilities
        class_names: Optional list of class names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    n_classes = len(np.unique(y_true))
    
    # Binarize labels for multi-class ROC
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        if n_classes == 2 and i == 1:
            # For binary classification, only plot one curve
            break
            
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1] if y_scores.ndim > 1 else y_scores)
            auc_score = auc(fpr, tpr)
            class_name = class_names[1] if class_names else 'Positive Class'
        else:
            # Multi-class classification
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
            auc_score = auc(fpr, tpr)
            class_name = class_names[i] if class_names else f'Class {i}'
        
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {auc_score:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curves(y_true: Union[np.ndarray, torch.Tensor], 
                                y_scores: Union[np.ndarray, torch.Tensor],
                                class_names: Optional[List[str]] = None,
                                title: str = 'Precision-Recall Curves',
                                figsize: Tuple[int, int] = (10, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Precision-Recall curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores/probabilities
        class_names: Optional list of class names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    n_classes = len(np.unique(y_true))
    
    # Binarize labels for multi-class PR curves
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        if n_classes == 2 and i == 1:
            # For binary classification, only plot one curve
            break
            
        if n_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(
                y_true, y_scores[:, 1] if y_scores.ndim > 1 else y_scores
            )
            auc_score = auc(recall, precision)
            class_name = class_names[1] if class_names else 'Positive Class'
        else:
            # Multi-class classification
            precision, recall, _ = precision_recall_curve(
                y_true_binarized[:, i], y_scores[:, i]
            )
            auc_score = auc(recall, precision)
            class_name = class_names[i] if class_names else f'Class {i}'
        
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{class_name} (AUC = {auc_score:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_metrics_summary_table(metrics_dict: Dict[str, float],
                                class_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create a summary table of metrics.
    
    Args:
        metrics_dict: Dictionary containing computed metrics
        class_names: Optional list of class names
        
    Returns:
        pd.DataFrame: Summary table of metrics
    """
    # Separate overall metrics from per-class metrics
    overall_metrics = {}
    per_class_metrics = {}
    
    for key, value in metrics_dict.items():
        if any(key.startswith(prefix) for prefix in ['precision_class_', 'recall_class_', 'f1_class_']):
            per_class_metrics[key] = value
        else:
            overall_metrics[key] = value
    
    # Create overall metrics DataFrame
    overall_df = pd.DataFrame([overall_metrics]).T
    overall_df.columns = ['Score']
    overall_df.index.name = 'Metric'
    
    # Create per-class metrics DataFrame if available
    if per_class_metrics and class_names:
        per_class_data = []
        for class_name in class_names:
            class_data = {}
            for metric_type in ['precision', 'recall', 'f1']:
                key = f'{metric_type}_class_{class_name}'
                if key in per_class_metrics:
                    class_data[metric_type] = per_class_metrics[key]
                else:
                    # Try with index-based naming
                    class_idx = class_names.index(class_name)
                    key_idx = f'{metric_type}_class_{class_idx}'
                    class_data[metric_type] = per_class_metrics.get(key_idx, 0.0)
            per_class_data.append(class_data)
        
        per_class_df = pd.DataFrame(per_class_data, index=class_names)
        per_class_df.index.name = 'Class'
        
        return overall_df, per_class_df
    
    return overall_df


def evaluate_model_performance(model, data_loader, device='cpu', 
                             class_names: Optional[List[str]] = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Comprehensive model performance evaluation.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        class_names: Optional list of class names
        
    Returns:
        Dict containing comprehensive evaluation results
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                inputs, labels = batch, None
            
            inputs = inputs.to(device)
            if labels is not None:
                labels = labels.to(device)
            
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take first output if tuple
            
            # Get predictions and scores
            if outputs.dim() > 1 and outputs.size(1) > 1:
                # Multi-class classification
                scores = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            else:
                # Binary classification
                scores = torch.sigmoid(outputs)
                predicted = (scores > 0.5).long()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            
            if labels is not None:
                all_labels.extend(labels.cpu().numpy())
    
    all_scores = np.vstack(all_scores)
    
    # Compute metrics
    if all_labels:
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        metrics = compute_metrics(all_labels, all_predictions, all_scores, class_names)
        
        # Add confusion matrix
        metrics['confusion_matrix'] = compute_confusion_matrix(all_labels, all_predictions)
        
        # Add classification report
        metrics['classification_report'] = generate_classification_report(
            all_labels, all_predictions, target_names=class_names, output_dict=True
        )
        
        return metrics
    else:
        return {
            'predictions': np.array(all_predictions),
            'scores': all_scores
        }


# Utility functions for backward compatibility with notebook code
def compute_accuracy_for_modality(y_true, y_pred):
    """Backward compatibility function for modality-specific accuracy calculation."""
    return calculate_accuracy(y_true, y_pred)


def compute_f1_micro_macro(y_true, y_pred):
    """Backward compatibility function for F1 micro and macro calculation."""
    f1_micro = calculate_f1_score(y_true, y_pred, average='micro')
    f1_macro = calculate_f1_score(y_true, y_pred, average='macro')
    return f1_micro, f1_macro