#!/usr/bin/env python3
"""
Training script for MentorPDM model.

This script implements the complete training pipeline for the MentorPDM model,
including contrastive learning pretraining and curriculum learning fine-tuning.
"""

import os
import sys
import argparse
import logging
import json
import copy
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from models.mentorpdm import (
    MentorPDM, 
    MentorPDMClassifier, 
    ContrastiveLoss, 
    ContrastiveDataset,
    create_mentorpdm_model,
    create_mentorpdm_classifier
)
from utils.metrics import compute_metrics, calculate_accuracy
from utils.losses import contrastive_loss_function
from data.data_loader import load_data, prepare_graph_data


def setup_logging(log_dir, log_level='INFO'):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint['epoch'], checkpoint['loss']


def contrastive_pretraining(model, train_data, args, logger):
    """
    Perform contrastive learning pretraining.
    
    Args:
        model: MentorPDM model
        train_data: Training data
        args: Training arguments
        logger: Logger instance
    """
    logger.info("Starting contrastive pretraining...")
    
    # Setup contrastive learning
    contrastive_loss = ContrastiveLoss(margin=args.contrastive_margin)
    contrastive_dataset = ContrastiveDataset(
        data_len=len(train_data),
        positive_margin_range=args.positive_margin_range,
        negative_margin_range=args.negative_margin_range
    )
    
    contrastive_loader = DataLoader(
        contrastive_dataset, 
        batch_size=args.contrastive_batch_size, 
        shuffle=True
    )
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.contrastive_lr, 
        weight_decay=args.weight_decay
    )
    
    model.train()
    training_losses = []
    
    for epoch in range(args.contrastive_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(contrastive_loader, desc=f"Contrastive Epoch {epoch+1}/{args.contrastive_epochs}")
        
        for batch_idx, (anchor_idx, positive_idx, negative_idx) in enumerate(pbar):
            optimizer.zero_grad()
            
            # Get embeddings for anchor, positive, and negative samples
            anchor_views, anchor_edges = train_data[anchor_idx]
            positive_views, positive_edges = train_data[positive_idx] 
            negative_views, negative_edges = train_data[negative_idx]
            
            # Move to device
            if torch.cuda.is_available():
                anchor_views = [v.cuda() for v in anchor_views]
                anchor_edges = [e.cuda() for e in anchor_edges]
                positive_views = [v.cuda() for v in positive_views]
                positive_edges = [e.cuda() for e in positive_edges]
                negative_views = [v.cuda() for v in negative_views]
                negative_edges = [e.cuda() for e in negative_edges]
            
            # Forward pass
            anchor_embeddings, anchor_diff_loss = model(anchor_views, anchor_edges, pretrain=True)
            positive_embeddings, _ = model(positive_views, positive_edges, pretrain=True)
            negative_embeddings, _ = model(negative_views, negative_edges, pretrain=True)
            
            # Calculate contrastive losses
            positive_loss = contrastive_loss(
                anchor_embeddings, positive_embeddings, 
                torch.zeros(anchor_embeddings.size(0)).cuda() if torch.cuda.is_available() 
                else torch.zeros(anchor_embeddings.size(0))
            )
            
            negative_loss = contrastive_loss(
                anchor_embeddings, negative_embeddings,
                torch.ones(anchor_embeddings.size(0)).cuda() if torch.cuda.is_available()
                else torch.ones(anchor_embeddings.size(0))
            )
            
            # Total loss includes contrastive loss and difference loss for modality diversity
            total_loss = positive_loss + negative_loss + args.diff_loss_weight * anchor_diff_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
        
        avg_epoch_loss = epoch_loss / num_batches
        training_losses.append(avg_epoch_loss)
        
        logger.info(f"Contrastive Epoch {epoch+1}: Average Loss = {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_epoch_loss,
                args.checkpoint_dir, is_best=False
            )
    
    logger.info("Contrastive pretraining completed.")
    return training_losses


def curriculum_training(model, train_data, train_labels, val_data, val_labels, args, logger):
    """
    Perform curriculum learning training with classification head.
    
    Args:
        model: Pretrained MentorPDM model
        train_data: Training data
        train_labels: Training labels
        val_data: Validation data  
        val_labels: Validation labels
        args: Training arguments
        logger: Logger instance
    """
    logger.info("Starting curriculum learning training...")
    
    # Create classifier with pretrained backbone
    classifier = create_mentorpdm_classifier(
        gnn_model=copy.deepcopy(model),
        output_dim=args.num_classes
    )
    
    if torch.cuda.is_available():
        classifier = classifier.cuda()
    
    optimizer = optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    
    # Curriculum learning coefficients
    coef1 = args.curriculum_coef1_start  # lambda_1
    coef2 = args.curriculum_coef2_start  # lambda_2
    
    for epoch in range(args.num_epochs):
        # Training phase
        classifier.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(
            range(len(train_data)), 
            desc=f"Training Epoch {epoch+1}/{args.num_epochs}"
        )
        
        for batch_idx in pbar:
            optimizer.zero_grad()
            
            # Get data
            views, edges = train_data[batch_idx]
            labels = train_labels[batch_idx]
            
            # Move to device
            if torch.cuda.is_available():
                views = [v.cuda() for v in views]
                edges = [e.cuda() for e in edges]
                labels = labels.cuda()
            
            # Forward pass with curriculum learning
            predictions, weighted_loss = classifier(views, edges, labels, coef2, coef1)
            
            # Backward pass
            weighted_loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip)
                
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            accuracy = calculate_accuracy(predicted.cpu(), labels.cpu())
            
            epoch_loss += weighted_loss.item()
            epoch_acc += accuracy
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{weighted_loss.item():.4f}',
                'Acc': f'{accuracy:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_acc / num_batches
        training_losses.append(avg_train_loss)
        training_accuracies.append(avg_train_acc)
        
        # Validation phase
        val_loss, val_acc, val_metrics = evaluate_model(
            classifier, val_data, val_labels, args, logger
        )
        validation_losses.append(val_loss)
        validation_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update curriculum coefficients
        if epoch % args.curriculum_update_freq == 0:
            coef1 = max(coef1 - args.curriculum_decay_rate, args.curriculum_coef1_min)
            coef2 = max(coef2 - args.curriculum_decay_rate, args.curriculum_coef2_min)
        
        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_checkpoint(
                classifier, optimizer, epoch, val_loss,
                args.checkpoint_dir, is_best=is_best
            )
        
        logger.info(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
            f"Curriculum: λ1={coef1:.3f}, λ2={coef2:.3f}"
        )
        
        # Early stopping
        if args.early_stopping > 0:
            if len(validation_losses) > args.early_stopping:
                if all(validation_losses[-i] >= validation_losses[-args.early_stopping-1] 
                       for i in range(1, args.early_stopping + 1)):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    return {
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'training_accuracies': training_accuracies,
        'validation_accuracies': validation_accuracies,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss
    }


def evaluate_model(model, data, labels, args, logger):
    """Evaluate model on given data."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(data)):
            views, edges = data[i]
            target = labels[i]
            
            if torch.cuda.is_available():
                views = [v.cuda() for v in views]
                edges = [e.cuda() for e in edges]
                target = target.cuda()
            
            # Forward pass
            if isinstance(model, MentorPDMClassifier):
                predictions, loss = model(views, edges, target)
            else:
                embeddings = model(views, edges, pretrain=False)
                # Simple classification for base model
                predictions = torch.softmax(embeddings.mean(dim=0), dim=-1)
                loss = nn.CrossEntropyLoss()(predictions.unsqueeze(0), target.unsqueeze(0))
            
            total_loss += loss.item()
            
            # Store predictions and labels
            _, predicted = torch.max(predictions.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(data)
    accuracy = calculate_accuracy(np.array(all_predictions), np.array(all_labels))
    
    # Compute additional metrics
    metrics = compute_metrics(np.array(all_labels), np.array(all_predictions))
    
    return avg_loss, accuracy, metrics


def plot_training_curves(results, save_dir):
    """Plot and save training curves."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['training_losses'], label='Training Loss')
    plt.plot(results['validation_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['training_accuracies'], label='Training Accuracy')
    plt.plot(results['validation_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train MentorPDM model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of output classes')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_modality', type=int, default=6,
                        help='Number of modalities/views')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    # Contrastive learning arguments
    parser.add_argument('--use_contrastive', action='store_true',
                        help='Use contrastive pretraining')
    parser.add_argument('--contrastive_epochs', type=int, default=100,
                        help='Number of contrastive pretraining epochs')
    parser.add_argument('--contrastive_lr', type=float, default=0.001,
                        help='Contrastive learning rate')
    parser.add_argument('--contrastive_batch_size', type=int, default=128,
                        help='Contrastive learning batch size')
    parser.add_argument('--contrastive_margin', type=float, default=2.0,
                        help='Contrastive loss margin')
    parser.add_argument('--positive_margin_range', type=int, nargs=2, default=[1, 5],
                        help='Positive margin range for contrastive learning')
    parser.add_argument('--negative_margin_range', type=int, nargs=2, default=[10, 50],
                        help='Negative margin range for contrastive learning')
    parser.add_argument('--diff_loss_weight', type=float, default=0.1,
                        help='Weight for modality difference loss')
    
    # Curriculum learning arguments
    parser.add_argument('--curriculum_coef1_start', type=float, default=1.0,
                        help='Initial curriculum coefficient 1 (lambda_1)')
    parser.add_argument('--curriculum_coef2_start', type=float, default=1.0,
                        help='Initial curriculum coefficient 2 (lambda_2)')
    parser.add_argument('--curriculum_coef1_min', type=float, default=0.1,
                        help='Minimum curriculum coefficient 1')
    parser.add_argument('--curriculum_coef2_min', type=float, default=0.1,
                        help='Minimum curriculum coefficient 2')
    parser.add_argument('--curriculum_decay_rate', type=float, default=0.01,
                        help='Curriculum coefficient decay rate')
    parser.add_argument('--curriculum_update_freq', type=int, default=10,
                        help='Frequency of curriculum coefficient updates')
    
    # Training control arguments
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold (0 to disable)')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frequency of saving checkpoints')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir / 'logs', args.log_level)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("Starting MentorPDM training...")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load data
        logger.info("Loading data...")
        train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data(
            args.data_dir, train_split=0.7, val_split=0.15
        )
        
        # Prepare graph data
        train_data = prepare_graph_data(train_data)
        val_data = prepare_graph_data(val_data) 
        test_data = prepare_graph_data(test_data)
        
        logger.info(f"Data loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Create model
        input_dims = [train_data[0][0][i].shape[-1] for i in range(len(train_data[0][0]))]
        logger.info(f"Input dimensions: {input_dims}")
        
        model = create_mentorpdm_model(
            input_dims=input_dims,
            hidden_dim=args.hidden_dim,
            output_dim=args.num_classes,
            num_modality=args.num_modality,
            num_heads=args.num_heads
        )
        
        if device == 'cuda':
            model = model.cuda()
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Contrastive pretraining
        if args.use_contrastive:
            contrastive_losses = contrastive_pretraining(model, train_data, args, logger)
        
        # Curriculum learning training
        results = curriculum_training(
            model, train_data, train_labels, val_data, val_labels, args, logger
        )
        
        # Plot training curves
        plot_training_curves(results, output_dir)
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        best_model_path = Path(args.checkpoint_dir) / 'best_model.pt'
        if best_model_path.exists():
            classifier = create_mentorpdm_classifier(
                gnn_model=model, output_dim=args.num_classes
            )
            if device == 'cuda':
                classifier = classifier.cuda()
            load_checkpoint(best_model_path, classifier)
            
            test_loss, test_acc, test_metrics = evaluate_model(
                classifier, test_data, test_labels, args, logger
            )
            
            logger.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
            logger.info(f"Test Metrics: {test_metrics}")
            
            # Save test results
            results['test_loss'] = test_loss
            results['test_accuracy'] = test_acc
            results['test_metrics'] = test_metrics
        
        # Save final results
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()