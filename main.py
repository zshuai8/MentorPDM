#!/usr/bin/env python3
"""
MentorPDM: Multi-modal Graph Neural Network for Predictive Maintenance

Main script for running experiments with the MentorPDM model.
This script provides a unified interface for training, evaluation, and inference.

Usage:
    python main.py --mode train --config configs/default_config.yaml
    python main.py --mode evaluate --model_path checkpoints/best_model.pth
    python main.py --mode inference --model_path checkpoints/best_model.pth --data_path data/test.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.data_loader import get_df_all, BearingDataLoader
from data.preprocessing import create_pytorch_geometric_data, FE_comprehensive
from models.mentorpdm import MentorPDM, create_mentorpdm_model
from training.train_mentorpdm import MentorPDMTrainer
from utils.metrics import evaluate_model, plot_confusion_matrix, calculate_metrics
from utils.losses import create_loss_function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mentorpdm.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def setup_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories for outputs."""
    directories = [
        config['model']['save_dir'],
        config['logging']['log_dir'],
        config['output']['results_dir'],
        config['output']['plots_dir']
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created/verified: {directory}")


def load_data(config: Dict[str, Any]) -> tuple:
    """Load and preprocess data according to configuration."""
    logger.info("Loading and preprocessing data...")
    
    data_config = config['data']
    
    # Load bearing data
    df_processed, df_raw = get_df_all(
        data_path=data_config['data_path'],
        data_cat=data_config['data_category'],
        segment_length=data_config['segment_length'],
        normalize=data_config['normalize']
    )
    
    logger.info(f"Loaded {len(df_processed)} processed samples")
    logger.info(f"Raw data shape: {df_raw.shape if df_raw is not None else 'N/A'}")
    
    # Create train/validation/test splits
    data_loader = BearingDataLoader(
        segment_length=data_config['segment_length'],
        overlap=data_config.get('overlap', 0.5)
    )
    
    train_data, val_data, test_data = data_loader.create_splits(
        df_processed, 
        test_size=data_config.get('test_size', 0.2),
        val_size=data_config.get('val_size', 0.1),
        random_state=config.get('seed', 42)
    )
    
    # Extract features if needed
    if data_config.get('extract_features', True):
        logger.info("Extracting comprehensive features...")
        train_features = FE_comprehensive(train_data)
        val_features = FE_comprehensive(val_data)
        test_features = FE_comprehensive(test_data)
    else:
        train_features = train_data
        val_features = val_data
        test_features = test_data
    
    # Create PyTorch Geometric data
    train_graph = create_pytorch_geometric_data(
        train_features, 
        train_data['label'], 
        k=config['model'].get('k_neighbors', 8)
    )
    val_graph = create_pytorch_geometric_data(
        val_features, 
        val_data['label'], 
        k=config['model'].get('k_neighbors', 8)
    )
    test_graph = create_pytorch_geometric_data(
        test_features, 
        test_data['label'], 
        k=config['model'].get('k_neighbors', 8)
    )
    
    return train_graph, val_graph, test_graph


def create_model(config: Dict[str, Any], input_dims: list) -> MentorPDM:
    """Create MentorPDM model according to configuration."""
    model_config = config['model']
    
    model = create_mentorpdm_model(
        input_dims=input_dims,
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        num_modality=model_config.get('num_modality', 1),
        num_heads=model_config.get('num_heads', 4),
        dropout=model_config.get('dropout', 0.1),
        use_curriculum=model_config.get('use_curriculum', True)
    )
    
    logger.info(f"Created MentorPDM model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def train_mode(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Training mode."""
    logger.info("Starting training mode...")
    
    # Load data
    train_data, val_data, test_data = load_data(config)
    
    # Determine input dimensions
    if hasattr(train_data, 'x'):
        input_dims = [train_data.x.shape[-1]]
    else:
        input_dims = [train_data[0].x.shape[-1] for _ in range(config['model'].get('num_modality', 1))]
    
    # Create model
    model = create_model(config, input_dims)
    
    # Create trainer
    trainer = MentorPDMTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
        device=torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    )
    
    # Train model
    trainer.train()
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_test()
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save final results
    results_path = Path(config['output']['results_dir']) / 'final_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(test_metrics, f, default_flow_style=False)
    logger.info(f"Results saved to {results_path}")


def evaluate_mode(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Evaluation mode."""
    logger.info("Starting evaluation mode...")
    
    if not args.model_path or not os.path.exists(args.model_path):
        logger.error("Model path not found. Please provide a valid --model_path")
        sys.exit(1)
    
    # Load data
    _, _, test_data = load_data(config)
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location='cpu')
    input_dims = checkpoint.get('input_dims', [128])  # Default fallback
    
    model = create_model(config, input_dims)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    model = model.to(device)
    
    logger.info(f"Model loaded from {args.model_path}")
    
    # Evaluate model
    metrics = evaluate_model(model, test_data, device)
    
    logger.info("Evaluation Results:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Plot confusion matrix
    plot_path = Path(config['output']['plots_dir']) / 'confusion_matrix_eval.png'
    plot_confusion_matrix(
        metrics['y_true'], 
        metrics['y_pred'], 
        save_path=str(plot_path)
    )
    logger.info(f"Confusion matrix saved to {plot_path}")


def inference_mode(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Inference mode."""
    logger.info("Starting inference mode...")
    
    if not args.model_path or not os.path.exists(args.model_path):
        logger.error("Model path not found. Please provide a valid --model_path")
        sys.exit(1)
    
    if not args.data_path or not os.path.exists(args.data_path):
        logger.error("Data path not found. Please provide a valid --data_path")
        sys.exit(1)
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location='cpu')
    input_dims = checkpoint.get('input_dims', [128])
    
    model = create_model(config, input_dims)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    model = model.to(device)
    
    # Load and process data
    # This is a simplified example - in practice, you'd want more sophisticated data loading
    data = pd.read_csv(args.data_path)
    logger.info(f"Loaded data with shape: {data.shape}")
    
    # Process data (this would depend on your specific data format)
    # For now, we'll assume the data is already preprocessed
    features = FE_comprehensive(data)
    graph_data = create_pytorch_geometric_data(
        features, 
        labels=None,  # No labels for inference
        k=config['model'].get('k_neighbors', 8)
    )
    
    # Run inference
    with torch.no_grad():
        graph_data = graph_data.to(device)
        predictions = model(graph_data.x, graph_data.edge_index)
        predictions = torch.softmax(predictions, dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)
    
    # Save predictions
    results_df = pd.DataFrame({
        'sample_id': range(len(predicted_classes)),
        'predicted_class': predicted_classes.cpu().numpy(),
        'confidence': torch.max(predictions, dim=1)[0].cpu().numpy()
    })
    
    output_path = Path(config['output']['results_dir']) / 'inference_results.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"Inference results saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='MentorPDM: Multi-modal Graph Neural Network for Predictive Maintenance')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'inference'], 
                       required=True, help='Running mode')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, help='Path to saved model (for evaluate/inference modes)')
    parser.add_argument('--data_path', type=str, help='Path to data file (for inference mode)')
    parser.add_argument('--gpu', type=int, help='GPU ID to use (overrides config)')
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.gpu is not None:
        config['gpu_id'] = args.gpu
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup directories
    setup_directories(config)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(yaml.dump(config, default_flow_style=False))
    
    # Run appropriate mode
    try:
        if args.mode == 'train':
            train_mode(config, args)
        elif args.mode == 'evaluate':
            evaluate_mode(config, args)
        elif args.mode == 'inference':
            inference_mode(config, args)
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {str(e)}")
        raise
    
    logger.info(f"{args.mode.capitalize()} mode completed successfully!")


if __name__ == '__main__':
    main()