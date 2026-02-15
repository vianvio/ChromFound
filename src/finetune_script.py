"""
Utility functions for fine-tuning ChromFound model with LoRA.
"""

import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from typing import Dict, Any

from src.models.finetune_model import create_finetune_model_from_pretrained, count_parameters
from src.data.dataset_ds import DatasetMultiPad

logging.basicConfig(level=logging.INFO)


def setup_finetune_parser():
    """Setup argument parser for fine-tuning"""
    parser = argparse.ArgumentParser(description="Fine-tune ChromFound model with LoRA")
    
    # Required paths
    parser.add_argument('--checkpoint_dir', type=str, required=True, 
                       help='Directory containing pretrained model files')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to training data (.h5ad file)')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory to save fine-tuned model')
    
    # LoRA parameters
    parser.add_argument('--lora_rank', type=int, default=8, 
                       help='Rank of LoRA adaptation (default: 8)')
    parser.add_argument('--lora_alpha', type=float, default=16.0, 
                       help='Alpha parameter for LoRA scaling (default: 16.0)')
    parser.add_argument('--lora_dropout', type=float, default=0.0, 
                       help='Dropout rate for LoRA layers (default: 0.0)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate for fine-tuning (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--warmup_epochs', type=int, default=1, 
                       help='Number of warmup epochs (default: 1)')
    
    # Task parameters
    parser.add_argument('--task_type', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Type of downstream task (default: classification)')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes for classification (default: 2)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training (default: cuda if available else cpu)')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='GPU rank for distributed training (default: 0)')
    
    return parser


def train_epoch(model, dataloader, optimizer, criterion, device, epoch_num):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num}")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # Unpack batch data - adjust based on your dataset format
        if len(batch_data) >= 5:
            value, chromosome, pos_start, pos_end, labels = batch_data[:5]
        else:
            # Adjust unpacking based on actual dataset structure
            value, chromosome, pos_start, pos_end = batch_data
            # Assuming labels are in the dataset - you may need to adjust this
            labels = torch.randint(0, 2, (value.size(0),))  # Placeholder - adjust as needed
        
        value = value.to(device)
        chromosome = chromosome.to(device)
        pos_start = pos_start.to(device)
        pos_end = pos_end.to(device)
        labels = labels.to(device).long()  # Convert to long for classification
        
        optimizer.zero_grad()
        
        outputs = model(value, chromosome, pos_start, pos_end)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_model(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Unpack batch data
            if len(batch_data) >= 5:
                value, chromosome, pos_start, pos_end, labels = batch_data[:5]
            else:
                value, chromosome, pos_start, pos_end = batch_data
                # Placeholder labels - adjust as needed
                labels = torch.randint(0, 2, (value.size(0),))
            
            value = value.to(device)
            chromosome = chromosome.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)
            labels = labels.to(device).long()
            
            outputs = model(value, chromosome, pos_start, pos_end)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy for classification
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    return avg_loss, accuracy


def finetune_chromfound():
    """Main function to fine-tune ChromFound model with LoRA"""
    parser = setup_finetune_parser()
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)
        torch.cuda.set_device(args.local_rank)
    
    logging.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create fine-tuning model
    logging.info("Creating fine-tuning model with LoRA...")
    model = create_finetune_model_from_pretrained(
        checkpoint_dir=args.checkpoint_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=args.task_type
    )
    
    # Update model args with task-specific parameters
    model.pretrain_model_args["num_classes"] = args.num_classes
    
    # Move model to device
    model = model.to(device)
    
    # Count parameters
    param_counts = count_parameters(model)
    logging.info(f"Total parameters: {param_counts['total_parameters']:,}")
    logging.info(f"Trainable parameters: {param_counts['trainable_parameters']:,}")
    logging.info(f"LoRA parameters: {param_counts['lora_parameters']:,}")
    logging.info(f"Frozen parameters: {param_counts['frozen_parameters']:,}")
    logging.info(f"Trainable percentage: {param_counts['trainable_parameters']/param_counts['total_parameters']*100:.4f}%")
    
    # Prepare data
    logging.info("Loading training data...")
    
    # Load the data configuration from the pretrained config
    config_path = os.path.join(args.checkpoint_dir, "chromfd_pretrain.yaml")
    with open(config_path, 'r') as file:
        pretrain_config = yaml.safe_load(file)
    pretrain_data_args = pretrain_config["data_args"]
    
    # Load chromosome vocabulary
    from src.utils.model_utils import ModelUtils
    chromosome_vocab = ModelUtils.get_chromosome_vocab(
        os.path.join(args.checkpoint_dir, "chromosome_vocab.yaml")
    )
    pretrain_data_args["chromosome_vocab"] = chromosome_vocab
    
    # Load training data
    import scanpy as sc
    adata = sc.read_h5ad(args.data_path)
    
    # Update data args with current data properties
    max_length = adata.shape[1]
    cell_types = list(set(adata.obs[args.get('cell_type_col', 'celltype')].unique().tolist()))
    cell_type_map = {ct: idx for idx, ct in enumerate(sorted(cell_types))}
    
    pretrain_data_args['cell_type_map'] = cell_type_map
    pretrain_data_args['cell_type_col'] = args.get('cell_type_col', 'celltype')
    pretrain_data_args["feature_num"] = adata.shape[1]
    pretrain_data_args["max_length"] = max_length
    pretrain_data_args["return_batch_label"] = False
    
    # Create dataset and dataloader
    train_dataset = DatasetMultiPad([adata], **pretrain_data_args)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    
    # Define loss function based on task type
    if args.task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported task type: {args.task_type}")
    
    # Get trainable parameters
    trainable_params = model.get_trainable_parameters()
    logging.info(f"Setting up optimizer for {len(trainable_params)} parameter groups...")
    
    # Setup optimizer for trainable parameters only
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logging.info("Starting fine-tuning...")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch+1)
        
        # Log metrics
        logging.info(f"Train Loss: {train_loss:.4f}")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
        }, checkpoint_path)
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "fine_tuned_model_lora.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'lora_rank': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'task_type': args.task_type,
            'num_classes': args.num_classes
        }
    }, final_model_path)
    
    logging.info(f"Fine-tuning completed! Model saved to {final_model_path}")


if __name__ == "__main__":
    finetune_chromfound()