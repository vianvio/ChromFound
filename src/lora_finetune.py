"""
Fine-tuning script for ChromFound model using LoRA.
This script enables parameter-efficient fine-tuning of the ChromFound model.
"""

import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.models.finetune_model import create_finetune_model_from_pretrained, count_parameters

logging.basicConfig(level=logging.INFO)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune ChromFound with LoRA")
    
    # Required paths
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing pretrained model files')
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
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for LoRA parameters (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    
    # Task parameters
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='Type of downstream task (default: classification)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes for classification (default: 2)')
    
    # Device
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Setup device
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the fine-tuning model with LoRA
    logging.info("Initializing ChromFound model with LoRA...")
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
    
    # Count and report parameters
    param_counts = count_parameters(model)
    logging.info(f"Total parameters: {param_counts['total_parameters']:,}")
    logging.info(f"Trainable parameters: {param_counts['trainable_parameters']:,}")
    logging.info(f"LoRA parameters: {param_counts['lora_parameters']:,}")
    logging.info(f"Frozen parameters: {param_counts['frozen_parameters']:,}")
    logging.info(f"Trainable percentage: {param_counts['trainable_parameters']/param_counts['total_parameters']*100:.4f}%")
    
    # Setup optimizer - only optimize LoRA and task-specific parameters
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Define loss function based on task type
    if args.task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:  # regression
        criterion = nn.MSELoss()
    
    logging.info("Starting training...")
    logging.info(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    
    # Training loop (simplified - in practice you'd load actual data)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        # In a real scenario, you would load your dataset here
        # For demonstration, we'll simulate a few batches
        num_batches = 5  # Simulated number of batches per epoch
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}"):
            # In practice, you would load actual batch data here
            # For now, we'll create dummy data to demonstrate the training loop
            batch_size = args.batch_size
            seq_len = 100  # Example sequence length
            feat_dim = model.pretrain_model_args.get("embedding_dim", 2560)
            
            # Create dummy input data
            value = torch.randn(batch_size, seq_len, 1).to(device)
            chromosome = torch.randint(0, 23, (batch_size, seq_len)).to(device)  # Assuming 23 chromosomes
            pos_start = torch.randint(0, 1000000, (batch_size, seq_len)).to(device)
            pos_end = pos_start + torch.randint(100, 1000, (batch_size, seq_len)).to(device)
            
            # Create dummy labels based on task type
            if args.task_type == "classification":
                labels = torch.randint(0, args.num_classes, (batch_size,)).to(device)
            else:  # regression
                labels = torch.randn(batch_size, args.num_classes if args.num_classes > 1 else 1).to(device)
            
            optimizer.zero_grad()
            
            outputs = model(value, chromosome, pos_start, pos_end)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        logging.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Save the fine-tuned model
    output_path = os.path.join(args.output_dir, "chromfound_finetuned_lora.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'lora_rank': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'task_type': args.task_type,
            'num_classes': args.num_classes,
            'model_args': model.pretrain_model_args
        },
        'epoch': args.epochs
    }, output_path)
    
    logging.info(f"Fine-tuning completed! Model saved to {output_path}")
    
    # Print final parameter statistics
    final_param_counts = count_parameters(model)
    logging.info(f"Final trainable parameters: {final_param_counts['trainable_parameters']:,}")


if __name__ == "__main__":
    main()