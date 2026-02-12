#!/usr/bin/env python
"""
Example script demonstrating the usage of enhanced fine-tuning methods for ChromFound
"""

import os
import subprocess
import argparse


def run_hierarchical_finetuning(data_path, checkpoint_path, output_dir):
    """Run fine-tuning with hierarchical learning rates"""
    cmd = [
        "python", "-m", "src.cell_type_annotation_improved",
        "--train_file_path", data_path,
        "--test_file_path", data_path,  # Using same data for demo purposes
        "--pretrain_checkpoint_path", checkpoint_path,
        "--pretrain_model_file", "model.pt",
        "--pretrain_config_file", "chromfd_pretrain.yaml",
        "--cell_type_col", "celltype",
        "--epoch", "5",
        "--batch_size", "8",
        "--learning_rate", "1e-4",
        "--backbone_lr", "1e-5",
        "--classifier_lr", "1e-3",
        "--log_path", os.path.join(output_dir, "hierarchical"),
    ]
    
    print("Running hierarchical learning rate fine-tuning...")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)


def run_lora_finetuning(data_path, checkpoint_path, output_dir):
    """Run fine-tuning with LoRA"""
    cmd = [
        "python", "-m", "src.cell_type_annotation_improved",
        "--train_file_path", data_path,
        "--test_file_path", data_path,  # Using same data for demo purposes
        "--pretrain_checkpoint_path", checkpoint_path,
        "--pretrain_model_file", "model.pt",
        "--pretrain_config_file", "chromfd_pretrain.yaml",
        "--cell_type_col", "celltype",
        "--epoch", "5",
        "--batch_size", "8",
        "--learning_rate", "1e-3",
        "--use_lora",
        "--lora_rank", "16",
        "--lora_alpha", "32.0",
        "--log_path", os.path.join(output_dir, "lora"),
    ]
    
    print("\nRunning LoRA fine-tuning...")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)


def run_full_enhanced_finetuning(data_path, checkpoint_path, output_dir):
    """Run fine-tuning with all enhancements"""
    cmd = [
        "python", "-m", "src.cell_type_annotation_improved",
        "--train_file_path", data_path,
        "--test_file_path", data_path,  # Using same data for demo purposes
        "--pretrain_checkpoint_path", checkpoint_path,
        "--pretrain_model_file", "model.pt",
        "--pretrain_config_file", "chromfd_pretrain.yaml",
        "--cell_type_col", "celltype",
        "--epoch", "10",
        "--batch_size", "8",
        "--learning_rate", "1e-3",
        "--use_lora",
        "--lora_rank", "16",
        "--lora_alpha", "32.0",
        "--gradient_clipping",
        "--max_grad_norm", "1.0",
        "--patience", "3",
        "--log_path", os.path.join(output_dir, "enhanced"),
    ]
    
    print("\nRunning fully enhanced fine-tuning...")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run enhanced ChromFound fine-tuning examples")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the input h5ad data file")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the pretrained checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="./fine_tuning_examples",
                        help="Directory to store output logs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting enhanced fine-tuning examples for ChromFound...")
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Run different fine-tuning strategies
    run_hierarchical_finetuning(args.data_path, args.checkpoint_path, args.output_dir)
    run_lora_finetuning(args.data_path, args.checkpoint_path, args.output_dir)
    run_full_enhanced_finetuning(args.data_path, args.checkpoint_path, args.output_dir)
    
    print("\nEnhanced fine-tuning examples completed!")
    print("Check the output directory for logs and results.")


if __name__ == "__main__":
    main()