#!/usr/bin/env python
"""
Script to run both baseline and improved models to compare performance
"""
import os
import subprocess
import sys
import time
from datetime import datetime

def run_baseline_model():
    """Run the original baseline model to establish metrics"""
    print("Running baseline model...")
    
    # Configuration for running baseline cell type annotation
    ATAC_FILE_PATH = "src/sample_data/PBMC169K"
    baseline_config = {
        "pretrain_checkpoint_path": "checkpoints",
        "pretrain_model_name": "model.pt",
        "pretrain_config_file": "chromfd_pretrain.yaml",
        "batch_size": 8,
        "device": 0,
        "output_path": str(os.path.join(ATAC_FILE_PATH, "cell_embedding")),
        "train_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad")),
        "test_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad")),
        "log_path": str(os.path.join(ATAC_FILE_PATH, "cell_type_annotation_baseline"))
    }

    # Create the log directory if it doesn't exist
    os.makedirs(baseline_config["log_path"], exist_ok=True)

    baseline_command = [
        sys.executable, '-m', 'src.cell_type_annotation',
        '--local_rank', f'{baseline_config["device"]}',
        '--batch_size', f'{baseline_config["batch_size"]}',
        '--learning_rate', '0.0003',
        '--pretrain_checkpoint_path', baseline_config['pretrain_checkpoint_path'],
        '--pretrain_model_file', baseline_config['pretrain_model_name'],
        '--pretrain_config_file', baseline_config['pretrain_config_file'],
        '--batch_size', f'{baseline_config["batch_size"]}',
        '--epoch', '5',  # Reduced epochs for faster testing
        '--train_file_path', baseline_config["train_file_path"],
        '--test_file_path', baseline_config["test_file_path"],
        '--log_path', baseline_config["log_path"],
        '--cell_type_col', 'celltype'
    ]

    print("Executing baseline command:", " ".join(baseline_command))
    result = subprocess.run(baseline_command)
    
    if result.returncode != 0:
        print(f"Baseline model failed with return code {result.returncode}")
        return False
    
    print("Baseline model completed successfully!")
    return True


def run_improved_model():
    """Run the improved multi-scale model"""
    print("Running improved multi-scale model...")
    
    # Configuration for running improved cell type annotation
    ATAC_FILE_PATH = "src/sample_data/PBMC169K"
    improved_config = {
        "pretrain_checkpoint_path": "checkpoints",
        "pretrain_model_name": "model.pt",
        "pretrain_config_file": "chromfd_pretrain.yaml",
        "batch_size": 8,
        "device": 0,
        "output_path": str(os.path.join(ATAC_FILE_PATH, "cell_embedding")),
        "train_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad")),
        "test_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad")),
        "log_path": str(os.path.join(ATAC_FILE_PATH, "cell_type_annotation_improved"))
    }

    # Create the log directory if it doesn't exist
    os.makedirs(improved_config["log_path"], exist_ok=True)

    improved_command = [
        sys.executable, '-m', 'src.cell_type_annotation_multiscale',
        '--local_rank', f'{improved_config["device"]}',
        '--batch_size', f'{improved_config["batch_size"]}',
        '--learning_rate', '0.0003',
        '--pretrain_checkpoint_path', improved_config['pretrain_checkpoint_path'],
        '--pretrain_model_file', improved_config['pretrain_model_name'],
        '--pretrain_config_file', improved_config['pretrain_config_file'],
        '--batch_size', f'{improved_config["batch_size"]}',
        '--epoch', '5',  # Reduced epochs for faster testing
        '--train_file_path', improved_config["train_file_path"],
        '--test_file_path', improved_config["test_file_path"],
        '--log_path', improved_config["log_path"],
        '--cell_type_col', 'celltype'
    ]

    print("Executing improved command:", " ".join(improved_command))
    result = subprocess.run(improved_command)
    
    if result.returncode != 0:
        print(f"Improved model failed with return code {result.returncode}")
        return False
    
    print("Improved model completed successfully!")
    return True


def extract_metrics_from_log(log_path):
    """Extract accuracy and F1 score from log file"""
    log_file = os.path.join(log_path, "finetune.log")
    if not os.path.exists(log_file):
        print(f"Log file does not exist: {log_file}")
        return None, None
    
    accuracy = None
    f1_score = None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        # Look for the last evaluation/test results
        for line in reversed(lines):
            if "cell type accuracy:" in line and "f1 score:" in line:
                # Extract accuracy and F1 score
                parts = line.split(',')
                for part in parts:
                    if "cell type accuracy:" in part:
                        accuracy = float(part.split(':')[1].strip())
                    if "f1 score:" in part:
                        f1_score = float(part.split(':')[1].strip())
                break
    
    return accuracy, f1_score


def compare_results():
    """Compare baseline and improved model results"""
    print("\nComparing results...")
    
    # Extract metrics from both models
    baseline_log_path = "src/sample_data/PBMC169K/cell_type_annotation_baseline"
    improved_log_path = "src/sample_data/PBMC169K/cell_type_annotation_improved"
    
    baseline_acc, baseline_f1 = extract_metrics_from_log(baseline_log_path)
    improved_acc, improved_f1 = extract_metrics_from_log(improved_log_path)
    
    print(f"\nBaseline Model:")
    print(f"  Accuracy: {baseline_acc}")
    print(f"  F1 Score: {baseline_f1}")
    
    print(f"\nImproved Model:")
    print(f"  Accuracy: {improved_acc}")
    print(f"  F1 Score: {improved_f1}")
    
    # Save comparison to summary file
    summary_path = "/root/idea-1768377409509-92/output/summary.log"
    with open(summary_path, 'w') as f:
        f.write(f"Experiment Summary - {datetime.now()}\n")
        f.write("="*50 + "\n")
        f.write(f"Baseline Model:\n")
        f.write(f"  Accuracy: {baseline_acc}\n")
        f.write(f"  F1 Score: {baseline_f1}\n")
        f.write(f"\nImproved Model (Multi-Scale Genomic Attention):\n")
        f.write(f"  Accuracy: {improved_acc}\n")
        f.write(f"  F1 Score: {improved_f1}\n")
        
        if baseline_acc is not None and improved_acc is not None:
            acc_improvement = ((improved_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc != 0 else 0
            f.write(f"\nAccuracy Improvement: {acc_improvement:.2f}%\n")
        
        if baseline_f1 is not None and improved_f1 is not None:
            f1_improvement = ((improved_f1 - baseline_f1) / baseline_f1) * 100 if baseline_f1 != 0 else 0
            f.write(f"F1 Score Improvement: {f1_improvement:.2f}%\n")
    
    print(f"\nSummary saved to: {summary_path}")
    

if __name__ == "__main__":
    print("Starting experiment to compare baseline vs improved model...")
    
    # Run baseline model
    baseline_success = run_baseline_model()
    
    if not baseline_success:
        print("Baseline model failed. Skipping improved model.")
        sys.exit(1)
    
    # Run improved model
    improved_success = run_improved_model()
    
    if not improved_success:
        print("Improved model failed.")
        sys.exit(1)
    
    # Compare results
    compare_results()
    
    print("\nExperiment completed!")