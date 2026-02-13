import os
import subprocess
import sys

# Configuration for running cell type annotation fine-tuning with advanced features
# - Implements gradient accumulation and dynamic learning rate scheduling
ATAC_FILE_PATH = "src/sample_data/PBMC169K"  # Updated path to use the symlink
OUTPUT_PATH = "output/idea1_results"  # Changed to use output directory
inference_config = {
    "pretrain_checkpoint_path": "checkpoints",  # Using symlink
    "pretrain_model_name": "model.pt",
    "pretrain_config_file": "chromfd_pretrain.yaml",
    "batch_size": 8,
    "device": 0,
    "output_path": str(os.path.join(OUTPUT_PATH, "cell_embedding")),
    "train_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad")),
    "test_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad")),
    "log_path": str(os.path.join(OUTPUT_PATH, "cell_type_annotation"))
}

# Ensure output directory exists
os.makedirs(inference_config["log_path"], exist_ok=True)

train_command = [
    sys.executable, '-m', 'src.cell_type_finetune_advanced',  # Use the advanced version
    '--local_rank', f'{inference_config["device"]}',
    '--batch_size', f'{inference_config["batch_size"]}',
    '--learning_rate', '0.0003',
    '--pretrain_checkpoint_path', inference_config['pretrain_checkpoint_path'],
    '--pretrain_model_file', inference_config['pretrain_model_name'],
    '--pretrain_config_file', inference_config['pretrain_config_file'],
    '--epoch', '5',
    '--train_file_path', inference_config["train_file_path"],
    '--test_file_path', inference_config["test_file_path"],
    '--log_path', inference_config["log_path"],
    '--cell_type_col', 'celltype',
    '--grad_accum_steps', '4',  # Added gradient accumulation
    '--lr_scheduler_type', 'warmup_cosine',  # Added dynamic learning rate scheduling
    '--warmup_steps', '200',
    '--min_lr_ratio', '0.1'
]

print("Starting advanced cell type annotation with gradient accumulation and dynamic learning rate scheduling...")
print(f"Command: {' '.join(train_command)}")

# Run the training process in the background and redirect output to log files
with open(os.path.join(inference_config["log_path"], "stdout.log"), "w") as stdout_file, \
     open(os.path.join(inference_config["log_path"], "stderr.log"), "w") as stderr_file:
    result = subprocess.run(train_command, stdout=stdout_file, stderr=stderr_file)

print("Training completed.")
print("Return code:", result.returncode)

# Print final metrics
metrics_file = os.path.join(inference_config["log_path"], "metrics.json")
if os.path.exists(metrics_file):
    import json
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    print("\nFinal Metrics:")
    print(f"Best F1 Score: {metrics.get('best_f1_score', 'N/A')}")
    print(f"Best Accuracy: {metrics.get('best_accuracy', 'N/A')}")
    print(f"Best ROC AUC: {metrics.get('best_roc_auc', 'N/A')}")
    print(f"Final Test F1 Score: {metrics.get('final_test_f1_score', 'N/A')}")
    print(f"Final Test Accuracy: {metrics.get('final_test_accuracy', 'N/A')}")
    print(f"Final Test ROC AUC: {metrics.get('final_test_roc_auc', 'N/A')}")
else:
    print(f"\nMetrics file not found at {metrics_file}")