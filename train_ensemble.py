import os
import subprocess
import sys

# Configuration for running cell embedding inference
# - pretrain_checkpoint_path: Directory containing the pretrained model checkpoint
# - pretrain_model_name: File name of the pretrained model
# - pretrain_config_file: Configuration file used for model architecture and settings
# - batch_size: Batch size for inference
# - device: GPU device ID for computation
# - output_path: Directory to save the inferred cell embeddings
ATAC_FILE_PATH = "src/sample_data/PBMC169K"
inference_config = {
    "pretrain_checkpoint_path": "checkpoints",
    "pretrain_model_name": "model.pt",
    "pretrain_config_file": "chromfd_pretrain.yaml",
    "batch_size": 8,
    "device": 0,
    "output_path": str(os.path.join(ATAC_FILE_PATH, "cell_embedding")),
    "train_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad")),
    "test_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad")),
    "log_path": str(os.path.join(ATAC_FILE_PATH, "cell_type_annotation_ensemble"))
}

train_command = [
    sys.executable, '-m', 'src.cell_type_annotation',
    '--local_rank', f'{inference_config["device"]}',
    '--batch_size', f'{inference_config["batch_size"]}',
    '--learning_rate', '0.0003',
    '--pretrain_checkpoint_path', inference_config['pretrain_checkpoint_path'],
    '--pretrain_model_file', inference_config['pretrain_model_name'],
    '--pretrain_config_file', inference_config['pretrain_config_file'],
    '--batch_size', f'{inference_config["batch_size"]}',
    '--epoch', '5',
    '--train_file_path', inference_config["train_file_path"],
    '--test_file_path', inference_config["test_file_path"],
    '--log_path', inference_config["log_path"],
    '--cell_type_col', 'celltype'
]

print("Running ensemble model training...")
print(f"Command: {' '.join(train_command)}")

# Check if the required files exist before running
checkpoint_dir = inference_config['pretrain_checkpoint_path']
required_files = [
    os.path.join(checkpoint_dir, inference_config['pretrain_model_name']),
    os.path.join(checkpoint_dir, inference_config['pretrain_config_file']),
    os.path.join(checkpoint_dir, 'chromosome_vocab.yaml')
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"Warning: Missing required files: {missing_files}")
    print("Please ensure the checkpoint files exist before running the training.")
else:
    print("All required files found. Starting training...")
    subprocess.run(train_command)