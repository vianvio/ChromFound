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
OUTPUT_PATH = "baseline/idea/idea-1" # change by idea id

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

inference_config = {
    "pretrain_checkpoint_path": "checkpoints",
    "pretrain_model_name": "model.pt",
    "pretrain_config_file": "chromfd_pretrain.yaml",
    "batch_size": 8,
    "device": 0,
    "output_path": str(os.path.join(OUTPUT_PATH, "cell_embedding")),
    "train_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad")),
    "test_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad")),
    "log_path": str(os.path.join(OUTPUT_PATH, "cell_type_annotation"))
}

# Check if train and test files exist
if not os.path.exists(inference_config["train_file_path"]):
    print(f"Warning: Train file does not exist: {inference_config['train_file_path']}")
    # Use the test file for both train and test as fallback
    inference_config["train_file_path"] = inference_config["test_file_path"]

if not os.path.exists(inference_config["test_file_path"]):
    print(f"Error: Test file does not exist: {inference_config['test_file_path']}")
    sys.exit(1)

# Create log directory
os.makedirs(inference_config["log_path"], exist_ok=True)

train_command = [
    sys.executable, '-m', 'src.cell_type_annotation',
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
    '--use_gradient_clipping',
    '--label_smoothing', '0.1',
    '--use_hierarchical_lr',
    '--lr_scheduler_type', 'cosine'
]

subprocess.run(train_command)
