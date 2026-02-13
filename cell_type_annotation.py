import os
import subprocess
import sys

# Configuration for running cell type annotation fine-tuning
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
os.makedirs(os.path.join(OUTPUT_PATH, "cell_type_annotation"), exist_ok=True)

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

# Check if required files exist
if not os.path.exists(inference_config["train_file_path"]):
    raise FileNotFoundError(f"Train file does not exist: {inference_config['train_file_path']}")
if not os.path.exists(inference_config["test_file_path"]):
    raise FileNotFoundError(f"Test file does not exist: {inference_config['test_file_path']}")
if not os.path.exists(os.path.join(inference_config["pretrain_checkpoint_path"], inference_config["pretrain_model_name"])):
    raise FileNotFoundError(f"Model file does not exist: {os.path.join(inference_config['pretrain_checkpoint_path'], inference_config['pretrain_model_name'])}")
if not os.path.exists(os.path.join(inference_config["pretrain_checkpoint_path"], inference_config["pretrain_config_file"])):
    raise FileNotFoundError(f"Config file does not exist: {os.path.join(inference_config['pretrain_checkpoint_path'], inference_config['pretrain_config_file'])}")

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
    '--cell_type_col', 'celltype'
]

# Run the training command and redirect output to log file
log_file = os.path.join("output", "cell_type_annotation.log")
os.makedirs("output", exist_ok=True)

with open(log_file, "w") as f:
    f.write("Starting cell type annotation training...\n")
    
# Run the command in the background
process = subprocess.Popen(train_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Continuously write output to log file and print to console
with open(log_file, "a") as f:
    for line in process.stdout:
        print(line, end='')
        f.write(line)
        
process.wait()
print(f"Training completed with return code: {process.returncode}")
print(f"Log output saved to: {log_file}")
