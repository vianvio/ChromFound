import os
import subprocess
import sys

# Configuration for running cell type annotation fine-tuning
# - pretrain_checkpoint_path: Directory containing the pretrained model checkpoint
# - pretrain_model_name: File name of the pretrained model
# - pretrain_config_file: Configuration file used for model architecture and settings
# - batch_size: Batch size for training
# - device: GPU device ID for computation
# - output_path: Directory to save the fine-tuned model and results
ATAC_FILE_PATH = "/inspire/cpfs/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/0815-shenlijie/data/sample_data/PBMC169K"
OUTPUT_PATH = "output/cell_type_annotation_idea1" # change by idea id
inference_config = {
    "pretrain_checkpoint_path": "/inspire/cpfs/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/0815-shenlijie/data/checkpoints",
    "pretrain_model_name": "model.pt",
    "pretrain_config_file": "chromfd_pretrain.yaml",
    "batch_size": 8,
    "device": 0,
    "output_path": str(os.path.join(OUTPUT_PATH, "cell_embedding")),
    "train_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad")),
    "test_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad")),
    "log_path": str(os.path.join(OUTPUT_PATH, "cell_type_annotation"))
}

# Create output directory if it doesn't exist
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
    '--cell_type_col', 'celltype'
]

subprocess.run(train_command)
