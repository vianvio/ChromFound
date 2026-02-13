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
ATAC_FILE_PATH = "/inspire/cpfs/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/0815-shenlijie/data/sample_data/PBMC169K"
inference_config = {
    "pretrain_checkpoint_path": "/inspire/cpfs/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/0815-shenlijie/data/checkpoints",
    "pretrain_model_name": "model.pt",
    "pretrain_config_file": "chromfd_pretrain.yaml",
    "batch_size": 8,
    "device": 0,
    "output_path": str(os.path.join(ATAC_FILE_PATH, "cell_embedding")),
    "train_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad")),
    "test_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad")),
    "log_path": str(os.path.join(ATAC_FILE_PATH, "cell_type_annotation"))
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

result = subprocess.run(train_command)
print(f"Training completed with return code: {result.returncode}")