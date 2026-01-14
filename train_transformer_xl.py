import os
import subprocess
import sys

# Configuration for running cell embedding inference with Transformer-XL
# - pretrain_checkpoint_path: Directory containing the pretrained model checkpoint
# - pretrain_model_name: File name of the pretrained model
# - pretrain_config_file: Configuration file used for model architecture and settings
# - batch_size: Batch size for inference
# - device: GPU device ID for computation
# - output_path: Directory to save the inferred cell embeddings
ATAC_FILE_PATH = "src/sample_data/PBMC169K"
inference_config = {
    "pretrain_checkpoint_path": "checkpoints",
    "pretrain_model_name": "transformer_xl_model.pt",
    "pretrain_config_file": "transformer_xl_config.yaml",
    "batch_size": 8,
    "device": 0,
    "output_path": str(os.path.join(ATAC_FILE_PATH, "cell_embedding")),
    "train_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad")),
    "test_file_path": str(os.path.join(ATAC_FILE_PATH, "atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad")),
    "log_path": str(os.path.join(ATAC_FILE_PATH, "cell_type_annotation_transformer_xl"))
}

# Create the training command for the Transformer-XL model
train_command = [
    sys.executable, '-m', 'src.cell_type_annotation_transformer_xl',
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

# Check if the required files exist, if not, create them
checkpoint_dir = inference_config["pretrain_checkpoint_path"]
os.makedirs(checkpoint_dir, exist_ok=True)

# Create a sample config file if it doesn't exist
config_path = os.path.join(checkpoint_dir, inference_config["pretrain_config_file"])
if not os.path.exists(config_path):
    import yaml
    sample_config = {
        "data_args": {
            "chromosome_vocab": {"chr1": 0, "chr2": 1, "chr3": 2, "chr4": 3, "chr5": 4, 
                                "chr6": 5, "chr7": 6, "chr8": 7, "chr9": 8, "chr10": 9,
                                "chr11": 10, "chr12": 11, "chr13": 12, "chr14": 13, "chr15": 14,
                                "chr16": 15, "chr17": 16, "chr18": 17, "chr19": 18, "chr20": 19,
                                "chr21": 20, "chr22": 21, "chrX": 22, "chrY": 23, "other": 24},
            "cell_type_col": "celltype",
            "feature_num": 40000,  # Approximate number of genomic features
            "max_length": 40000,
            "return_batch_label": False
        },
        "model_args": {
            "embedding_dim": 256,
            "chromosome_size": 25,
            "embedding_dropout": 0.1,
            "positional_embedding_type": "sinusoidal",
            "positional_temp": 10000,
            "batch_size": 8,
            "seq_length": 40000,
            "device": "cuda:0",
            "chromatin_embedding": True,
            "encoder_layers": 12,
            "value_size": 1,
            "cell_type_num": 10,  # Placeholder - will be updated based on data
            "feature_num": 40000,
            "mask_ratio": 0.0,
            "n_head": 8,
            "mem_len": 512,
            "max_pos_len": 100000
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)

# Create a sample chromosome vocab file if it doesn't exist
vocab_path = os.path.join(checkpoint_dir, "chromosome_vocab.yaml")
if not os.path.exists(vocab_path):
    chromosome_vocab = {"chr1": 0, "chr2": 1, "chr3": 2, "chr4": 3, "chr5": 4, 
                        "chr6": 5, "chr7": 6, "chr8": 7, "chr9": 8, "chr10": 9,
                        "chr11": 11, "chr12": 12, "chr13": 13, "chr14": 14, "chr15": 15,
                        "chr16": 16, "chr17": 17, "chr18": 18, "chr19": 19, "chr20": 20,
                        "chr21": 21, "chr22": 22, "chrX": 23, "chrY": 24, "other": 25}
    
    with open(vocab_path, 'w') as f:
        yaml.dump(chromosome_vocab, f)

# Create the log directory
log_path = inference_config["log_path"]
os.makedirs(log_path, exist_ok=True)

# Run the training command
result = subprocess.run(train_command)

# Check if training was successful
if result.returncode == 0:
    print("Training completed successfully!")
else:
    print(f"Training failed with return code: {result.returncode}")
    print("Note: This may happen if the required data files don't exist.")