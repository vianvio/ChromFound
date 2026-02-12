# ChromFound - Enhanced Fine-Tuning

ChromFound is a foundation model for scATAC-seq that leverages a hybrid architecture and genome-aware tokenization to capture genome-wide regulatory dynamics from chromatin accessibility profiles. Trained on 1.97 million cells spanning 30 tissues and 6 disease contexts, it delivers strong zero-shot and transfer performance across diverse tasks, providing a powerful framework for decoding enhancer–gene regulation and noncoding variant functions.

ChromFound has been accepted as a poster at NeurIPS 2025. See the preprint: [arXiv:2505.12638](https://arxiv.org/abs/2505.12638).

## Model Architecture

![Model architecture](model_architecture.png)

## Enhanced Fine-Tuning Features

The enhanced version of ChromFound includes three key improvements for fine-tuning:

### 1. Hierarchical Learning Rate Strategy
- Uses different learning rates for different parts of the model
- Lower learning rate (1e-5) for pre-trained backbone to preserve learned features
- Higher learning rate (1e-3) for new classifier heads to accelerate learning on new tasks

### 2. Gradient Clipping and Early Stopping
- Gradient clipping to prevent gradient explosion and improve training stability
- Early stopping mechanism that monitors validation F1 score
- Stops training if F1 score doesn't improve for 5 consecutive epochs

### 3. Parameter-Efficient Fine-Tuning (LoRA)
- Low-Rank Adaptation (LoRA) to significantly reduce computational overhead
- Only trains a small subset of parameters while freezing the pre-trained model
- Dramatically reduces memory usage and training time

## Installation Requirements

### System Requirements
- Python 3.10+
- CUDA 12.1+
- GPUs supporting [FlashAttention](https://github.com/Dao-AILab/flash-attention)

### Installation Steps

Please ensure your Conda installation is properly configured and active before running the following commands.

```bash
# 1) Create and activate the conda environment
conda env create -f environment.yml
conda activate chromfound

# 2) Install PyTorch (CUDA 12.1 wheels)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 3) Install core dependencies
pip install mamba-ssm==2.2.4
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

For platform-specific notes and troubleshooting when installing Mamba and FlashAttention, see the official installation guides for [mamba-ssm](https://github.com/state-spaces/mamba) and [FlashAttention](https://github.com/Dao-AILab/flash-attention).

## Quick Start

Note: The GitHub repo does NOT include large assets (`src/checkpoints/` and `sample_data/`). Download them from [Hugging Face](https://huggingface.co/YifengJiao/ChromFound) or [Google Drive](https://drive.google.com/drive/folders/1wSq9gPwnUmSiw3obz1mjyX2ZiXS8sWbf?usp=sharing) and place them locally as follows:
- Pretrained weights: download `model.pt`, `chromfd_pretrain.yaml`, `chromosome_vocab.yaml` into `src/checkpoints/`
- Sample data: download `PBMC169K/*.h5ad` into `sample_data/PBMC169K/`

### Pretrained Weights

Ensure the following files exist in src/checkpoints:
- model.pt (pretrained model weights)
- chromfd_pretrain.yaml (pretrained model config)
- chromosome_vocab.yaml (chromosome index mapping)

### Generate Cell Embeddings
```bash
python -m src/cell_embedding \
    --data_path sample_data/PBMC169K/atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad \
    --output_path sample_data/PBMC169K/cell_embedding \
    --pretrain_checkpoint_path src/checkpoints \
    --pretrain_model_file model.pt \
    --pretrain_config_file chromfd_pretrain.yaml \
    --batch_size 16 \
    --cell_type_col celltype
```

### Enhanced Cell Type Annotation with Improved Fine-Tuning

#### Using Hierarchical Learning Rates (Default)
```bash
python -m src/cell_type_annotation_improved \
    --data_path sample_data/PBMC169K/atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad \
    --output_path sample_data/PBMC169K/cell_type_annotation \
    --pretrain_checkpoint_path src/checkpoints \
    --pretrain_model_file model.pt \
    --pretrain_config_file chromfd_pretrain.yaml \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --backbone_lr 1e-5 \
    --classifier_lr 1e-3 \
    --cell_type_col celltype \
    --epoch 10 \
    --log_path logs/cell_type_annotation_hierarchical
```

#### Using LoRA for Parameter-Efficient Fine-Tuning
```bash
python -m src/cell_type_annotation_improved \
    --data_path sample_data/PBMC169K/atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad \
    --output_path sample_data/PBMC169K/cell_type_annotation \
    --pretrain_checkpoint_path src/checkpoints \
    --pretrain_model_file model.pt \
    --pretrain_config_file chromfd_pretrain.yaml \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --cell_type_col celltype \
    --epoch 10 \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --log_path logs/cell_type_annotation_lora
```

#### Using All Enhancements (Recommended)
```bash
python -m src/cell_type_annotation_improved \
    --data_path sample_data/PBMC169K/atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad \
    --output_path sample_data/PBMC169K/cell_type_annotation \
    --pretrain_checkpoint_path src/checkpoints \
    --pretrain_model_file model.pt \
    --pretrain_config_file chromfd_pretrain.yaml \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --cell_type_col celltype \
    --epoch 20 \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --gradient_clipping \
    --max_grad_norm 1.0 \
    --patience 5 \
    --log_path logs/cell_type_annotation_enhanced
```

### Additional Arguments for Enhanced Fine-Tuning

- `--use_lora`: Enable Low-Rank Adaptation for parameter-efficient fine-tuning
- `--lora_rank`: Rank of the LoRA decomposition (default: 16)
- `--lora_alpha`: Alpha parameter for LoRA scaling (default: 32.0)
- `--gradient_clipping`: Enable gradient clipping to prevent exploding gradients
- `--max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)
- `--patience`: Number of epochs to wait before early stopping (default: 5)
- `--backbone_lr`: Learning rate for the pre-trained backbone (default: 1e-5)
- `--classifier_lr`: Learning rate for the new classifier head (default: 1e-3)

For an interactive walkthrough and examples, see the tutorial notebook `cell_embedding.ipynb`.

## Data Format

### Input Data Requirements
- **Format**: H5AD (AnnData) format
- **Required Columns**:
  - `obs`: Cell-level metadata; must include the cell type column used via `--cell_type_col` (e.g., `celltype`).
  - `var`: Feature metadata containing chromosome position information
    - `#Chromosome`: Integer chromosome index as defined in `src/conf/chromosome_vocab.yaml`.
    - `hg38_Start`: 0-based, inclusive genomic start coordinate (int) on the hg38 reference (base pairs).
    - `hg38_End`: 0-based, exclusive genomic end coordinate (int) on the hg38 reference (base pairs).

## Citation
If you use ChromFound, please cite our paper:

```bibtex
@article{jiao2025chromfound,
  title={ChromFound: Towards A Universal Foundation Model for Single-Cell Chromatin Accessibility Data},
  author={Jiao, Yifeng and Liu, Yuchen and Zhang, Yu and Guo, Xin and Wu, Yushuai and Jiang, Chen and Li, Jiyang and Zhang, Hongwei and Han, Limei and Gao, Xin and Qi, yuan and Cheng, yuan},
  journal={arXiv preprint arXiv:2505.12638},
  year={2025}
}
```

## Changelog

### v1.1.0 (2025-10-17)
- Added hierarchical learning rate strategy for fine-tuning
- Implemented gradient clipping and early stopping mechanisms
- Integrated LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Created enhanced cell type annotation script with all improvements
- Updated documentation with usage examples for new features

### v1.0.0 (2025-10-16)
- Initial release
- Support for basic cell embedding and cell type annotation functionality
- update README.md