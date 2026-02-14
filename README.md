# ChromFound

ChromFound is a foundation model for scATAC-seq that leverages a hybrid architecture and genome-aware tokenization to capture genome-wide regulatory dynamics from chromatin accessibility profiles. Trained on 1.97 million cells spanning 30 tissues and 6 disease contexts, it delivers strong zero-shot and transfer performance across diverse tasks, providing a powerful framework for decoding enhancer–gene regulation and noncoding variant functions. 

ChromFound has been accepted as a poster at NeurIPS 2025. See the preprint: [arXiv:2505.12638](https://arxiv.org/abs/2505.12638). 

## Model Architecture

![Model architecture](model_architecture.png)


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

Then follow the command in Generate Cell Embeddings (below) by setting:
- `--pretrain_checkpoint_path src/checkpoints`
- `--pretrain_model_file model.pt`
- `--pretrain_config_file chromfd_pretrain.yaml`

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

## Advanced Training Techniques

ChromFound now supports advanced training techniques to improve model convergence, stability, and generalization:

### 1. Gradient Clipping and Dynamic Learning Rate Adjustment
- Gradient clipping to prevent exploding gradients during fine-tuning
- Dynamic learning rate scheduling with cosine annealing or exponential decay
- Monitoring of gradient norms during training

### 2. Label Smoothing
- Implementation of label smoothing to reduce overfitting
- Adjustable smoothing factor to balance between confidence and generalization

### 3. Hierarchical Learning Rate Fine-tuning
- Different learning rates for different model layers
- Lower learning rates for early layers to preserve general features
- Higher learning rates for later layers to adapt to specific tasks

## Usage Examples

### Enhanced Fine-tuning with Advanced Techniques
```bash
python -m src/enhanced_cell_type_finetune \
    --data_path sample_data/PBMC169K/atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad \
    --output_path sample_data/PBMC169K/cell_embedding \
    --pretrain_checkpoint_path src/checkpoints \
    --pretrain_model_file model.pt \
    --pretrain_config_file chromfd_pretrain.yaml \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --epoch 10 \
    --cell_type_col celltype \
    --log_path ./logs \
    --use_gradient_clipping \
    --gradient_clip_threshold 1.0 \
    --label_smoothing 0.1 \
    --use_hierarchical_lr \
    --layer_decay 0.9 \
    --lr_scheduler_type cosine \
    --min_lr 1e-6
```

### Standard Fine-tuning with Selected Features
```bash
python -m src/cell_type_annotation \
    --data_path sample_data/PBMC169K/atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad \
    --pretrain_checkpoint_path src/checkpoints \
    --pretrain_model_file model.pt \
    --pretrain_config_file chromfd_pretrain.yaml \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --epoch 10 \
    --cell_type_col celltype \
    --log_path ./logs \
    --use_gradient_clipping \
    --label_smoothing 0.1 \
    --use_hierarchical_lr
```

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

### v1.1.0 (2025-10-20)
- Added advanced training techniques:
  - Gradient clipping and dynamic learning rate adjustment
  - Label smoothing implementation
  - Hierarchical learning rate fine-tuning
- New enhanced fine-tuning script with all improvements
- Updated documentation with usage examples

### v1.0.0 (2025-10-16)
- Initial release
- Support for basic cell embedding and cell type annotation functionality
- update README.md
