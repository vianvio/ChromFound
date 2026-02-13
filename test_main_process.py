#!/usr/bin/env python
"""Test the exact data processing from main function"""

import scanpy as sc
import numpy as np
import random
from src.utils.model_utils import ModelUtils

# Load the data (same as in main function)
train_file_path = "src/sample_data/PBMC169K/atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad"
test_file_path = "src/sample_data/PBMC169K/atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad"

print("Loading data...")
adata_train_val = sc.read_h5ad(train_file_path)
adata_test = sc.read_h5ad(test_file_path)

print("Before concatenation:")
print(f"Train data shape: {adata_train_val.shape}")
print(f"Test data shape: {adata_test.shape}")
print(f"Train var columns: {list(adata_train_val.var.columns)}")
print(f"Test var columns: {list(adata_test.var.columns)}")

# Same process as in main function (updated)
adata_train_val.obs["tag"] = "train"
adata_test.obs["tag"] = "test"
import anndata
adata_concat = anndata.concat([adata_train_val, adata_test], axis=0, label='tag', keys=['train', 'test'], 
                              join='outer', merge='first', uns_merge='unique')
adata_train_val = adata_concat[adata_concat.obs["tag"] == "train"]
adata_test = adata_concat[adata_concat.obs["tag"] == "test"]

print("\nAfter concatenation and re-splitting:")
print(f"New train data shape: {adata_train_val.shape}")
print(f"New test data shape: {adata_test.shape}")
print(f"New train var columns: {list(adata_train_val.var.columns)}")
print(f"New test var columns: {list(adata_test.var.columns)}")

# Check if required columns exist
required_cols = ["#Chromosome", "hg38_Start", "hg38_End"]
for col in required_cols:
    if col in adata_train_val.var.columns:
        print(f"✓ New train data has '{col}' column")
    else:
        print(f"✗ New train data missing '{col}' column")
        
    if col in adata_test.var.columns:
        print(f"✓ New test data has '{col}' column")
    else:
        print(f"✗ New test data missing '{col}' column")

# Split into train and validation (same as main function)
idx_list = [i for i in range(adata_train_val.X.shape[0])]
random.shuffle(idx_list)
split_idx = int(len(idx_list) * 0.9)
train_idx = idx_list[:split_idx]
val_idx = idx_list[split_idx:]
adata_train = adata_train_val[train_idx]
adata_val = adata_train_val[val_idx]

print(f"\nFinal train data shape: {adata_train.shape}")
print(f"Final val data shape: {adata_val.shape}")
print(f"Final train var columns: {list(adata_train.var.columns)}")
print(f"Final val var columns: {list(adata_val.var.columns)}")

for col in required_cols:
    if col in adata_train.var.columns:
        print(f"✓ Final train data has '{col}' column")
    else:
        print(f"✗ Final train data missing '{col}' column")
        
    if col in adata_val.var.columns:
        print(f"✓ Final val data has '{col}' column")
    else:
        print(f"✗ Final val data missing '{col}' column")

# Load chromosome vocab (same as main function)
chromosome_vocab = ModelUtils.get_chromosome_vocab(
    "checkpoints/chromosome_vocab.yaml"
)

print(f"\nChromosome vocab keys: {list(chromosome_vocab.keys())[:10]}")

# Prepare data args (similar to main function)
cell_type = list(set(adata_train_val.obs["celltype"].unique().tolist() + adata_test.obs["celltype"].unique().tolist()))
cell_type_map = {cell_type: idx for idx, cell_type in enumerate(sorted(cell_type))}

max_length = adata_concat.shape[1]

data_args = {
    'cell_type_map': cell_type_map,
    'cell_type_col': 'celltype',
    'chromosome_vocab': chromosome_vocab,
    'max_length': max_length,
    'return_batch_label': False,
    'feature_num': adata_train_val.shape[1]
}

# Test accessing the chromosome data
try:
    print("\nTesting chromosome access...")
    chrom_values = adata_train.var["#Chromosome"].tolist()
    print(f"✓ Successfully accessed '#Chromosome' column. First 5 values: {chrom_values[:5]}")
    
    # Test with a few values to see if they're in the vocab
    for val in chrom_values[:5]:
        if val in chromosome_vocab:
            print(f"✓ '{val}' is in chromosome vocab (index: {chromosome_vocab[val]})")
        else:
            print(f"✗ '{val}' is NOT in chromosome vocab")
            
except Exception as e:
    print(f"✗ Error accessing '#Chromosome' column: {e}")
    import traceback
    traceback.print_exc()