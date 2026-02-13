#!/usr/bin/env python
"""Test AnnData subsetting"""

import scanpy as sc
import numpy as np

# Load the data
train_file_path = "src/sample_data/PBMC169K/atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad"
adata_train_val = sc.read_h5ad(train_file_path)

print("Original data shape:", adata_train_val.shape)
print("Original var columns:", list(adata_train_val.var.columns))

# Test subsetting
idx_list = [i for i in range(adata_train_val.X.shape[0])]
np.random.seed(42)  # For reproducible results
np.random.shuffle(idx_list)
split_idx = int(len(idx_list) * 0.9)
train_idx = idx_list[:split_idx]
val_idx = idx_list[split_idx:]

adata_train = adata_train_val[train_idx]
adata_val = adata_train_val[val_idx]

print("\nAfter subsetting:")
print("Train data shape:", adata_train.shape)
print("Train var columns:", list(adata_train.var.columns))
print("Val data shape:", adata_val.shape)
print("Val var columns:", list(adata_val.var.columns))

# Check if required columns exist
required_cols = ["#Chromosome", "hg38_Start", "hg38_End"]
for col in required_cols:
    if col in adata_train.var.columns:
        print(f"✓ Train data has '{col}' column")
    else:
        print(f"✗ Train data missing '{col}' column")
        
    if col in adata_val.var.columns:
        print(f"✓ Val data has '{col}' column")
    else:
        print(f"✗ Val data missing '{col}' column")

# Test accessing the column
try:
    chrom_values = adata_train.var["#Chromosome"].tolist()
    print(f"\n✓ Successfully accessed '#Chromosome' column. First 5 values: {chrom_values[:5]}")
except Exception as e:
    print(f"\n✗ Error accessing '#Chromosome' column: {e}")