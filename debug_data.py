#!/usr/bin/env python
"""Debug script to test data loading"""

import scanpy as sc
import numpy as np
from src.data.dataset_ds import DatasetMultiPad

# Load the data
train_file_path = "src/sample_data/PBMC169K/atac_pbmc_benchmark_EPF_hydrop_1_qc_deepen_norm_log.h5ad"
test_file_path = "src/sample_data/PBMC169K/atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad"

print("Loading data...")
adata_train_val = sc.read_h5ad(train_file_path)
adata_test = sc.read_h5ad(test_file_path)

print("Train data shape:", adata_train_val.shape)
print("Test data shape:", adata_test.shape)
print("Train var columns:", list(adata_train_val.var.columns))
print("Test var columns:", list(adata_test.var.columns))

# Check if required columns exist
required_cols = ["#Chromosome", "hg38_Start", "hg38_End"]
for col in required_cols:
    if col in adata_train_val.var.columns:
        print(f"✓ Train data has '{col}' column")
    else:
        print(f"✗ Train data missing '{col}' column")
        
    if col in adata_test.var.columns:
        print(f"✓ Test data has '{col}' column")
    else:
        print(f"✗ Test data missing '{col}' column")

# Create a simple dataset to test
cell_types = list(set(adata_train_val.obs["celltype"].unique().tolist() + adata_test.obs["celltype"].unique().tolist()))
cell_type_map = {ct: idx for idx, ct in enumerate(sorted(cell_types))}

print(f"Cell types: {cell_type_map}")

# Prepare data args
chromosome_vocab = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, 
    "19": 19, "20": 20, "21": 21, "22": 22, "X": 23, "Y": 24, "pad": 0
}

data_args = {
    'cell_type_map': cell_type_map,
    'cell_type_col': 'celltype',
    'chromosome_vocab': chromosome_vocab,
    'max_length': adata_train_val.shape[1],
    'return_batch_label': False
}

# Load the real chromosome vocab
import yaml
with open("checkpoints/chromosome_vocab.yaml") as file:
    chromosome_vocab_orig = yaml.safe_load(file)
    chromosome_vocab_real = {chr_: idx for idx, chr_ in enumerate(chromosome_vocab_orig["chromosome"])}

print("Real chromosome vocab keys:", list(chromosome_vocab_real.keys())[:10])

# Update data_args with real chromosome vocab
data_args['chromosome_vocab'] = chromosome_vocab_real

print("Creating dataset...")
try:
    train_dataset = DatasetMultiPad(adata_train_val, **data_args)
    print(f"Dataset created successfully. Length: {len(train_dataset)}")
    
    print("Testing first few samples...")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Sample {i} shape: {[s.shape if hasattr(s, 'shape') else len(s) for s in sample]}")
        
    print("Dataset test completed successfully!")
    
except Exception as e:
    print(f"Error creating/accessing dataset: {e}")
    import traceback
    traceback.print_exc()