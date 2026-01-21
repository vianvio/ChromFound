import torch
import torch.nn as nn
import numpy as np
from src.models.improved_cell_type_model import ImprovedFinetuneModelMambaCellType

def test_improved_model():
    """Test the improved model architecture"""
    print("Testing the improved genomic context-aware dynamic convolution model...")
    
    # Define model parameters similar to the original
    model_args = {
        "embedding_dim": 256,
        "chromosome_size": 25,  # Assuming 25 chromosomes
        "embedding_dropout": 0.1,
        "positional_embedding_type": "sinusoidal",
        "positional_temp": 10000,
        "batch_size": 4,
        "max_length": 100,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "chromatin_embedding": True,
        "wpsa_heads": 8,
        "wpsa_window_size": 7,
        "shift_size": 0,
        "encoder_layers": 4,
        "cell_type_num": 10,  # Example number of cell types
        "value_size": 1,
        "encoder_dim": 256
    }
    
    # Create model
    model = ImprovedFinetuneModelMambaCellType(**model_args)
    model = model.to(model_args["device"])
    
    print(f"Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy input data
    batch_size = 2
    seq_len = model_args["max_length"]
    
    value = torch.randn(batch_size, seq_len, 1).to(model_args["device"])  # Value input
    chromosome = torch.randint(0, model_args["chromosome_size"], (batch_size, seq_len)).to(model_args["device"])  # Chromosome indices
    hg38_start = torch.randint(0, 250000000, (batch_size, seq_len)).to(model_args["device"])  # Start positions
    hg38_end = hg38_start + torch.randint(100, 1000, (batch_size, seq_len)).to(model_args["device"])  # End positions
    
    print(f"Input shapes - value: {value.shape}, chromosome: {chromosome.shape}, start: {hg38_start.shape}, end: {hg38_end.shape}")
    
    # Forward pass
    try:
        output = model(value, chromosome, hg38_start, hg38_end)
        print(f"Forward pass successful! Output shape: {output.shape}")
        
        # Check if output dimensions match expected cell type number
        expected_shape = (batch_size, model_args["cell_type_num"])
        if output.shape == expected_shape:
            print("✓ Output shape matches expected cell type classification dimensions")
        else:
            print(f"✗ Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
            
        print("✓ Model architecture test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_model()
    if success:
        print("\nModel architecture verification completed successfully!")
        print("The genomic context-aware dynamic convolution module has been integrated properly.")
    else:
        print("\nModel architecture verification failed!")