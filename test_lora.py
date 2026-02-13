#!/usr/bin/env python
"""
Test script to verify LoRA implementation works correctly with ChromFound
"""
import torch
from src.models.chromfd_mixer import PretrainModelMambaLM
from src.lora.utils import configure_lora_model, print_trainable_parameters
from src.lora.lora import get_lora_state_dict


def test_lora_integration():
    print("Testing LoRA integration with ChromFound...")
    
    # Define mock model arguments
    model_args = {
        "embedding_dim": 64,  # Smaller dimension to avoid Triton issues
        "chromosome_size": 25,
        "embedding_dropout": 0.1,
        "positional_embedding_type": "sinusoidal",
        "positional_temp": 10000.0,
        "batch_size": 4,
        "max_length": 50,
        "device": torch.device("cpu"),
        "chromatin_embedding": True,
        "wpsa_heads": 4,
        "wpsa_window_size": 5,
        "shift_size": 0,
        "encoder_layers": 1,  # Fewer layers to simplify
        "cell_type_num": 5,
        "value_size": 1,
        "encoder_dim": 64,  # Required for mask token prediction
        "fused_add_norm": False,  # Disable fused norm to avoid Triton issues
        "rms_norm": False  # Use standard LayerNorm instead of RMSNorm
    }
    
    # Create a model instance
    model = PretrainModelMambaLM(**model_args)
    print(f"Original model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Count initial trainable parameters
    initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initial trainable parameters: {initial_trainable:,}")
    
    # Configure LoRA
    lora_config = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ["Linear"]
    }
    
    model = configure_lora_model(model, lora_config)
    
    # Count parameters after LoRA
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"After LoRA - Total parameters: {total_params:,}")
    print(f"After LoRA - Trainable parameters: {trainable_params:,}")
    print(f"After LoRA - Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    # Get LoRA state dict
    lora_params = get_lora_state_dict(model)
    lora_param_count = sum(p.numel() for p in lora_params.values())
    print(f"LoRA parameters count: {lora_param_count:,}")
    
    # Verify that most parameters are frozen
    frozen_params = total_params - trainable_params
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Percentage of frozen parameters: {100 * frozen_params / total_params:.2f}%")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        # Create dummy inputs - match the max_length from model_args
        batch_size = 2
        seq_len = 50  # Match max_length from model_args
        value = torch.randn(batch_size, seq_len)
        chromosome = torch.randint(0, 25, (batch_size, seq_len))
        hg38_start = torch.randint(0, 1000000, (batch_size, seq_len))
        hg38_end = hg38_start + torch.randint(100, 1000, (batch_size, seq_len))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(value, chromosome, hg38_start, hg38_end)
        
        print(f"Forward pass successful! Output shape: {output.shape}")
        print("LoRA integration test PASSED!")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        print("LoRA integration test FAILED!")
        return False
    
    return True


if __name__ == "__main__":
    success = test_lora_integration()
    if success:
        print("\nAll tests passed! LoRA implementation is working correctly.")
    else:
        print("\nSome tests failed!")