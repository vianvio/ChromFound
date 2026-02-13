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
        "residual_in_fp32": False,  # Avoid fp32 requirements
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
    
    # Basic functionality test - check that LoRA parameters exist and are trainable
    has_lora_params = len(lora_params) > 0
    lora_params_trainable = all(p.requires_grad for p in lora_params.values())
    non_lora_params_frozen = all(not p.requires_grad for name, p in model.named_parameters() if 'lora_' not in name)
    
    print(f"\nFunctionality checks:")
    print(f"- Has LoRA parameters: {has_lora_params}")
    print(f"- LoRA parameters are trainable: {lora_params_trainable}")
    print(f"- Non-LoRA parameters are frozen: {non_lora_params_frozen}")
    
    if has_lora_params and lora_params_trainable and non_lora_params_frozen:
        print("\nLoRA integration test PASSED!")
        return True
    else:
        print("\nLoRA integration test FAILED!")
        return False


if __name__ == "__main__":
    success = test_lora_integration()
    if success:
        print("\nAll tests passed! LoRA implementation is working correctly.")
    else:
        print("\nSome tests failed!")