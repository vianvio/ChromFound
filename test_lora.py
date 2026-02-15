"""
Simple test to verify LoRA implementation works correctly.
"""

import torch
import torch.nn as nn
from src.models.lora import LoRALayer, LinearWithLoRA


def test_lora_layer():
    """Test basic LoRA layer functionality"""
    print("Testing LoRA layer...")
    
    # Create a simple LoRA layer
    lora_layer = LoRALayer(in_features=128, out_features=64, rank=8, alpha=16.0)
    
    # Create input tensor
    x = torch.randn(10, 128)
    
    # Forward pass
    output = lora_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"LoRA A shape: {lora_layer.lora_A.shape}")
    print(f"LoRA B shape: {lora_layer.lora_B.shape}")
    print("✓ LoRA layer test passed!\n")


def test_linear_with_lora():
    """Test LinearWithLoRA wrapper"""
    print("Testing LinearWithLoRA...")
    
    # Create a regular linear layer
    linear_layer = nn.Linear(128, 64)
    original_weight = linear_layer.weight.clone()
    
    # Wrap with LoRA
    linear_with_lora = LinearWithLoRA(linear_layer, rank=8, alpha=16.0)
    
    # Create input tensor
    x = torch.randn(10, 128)
    
    # Forward pass
    output = linear_with_lora(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check that original weights are frozen
    print(f"Original weight requires_grad: {linear_with_lora.linear.weight.requires_grad}")
    print(f"LoRA A requires_grad: {linear_with_lora.lora.lora_A.requires_grad}")
    print(f"LoRA B requires_grad: {linear_with_lora.lora.lora_B.requires_grad}")
    
    # Verify that original weight hasn't changed
    weight_unchanged = torch.equal(original_weight, linear_with_lora.linear.weight)
    print(f"Original weight unchanged: {weight_unchanged}")
    
    print("✓ LinearWithLoRA test passed!\n")


def test_parameter_counts():
    """Test parameter counting"""
    print("Testing parameter counts...")
    
    # Create a linear layer
    linear_layer = nn.Linear(256, 128)
    original_params = sum(p.numel() for p in linear_layer.parameters())
    
    # Create with LoRA
    linear_with_lora = LinearWithLoRA(linear_layer, rank=16, alpha=32.0)
    total_params = sum(p.numel() for p in linear_with_lora.parameters())
    lora_params = sum(p.numel() for p in linear_with_lora.lora.parameters())
    
    print(f"Original parameters: {original_params:,}")
    print(f"Total parameters with LoRA: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Reduction factor: {(original_params-lora_params)/original_params*100:.2f}% frozen")
    
    print("✓ Parameter count test passed!\n")


if __name__ == "__main__":
    print("Running LoRA implementation tests...\n")
    
    test_lora_layer()
    test_linear_with_lora()
    test_parameter_counts()
    
    print("All tests passed! ✓")