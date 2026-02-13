"""
LoRA (Low-Rank Adaptation) implementation for ChromFound model
This module provides both basic and PEFT-compatible LoRA implementations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from .peft_lora import replace_linear_with_lora, freeze_non_lora_parameters as peft_freeze_non_lora_parameters


class LoRALayer(nn.Module):
    """
    Implements LoRA (Low-Rank Adaptation) for linear layers
    """
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.05):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Initialize low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Scaling factor
        self.scaling = alpha / rank

    def forward(self, x):
        # Apply dropout if enabled
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Compute: x @ A.T @ B.T * scaling
        # This is equivalent to: x @ (A.T @ B.T) * scaling
        result = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result


class LinearWithLoRA(nn.Module):
    """
    Linear layer with integrated LoRA adaptation
    """
    def __init__(self, in_features, out_features, bias=True, rank=16, alpha=32, dropout=0.05):
        super(LinearWithLoRA, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_layer = LoRALayer(in_features, out_features, rank, alpha, dropout)
        
        # Flag to enable/disable LoRA
        self.lora_enabled = True

    def forward(self, x):
        # Standard linear transformation
        output = self.linear(x)
        
        # Add LoRA adaptation if enabled
        if self.lora_enabled:
            output = output + self.lora_layer(x)
        
        return output
    
    def enable_lora(self):
        """Enable LoRA adaptation"""
        self.lora_enabled = True
    
    def disable_lora(self):
        """Disable LoRA adaptation"""
        self.lora_enabled = False


def inject_lora_into_model(model, target_modules=["Linear"], rank=16, alpha=32, dropout=0.05):
    """
    Recursively inject LoRA layers into the model using PEFT-compatible approach
    
    Args:
        model: The model to inject LoRA into
        target_modules: List of module types to replace with LoRA versions
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor for LoRA
        dropout: Dropout probability for LoRA layers
    """
    # Use the PEFT-compatible implementation
    replace_linear_with_lora(
        model=model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules
    )


def get_lora_state_dict(model):
    """
    Get only the LoRA parameters from the model state dict
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param
    return lora_state_dict


def freeze_non_lora_parameters(model):
    """
    Freeze all parameters except LoRA parameters using PEFT approach
    """
    peft_freeze_non_lora_parameters(model)


def unfreeze_all_parameters(model):
    """
    Unfreeze all parameters in the model
    """
    for param in model.parameters():
        param.requires_grad = True