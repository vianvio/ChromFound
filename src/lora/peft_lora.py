"""
Enhanced LoRA implementation compatible with PEFT-style approaches
"""
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
import math


class LoraLinear(nn.Module):
    """
    A Linear layer with integrated LoRA (Low-Rank Adaptation) 
    This implementation follows the PEFT library approach
    """
    def __init__(
        self,
        linear_module: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        lora_scaling: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        
        # Store the original linear layer
        self.linear = linear_module
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        
        # LoRA parameters
        self.rank = rank
        self.alpha = alpha
        self.scaling = lora_scaling or (alpha / rank)
        
        # Initialize LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Zero initialization for LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Mark the LoRA parameters as trainable
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        
        # Freeze the original linear layer parameters
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear transformation
        output = self.linear(x)
        
        # LoRA transformation: (x @ A.T @ B.T) * scaling
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        # Add LoRA contribution to original output
        return output + lora_output

    def merge_lora_weights(self):
        """
        Merge LoRA weights back into the original linear layer
        """
        with torch.no_grad():
            # Calculate the merged weight
            merged_weight = self.linear.weight + (
                self.lora_B @ self.lora_A
            ) * self.scaling
            
            # Update the original layer
            self.linear.weight.copy_(merged_weight)
            
            # Reset LoRA parameters
            self.lora_A.zero_()
            self.lora_B.zero_()

    def unmerge_lora_weights(self):
        """
        Unmerge LoRA weights from the original linear layer
        """
        with torch.no_grad():
            # Subtract LoRA contribution from the original weight
            unmerged_weight = self.linear.weight - (
                self.lora_B @ self.lora_A
            ) * self.scaling
            
            # Update the original layer
            self.linear.weight.copy_(unmerged_weight)


def replace_linear_with_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: List[str] = ["Linear"],
    lora_scaling: Optional[float] = None
) -> nn.Module:
    """
    Recursively replace Linear layers in the model with LoraLinear layers
    
    Args:
        model: The model to modify
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: Dropout rate for LoRA layers
        target_modules: List of module types to replace
        lora_scaling: Optional custom scaling factor
    
    Returns:
        Modified model with LoRA layers
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            # Recursively apply to child modules
            replace_linear_with_lora(
                module, rank, alpha, dropout, target_modules, lora_scaling
            )
        else:
            # Check if this module should be replaced
            if isinstance(module, nn.Linear):
                # Replace with LoraLinear
                lora_module = LoraLinear(
                    module, rank, alpha, dropout, lora_scaling
                )
                setattr(model, name, lora_module)
                
    return model


def get_lora_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    """
    Get only the LoRA parameters from the model
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def get_non_lora_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    """
    Get all non-LoRA parameters from the model
    """
    non_lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            non_lora_params.append(param)
    return non_lora_params


def freeze_non_lora_parameters(model: nn.Module):
    """
    Freeze all parameters except LoRA parameters
    """
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_all_parameters(model: nn.Module):
    """
    Unfreeze all parameters in the model
    """
    for param in model.parameters():
        param.requires_grad = True