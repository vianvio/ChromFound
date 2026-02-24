"""
LoRA (Low-Rank Adaptation) Adapter for Parameter-Efficient Finetuning

This module provides LoRA adapters that can be injected into Mamba and MHA layers
to enable parameter-efficient finetuning of the ChromFound model.
"""

import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """
    LoRA adapter for Mamba/MHA layers.
    
    Uses low-rank decomposition to reduce trainable parameters:
    - Decomposes weight matrix W into W + BA where B is r x out_features and A is in_features x r
    - Only A and B are trainable, original weights are frozen
    - Scaling factor alpha/r controls the contribution of LoRA branch
    
    Reference: https://arxiv.org/abs/2106.09685
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 16, 
        alpha: int = 32, 
        dropout: float = 0.1
    ):
        """
        Initialize LoRA adapter.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            r: Rank for low-rank decomposition (smaller = fewer params)
            alpha: Scaling factor (typically alpha = 2r or alpha = r)
            dropout: Dropout rate for LoRA branch
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize: A with Kaiming uniform, B with zeros
        # This ensures initial state is identity mapping (no perturbation)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, in_features)
            
        Returns:
            LoRA-adapted output of shape (batch, seq_len, out_features)
        """
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


class LoRAInjectedMamba(nn.Module):
    """
    Wrapper that injects LoRA adapters into Mamba layers.
    
    This wrapper wraps the original Mamba module and applies LoRA
    adaptation to the in_proj linear projection.
    """
    def __init__(self, mamba_module: nn.Module, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1):
        """
        Wrap a Mamba module with LoRA adapters.
        
        Args:
            mamba_module: Original Mamba module to wrap
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout rate
        """
        super().__init__()
        self.mamba = mamba_module
        self.lora_adapter = None
        
        # Find and inject LoRA into in_proj if it exists
        if hasattr(mamba_module, 'in_proj') and mamba_module.in_proj is not None:
            d_model = mamba_module.in_proj.in_features
            self.lora_adapter = LoRAAdapter(
                d_model, d_model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
            )
            # Move adapter to same device as mamba
            self.lora_adapter.to(next(mamba_module.parameters()).device)
        elif hasattr(mamba_module, 'Wqkv') and mamba_module.Wqkv is not None:
            d_model = mamba_module.Wqkv.in_features
            self.lora_adapter = LoRAAdapter(
                d_model, d_model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
            )
            self.lora_adapter.to(next(mamba_module.parameters()).device)
    
    def forward(self, x, inference_params=None, **kwargs):
        """
        Forward pass with optional LoRA adaptation.
        
        Args:
            x: Input tensor
            inference_params: Parameters for inference caching
            **kwargs: Additional arguments passed to Mamba
            
        Returns:
            Output tensor
        """
        # Standard Mamba forward
        output = self.mamba(x, inference_params=inference_params, **kwargs)
        
        # Add LoRA contribution if adapter exists
        if self.lora_adapter is not None:
            lora_output = self.lora_adapter(x)
            output = output + lora_output
        
        return output


def inject_lora_into_model(model: nn.Module, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1) -> int:
    """
    Inject LoRA adapters into all Mamba/MHA layers of a model.
    
    This function freezes the backbone parameters and injects LoRA adapters
    into Mamba and MHA layers for parameter-efficient finetuning.
    
    Args:
        model: Model to inject LoRA adapters into
        lora_r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout rate
        
    Returns:
        Number of LoRA adapters injected
    """
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mha import MHA
    
    lora_count = 0
    
    # Freeze all backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Inject LoRA into each layer's mixer
    for layer_idx, layer in enumerate(model.backbone.layers):
        if isinstance(layer.mixer, (Mamba, MHA)):
            # Replace mixer with LoRA-injected version
            original_mixer = layer.mixer
            layer.mixer = LoRAInjectedMamba(
                original_mixer, 
                lora_r=lora_r, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout
            )
            lora_count += 1
    
    return lora_count


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count trainable and total parameters in a model.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params
