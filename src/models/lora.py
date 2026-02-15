"""
LoRA (Low-Rank Adaptation) implementation for ChromFound model.
This module adds LoRA layers to the attention mechanisms in the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class LoRALayer(nn.Module):
    """Base LoRA layer for linear transformations"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super(LoRALayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Initialize low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Optional dropout
        if dropout > 0.0:
            self.lora_dropout = nn.Dropout(dropout)
        else:
            self.lora_dropout = lambda x: x
            
        # Initialize A with random Gaussian and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout to input
        x = self.lora_dropout(x)
        
        # Compute (x @ A.T @ B.T) * scaling
        x = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        return x


class LinearWithLoRA(nn.Module):
    """Linear layer with integrated LoRA adaptation"""
    
    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super(LinearWithLoRA, self).__init__()
        
        # Store the original linear layer (frozen)
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        
        # Add LoRA layer
        self.lora = LoRALayer(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear transformation
        output = self.linear(x)
        
        # Add LoRA adaptation
        output = output + self.lora(x)
        
        return output
    
    def merge_lora_weights(self):
        """Merge LoRA weights back into the original linear layer"""
        # Calculate the merged weight: W_original + (B @ A) * scaling
        lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
        merged_weight = self.linear.weight + lora_weight
        self.linear.weight.data.copy_(merged_weight)
        
        # Reset LoRA parameters
        self.lora.lora_A.data.zero_()
        self.lora.lora_B.data.zero_()
        
    def unmerge_lora_weights(self):
        """Unmerge LoRA weights from the original linear layer"""
        # Calculate the LoRA contribution: (B @ A) * scaling
        lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
        original_weight = self.linear.weight - lora_weight
        self.linear.weight.data.copy_(original_weight)


class MambaWithLoRA(nn.Module):
    """Mamba layer with LoRA applied to selective linear layers"""
    
    def __init__(self, mamba_module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super(MambaWithLoRA, self).__init__()
        
        # Store the original Mamba module
        self.mamba = mamba_module
        
        # Apply LoRA to key linear layers in Mamba
        # These are the typical linear layers in Mamba that we want to adapt
        if hasattr(mamba_module, 'in_proj'):
            self.mamba.in_proj = LinearWithLoRA(mamba_module.in_proj, rank, alpha, dropout)
            
        if hasattr(mamba_module, 'x_proj'):
            self.mamba.x_proj = LinearWithLoRA(mamba_module.x_proj, rank, alpha, dropout)
            
        if hasattr(mamba_module, 'dt_proj'):
            self.mamba.dt_proj = LinearWithLoRA(mamba_module.dt_proj, rank, alpha, dropout)
            
        if hasattr(mamba_module, 'out_proj'):
            self.mamba.out_proj = LinearWithLoRA(mamba_module.out_proj, rank, alpha, dropout)
            
        # Also apply to conv layers if they are linear-like
        if hasattr(mamba_module, 'conv1d') and hasattr(mamba_module.conv1d, 'weight'):
            # For convolutional layers, we can apply a similar low-rank adaptation
            conv_weight = mamba_module.conv1d.weight
            if conv_weight.dim() == 3:  # (out_channels, in_channels, kernel_size)
                # Apply LoRA to the conv weight by treating it as a linear transformation
                conv_out, conv_in, kernel_size = conv_weight.shape
                # We'll create a low-rank adaptation for the spatial dimensions
                self.conv_lora_A = nn.Parameter(torch.zeros(rank, conv_in * kernel_size))
                self.conv_lora_B = nn.Parameter(torch.zeros(conv_out * kernel_size, rank))
                self.conv_scaling = alpha / rank
                nn.init.kaiming_uniform_(self.conv_lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.conv_lora_B)
                
    def forward(self, *args, **kwargs):
        return self.mamba.forward(*args, **kwargs)


class WindowAttentionWithLoRA(nn.Module):
    """WindowAttention1D with LoRA applied to QKV and projection layers"""
    
    def __init__(self, attention_module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super(WindowAttentionWithLoRA, self).__init__()
        
        # Store the original attention module
        self.attention = attention_module
        
        # Apply LoRA to the QKV linear layer
        if hasattr(attention_module, 'qkv'):
            self.attention.qkv = LinearWithLoRA(attention_module.qkv, rank, alpha, dropout)
        
        # Apply LoRA to the output projection layer
        if hasattr(attention_module, 'proj'):
            self.attention.proj = LinearWithLoRA(attention_module.proj, rank, alpha, dropout)
        elif hasattr(attention_module, 'projection'):  # Alternative name
            self.attention.projection = LinearWithLoRA(attention_module.projection, rank, alpha, dropout)
        elif hasattr(attention_module, 'out_proj'):  # Another alternative name
            self.attention.out_proj = LinearWithLoRA(attention_module.out_proj, rank, alpha, dropout)
    
    def forward(self, *args, **kwargs):
        return self.attention.forward(*args, **kwargs)


class BlockWithLoRA(nn.Module):
    """Block with LoRA applied to attention and other linear layers"""
    
    def __init__(self, block_module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super(BlockWithLoRA, self).__init__()
        
        # Store the original block
        self.block = block_module
        
        # Apply LoRA to the mixer (could be Mamba or attention)
        if hasattr(block_module, 'mixer'):
            if 'mamba' in str(type(block_module.mixer)).lower():
                self.block.mixer = MambaWithLoRA(block_module.mixer, rank, alpha, dropout)
            elif 'attention' in str(type(block_module.mixer)).lower():
                # If mixer is attention-based
                self.block.mixer = WindowAttentionWithLoRA(block_module.mixer, rank, alpha, dropout)
        
        # Apply LoRA to reduction and expansion layers
        if hasattr(block_module, 'reduction_layer'):
            self.block.reduction_layer = LinearWithLoRA(
                block_module.reduction_layer, rank, alpha, dropout
            )
        
        if hasattr(block_module, 'expansion_layer'):
            self.block.expansion_layer = LinearWithLoRA(
                block_module.expansion_layer, rank, alpha, dropout
            )
        
        # Apply LoRA to MLP layers if they exist
        if hasattr(block_module, 'mlp') and hasattr(block_module.mlp, 'fc1'):
            self.block.mlp.fc1 = LinearWithLoRA(block_module.mlp.fc1, rank, alpha, dropout)
        if hasattr(block_module, 'mlp') and hasattr(block_module.mlp, 'fc2'):
            self.block.mlp.fc2 = LinearWithLoRA(block_module.mlp.fc2, rank, alpha, dropout)
        
        # Apply LoRA to transformer block if present
        if hasattr(block_module, 'chromfound_block'):
            if hasattr(block_module.chromfound_block, 'attn'):
                block_module.chromfound_block.attn = WindowAttentionWithLoRA(
                    block_module.chromfound_block.attn, rank, alpha, dropout
                )
            if hasattr(block_module.chromfound_block, 'mlp'):
                if hasattr(block_module.chromfound_block.mlp, 'fc1'):
                    block_module.chromfound_block.mlp.fc1 = LinearWithLoRA(
                        block_module.chromfound_block.mlp.fc1, rank, alpha, dropout
                    )
                if hasattr(block_module.chromfound_block.mlp, 'fc2'):
                    block_module.chromfound_block.mlp.fc2 = LinearWithLoRA(
                        block_module.chromfound_block.mlp.fc2, rank, alpha, dropout
                    )
    
    def forward(self, *args, **kwargs):
        return self.block.forward(*args, **kwargs)


class MambaMixerWithLoRA(nn.Module):
    """MambaMixer with LoRA applied to all blocks"""
    
    def __init__(self, mixer_module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super(MambaMixerWithLoRA, self).__init__()
        
        # Store the original mixer
        self.mixer = mixer_module
        
        # Apply LoRA to each layer/block
        for i, layer in enumerate(mixer_module.layers):
            self.mixer.layers[i] = BlockWithLoRA(layer, rank, alpha, dropout)
    
    def forward(self, *args, **kwargs):
        return self.mixer.forward(*args, **kwargs)


class PretrainModelMambaLMWithLoRA(nn.Module):
    """Main model with LoRA applied to backbone and other components"""
    
    def __init__(self, model, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super(PretrainModelMambaLMWithLoRA, self).__init__()
        
        # Store the original model components
        self.model = model
        
        # Apply LoRA to the backbone (MambaMixer)
        if hasattr(model, 'backbone'):
            self.model.backbone = MambaMixerWithLoRA(model.backbone, rank, alpha, dropout)
        
        # Apply LoRA to mask token prediction head if it has linear layers
        if hasattr(model, 'mask_token_prediction'):
            if hasattr(model.mask_token_prediction, 'projection'):
                self.model.mask_token_prediction.projection = LinearWithLoRA(
                    model.mask_token_prediction.projection, rank, alpha, dropout
                )
        
        # Apply LoRA to embedding layers if they have linear components
        if hasattr(model, 'embedding'):
            if hasattr(model.embedding, 'value_embedding') and hasattr(model.embedding.value_embedding, 'embedding'):
                self.model.embedding.value_embedding.embedding = LinearWithLoRA(
                    model.embedding.value_embedding.embedding, rank, alpha, dropout
                )
            if hasattr(model.embedding, 'chromosome_embedding'):
                # For embedding layers, we can't directly apply LoRA in the same way
                # So we'll handle this specially if needed
                pass
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def get_lora_parameters(self):
        """Get all LoRA parameters for optimizer"""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_params.append(param)
        return lora_params
    
    def get_non_lora_parameters(self):
        """Get all non-LoRA parameters (frozen)"""
        non_lora_params = []
        for name, param in self.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                non_lora_params.append(param)
        return non_lora_params


def inject_lora_to_model(model, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
    """
    Inject LoRA layers into the given model.
    
    Args:
        model: The original model to inject LoRA into
        rank: Rank of the low-rank adaptation
        alpha: Scaling factor for LoRA
        dropout: Dropout rate for LoRA layers
    
    Returns:
        Model with LoRA injected
    """
    return PretrainModelMambaLMWithLoRA(model, rank, alpha, dropout)