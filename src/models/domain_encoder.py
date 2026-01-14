import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from .chromfd_block import Block


class ChromatinDomainEncoder(nn.Module):
    """Specialized encoder for large-scale structural patterns"""
    def __init__(self, d_model, seq_length, window_size=2048, n_layer=32, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # Specialized architecture for chromatin domains
        # Uses very large window sizes to capture structural patterns
        self.backbone = nn.ModuleList([
            self._create_domain_block(d_model, seq_length, window_size, i, **kwargs)
            for i in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def _create_domain_block(self, d_model, seq_length, window_size, layer_idx, **kwargs):
        # Use combination of Mamba and attention for structural patterns
        mixer_cls = lambda **factory_kwargs: Mamba(
            layer_idx=layer_idx,
            **factory_kwargs
        )
        
        norm_cls = lambda **factory_kwargs: nn.LayerNorm(eps=1e-5, **factory_kwargs)
        
        mlp_cls = lambda **factory_kwargs: GatedMLP(
            hidden_features=d_model * 4,
            out_features=d_model,
            **factory_kwargs
        )
        
        block = Block(
            d_model,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            fused_add_norm=False,
            residual_in_fp32=True,
            wpsa_window_size=window_size,
            shift_size=window_size//2,
            wpsa_heads=max(1, d_model // 256),
            seq_length=seq_length,
        )
        block.layer_idx = layer_idx
        return block
    
    def forward(self, x):
        residual = None
        for layer in self.backbone:
            x, residual = layer(x, residual)
        
        if residual is not None:
            x = (x + residual)
        x = self.norm(x)
        return x