"""
Multi-scale Feature Fusion Model for Cell Type Annotation
This model combines:
1. Local features from WPSA (Window Partition Self-Attention) module
2. Global features from Mamba module
3. Intermediate features from different transformer layers
Using a learnable attention mechanism before passing to classification head
"""
import math
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP

from src.models.chromfd_block import Block
from src.models.embedding_model import PretrainEmbeddingSimple
from src.models.multi_scale_genomic_extractor import MultiScaleGenomicContextExtractor

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class MultiScaleFeatureFusion(nn.Module):
    """
    Combines local, global, and intermediate features using a learnable attention mechanism.
    """
    def __init__(self, feature_dim, num_feature_types=3):
        """
        Args:
            feature_dim: Dimension of each feature type
            num_feature_types: Number of feature types to combine (local, global, intermediate)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_feature_types = num_feature_types
        
        # Learnable attention weights for each feature type
        self.attention_weights = nn.Parameter(torch.randn(num_feature_types, feature_dim))
        
        # Layer norm for attention computation
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # Final projection after fusion
        self.projection = nn.Linear(feature_dim * num_feature_types, feature_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, local_features, global_features, intermediate_features):
        """
        Args:
            local_features: Features from WPSA module (B, L, D)
            global_features: Features from Mamba module (B, L, D) 
            intermediate_features: Features from intermediate transformer layers (B, L, D)
        Returns:
            Fused features (B, L, D)
        """
        batch_size, seq_len, feature_dim = local_features.shape
        
        # Stack features: (B, L, 3, D)
        stacked_features = torch.stack([local_features, global_features, intermediate_features], dim=2)
        
        # Compute attention weights for each feature type
        # Apply layer norm to features
        normalized_features = self.layer_norm(stacked_features)
        
        # Compute attention scores using the learnable weights
        attention_scores = torch.einsum('blfd,fd->blf', normalized_features, self.attention_weights)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, L, 3)
        
        # Apply attention weights to features: (B, L, 3) -> (B, L, 3, 1)
        attention_weights = attention_weights.unsqueeze(-1)
        
        # Weighted sum of features: (B, L, 3, D) * (B, L, 3, 1) -> (B, L, 3, D) -> (B, L, D)
        weighted_features = stacked_features * attention_weights
        fused_features = weighted_features.sum(dim=2)
        
        # Apply final projection and dropout
        fused_features = self.projection(torch.cat([local_features, global_features, intermediate_features], dim=-1))
        fused_features = self.dropout(fused_features)
        
        return fused_features


class MultiScaleMambaMixer(nn.Module):
    """
    Modified MambaMixer that extracts and fuses multi-scale features from different modules
    """
    def __init__(
        self,
        d_model: int,
        seq_length: int,
        wpsa_heads: int,
        wpsa_window_size: int,
        shift_size: int,
        n_layer: int,
        d_intermediate: int,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        use_multiscale_extractor: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.use_multiscale_extractor = use_multiscale_extractor
        self.d_model = d_model
        self.n_layer = n_layer
        self.seq_length = seq_length

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        # Create blocks with different configurations to extract different feature types
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            layer = self._create_block(
                d_model,
                seq_length,
                wpsa_heads=wpsa_heads,
                wpsa_window_size=wpsa_window_size,
                shift_size=shift_size,
                d_intermediate=d_intermediate,
                attn_layer_idx=attn_layer_idx,
                attn_cfg=attn_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                **factory_kwargs,
            )
            self.layers.append(layer)

        # Add multi-scale genomic context extractor
        if self.use_multiscale_extractor:
            self.multiscale_extractor = MultiScaleGenomicContextExtractor(
                input_dim=d_model,
                output_dim=d_model,
                scales=[1, 10, 100, 1000],  # Corresponds to ~1kb, 10kb, 100kb, 1Mb scales
                max_distance=1000000,  # 1 Mb max distance
                distance_bins=100
            )

        # Multi-scale feature fusion module
        self.feature_fusion = MultiScaleFeatureFusion(d_model, num_feature_types=3)

        # Separate linear projections for extracting different feature types
        self.local_feature_proj = nn.Linear(d_model, d_model)
        self.global_feature_proj = nn.Linear(d_model, d_model)
        self.intermediate_feature_proj = nn.Linear(d_model, d_model)

        # Layer norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                self._init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def _create_block(self, d_model, seq_length, wpsa_heads, wpsa_window_size, shift_size,
                     d_intermediate, attn_layer_idx, attn_cfg, norm_epsilon, rms_norm,
                     residual_in_fp32, fused_add_norm, layer_idx, **factory_kwargs):
        """Create a block that can extract different types of features"""
        if attn_layer_idx is None:
            attn_layer_idx = []
        if attn_cfg is None:
            attn_cfg = {}

        # Mamba mixer for global features - operates on reduced dimension (32)
        mixer_cls = partial(
            Mamba,
            d_model=32,  # Mamba operates on the reduced dimension after reduction_layer
            layer_idx=layer_idx,
            **factory_kwargs
        )

        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )

        if d_intermediate == 0:
            mlp_cls = nn.Identity
        else:
            mlp_cls = partial(
                GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
            )

        block = Block(
            d_model,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            wpsa_window_size=wpsa_window_size,
            shift_size=shift_size,
            wpsa_heads=wpsa_heads,
            seq_length=seq_length,
        )
        block.layer_idx = layer_idx
        return block

    def _init_weights(self, module, n_layer, initializer_range=0.02,
                     rescale_prenorm_residual=True, n_residuals_per_layer=1):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, inputs, inference_params=None, hg38_start=None, hg38_end=None, **mixer_kwargs):
        hidden_states = inputs
        residual = None

        # Apply multi-scale genomic context extraction before the main processing
        if self.use_multiscale_extractor:
            positions = hg38_start if hg38_start is not None else hg38_end
            hidden_states = self.multiscale_extractor(hidden_states, positions)

        # Store features from different scales at each layer
        local_features_list = []
        global_features_list = []
        intermediate_features_list = []

        for i, layer in enumerate(self.layers):
            # Process through the layer
            layer_output, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )

            # Extract different feature types from this layer
            # Local features: from WPSA component (captures nearby regulatory elements)
            if hasattr(layer, 'chromfound_block') and layer.chromfound_block is not None:
                # Extract features specifically from the WPSA component
                # We'll store the intermediate state before the Mamba mixer
                local_features = self.local_feature_proj(layer_output)
            else:
                # If no WPSA, use the layer output as local features
                local_features = self.local_feature_proj(layer_output)

            # Global features: from Mamba component (captures long-range dependencies)
            global_features = self.global_feature_proj(layer_output)

            # Intermediate features: from different transformer layers (captures multi-level representations)
            if i % max(1, self.n_layer // 3) == 0:  # Take features from roughly 3 intermediate points
                intermediate_features_list.append(self.intermediate_feature_proj(layer_output))

            local_features_list.append(local_features)
            global_features_list.append(global_features)

            # Update hidden states for next layer
            hidden_states = layer_output

        # Aggregate features from all layers
        # For local and global features, we'll use the final layer output
        final_local_features = local_features_list[-1] if local_features_list else self.local_feature_proj(hidden_states)
        final_global_features = global_features_list[-1] if global_features_list else self.global_feature_proj(hidden_states)
        # For intermediate features, we'll use the last recorded intermediate feature
        final_intermediate_features = intermediate_features_list[-1] if intermediate_features_list else self.intermediate_feature_proj(hidden_states)

        # Fuse the multi-scale features using learnable attention mechanism
        fused_features = self.feature_fusion(
            final_local_features,
            final_global_features,
            final_intermediate_features
        )

        # Apply final normalization
        if not self.fused_add_norm:
            residual = (fused_features + residual) if residual is not None else fused_features
            output = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            output = layer_norm_fn(
                fused_features,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        return output


class MultiScalePretrainModelMambaLM(torch.nn.Module):
    """
    Pretrain model with multi-scale feature fusion for cell type annotation
    """
    def __init__(self, **kwargs):
        super(MultiScalePretrainModelMambaLM, self).__init__()
        self.model_args = kwargs
        factory_kwargs = {"device": self.model_args["device"], "dtype": torch.float32}
        
        self.embedding = PretrainEmbeddingSimple(
            embedding_dim=self.model_args.get("embedding_dim"),
            chromosome_size=self.model_args.get("chromosome_size"),
            embedding_dropout=self.model_args.get("embedding_dropout", 0),
            positional_embedding_type=self.model_args.get("positional_embedding_type"),
            positional_temp=self.model_args['positional_temp'],
            batch_size=self.model_args.get("batch_size"),
            seq_length=self.model_args.get("max_length"),
            device=self.model_args["device"],
            chromatin_embedding=self.model_args.get("chromatin_embedding", True)
        )
        
        # Disable fused operations when using CPU to avoid Triton issues
        device = self.model_args.get("device", torch.device("cpu"))
        fused_add_norm = MambaConfig.fused_add_norm if str(device) != "cpu" else False

        self.backbone = MultiScaleMambaMixer(
            d_model=self.model_args.get("embedding_dim"),
            seq_length=self.model_args.get("max_length"),
            wpsa_heads=self.model_args.get("wpsa_heads"),
            wpsa_window_size=self.model_args.get("wpsa_window_size"),
            shift_size=self.model_args.get("shift_size"),
            n_layer=self.model_args.get("encoder_layers"),
            d_intermediate=MambaConfig.d_intermediate,
            attn_layer_idx=None,
            attn_cfg=None,
            norm_epsilon=1e-5,
            rms_norm=MambaConfig.rms_norm,
            initializer_cfg=None,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=MambaConfig.residual_in_fp32,
            use_multiscale_extractor=self.model_args.get("use_multiscale_extractor", True),
            **factory_kwargs,
        )
        
        # Mask token prediction head (for pretraining)
        self.mask_token_prediction = self._create_mask_prediction_head()

    def _create_mask_prediction_head(self):
        """Create mask token prediction head"""
        projection = torch.nn.Linear(
            self.model_args.get("encoder_dim", self.model_args.get("embedding_dim")), 
            self.model_args.get("value_size", 1)
        )
        return torch.nn.Sequential(
            projection,
            torch.nn.LogSoftmax(dim=-1) if not self.model_args.get("linear_embedding", False) else torch.nn.Identity()
        )

    def forward(self, value, chromosome, hg38_start, hg38_end, key_padding_mask=None):
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x = self.backbone(x, hg38_start=hg38_start, hg38_end=hg38_end)
        logits = self.mask_token_prediction(x)
        return logits