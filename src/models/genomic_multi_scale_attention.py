import math
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from functools import partial


class GenomicContextAttention(nn.Module):
    """Genomic context attention mechanism that weighs features based on regulatory element annotations"""
    
    def __init__(self, embedding_dim, num_genomic_contexts=4):
        """
        Args:
            embedding_dim: Dimension of the input embeddings
            num_genomic_contexts: Number of genomic contexts (promoter, gene_body, enhancer, intergenic)
        """
        super(GenomicContextAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_genomic_contexts = num_genomic_contexts
        
        # Attention weights for different genomic contexts
        self.context_query = nn.Linear(embedding_dim, embedding_dim)
        self.context_key = nn.Linear(embedding_dim, embedding_dim)
        self.context_value = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, genomic_contexts=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            genomic_contexts: Tensor indicating genomic context for each position (optional)
        """
        batch_size, seq_len, embedding_dim = x.size()
        
        # Apply layer norm
        x_norm = self.layer_norm(x)
        
        # Generate queries, keys, values
        Q = self.context_query(x_norm)  # (batch_size, seq_len, embedding_dim)
        K = self.context_key(x_norm)    # (batch_size, seq_len, embedding_dim)
        V = self.context_value(x_norm)  # (batch_size, seq_len, embedding_dim)
        
        # Calculate attention scores
        attention_scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(embedding_dim)
        
        # If genomic contexts are provided, apply context-specific masking
        if genomic_contexts is not None:
            # Expand genomic contexts to match attention matrix dimensions
            # genomic_contexts shape: (batch_size, seq_len)
            genomic_contexts_expanded = genomic_contexts.unsqueeze(1).expand(-1, seq_len, -1)
            context_mask = (genomic_contexts_expanded == genomic_contexts_expanded.transpose(-2, -1)).float()
            attention_scores = attention_scores * context_mask
            
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.bmm(attention_weights, V)
        
        # Project output
        output = self.output_projection(attended_values)
        
        # Residual connection
        return x + output


class MultiScaleFeatureExtractor(nn.Module):
    """Extracts features at multiple genomic scales: promoter, gene body, enhancer regions"""
    
    def __init__(self, embedding_dim, kernel_sizes=[3, 7, 15]):
        """
        Args:
            embedding_dim: Dimension of the input embeddings
            kernel_sizes: List of kernel sizes for different genomic scales
        """
        super(MultiScaleFeatureExtractor, self).__init__()
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        
        # Convolutional layers for different scales
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=k, padding=k//2, groups=embedding_dim//4)
            for k in kernel_sizes
        ])
        
        # Projection layers to combine multi-scale features
        self.scale_projections = nn.ModuleList([
            nn.Linear(embedding_dim * 2, embedding_dim)  # Concatenate original + convolved features
            for _ in kernel_sizes
        ])
        
        # Final fusion layer
        self.fusion_layer = nn.Linear(embedding_dim * len(kernel_sizes), embedding_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, embedding_dim = x.size()
        
        # Transpose for convolution: (batch_size, embedding_dim, seq_len)
        x_conv = x.transpose(1, 2)
        
        scale_outputs = []
        
        for i, (conv, proj) in enumerate(zip(self.scale_convs, self.scale_projections)):
            # Apply convolution at different scales
            conv_out = conv(x_conv)  # (batch_size, embedding_dim, seq_len)
            
            # Transpose back: (batch_size, seq_len, embedding_dim)
            conv_out = conv_out.transpose(1, 2)
            
            # Concatenate original and convolved features
            combined_features = torch.cat([x, conv_out], dim=-1)  # (batch_size, seq_len, 2*embedding_dim)
            
            # Project back to original dimension
            projected = proj(combined_features)  # (batch_size, seq_len, embedding_dim)
            
            # Apply activation
            activated = torch.relu(projected)
            
            scale_outputs.append(activated)
        
        # Concatenate all scale outputs
        all_scales = torch.cat(scale_outputs, dim=-1)  # (batch_size, seq_len, len(kernel_sizes)*embedding_dim)
        
        # Fuse all scales
        fused_output = self.fusion_layer(all_scales)  # (batch_size, seq_len, embedding_dim)
        
        # Apply layer norm with residual connection
        return self.layer_norm(x + fused_output)


class MultiScaleGenomicModel(nn.Module):
    """Complete model with multi-scale feature extraction and genomic context attention"""

    def __init__(self, embedding_dim, num_classes, max_length, encoder_layers=64,
                 kernel_sizes=[3, 7, 15], num_genomic_contexts=4):
        super(MultiScaleGenomicModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes

        # Multi-scale feature extractor
        self.multi_scale_extractor = MultiScaleFeatureExtractor(embedding_dim, kernel_sizes)

        # Genomic context attention
        self.genomic_attention = GenomicContextAttention(embedding_dim, num_genomic_contexts)

        # Mamba backbone (using the same structure as the original)
        self.mamba_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(encoder_layers)
        ])

        # Post-processing layers for cell type prediction
        self.post_backbone_dropout = nn.Dropout(p=0.3)
        self.feature_projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            nn.GELU()
        )

        # Initialize feature projection weights
        self._init_weights(self.feature_projection)

        in_feature = max_length
        self.ft_cell_type_projection = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

        # Initialize cell type projection weights
        self._init_weights(self.ft_cell_type_projection)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, genomic_contexts=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim) - this is the embedded representation
            genomic_contexts: Optional genomic context information
        """
        # Apply multi-scale feature extraction
        x = self.multi_scale_extractor(x)

        # Apply genomic context attention
        x = self.genomic_attention(x, genomic_contexts)

        # Pass through Mamba-like transformer blocks
        for block in self.mamba_blocks:
            x = block(x)

        # Apply feature projection
        x = self.feature_projection(x)
        x = torch.squeeze(x, dim=-1)  # Remove last dimension: (batch_size, seq_len)
        x = self.post_backbone_dropout(x)

        # Final projection for cell type prediction
        x_cell_type_prediction = self.ft_cell_type_projection(x)

        return x_cell_type_prediction