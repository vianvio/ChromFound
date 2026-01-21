"""
Genomic Context-Aware Dynamic Convolution Module
Implements a novel approach to capture local patterns in genomic sequences
by dynamically adjusting receptive fields based on genomic context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GenomicContextEncoder(nn.Module):
    """
    Encodes genomic context by fusing position information and sequence features
    """
    def __init__(self, embedding_dim, hidden_dim=None):
        super(GenomicContextEncoder, self).__init__()
        if hidden_dim is None:
            hidden_dim = embedding_dim
        
        self.position_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # 2 for start/end positions
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        self.sequence_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, sequence_features, pos_start, pos_end):
        """
        Args:
            sequence_features: (B, L, D) - embedded sequence features
            pos_start: (B, L) - start positions
            pos_end: (B, L) - end positions
        """
        B, L, D = sequence_features.shape
        
        # Encode positional information
        pos_info = torch.stack([pos_start, pos_end], dim=-1)  # (B, L, 2)
        pos_encoded = self.position_encoder(pos_info)  # (B, L, hidden_dim//2)
        
        # Encode sequence features
        seq_encoded = self.sequence_encoder(sequence_features)  # (B, L, hidden_dim//2)
        
        # Concatenate and fuse
        context_input = torch.cat([seq_encoded, pos_encoded], dim=-1)  # (B, L, hidden_dim)
        genomic_context = self.context_fusion(context_input)  # (B, L, hidden_dim)
        
        return genomic_context


class DynamicConvKernelGenerator(nn.Module):
    """
    Generates adaptive convolution kernels based on genomic context
    """
    def __init__(self, embedding_dim, kernel_size=3, num_kernels=8):
        super(DynamicConvKernelGenerator, self).__init__()
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.embedding_dim = embedding_dim
        
        # Kernel generator network
        self.kernel_generator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU()
        )
        
        # Generate kernels for different genomic regions
        self.kernel_weights = nn.Parameter(
            torch.randn(num_kernels, embedding_dim, embedding_dim, kernel_size)
        )
        nn.init.xavier_uniform_(self.kernel_weights)
        
        # Kernel selector based on context
        self.kernel_selector = nn.Linear(embedding_dim // 4, num_kernels)
        
    def forward(self, genomic_context):
        """
        Args:
            genomic_context: (B, L, D) - genomic context from encoder
        Returns:
            kernels: (B, L, D, D, kernel_size) - dynamic kernels
        """
        B, L, D = genomic_context.shape
        
        # Generate kernel parameters from context
        kernel_params = self.kernel_generator(genomic_context)  # (B, L, D//4)
        
        # Select appropriate kernels based on context
        kernel_selection_weights = torch.softmax(
            self.kernel_selector(kernel_params), dim=-1
        )  # (B, L, num_kernels)
        
        # Weighted combination of kernels
        # Expand kernel weights: (num_kernels, D, D, kernel_size) -> (B, L, num_kernels, D, D, kernel_size)
        expanded_kernels = self.kernel_weights.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1, -1, -1)
        
        # Apply selection weights: (B, L, num_kernels) -> (B, L, num_kernels, 1, 1, 1)
        selection_expanded = kernel_selection_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Weighted sum to get dynamic kernels: (B, L, D, D, kernel_size)
        dynamic_kernels = torch.sum(selection_expanded * expanded_kernels, dim=2)
        
        return dynamic_kernels


class GenomicAwarePositionEmbedding(nn.Module):
    """
    Enhanced position embedding that incorporates genomic structure awareness
    """
    def __init__(self, d_model, max_positions=5000):
        super(GenomicAwarePositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_positions = max_positions
        
        # Standard positional encoding
        pe = torch.zeros(max_positions, d_model)
        position = torch.arange(0, max_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_pos, d_model)
        self.register_buffer('pe', pe)
        
        # Genomic structure awareness layer
        self.genomic_structure_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x, pos_start, pos_end):
        """
        Args:
            x: (B, L, D) - input sequence
            pos_start: (B, L) - start positions
            pos_end: (B, L) - end positions
        """
        B, L, D = x.shape
        pe = self.pe[:, :L, :].expand(B, -1, -1)  # (B, L, D)
        
        # Incorporate genomic structure awareness
        genomic_structure_weights = self.genomic_structure_net(pe)  # (B, L, D)
        
        # Adjust positional encoding based on genomic structure
        adjusted_pe = pe * genomic_structure_weights
        
        return adjusted_pe


class GenomicDynamicConvLayer(nn.Module):
    """
    A single layer of genomic context-aware dynamic convolution
    """
    def __init__(self, embedding_dim, kernel_size=3, dilation=1, dropout=0.1):
        super(GenomicDynamicConvLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Components
        self.context_encoder = GenomicContextEncoder(embedding_dim)
        self.kernel_generator = DynamicConvKernelGenerator(embedding_dim, kernel_size)

        # Normalization and output processing
        self.norm = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_start, pos_end):
        """
        Args:
            x: (B, L, D) - input sequence
            pos_start: (B, L) - start positions
            pos_end: (B, L) - end positions
        """
        B, L, D = x.shape

        # Convert positions to float if they're not already
        pos_start_float = pos_start.float()
        pos_end_float = pos_end.float()

        # Encode genomic context
        genomic_context = self.context_encoder(x, pos_start_float, pos_end_float)  # (B, L, D)

        # Generate dynamic kernels
        dynamic_kernels = self.kernel_generator(genomic_context)  # (B, L, D, D, kernel_size)

        # Pad input for convolution
        pad_total = self.kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        x_padded = F.pad(x, (0, 0, pad_left, pad_right), mode='reflect')  # (B, L+pad_total, D)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Efficient computation using grouped convolution for each position
        for l in range(L):
            # Extract input segment: (B, kernel_size, D)
            x_seg = x_padded[:, l:l + self.kernel_size, :].transpose(1, 2)  # (B, D, kernel_size)

            # Extract kernels for this position: (B, D, D, kernel_size)
            kernels = dynamic_kernels[:, l, :, :, :]  # (B, D, D, kernel_size)

            # Compute output for each position using Einstein summation
            # x_seg: (B, D, kernel_size)
            # kernels: (B, D, D, kernel_size)
            # Result: (B, D) - for each output dimension, compute weighted sum
            output[:, l, :] = torch.einsum('bik,bdik->bd', x_seg, kernels)

        # Apply normalization and projection with residual connection
        output = self.norm(output + x)  # Residual connection
        output = self.dropout(output)
        output = self.output_proj(output)
        output = self.dropout(output)

        return output


class GenomicDynamicConvBlock(nn.Module):
    """
    A block combining genomic dynamic convolution with existing Mamba/WPSA modules
    """
    def __init__(self, embedding_dim, kernel_size=3, num_layers=2, dropout=0.1):
        super(GenomicDynamicConvBlock, self).__init__()
        self.layers = nn.ModuleList([
            GenomicDynamicConvLayer(embedding_dim, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        
        # Genomic-aware position embedding
        self.pos_embedding = GenomicAwarePositionEmbedding(embedding_dim)
        
        # Gating mechanism to balance dynamic conv and other modules
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, pos_start, pos_end, other_module_output=None):
        """
        Args:
            x: (B, L, D) - input sequence
            pos_start: (B, L) - start positions
            pos_end: (B, L) - end positions
            other_module_output: (B, L, D) - output from Mamba/WPSA modules to combine with
        """
        # Add genomic-aware positional embedding
        pos_emb = self.pos_embedding(x, pos_start, pos_end)
        x_with_pos = x + pos_emb
        
        # Apply dynamic convolution layers
        conv_output = x_with_pos
        for layer in self.layers:
            conv_output = layer(conv_output, pos_start, pos_end)
        
        # If other module output is provided, combine them
        if other_module_output is not None:
            gate_input = torch.cat([conv_output, other_module_output], dim=-1)
            gate = self.gate_network(gate_input)
            final_output = gate * conv_output + (1 - gate) * other_module_output
        else:
            final_output = conv_output
            
        return final_output