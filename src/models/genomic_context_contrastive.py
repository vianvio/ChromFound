import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import MultiheadAttention


class GenomicDistanceCalculator:
    """Calculate genomic distances between regions"""
    
    @staticmethod
    def calculate_genomic_distance(chromosome1, start1, end1, chromosome2, start2, end2):
        """
        Calculate genomic distance between two genomic regions
        Returns 0 if same chromosome and overlapping, otherwise distance
        """
        # If on different chromosomes, return large distance
        if chromosome1 != chromosome2:
            return float('inf')
        
        # If regions overlap, distance is 0
        if start1 <= end2 and start2 <= end1:
            return 0
        
        # Calculate distance between regions
        if start1 > end2:
            return start1 - end2
        else:
            return start2 - end1


class GenomicProximitySampler:
    """Sample positive pairs based on genomic proximity"""
    
    def __init__(self, proximity_threshold=1000000):  # 1MB threshold
        self.proximity_threshold = proximity_threshold
    
    def sample_positive_pairs(self, chromosomes, starts, ends):
        """
        Sample positive pairs based on genomic proximity
        Args:
            chromosomes: tensor of chromosome indices
            starts: tensor of start positions
            ends: tensor of end positions
        Returns:
            adjacency matrix indicating positive pairs
        """
        batch_size = chromosomes.size(0)
        adj_matrix = torch.zeros(batch_size, batch_size)
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                dist = GenomicDistanceCalculator.calculate_genomic_distance(
                    chromosomes[i], starts[i], ends[i],
                    chromosomes[j], starts[j], ends[j]
                )
                
                if dist < self.proximity_threshold:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        return adj_matrix


class CellTypeSpecificAnchor(nn.Module):
    """Learnable genomic anchors specific to each cell type"""
    
    def __init__(self, embedding_dim, num_cell_types, num_anchors=10):
        super(CellTypeSpecificAnchor, self).__init__()
        self.num_cell_types = num_cell_types
        self.num_anchors = num_anchors
        self.embedding_dim = embedding_dim
        
        # Learnable anchors for each cell type
        self.anchors = nn.Parameter(
            torch.randn(num_cell_types, num_anchors, embedding_dim) * 0.1
        )
        
        # Anchor weights for each cell type
        self.anchor_weights = nn.Parameter(
            torch.ones(num_cell_types, num_anchors)
        )
        
    def forward(self, cell_type_indices):
        """
        Get anchors for specific cell types
        Args:
            cell_type_indices: tensor of cell type indices
        Returns:
            anchors for each cell type in the batch
        """
        batch_anchors = []
        batch_weights = []
        
        for ct_idx in cell_type_indices:
            anchor = self.anchors[ct_idx]
            weight = self.anchor_weights[ct_idx]
            batch_anchors.append(anchor)
            batch_weights.append(weight)
            
        return torch.stack(batch_anchors), torch.stack(batch_weights)


class GenomicContextAwareContrastiveLoss(nn.Module):
    """Multi-scale contrastive loss with genomic context awareness"""
    
    def __init__(self, temperature=0.07, genomic_weight=0.5, regional_weight=0.3):
        super(GenomicContextAwareContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.genomic_weight = genomic_weight
        self.regional_weight = regional_weight
        
    def forward(self, features, labels, genomic_adjacency=None, cell_type_features=None):
        """
        Compute contrastive loss with genomic context
        Args:
            features: feature representations [batch_size, feature_dim]
            labels: ground truth labels [batch_size]
            genomic_adjacency: genomic proximity adjacency matrix [batch_size, batch_size]
            cell_type_features: cell type specific features [batch_size, feature_dim]
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create label-based positive pairs mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Exclude self-similarity
        mask_diag = torch.eye(batch_size, dtype=torch.bool)
        mask[mask_diag] = 0
        
        # Genomic context mask (if provided)
        genomic_mask = genomic_adjacency if genomic_adjacency is not None else torch.zeros_like(mask)
        
        # Combine label-based and genomic-based masks
        combined_mask = mask + self.genomic_weight * genomic_mask
        combined_mask = (combined_mask > 0).float()
        
        # Compute contrastive loss
        # Log-sum-exp for denominator
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Compute mean log-likelihood for positive pairs
        mean_log_prob_pos = (combined_mask * log_prob).sum(1) / (combined_mask.sum(1) + 1e-8)
        
        # Overall loss (negative mean log-likelihood)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class GenomicAttention(nn.Module):
    """Attention mechanism guided by genomic context"""
    
    def __init__(self, embed_dim, num_heads=8):
        super(GenomicAttention, self).__init__()
        self.multihead_attn = MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.genomic_position_encoder = nn.Linear(2, embed_dim)  # Encode start/end positions
        
    def forward(self, x, genomic_positions, key_padding_mask=None):
        """
        Apply attention with genomic context
        Args:
            x: input features [batch_size, seq_len, embed_dim]
            genomic_positions: [batch_size, seq_len, 2] (start, end positions)
            key_padding_mask: mask for padded positions
        """
        # Encode genomic positions
        pos_encoding = self.genomic_position_encoder(genomic_positions.float())
        
        # Add genomic position encoding to input
        x_with_pos = x + pos_encoding
        
        # Apply multi-head attention
        attn_output, attn_weights = self.multihead_attn(
            x_with_pos, x_with_pos, x_with_pos, 
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        return attn_output, attn_weights


class RegionalLevelContrastive(nn.Module):
    """Contrastive learning at regional level"""
    
    def __init__(self, region_size=100, embedding_dim=256):
        super(RegionalLevelContrastive, self).__init__()
        self.region_size = region_size
        self.embedding_dim = embedding_dim
        
        # Regional feature aggregation
        self.region_aggregator = nn.Conv1d(embedding_dim, embedding_dim, 
                                          kernel_size=region_size, 
                                          stride=region_size, 
                                          padding=0)
        
        # Regional contrastive projector
        self.region_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )
    
    def forward(self, sequence_features):
        """
        Apply regional level contrastive learning
        Args:
            sequence_features: [batch_size, seq_len, embedding_dim]
        Returns:
            regional representations for contrastive learning
        """
        batch_size, seq_len, embed_dim = sequence_features.shape
        
        # Reshape to apply convolution: [batch_size * num_regions, embed_dim, region_size]
        num_regions = seq_len // self.region_size
        if num_regions == 0:
            # If sequence is too short, return global representation
            return sequence_features.mean(dim=1, keepdim=True)
        
        # Pad sequence if needed
        if seq_len % self.region_size != 0:
            pad_len = self.region_size - (seq_len % self.region_size)
            padding = torch.zeros(batch_size, pad_len, embed_dim, device=sequence_features.device)
            sequence_features = torch.cat([sequence_features, padding], dim=1)
        
        # Reshape for convolution: [batch_size, embedding_dim, new_seq_len]
        x = sequence_features.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
        
        # Apply regional aggregation
        regional_features = self.region_aggregator(x)  # [batch_size, embed_dim, num_regions]
        
        # Transpose back: [batch_size, num_regions, embed_dim]
        regional_features = regional_features.transpose(1, 2)
        
        # Apply projector to get contrastive representations
        contrastive_regional = self.region_projector(regional_features)
        
        return contrastive_regional