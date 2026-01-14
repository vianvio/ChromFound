import torch
import torch.nn as nn


class MetaLearner(nn.Module):
    """Meta-learner that combines predictions based on cell-type-specific patterns"""
    def __init__(self, d_model, cell_type_num, hidden_dim=512):
        super().__init__()
        self.d_model = d_model
        self.cell_type_num = cell_type_num
        
        # Projection layers for meta-learning
        self.projection = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, cell_type_num)
        )
        
        # Cell type-specific adapters
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
                nn.Dropout(0.1)
            ) for _ in range(cell_type_num)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, combined_representation, cell_type_mask=None):
        """
        Args:
            combined_representation: Combined output from attention mechanism [batch_size, seq_len, d_model]
            cell_type_mask: Optional mask for cell-type-specific processing
        Returns:
            Final predictions for cell types [batch_size, cell_type_num]
        """
        # Global pooling to get sequence-level representation
        pooled_repr = torch.mean(combined_representation, dim=1)  # [batch_size, d_model]
        
        # Apply cell type-specific adapters if mask is provided
        if cell_type_mask is not None:
            adapted_repr = []
            for i, adapter in enumerate(self.adapters):
                # Apply adapter to corresponding samples
                mask_indices = torch.where(cell_type_mask == i)[0]
                if len(mask_indices) > 0:
                    masked_repr = pooled_repr[mask_indices]
                    adapted_repr.append(adapter(masked_repr))
                else:
                    # If no samples for this cell type, append zeros
                    adapted_repr = torch.zeros(0, self.d_model, device=pooled_repr.device)
            
            # Concatenate and process
            if all(len(r) > 0 for r in adapted_repr):
                adapted_repr = torch.cat(adapted_repr, dim=0)
            else:
                adapted_repr = pooled_repr
        else:
            adapted_repr = pooled_repr
        
        # Final projection to cell type predictions
        predictions = self.projection(adapted_repr)
        
        return predictions