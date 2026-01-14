import torch
import torch.nn as nn


class LearnedAttentionCombiner(nn.Module):
    """Learned attention mechanism to combine encoder outputs"""
    def __init__(self, d_model, num_encoders=3):
        super().__init__()
        self.num_encoders = num_encoders
        self.d_model = d_model
        
        # Attention weights for combining encoder outputs
        self.attention_weights = nn.Parameter(torch.randn(num_encoders, d_model))
        self.query_transform = nn.Linear(d_model, d_model)
        self.key_transform = nn.Linear(d_model, d_model)
        self.value_transform = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize attention weights
        nn.init.xavier_uniform_(self.attention_weights)
    
    def forward(self, encoder_outputs):
        """
        Args:
            encoder_outputs: List of tensors from different encoders, each of shape [batch_size, seq_len, d_model]
        Returns:
            Combined representation of shape [batch_size, seq_len, d_model]
        """
        # Stack encoder outputs: [num_encoders, batch_size, seq_len, d_model]
        stacked_outputs = torch.stack(encoder_outputs, dim=0)
        
        batch_size, seq_len = encoder_outputs[0].shape[:2]
        
        # Apply transformations for attention computation
        queries = self.query_transform(stacked_outputs)  # [num_encoders, batch_size, seq_len, d_model]
        keys = self.key_transform(stacked_outputs)      # [num_encoders, batch_size, seq_len, d_model]
        values = self.value_transform(stacked_outputs)  # [num_encoders, batch_size, seq_len, d_model]
        
        # Compute attention scores
        attention_scores = torch.einsum('nbsk,nbsk->nbs', queries, keys)  # [num_encoders, batch_size, seq_len]
        
        # Apply learned attention weights
        attention_weights = self.attention_weights.unsqueeze(1).unsqueeze(2)  # [num_encoders, 1, 1, d_model]
        weighted_scores = attention_scores.unsqueeze(-1) * attention_weights.expand_as(values)
        
        # Apply softmax along encoder dimension
        attention_weights_normalized = torch.softmax(weighted_scores, dim=0)
        
        # Weighted sum of values
        combined_output = torch.sum(attention_weights_normalized * values, dim=0)  # [batch_size, seq_len, d_model]
        
        # Apply layer norm
        combined_output = self.norm(combined_output)
        
        return combined_output