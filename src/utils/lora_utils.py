"""
LoRA (Low-Rank Adaptation) implementation for ChromFound model
"""
import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that adds low-rank adaptation to linear layers
    """
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.05):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Initialize low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * math.sqrt(1 / rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Zero initialization for B matrix
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Apply dropout to input
        x = self.dropout(x)
        # Compute (BA)x where A is (rank, in_features) and B is (out_features, rank)
        result = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result


class LinearWithLoRA(nn.Module):
    """
    Linear layer with integrated LoRA adaptation
    """
    def __init__(self, in_features, out_features, bias=True, rank=16, alpha=32, dropout=0.05):
        super(LinearWithLoRA, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_layer = LoRALayer(in_features, out_features, rank, alpha, dropout)
    
    def forward(self, x):
        # Original linear transformation
        linear_out = self.linear(x)
        # Add LoRA adaptation
        lora_out = self.lora_layer(x)
        return linear_out + lora_out
    
    def merge_lora_weights(self):
        """Merge LoRA weights back into the original linear layer"""
        with torch.no_grad():
            # Calculate the merged weight
            merged_weight = self.lora_A.T @ self.lora_B.T * (self.alpha / self.rank)
            self.linear.weight.data += merged_weight
            # Reset LoRA parameters
            self.lora_A.zero_()
            self.lora_B.zero_()


def inject_lora_to_model(model, target_modules=["Linear"], rank=16, alpha=32, dropout=0.05):
    """
    Inject LoRA layers into specified modules of the model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should be targeted
            should_replace = False
            for target_module in target_modules:
                if target_module.lower() in name.lower():
                    should_replace = True
                    break
            
            if should_replace:
                # Get the original module's parameters
                in_features = module.in_features
                out_features = module.out_features
                bias = module.bias is not None
                
                # Create new LinearWithLoRA module
                new_module = LinearWithLoRA(
                    in_features, out_features, bias=bias,
                    rank=rank, alpha=alpha, dropout=dropout
                )
                
                # Copy original weights
                new_module.linear.weight.data.copy_(module.weight.data)
                if bias and module.bias is not None:
                    new_module.linear.bias.data.copy_(module.bias.data)
                
                # Replace the original module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name == '':
                    setattr(model, child_name, new_module)
                else:
                    parent_module = model
                    for parent_part in parent_name.split('.'):
                        parent_module = getattr(parent_module, parent_part)
                    setattr(parent_module, child_name, new_module)
    
    return model


def get_lora_state_dict(model):
    """
    Get only the LoRA parameters from the model
    """
    lora_params = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params[name] = param
    return lora_params


def set_lora_requires_grad(model, requires_grad=True):
    """
    Set requires_grad for LoRA parameters only
    """
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = requires_grad
        else:
            param.requires_grad = False  # Freeze non-LoRA parameters


def count_lora_params(model):
    """
    Count the total number of LoRA parameters
    """
    total_params = 0
    for name, param in model.named_parameters():
        if 'lora_' in name:
            total_params += param.numel()
    return total_params