"""
LoRA (Low-Rank Adaptation) implementation for PyTorch models
"""
import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that adds low-rank adaptation to a linear layer
    """
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Initialize the low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * math.sqrt(1 / rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Zero-initialize B matrix to ensure initial output remains unchanged
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Apply dropout to input
        x = self.dropout(x)
        # Compute (x @ A.T @ B.T) * scaling
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora_to_linear(layer, rank=16, alpha=32, dropout=0.0):
    """
    Replace a Linear layer with a Linear layer + LoRA
    """
    in_features = layer.in_features
    out_features = layer.out_features
    
    # Create a new module that combines the original linear layer with LoRA
    class LinearWithLoRA(nn.Module):
        def __init__(self, original_layer, lora_layer):
            super(LinearWithLoRA, self).__init__()
            self.linear = original_layer
            self.lora = lora_layer
            
        def forward(self, x):
            return self.linear(x) + self.lora(x)
    
    lora_layer = LoRALayer(in_features, out_features, rank, alpha, dropout)
    return LinearWithLoRA(layer, lora_layer)


def add_lora_to_model(model, target_modules=["Linear"], rank=16, alpha=32, dropout=0.0):
    """
    Add LoRA to specified modules in the model
    """
    for name, module in model.named_modules():
        if any(target_module in str(type(module)) for target_module in target_modules):
            # Skip the LoRA layers themselves if they contain Linear layers
            if not isinstance(module, LoRALayer):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                # Get parent module
                parent_module = model
                if parent_name:
                    for parent_attr in parent_name.split("."):
                        parent_module = getattr(parent_module, parent_attr)
                
                # Replace the original module with LoRA-enhanced version
                original_module = getattr(parent_module, child_name)
                lora_enhanced_module = inject_lora_to_linear(original_module, rank, alpha, dropout)
                setattr(parent_module, child_name, lora_enhanced_module)
    
    return model


def get_lora_parameters(model):
    """
    Get only the LoRA parameters for optimization
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            lora_params.append(param)
    return lora_params


def freeze_non_lora_parameters(model):
    """
    Freeze all parameters except LoRA parameters
    """
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True