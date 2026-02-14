"""
Training utilities for ChromFound model
Contains implementations for gradient clipping, dynamic learning rate scheduling,
label smoothing, and hierarchical learning rates
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
import math


def compute_gradient_norm(model):
    """Compute the total gradient norm across all parameters."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def gradient_clip_by_norm(model, max_norm):
    """Clip gradients by global norm."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def gradient_clip_by_value(model, clip_value):
    """Clip gradients by value."""
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)


class DynamicLRScheduler:
    """Dynamic learning rate scheduler with cosine annealing and exponential decay options."""
    
    def __init__(self, optimizer, scheduler_type='cosine', T_max=None, gamma=0.95, eta_min=1e-8):
        """
        Args:
            optimizer: Optimizer to adjust learning rate for
            scheduler_type: 'cosine' for cosine annealing, 'exponential' for exponential decay
            T_max: Maximum number of iterations for cosine annealing
            gamma: Multiplicative factor for exponential decay
            eta_min: Minimum learning rate for cosine annealing
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'cosine':
            if T_max is None:
                raise ValueError("T_max must be provided for cosine annealing")
            self.scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()
    
    def get_last_lr(self):
        """Get the last learning rate."""
        return self.scheduler.get_last_lr()


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss."""
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        """
        Args:
            smoothing: Label smoothing factor (0.0 means no smoothing)
            reduction: Specifies the reduction to apply to the output
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions from model (before softmax) with shape (N, C)
            target: Ground truth labels with shape (N,)
        """
        logprobs = torch.nn.functional.log_softmax(pred, dim=-1)
        
        # Create one-hot encoding of targets
        n_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        if self.reduction == 'sum':
            return (-true_dist * logprobs).sum(dim=-1).sum()
        elif self.reduction == 'mean':
            return (-true_dist * logprobs).sum(dim=-1).mean()
        else:
            return (-true_dist * logprobs).sum(dim=-1)


def get_param_groups_with_different_lr(model, base_lr, layer_decay=0.9):
    """
    Create parameter groups with different learning rates for different layers.
    Lower layers (closer to input) get smaller learning rates, higher layers get larger rates.
    
    Args:
        model: The model to create parameter groups for
        base_lr: Base learning rate for the topmost layer
        layer_decay: Factor by which to multiply LR for each lower layer
    """
    # Assuming the model has named modules that we can identify
    param_groups = []
    
    # Find all the major components of the model
    embedding_params = []
    backbone_params = []
    prediction_head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'embedding' in name.lower():
                embedding_params.append(param)
            elif 'backbone' in name.lower() or 'mamba' in name.lower() or 'block' in name.lower():
                backbone_params.append(param)
            else:
                prediction_head_params.append(param)
    
    # Assign different learning rates to different parts of the model
    param_groups.append({
        'params': embedding_params,
        'lr': base_lr * (layer_decay ** 2),  # Smallest LR for embedding layer
        'name': 'embedding'
    })
    
    param_groups.append({
        'params': backbone_params,
        'lr': base_lr * layer_decay,  # Medium LR for backbone
        'name': 'backbone'
    })
    
    param_groups.append({
        'params': prediction_head_params,
        'lr': base_lr,  # Full LR for prediction head
        'name': 'prediction_head'
    })
    
    return param_groups


def get_layerwise_param_groups(model, base_lr, layer_decay=0.9):
    """
    Create parameter groups with different learning rates for different layers in the backbone.
    Each layer gets a different learning rate based on its depth.
    
    Args:
        model: The model to create parameter groups for
        base_lr: Base learning rate for the topmost layer
        layer_decay: Factor by which to multiply LR for each lower layer
    """
    param_groups = []
    
    # Check if model has backbone attribute with layers
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        num_layers = len(model.backbone.layers)
        
        # Process each layer in the backbone separately
        for i, layer in enumerate(model.backbone.layers):
            layer_lr = base_lr * (layer_decay ** (num_layers - 1 - i))
            
            layer_params = []
            for param_name, param in layer.named_parameters():
                if param.requires_grad:
                    layer_params.append(param)
            
            if layer_params:  # Only add if there are parameters
                param_groups.append({
                    'params': layer_params,
                    'lr': layer_lr,
                    'name': f'layer_{i}'
                })
    
    # Add remaining parameters (embedding, prediction head, etc.) with appropriate LRs
    embedding_params = []
    prediction_head_params = []

    # Collect IDs of parameters already added to layer groups
    added_param_ids = set()
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        for group in param_groups:
            for param in group['params']:
                added_param_ids.add(id(param))

    for name, param in model.named_parameters():
        # Check if this parameter is already included in layer groups
        if id(param) in added_param_ids or not param.requires_grad:
            continue

        if 'embedding' in name.lower():
            embedding_params.append(param)
        else:
            prediction_head_params.append(param)
    
    # Add embedding parameters with lower LR
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': base_lr * (layer_decay ** 2),
            'name': 'embedding'
        })
    
    # Add prediction head parameters with full LR
    if prediction_head_params:
        param_groups.append({
            'params': prediction_head_params,
            'lr': base_lr,
            'name': 'prediction_head'
        })
    
    return param_groups