"""
Dynamic Learning Rate Scheduler with Adaptive Weight Decay for ChromFound
"""
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing scheduler with warmup and restarts
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]


class AdaptiveWeightDecayScheduler:
    """
    Adaptive weight decay scheduler based on gradient magnitude
    """
    def __init__(self, optimizer, initial_weight_decay=1e-2, target_decay_range=(1e-6, 1e-2)):
        self.optimizer = optimizer
        self.initial_weight_decay = initial_weight_decay
        self.target_decay_range = target_decay_range
        self.step_count = 0
        
        # Store initial weight decay values for each parameter group
        self.initial_param_wds = []
        for group in optimizer.param_groups:
            self.initial_param_wds.append(group.get('weight_decay', 0.0))
    
    def update_weight_decay(self, model):
        """
        Update weight decay based on gradient statistics
        """
        self.step_count += 1
        
        # Calculate gradient norms
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        if grad_norms:
            # Calculate mean gradient norm
            mean_grad_norm = np.mean(grad_norms)
            
            # Adjust weight decay based on gradient magnitude
            # Higher gradients -> lower weight decay
            # Lower gradients -> higher weight decay
            adaptive_factor = max(0.1, min(10.0, 1.0 / (1.0 + mean_grad_norm)))
            
            # Map to target range
            current_min, current_max = self.target_decay_range
            new_weight_decay = max(current_min, min(current_max, self.initial_weight_decay * adaptive_factor))
            
            # Apply to optimizer
            for i, group in enumerate(self.optimizer.param_groups):
                group['weight_decay'] = new_weight_decay


class DynamicLRScheduler:
    """
    Dynamic learning rate scheduler with adaptive weight decay
    """
    def __init__(self, optimizer, warmup_steps, total_steps, 
                 min_lr=1e-7, initial_weight_decay=1e-2, 
                 target_decay_range=(1e-6, 1e-2)):
        self.lr_scheduler = CosineAnnealingWithWarmup(
            optimizer, warmup_steps, total_steps, min_lr
        )
        self.wd_scheduler = AdaptiveWeightDecayScheduler(
            optimizer, initial_weight_decay, target_decay_range
        )
    
    def step(self, model=None):
        """
        Step both learning rate and weight decay schedulers
        """
        self.lr_scheduler.step()
        if model is not None:
            self.wd_scheduler.update_weight_decay(model)
    
    def get_last_lr(self):
        """
        Get the last learning rate
        """
        return self.lr_scheduler.get_last_lr()


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss for fine-tuning ChromFound model
    """
    def __init__(self, temperature=0.07, negative_sampling_ratio=1.0, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.negative_sampling_ratio = negative_sampling_ratio
        self.reduction = reduction
        
    def forward(self, embeddings, labels):
        """
        Compute contrastive loss
        Args:
            embeddings: tensor of shape (batch_size, embedding_dim)
            labels: tensor of shape (batch_size,) containing class indices
        """
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create mask for positive samples (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(mask.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_similarities = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_similarities.sum(1, keepdim=True))
        
        # Compute mean of log likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Return contrastive loss
        loss = -mean_log_prob_pos
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


class CombinedLoss(torch.nn.Module):
    """
    Combined loss function including focal loss and contrastive loss
    """
    def __init__(self, alpha=1.0, gamma=2.0, temperature=0.07, 
                 contrastive_weight=0.1, focal_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.contrastive_weight = contrastive_weight
        self.focal_weight = focal_weight
    
    def forward(self, logits, labels, embeddings=None):
        # Standard focal loss
        focal_loss = self.focal_loss(logits, labels)
        
        # Contrastive loss if embeddings are provided
        contrastive_loss = 0.0
        if embeddings is not None:
            contrastive_loss = self.contrastive_loss(embeddings, labels)
        
        # Combined loss
        total_loss = self.focal_weight * focal_loss + self.contrastive_weight * contrastive_loss
        return total_loss


class FocalLoss(torch.nn.Module):
    """
    Focal Loss as described in https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha          # Balance factor
        self.gamma = gamma          # Modulating factor
        self.reduction = reduction  # Reduction method: 'mean', 'sum', 'none'

    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilities of the predicted classes
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss