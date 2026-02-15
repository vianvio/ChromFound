"""
Fine-tuning model with LoRA for ChromFound.
This module provides functionality for parameter-efficient fine-tuning using LoRA.
"""

import torch
import torch.nn as nn
import yaml
import os
from typing import Optional, Dict, Any
from .chromfd_mixer import PretrainModelMambaLM
from .lora import inject_lora_to_model


class ChromFoundFineTuneModel(nn.Module):
    """ChromFound model with LoRA for fine-tuning"""
    
    def __init__(self, 
                 pretrained_model_path: str,
                 config_path: str,
                 chromosome_vocab_path: str,
                 lora_rank: int = 8,
                 lora_alpha: float = 16.0,
                 lora_dropout: float = 0.0,
                 freeze_backbone: bool = True,
                 task_type: str = "classification"):
        super(ChromFoundFineTuneModel, self).__init__()
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.pretrain_config = yaml.safe_load(file)
        self.pretrain_model_args = self.pretrain_config['model_args']
        
        # Load chromosome vocabulary
        with open(chromosome_vocab_path) as file:
            chromosome_vocab = yaml.safe_load(file)
            chromosome_vocab = {chr_: idx for idx, chr_ in enumerate(chromosome_vocab["chromosome"])}
        self.pretrain_model_args["chromosome_vocab"] = chromosome_vocab
        
        # Set device to CPU initially, will be moved later
        self.pretrain_model_args["device"] = torch.device("cpu")
        
        # Load the pretrained model
        self.model = PretrainModelMambaLM(**self.pretrain_model_args)
        
        # Load pretrained weights
        state_dict = torch.load(pretrained_model_path, map_location='cpu')
        if 'module' in state_dict:
            state_dict = state_dict['module']
        self.model.load_state_dict(state_dict, strict=False)  # Use strict=False to allow for new LoRA params
        
        # Inject LoRA layers
        self.model_with_lora = inject_lora_to_model(
            self.model, 
            rank=lora_rank, 
            alpha=lora_alpha, 
            dropout=lora_dropout
        )
        
        # Freeze the original model parameters if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Add task-specific head based on task type
        self.task_type = task_type
        embedding_dim = self.pretrain_model_args.get("embedding_dim", 2560)  # Default to common dimension
        
        if task_type == "classification":
            # Classification head
            num_classes = self.pretrain_model_args.get("num_classes", 2)  # Default binary classification
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, embedding_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim // 4, num_classes)
            )
        elif task_type == "regression":
            # Regression head
            output_dim = self.pretrain_model_args.get("output_dim", 1)  # Default single output
            self.regressor = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, embedding_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim // 4, output_dim)
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        # Get embeddings from the backbone model with LoRA
        embeddings = self.model_with_lora(value, chromosome, hg38_start, hg38_end, **kwargs)
        
        # Take mean across sequence dimension for classification/regression
        if embeddings.dim() > 2:
            embeddings = embeddings.mean(dim=1)  # Average pooling over sequence length
        
        # Apply task-specific head
        if self.task_type == "classification":
            return self.classifier(embeddings)
        elif self.task_type == "regression":
            return self.regressor(embeddings)
    
    def get_trainable_parameters(self):
        """Get parameters that should be trained (LoRA parameters + task head)"""
        trainable_params = []
        
        # Add LoRA parameters
        for name, param in self.model_with_lora.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                trainable_params.append(param)
        
        # Add task-specific head parameters
        if self.task_type == "classification":
            trainable_params.extend(list(self.classifier.parameters()))
        elif self.task_type == "regression":
            trainable_params.extend(list(self.regressor.parameters()))
        
        return trainable_params
    
    def get_lora_parameters(self):
        """Get only LoRA parameters"""
        return self.model_with_lora.get_lora_parameters()
    
    def get_non_lora_parameters(self):
        """Get non-LoRA parameters (should be frozen)"""
        return self.model_with_lora.get_non_lora_parameters()


def create_finetune_model_from_pretrained(
    checkpoint_dir: str,
    model_filename: str = "model.pt",
    config_filename: str = "chromfd_pretrain.yaml",
    vocab_filename: str = "chromosome_vocab.yaml",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    freeze_backbone: bool = True,
    task_type: str = "classification"
) -> ChromFoundFineTuneModel:
    """
    Create a fine-tuning model from pretrained checkpoints.
    
    Args:
        checkpoint_dir: Directory containing pretrained model files
        model_filename: Name of the model file
        config_filename: Name of the config file
        vocab_filename: Name of the chromosome vocabulary file
        lora_rank: Rank of LoRA adaptation
        lora_alpha: Alpha parameter for LoRA scaling
        lora_dropout: Dropout rate for LoRA layers
        freeze_backbone: Whether to freeze the original model parameters
        task_type: Type of task ('classification' or 'regression')
    
    Returns:
        ChromFoundFineTuneModel instance
    """
    model_path = os.path.join(checkpoint_dir, model_filename)
    config_path = os.path.join(checkpoint_dir, config_filename)
    vocab_path = os.path.join(checkpoint_dir, vocab_filename)
    
    return ChromFoundFineTuneModel(
        pretrained_model_path=model_path,
        config_path=config_path,
        chromosome_vocab_path=vocab_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        freeze_backbone=freeze_backbone,
        task_type=task_type
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total, trainable, and LoRA parameters in the model.
    
    Args:
        model: The model to analyze
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count LoRA parameters specifically
    if hasattr(model, 'get_lora_parameters'):
        lora_params = sum(p.numel() for p in model.get_lora_parameters())
    else:
        lora_params = 0
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "lora_parameters": lora_params,
        "frozen_parameters": total_params - trainable_params
    }