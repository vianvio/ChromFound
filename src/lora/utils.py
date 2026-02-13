"""
Configuration and utility functions for LoRA integration in ChromFound
"""
import torch
import torch.nn as nn
from .lora import inject_lora_into_model, freeze_non_lora_parameters


def configure_lora_model(model, lora_config):
    """
    Configure the model with LoRA settings
    
    Args:
        model: The model to configure
        lora_config: Dictionary containing LoRA configuration parameters
    """
    # Extract LoRA configuration
    rank = lora_config.get('rank', 16)
    alpha = lora_config.get('alpha', 32)
    dropout = lora_config.get('dropout', 0.05)
    target_modules = lora_config.get('target_modules', ['Linear'])
    
    # Inject LoRA layers into the model
    inject_lora_into_model(
        model=model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )
    
    # Freeze non-LoRA parameters to only train LoRA parameters
    freeze_non_lora_parameters(model)
    
    return model


def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    """
    Count the total number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters())


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters and their percentage
    """
    trainable_params = count_trainable_parameters(model)
    total_params = count_total_parameters(model)
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")