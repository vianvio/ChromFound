"""
New finetune model that uses the multi-scale architecture combining:
1. Local features from WPSA (Window Partition Self-Attention) module
2. Global features from Mamba module  
3. Intermediate features from different transformer layers
Using a learnable attention mechanism before passing to classification head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.multiscale_fusion_model import MultiScalePretrainModelMambaLM


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class MultiScaleFinetuneModelMambaCellType(MultiScalePretrainModelMambaLM):
    """
    Fine-tuning model for cell type annotation using multi-scale feature fusion
    """
    def __init__(self, **kwargs):
        # Set default to use multiscale extractor if not specified
        if "use_multiscale_extractor" not in kwargs:
            kwargs["use_multiscale_extractor"] = True
        super().__init__(**kwargs)
        
        max_length = self.model_args["max_length"]
        self.post_backbone_dropout = torch.nn.Dropout(p=0.3)
        
        # Feature projection layer to reduce sequence dimensionality
        self.feature_projection = torch.nn.Sequential(
            torch.nn.Linear(self.model_args["embedding_dim"], 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 1),
            torch.nn.GELU()
        )
        self.feature_projection.apply(init_weight)
        
        in_feature = max_length
        # Classification head for cell type prediction
        self.ft_cell_type_projection = torch.nn.Sequential(
            torch.nn.Linear(in_feature, 1024),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(512, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(128, self.model_args["cell_type_num"])
        )
        self.ft_cell_type_projection.apply(init_weight)

        # Freeze the pre-trained components during fine-tuning
        for name, param in self.mask_token_prediction.named_parameters():
            param.requires_grad = False

    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        # Forward pass through the multi-scale backbone
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x = self.backbone(x, hg38_start=hg38_start, hg38_end=hg38_end)
        
        # Project features to lower dimension
        x = self.feature_projection(x)
        x = torch.squeeze(x, dim=-1)  # Remove the last dimension
        x = self.post_backbone_dropout(x)
        
        # Final classification
        x_cell_type_prediction = self.ft_cell_type_projection(x)
        return x_cell_type_prediction