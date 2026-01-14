import math
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP

from .chromfd_block import Block
from .embedding_model import PretrainEmbeddingSimple
from .promoter_encoder import PromoterFocusedEncoder
from .enhancer_encoder import EnhancerFocusedEncoder
from .domain_encoder import ChromatinDomainEncoder
from .attention_combiner import LearnedAttentionCombiner
from .meta_learner import MetaLearner

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class EnsembleCellTypeModel(nn.Module):
    """Ensemble model combining specialized encoders with attention and meta-learner"""
    def __init__(self, **kwargs):
        super().__init__()
        self.model_args = kwargs
        self.d_model = self.model_args.get("embedding_dim", 2560)
        self.seq_length = self.model_args.get("max_length", 10000)
        self.cell_type_num = self.model_args.get("cell_type_num", 10)
        
        # Device and dtype
        self.factory_kwargs = {"device": self.model_args["device"], "dtype": torch.float32}
        
        # Embedding layer
        self.embedding = PretrainEmbeddingSimple(
            embedding_dim=self.model_args.get("embedding_dim"),
            chromosome_size=self.model_args.get("chromosome_size"),
            embedding_dropout=self.model_args.get("embedding_dropout", 0),
            positional_embedding_type=self.model_args.get("positional_embedding_type"),
            positional_temp=self.model_args['positional_temp'],
            batch_size=self.model_args.get("batch_size"),
            seq_length=self.model_args.get("max_length"),
            device=self.model_args["device"],
            chromatin_embedding=self.model_args.get("chromatin_embedding", True)
        )
        
        # Specialized encoders
        self.promoter_encoder = PromoterFocusedEncoder(
            d_model=self.d_model,
            seq_length=self.seq_length,
            window_size=self.model_args.get("promoter_window_size", 512),
            n_layer=self.model_args.get("promoter_layers", 32),
            **self.factory_kwargs
        )
        
        self.enhancer_encoder = EnhancerFocusedEncoder(
            d_model=self.d_model,
            seq_length=self.seq_length,
            window_size=self.model_args.get("enhancer_window_size", 1024),
            n_layer=self.model_args.get("enhancer_layers", 32),
            **self.factory_kwargs
        )
        
        self.domain_encoder = ChromatinDomainEncoder(
            d_model=self.d_model,
            seq_length=self.seq_length,
            window_size=self.model_args.get("domain_window_size", 2048),
            n_layer=self.model_args.get("domain_layers", 32),
            **self.factory_kwargs
        )
        
        # Attention combiner
        self.attention_combiner = LearnedAttentionCombiner(
            d_model=self.d_model,
            num_encoders=3
        )
        
        # Meta-learner
        self.meta_learner = MetaLearner(
            d_model=self.d_model,
            cell_type_num=self.cell_type_num,
            hidden_dim=self.model_args.get("meta_hidden_dim", 512)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.cell_type_num)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, value, chromosome, hg38_start, hg38_end, cell_type_mask=None):
        # Embed the input
        embedded = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        
        # Pass through specialized encoders
        promoter_out = self.promoter_encoder(embedded)
        enhancer_out = self.enhancer_encoder(embedded)
        domain_out = self.domain_encoder(embedded)
        
        # Combine encoder outputs using learned attention
        combined_repr = self.attention_combiner([promoter_out, enhancer_out, domain_out])
        
        # Apply dropout
        combined_repr = self.dropout(combined_repr)
        
        # Get final predictions from meta-learner
        predictions = self.meta_learner(combined_repr, cell_type_mask)
        
        return predictions