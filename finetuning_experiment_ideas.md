# ChromFound 微调优化实验想法

## Idea 1、基于分层学习率 (Layer-wise Learning Rate Decay) 的微调策略优化

**核心思路：** 对 backbone 的不同层应用递减的学习率，底层（靠近输入）使用较小的学习率以保留预训练知识，顶层使用较大的学习率以适应下游任务。

**具体实现：**
```python
# 在 cell_type_annotation.py 中修改 optimizer 参数设置
def get_layerwise_lr_params(model, base_lr, decay_rate=0.9):
    """
    为 MambaMixer 的 64 层应用指数衰减的学习率
    decay_rate = 0.9 表示每层学习率乘以 0.9^layer_idx
    """
    param_groups = []
    
    # 获取 backbone 的 64 层参数
    backbone_layers = model.backbone.layers
    n_layers = len(backbone_layers)
    
    for layer_idx in range(n_layers):
        layer_params = list(backbone_layers[layer_idx].parameters())
        lr = base_lr * (decay_rate ** (n_layers - layer_idx))  # 底层 lr 小，顶层 lr 大
        param_groups.append({"params": layer_params, "lr": lr})
    
    # classification head 使用基础学习率
    head_params = list(model.ft_cell_type_projection.parameters())
    param_groups.append({"params": head_params, "lr": base_lr})
    
    # embedding 层使用最小学习率
    embedding_params = list(model.embedding.parameters())
    param_groups.append({"params": embedding_params, "lr": base_lr * (decay_rate ** n_layers)})
    
    return param_groups

# 使用示例
optimizer_params = get_layerwise_lr_params(model, base_lr=args.learning_rate, decay_rate=0.95)
optimizer = torch.optim.AdamW(optimizer_params, betas=(0.8, 0.999), eps=1e-8, weight_decay=1e-6)
```

**预期效果：** 防止底层预训练特征被过度破坏，同时允许高层更快适应新任务，提升小样本场景下的泛化能力。

---

## Idea 2、基于 LoRA (Low-Rank Adaptation) 的参数高效微调

**核心思路：** 冻结 backbone 全部参数，仅在 Mamba 和 MHA 层中插入低秩适配器 (LoRA)，大幅减少可训练参数量。

**具体实现：**
```python
# 新建 src/models/lora_adapter.py
import torch.nn as nn

class LoRAAdapter(nn.Module):
    """
    LoRA adapter for Mamba/MHA layers
    r: rank for low-rank decomposition (e.g., r=8, 16, 32)
    alpha: scaling factor (typically alpha = 2r or alpha = r)
    """
    def __init__(self, in_features, out_features, r=16, alpha=32, dropout=0.1):
        super().__init__()
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        
        # 初始化：A 用高斯分布，B 用零初始化 (确保初始状态为恒等映射)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


# 在 FinetuneModelMambaCellType 中应用 LoRA
class FinetuneModelMambaCellTypeWithLoRA(PretrainModelMambaLM):
    def __init__(self, lora_r=16, lora_alpha=32, **kwargs):
        super().__init__(**kwargs)
        
        # 1. 冻结 backbone 所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 2. 为每个 Mamba/MHA 层注入 LoRA
        from functools import partial
        from mamba_ssm.modules.mamba_simple import Mamba
        from mamba_ssm.modules.mha import MHA
        
        for layer_idx, layer in enumerate(self.backbone.layers):
            # 为 Mamba 层的 mixer 添加 LoRA
            if isinstance(layer.mixer, (Mamba, MHA)):
                # 在 Mamba 的 in_proj 或 MHA 的 Wqkv 上添加 LoRA
                if hasattr(layer.mixer, 'in_proj'):
                    d_model = layer.mixer.in_proj.in_features
                    layer.mixer.lora_adapter = LoRAAdapter(d_model, d_model, r=lora_r, alpha=lora_alpha)
                    layer.mixer.lora_adapter.to(layer.mixer.in_proj.weight.device)
                elif hasattr(layer.mixer, 'Wqkv'):
                    d_model = layer.mixer.Wqkv.in_features
                    layer.mixer.lora_adapter = LoRAAdapter(d_model, d_model, r=lora_r, alpha=lora_alpha)
                    layer.mixer.lora_adapter.to(layer.mixer.Wqkv.weight.device)
        
        # 3. 修改 forward 以应用 LoRA
        original_backbone_forward = self.backbone.forward
        
        def modified_backbone_forward(inputs, **kwargs):
            # 需要在 MambaMixer 的 forward 中注入 LoRA 逻辑
            return original_backbone_forward(inputs, **kwargs)
        
        self.backbone.forward = modified_backbone_forward
        
        # 4. 保持 classification head 可训练
        max_length = self.model_args["max_length"]
        self.post_backbone_dropout = nn.Dropout(p=0.3)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.model_args["embedding_dim"], 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            nn.GELU()
        )
        self.ft_cell_type_projection = nn.Sequential(
            nn.Linear(max_length, 1024),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.model_args["cell_type_num"])
        )
    
    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x = self.backbone(x)
        x = self.feature_projection(x)
        x = torch.squeeze(x, dim=-1)
        x = self.post_backbone_dropout(x)
        return self.ft_cell_type_projection(x)
```

**配置建议：**
- `r=16` 或 `r=32`（秩越小参数越少）
- `alpha=2*r`（常见设置）
- 可训练参数减少约 95%，显存占用降低 60%+

**预期效果：** 在保持与全量微调相当性能的同时，显著降低显存需求和训练时间，适合资源受限场景。

---

## Idea 3、基于对比学习辅助损失的微调增强

**核心思路：** 在交叉熵/Focal Loss 基础上添加对比学习损失 (Contrastive Loss)，使同类细胞在 embedding 空间中更紧凑，异类细胞更分离。

**具体实现：**
```python
# 在 cell_type_annotation.py 中添加 NT-Xent 对比损失
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss
    用于对比学习的 InfoNCE 损失变体
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        features: (batch_size, hidden_dim) - L2 normalized embeddings
        labels: (batch_size,) - cell type labels
        """
        batch_size = features.shape[0]
        
        # 计算余弦相似度矩阵
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), 
                                                 features.unsqueeze(0), dim=2)
        similarity_matrix = similarity_matrix / self.temperature
        
        # 创建正样本对 mask (相同 label 的样本对)
        labels = labels.unsqueeze(0)
        positive_mask = (labels == labels.T).float()
        
        # 移除对角线 (自己和自己不是正样本对)
        positive_mask = positive_mask - torch.eye(batch_size, device=features.device)
        
        # 计算负样本对 mask
        negative_mask = 1.0 - positive_mask - torch.eye(batch_size, device=features.device)
        
        # 对于每个样本，计算 log(exp(sim_pos) / sum(exp(sim_all)))
        exp_sim = torch.exp(similarity_matrix)
        
        # 正样本的 exp 和 (每个样本可能有多个正样本)
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        
        # 所有非自身样本的 exp 和
        all_sum = (exp_sim * (positive_mask + negative_mask)).sum(dim=1)
        
        # NT-Xent loss
        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
        
        return loss.mean()


class FinetuneModelWithContrastiveLoss(PretrainModelMambaLM):
    def __init__(self, contrastive_weight=0.1, contrastive_temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_weight = contrastive_weight
        self.contrastive_loss_fn = NTXentLoss(temperature=contrastive_temperature)
        
        max_length = self.model_args["max_length"]
        self.post_backbone_dropout = nn.Dropout(p=0.3)
        
        # 修改 feature_projection 输出用于对比学习的 embedding
        self.feature_projection = nn.Sequential(
            nn.Linear(self.model_args["embedding_dim"], 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            nn.GELU()
        )
        
        # 用于对比学习的 projection head (将 backbone 输出映射到对比空间)
        self.contrastive_projection = nn.Sequential(
            nn.Linear(max_length, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        self.ft_cell_type_projection = nn.Sequential(
            nn.Linear(max_length, 1024),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.model_args["cell_type_num"])
        )
    
    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x_backbone = self.backbone(x)
        x = self.feature_projection(x_backbone)
        x = torch.squeeze(x, dim=-1)
        x = self.post_backbone_dropout(x)
        
        # 分类预测
        x_cell_type = self.ft_cell_type_projection(x)
        
        # 对比学习 embedding (L2 normalized)
        contrastive_embedding = self.contrastive_projection(x)
        contrastive_embedding = F.normalize(contrastive_embedding, p=2, dim=1)
        
        return x_cell_type, contrastive_embedding


# 修改训练循环
def cell_type_finetune_with_contrastive(...):
    cell_type_criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    contrastive_criterion = NTXentLoss(temperature=0.1)
    contrastive_weight = 0.1  # 对比损失权重
    
    for batch in train_dataloader:
        # ... 数据准备 ...
        cell_type_output, contrastive_embedding = model(value, chromosome, pos_start, pos_end)
        
        # 分类损失
        loss_cell_type = cell_type_criterion(cell_type_output, cell_type)
        
        # 对比损失
        loss_contrastive = contrastive_criterion(contrastive_embedding, cell_type)
        
        # 总损失
        total_loss = loss_cell_type + contrastive_weight * loss_contrastive
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**配置建议：**
- `contrastive_weight`: 0.05 ~ 0.2（从较小值开始）
- `temperature`: 0.05 ~ 0.2（较低温度使模型更关注困难负样本）
- 需要较大的 batch size (≥32) 以保证足够的负样本对

**预期效果：** 学习更具判别性的细胞类型 embedding，提升模型在少样本细胞类型和跨数据集泛化场景下的表现。
