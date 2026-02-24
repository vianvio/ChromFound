import argparse
import json
import os
import pickle
import random

import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.dataset_ds import DatasetMultiPad
from src.models.chromfd_mixer import PretrainModelMambaLM
from src.models.lora_adapter import inject_lora_into_model, count_trainable_parameters
from src.utils.model_utils import ModelUtils
from src.utils.tb_utils import setup_logging


def warmup_lambda(current_step, warmup_steps=1000):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0


def load_data(file_path):
    if file_path.endswith('.h5ad'):
        print(f"Reading h5ad file from {file_path}")
        adata = sc.read_h5ad(file_path)
        return adata
    else:
        raise ValueError("Unsupported file format. Please provide a .h5ad file.")


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


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
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilities of the predicted classes
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_layerwise_lr_params(model, base_lr, decay_rate=0.95):
    """
    Generate parameter groups with layer-wise learning rate decay.
    
    Applies exponentially decaying learning rates across backbone layers:
    - Lower layers (closer to input) get smaller LR to preserve pretrained knowledge
    - Higher layers get larger LR to adapt faster to downstream tasks
    - Classification head uses base LR for rapid task adaptation
    
    Args:
        model: FinetuneModelMambaCellType or FinetuneModelWithContrastiveLoss
        base_lr: Base learning rate for classification head
        decay_rate: Decay factor per layer (0.95 means each lower layer gets 0.95x LR)
        
    Returns:
        List of parameter groups with different learning rates
    """
    param_groups = []
    
    # Get backbone layers
    backbone_layers = model.backbone.layers
    n_layers = len(backbone_layers)
    
    # Create parameter groups for each backbone layer with decaying LR
    for layer_idx in range(n_layers):
        layer_params = list(backbone_layers[layer_idx].parameters())
        # Skip if layer has no trainable parameters (e.g., frozen in LoRA mode)
        trainable_params = [p for p in layer_params if p.requires_grad]
        if trainable_params:
            lr = base_lr * (decay_rate ** (n_layers - layer_idx))
            param_groups.append({"params": trainable_params, "lr": lr})
    
    # Classification head uses base learning rate
    head_params = list(model.ft_cell_type_projection.parameters())
    trainable_head_params = [p for p in head_params if p.requires_grad]
    if trainable_head_params:
        param_groups.append({"params": trainable_head_params, "lr": base_lr})
    
    # Feature projection layers
    if hasattr(model, 'feature_projection'):
        proj_params = list(model.feature_projection.parameters())
        trainable_proj_params = [p for p in proj_params if p.requires_grad]
        if trainable_proj_params:
            param_groups.append({"params": trainable_proj_params, "lr": base_lr})
    
    # Contrastive projection (if using contrastive learning)
    if hasattr(model, 'contrastive_projection'):
        contrastive_params = list(model.contrastive_projection.parameters())
        trainable_contrastive_params = [p for p in contrastive_params if p.requires_grad]
        if trainable_contrastive_params:
            param_groups.append({"params": trainable_contrastive_params, "lr": base_lr})
    
    return param_groups


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss.
    
    InfoNCE-style contrastive loss for learning discriminative embeddings.
    Pulls same-class samples together and pushes different-class samples apart.
    
    Reference: https://arxiv.org/abs/2002.05709
    """
    def __init__(self, temperature=0.1):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature scaling factor (lower = more focused on hard negatives)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Compute NT-Xent contrastive loss.
        
        Args:
            features: L2-normalized embeddings of shape (batch_size, hidden_dim)
            labels: Class labels of shape (batch_size,)
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = features.shape[0]
        
        # Compute pairwise cosine similarity matrix
        # similarity_matrix[i,j] = cos(features[i], features[j]) / temperature
        similarity_matrix = F.cosine_similarity(
            features.unsqueeze(1), 
            features.unsqueeze(0), 
            dim=2
        ) / self.temperature
        
        # Create positive pair mask (same label pairs)
        labels = labels.unsqueeze(0)
        positive_mask = (labels == labels.T).float()
        
        # Remove self-similarity from positive mask (diagonal)
        positive_mask = positive_mask - torch.eye(batch_size, device=features.device)
        
        # Negative mask: all pairs that are not positive and not self
        negative_mask = 1.0 - positive_mask - torch.eye(batch_size, device=features.device)
        
        # Compute exp(similarity) for all pairs
        exp_sim = torch.exp(similarity_matrix)
        
        # Sum of positive pair similarities (excluding self)
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        
        # Sum of all non-self similarities (positive + negative)
        all_sum = (exp_sim * (positive_mask + negative_mask)).sum(dim=1)
        
        # NT-Xent loss: -log(sum_pos / sum_all)
        # Add epsilon for numerical stability
        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
        
        return loss.mean()


class FinetuneModelMambaCellType(PretrainModelMambaLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        max_length = self.model_args["max_length"]
        self.post_backbone_dropout = torch.nn.Dropout(p=0.3)
        self.feature_projection = torch.nn.Sequential(
            torch.nn.Linear(self.model_args["embedding_dim"], 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 1),
            torch.nn.GELU()
        )
        self.feature_projection.apply(init_weight)
        in_feature = max_length
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

        for name, param in self.mask_token_prediction.named_parameters():
            param.requires_grad = False

    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x = self.backbone(x)
        x = self.feature_projection(x)
        x = torch.squeeze(x, dim=-1)
        x = self.post_backbone_dropout(x)
        x_cell_type_prediction = self.ft_cell_type_projection(x)
        return x_cell_type_prediction


class FinetuneModelWithLoRA(PretrainModelMambaLM):
    """
    Finetuning model with LoRA (Low-Rank Adaptation) for parameter-efficient training.
    
    Freezes backbone parameters and injects LoRA adapters into Mamba/MHA layers.
    Reduces trainable parameters by ~95% while maintaining comparable performance.
    """
    def __init__(self, lora_r=16, lora_alpha=32, lora_dropout=0.1, **kwargs):
        """
        Initialize finetuning model with LoRA.
        
        Args:
            lora_r: LoRA rank (smaller = fewer params, typical: 8, 16, 32)
            lora_alpha: LoRA scaling factor (typical: 2*r or r)
            lora_dropout: Dropout rate for LoRA adapters
            **kwargs: Additional arguments passed to PretrainModelMambaLM
        """
        super().__init__(**kwargs)
        max_length = self.model_args["max_length"]
        
        # Inject LoRA adapters into backbone
        lora_count = inject_lora_into_model(
            self, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        
        self.post_backbone_dropout = torch.nn.Dropout(p=0.3)
        self.feature_projection = torch.nn.Sequential(
            torch.nn.Linear(self.model_args["embedding_dim"], 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 1),
            torch.nn.GELU()
        )
        self.feature_projection.apply(init_weight)
        in_feature = max_length
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

        for name, param in self.mask_token_prediction.named_parameters():
            param.requires_grad = False
        
        # Log LoRA injection info
        trainable_params, total_params = count_trainable_parameters(self)
        print(f"[LoRA] Injected {lora_count} LoRA adapters")
        print(f"[LoRA] Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100*trainable_params/total_params:.2f}%)")

    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x = self.backbone(x)
        x = self.feature_projection(x)
        x = torch.squeeze(x, dim=-1)
        x = self.post_backbone_dropout(x)
        x_cell_type_prediction = self.ft_cell_type_projection(x)
        return x_cell_type_prediction


class FinetuneModelWithContrastiveLoss(PretrainModelMambaLM):
    """
    Finetuning model with contrastive learning auxiliary loss.
    
    Adds NT-Xent contrastive loss to improve embedding discriminability.
    Enhances few-shot generalization and cross-dataset transfer.
    """
    def __init__(
        self, 
        contrastive_weight=0.1, 
        contrastive_temperature=0.1,
        contrastive_hidden_dim=128,
        **kwargs
    ):
        """
        Initialize finetuning model with contrastive learning.
        
        Args:
            contrastive_weight: Weight for contrastive loss in total loss (typical: 0.05-0.2)
            contrastive_temperature: Temperature for NT-Xent loss (typical: 0.05-0.2)
            contrastive_hidden_dim: Hidden dimension for contrastive projection head
            **kwargs: Additional arguments passed to PretrainModelMambaLM
        """
        super().__init__(**kwargs)
        max_length = self.model_args["max_length"]
        
        self.contrastive_weight = contrastive_weight
        self.contrastive_loss_fn = NTXentLoss(temperature=contrastive_temperature)
        
        self.post_backbone_dropout = torch.nn.Dropout(p=0.3)
        self.feature_projection = torch.nn.Sequential(
            torch.nn.Linear(self.model_args["embedding_dim"], 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 1),
            torch.nn.GELU()
        )
        self.feature_projection.apply(init_weight)
        
        # Contrastive projection head (maps backbone output to contrastive space)
        self.contrastive_projection = torch.nn.Sequential(
            torch.nn.Linear(max_length, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, contrastive_hidden_dim)
        )
        self.contrastive_projection.apply(init_weight)
        
        in_feature = max_length
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

        for name, param in self.mask_token_prediction.named_parameters():
            param.requires_grad = False

    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x_backbone = self.backbone(x)
        x = self.feature_projection(x_backbone)
        x = torch.squeeze(x, dim=-1)
        x = self.post_backbone_dropout(x)
        
        # Classification prediction
        x_cell_type = self.ft_cell_type_projection(x)
        
        # Contrastive embedding (L2 normalized)
        contrastive_embedding = self.contrastive_projection(x)
        contrastive_embedding = F.normalize(contrastive_embedding, p=2, dim=1)
        
        return x_cell_type, contrastive_embedding


def evaluate_finetune_model(model, val_dataloader, criterion, device, use_contrastive=False):
    """
    Evaluate finetuning model on validation/test data.
    
    Args:
        model: Finetuning model (standard or with contrastive learning)
        val_dataloader: Data loader for evaluation
        criterion: Loss function for classification
        device: Device to run evaluation on
        use_contrastive: Whether model returns contrastive embeddings
        
    Returns:
        Tuple of (eval_loss, eval_f1_score, accuracy, label_list, pred_list)
    """
    model.eval()
    eval_loss = 0
    accuracy = 0
    data_shape = 0
    eval_steps = 0
    eval_f1_score = 0
    cell_type_label_list = []
    cell_type_pred_list = []
    with torch.no_grad():
        for val_batch in val_dataloader:
            value, chromosome, pos_start, pos_end, cell_type = val_batch
            value = value.to(device)
            chromosome = chromosome.to(device)
            cell_type_gpu = cell_type.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)
            
            model_output = model(value, chromosome, pos_start, pos_end)
            
            # Handle contrastive learning model output
            if use_contrastive:
                cell_type_output, contrastive_embedding = model_output
            else:
                cell_type_output = model_output
            
            tmp_loss_cell_type_prediction = criterion(cell_type_output, cell_type_gpu)
            cell_type_pred = torch.argmax(cell_type_output, dim=-1)

            cell_type_label_list.extend(cell_type.detach().cpu().numpy().tolist())
            cell_type_pred_list.extend(cell_type_pred.detach().cpu().numpy().tolist())

            tmp_f1_score = f1_score(cell_type, cell_type_pred.cpu().numpy(), average='macro')
            eval_f1_score += tmp_f1_score
            accuracy += torch.sum(cell_type_pred == cell_type_gpu).item()
            data_shape += cell_type.size(0)
            eval_loss += tmp_loss_cell_type_prediction.item()
            eval_steps += 1
        eval_loss = eval_loss / eval_steps

    eval_loss_tensor = torch.tensor(eval_loss).to(device)
    eval_loss = eval_loss_tensor.item()

    eval_f1_score = f1_score(cell_type_label_list, cell_type_pred_list, average='macro')
    eval_f1_score_tensor = torch.tensor(eval_f1_score).to(device)
    eval_f1_score = eval_f1_score_tensor.item()

    accuracy = accuracy_score(cell_type_label_list, cell_type_pred_list)
    accuracy_tensor = torch.tensor(accuracy).to(device)
    accuracy = accuracy_tensor.item()

    return eval_loss, eval_f1_score, accuracy, cell_type_label_list, cell_type_pred_list


def cell_type_finetune(
        model,
        finetune_args,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device,
        logger
):
    """
    Main finetuning loop with support for contrastive learning.
    
    Args:
        model: Finetuning model (standard, LoRA, or contrastive)
        finetune_args: Configuration dictionary for finetuning
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        test_dataloader: Test data loader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        device: Device to run training on
        logger: Logger instance
    """
    model = model.to(device)
    cell_type_criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    
    # Check if using contrastive learning
    use_contrastive = hasattr(model, 'contrastive_loss_fn')
    contrastive_weight = getattr(model, 'contrastive_weight', 0.1)
    
    if use_contrastive:
        logger.info(f"[Config] Using contrastive learning with weight={contrastive_weight}")
        contrastive_criterion = NTXentLoss(temperature=0.1)
    
    step = 0
    best_f1_score = 0.0
    
    for eph in range(finetune_args.get("epoch")):
        for batch in train_dataloader:
            model.train()
            value, chromosome, pos_start, pos_end, cell_type = batch
            value = value.to(device)
            chromosome = chromosome.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)
            cell_type = cell_type.to(device)
            
            model_output = model(value, chromosome, pos_start, pos_end)
            
            # Handle contrastive learning model output
            if use_contrastive:
                cell_type_output, contrastive_embedding = model_output
                # Compute classification loss
                loss_cell_type = cell_type_criterion(cell_type_output, cell_type)
                # Compute contrastive loss
                loss_contrastive = contrastive_criterion(contrastive_embedding, cell_type)
                # Total loss
                loss_total = loss_cell_type + contrastive_weight * loss_contrastive
                loss_total.backward()
            else:
                cell_type_output = model_output
                loss_cell_type_prediction = cell_type_criterion(cell_type_output, cell_type)
                loss_cell_type_prediction.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            if step % finetune_args.get("loss_evaluate", 10) == 0:
                accuracy = torch.sum(torch.argmax(cell_type_output, dim=-1) == cell_type).item() / cell_type.size(0)
                if use_contrastive:
                    logger.info(
                        f"[Train] loss at epoch {eph} step {step}: {loss_total.item():.4f} "
                        f"(cls: {loss_cell_type.item():.4f}, cont: {loss_contrastive.item():.4f}), "
                        f"accuracy: {accuracy:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    logger.info(
                        f"[Train] loss at epoch {eph} step {step}: {loss_cell_type_prediction.item():.4f}, "
                        f"accuracy: {accuracy:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}"
                    )
            
            if step % finetune_args.get("val_evaluate", 10) == 0:
                eval_loss, eval_f1_score, eval_accuracy, eval_cell_type_label_list, eval_cell_type_pred_list = \
                    evaluate_finetune_model(model, val_dataloader, cell_type_criterion, device, use_contrastive=use_contrastive)
                test_loss, test_f1_score, test_accuracy, test_cell_type_label_list, test_cell_type_pred_list = \
                    evaluate_finetune_model(model, test_dataloader, cell_type_criterion, device, use_contrastive=use_contrastive)
                logger.info(
                    f"[Evaluate] loss at epoch {eph} step {step}: {eval_loss:.4f}, "
                    f"cell type accuracy: {eval_accuracy:.4f}, f1 score: {eval_f1_score:.4f}, "
                    f"lr: {optimizer.param_groups[0]['lr']:.6f}"
                )
                logger.info(
                    f"[Test] loss at epoch {eph} step {step}: {test_loss:.4f}, "
                    f"cell type accuracy: {test_accuracy:.4f}, f1 score: {test_f1_score:.4f}, "
                    f"lr: {optimizer.param_groups[0]['lr']:.6f}"
                )
                if eval_f1_score > best_f1_score:
                    best_f1_score = eval_f1_score
                    with open(os.path.join(
                            finetune_args["log_path"], f"cell_type_label_pred.pkl"), "wb") as f:
                        pickle.dump((eval_cell_type_label_list, eval_cell_type_pred_list), f)
                    logger.info(
                        f"[Test] best validation f1_score: {best_f1_score:.4f} at epoch {eph} step {step}, "
                        f"test accuracy: {test_accuracy:.4f}, f1_score: {test_f1_score:.4f}"
                    )
                    torch.save(model.state_dict(), os.path.join(finetune_args["log_path"], "best_model.pt"))
            step += 1
        torch.save(model.state_dict(), os.path.join(finetune_args["log_path"], f"epoch_{eph}.pt"))


def main_finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help='local rank passed from distributed launcher', default=0)
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")
    parser.add_argument("--learning_rate", type=float, required=True, help="learning rate for finetune")
    parser.add_argument("--pretrain_checkpoint_path", type=str, required=True, help="path to pretrain checkpoint")
    parser.add_argument("--pretrain_model_file", type=str, required=True, help="file name of pre-trained model")
    parser.add_argument('--pretrain_config_file', type=str, required=True, help='file name of pre-trained config')
    parser.add_argument("--cell_type_col", type=str, required=True, help="cell type column name")
    parser.add_argument("--epoch", type=int, required=True, help="epoch for training")
    parser.add_argument("--train_file_path", type=str, required=True, help="train file path")
    parser.add_argument("--test_file_path", type=str, required=True, help="validation file path")
    parser.add_argument("--log_path", type=str, required=True, help="log path")
    parser.add_argument("--load_pretrain_ckpt", action="store_true", default=True, help="load pre-trained model")
    
    # Idea 1: Layer-wise Learning Rate Decay
    parser.add_argument("--use_layerwise_lr", action="store_true", default=False,
                        help="Use layer-wise learning rate decay for finetuning")
    parser.add_argument("--layerwise_lr_decay_rate", type=float, default=0.95,
                        help="Decay rate for layer-wise LR (default: 0.95)")
    
    # Idea 2: LoRA Parameter-Efficient Finetuning
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Use LoRA for parameter-efficient finetuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA scaling factor (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate (default: 0.1)")
    
    # Idea 3: Contrastive Learning
    parser.add_argument("--use_contrastive", action="store_true", default=False,
                        help="Use contrastive learning auxiliary loss")
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                        help="Weight for contrastive loss (default: 0.1)")
    parser.add_argument("--contrastive_temperature", type=float, default=0.1,
                        help="Temperature for NT-Xent loss (default: 0.1)")
    parser.add_argument("--contrastive_hidden_dim", type=int, default=128,
                        help="Hidden dimension for contrastive projection (default: 128)")
    
    args = parser.parse_args()

    with open(os.path.join(args.pretrain_checkpoint_path, args.pretrain_config_file), 'r') as file:
        pretrain_config = yaml.safe_load(file)

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    pretrain_data_args = pretrain_config["data_args"]
    pretrain_model_args = pretrain_config["model_args"]
    log_path = args.log_path
    chromosome_vocab = ModelUtils.get_chromosome_vocab(
        os.path.join(args.pretrain_checkpoint_path, "chromosome_vocab.yaml")
    )
    pretrain_data_args["chromosome_vocab"] = chromosome_vocab
    adata_train_val = load_data(args.train_file_path)
    adata_test = load_data(args.test_file_path)

    adata_train_val.obs["tag"] = "train"
    adata_test.obs["tag"] = "test"
    adata_concat = sc.AnnData.concatenate(adata_train_val, adata_test)
    adata_train_val = adata_concat[adata_concat.obs["tag"] == "train"]
    adata_test = adata_concat[adata_concat.obs["tag"] == "test"]
    max_length = adata_concat.shape[1]

    cell_type = list(set(adata_train_val.obs[args.cell_type_col].unique().tolist() + adata_test.obs[
        args.cell_type_col].unique().tolist()))
    cell_type_map = {cell_type: idx for idx, cell_type in enumerate(sorted(cell_type))}

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    os.system(f"cp {os.path.join(args.pretrain_checkpoint_path, args.pretrain_config_file)} {log_path}")
    os.system(f"cp {os.path.join(args.pretrain_checkpoint_path, 'chromosome_vocab.yaml')} {log_path}")

    log_file_path = os.path.join(log_path, "finetune.log")

    finetune_logger = setup_logging(log_file_path)

    finetune_logger.info('PretrainLogger is configured and ready.')
    finetune_logger.info(f"args from parser: {args}")
    finetune_logger.info(f"max length for cell type finetune: {max_length}")
    
    # Log improvement method configuration
    if args.use_layerwise_lr:
        finetune_logger.info(f"[Idea 1] Using layer-wise LR decay with rate={args.layerwise_lr_decay_rate}")
    if args.use_lora:
        finetune_logger.info(f"[Idea 2] Using LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    if args.use_contrastive:
        finetune_logger.info(f"[Idea 3] Using contrastive learning with weight={args.contrastive_weight}, "
                            f"temperature={args.contrastive_temperature}")

    with open(os.path.join(log_path, "cell_type_map.json"), "w") as f:
        json.dump(cell_type_map, f)

    pretrain_data_args['cell_type_map'] = cell_type_map
    pretrain_model_args["cell_type_num"] = len(cell_type_map)
    pretrain_data_args['cell_type_col'] = args.cell_type_col
    pretrain_data_args["feature_num"] = adata_train_val.shape[1]
    pretrain_model_args["feature_num"] = adata_train_val.shape[1]
    pretrain_model_args["batch_size"] = args.batch_size
    pretrain_data_args["max_length"] = max_length
    pretrain_model_args["max_length"] = max_length
    pretrain_model_args["device"] = device
    pretrain_model_args["mask_ratio"] = 0.0
    pretrain_data_args["return_batch_label"] = False

    idx_list = [i for i in range(adata_train_val.X.shape[0])]
    random.shuffle(idx_list)
    split_idx = int(len(idx_list) * 0.9)
    train_idx = idx_list[:split_idx]
    val_idx = idx_list[split_idx:]
    adata_train = adata_train_val[train_idx]
    adata_val = adata_train_val[val_idx]

    train_dataset = DatasetMultiPad(*[adata_train], **pretrain_data_args)
    val_dataset = DatasetMultiPad(*[adata_val], **pretrain_data_args)
    test_dataset = DatasetMultiPad(*[adata_test], **pretrain_data_args)
    # Print dataset lengths
    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Validation Dataset Length: {len(val_dataset)}")
    print(f"Test Dataset Length: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    # Select model type based on improvement method
    if args.use_lora and args.use_contrastive:
        # Note: LoRA + Contrastive combination can be supported by extending FinetuneModelWithContrastiveLoss
        finetune_logger.warning("LoRA + Contrastive combination not directly supported. Using Contrastive only.")
        model = FinetuneModelWithContrastiveLoss(
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_hidden_dim=args.contrastive_hidden_dim,
            **pretrain_model_args
        )
    elif args.use_lora:
        model = FinetuneModelWithLoRA(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            **pretrain_model_args
        )
    elif args.use_contrastive:
        model = FinetuneModelWithContrastiveLoss(
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_hidden_dim=args.contrastive_hidden_dim,
            **pretrain_model_args
        )
    else:
        model = FinetuneModelMambaCellType(**pretrain_model_args)
    
    model = model.to(device)
    finetune_logger.info(f'Model parameters: {model}')
    
    # Setup optimizer with or without layer-wise LR
    if args.use_layerwise_lr:
        optimizer_params = get_layerwise_lr_params(
            model, 
            base_lr=args.learning_rate, 
            decay_rate=args.layerwise_lr_decay_rate
        )
        finetune_logger.info(f"[Optimizer] Using layer-wise LR with {len(optimizer_params)} parameter groups")
        for i, group in enumerate(optimizer_params):
            finetune_logger.info(f"  Group {i}: LR={group['lr']:.6f}, params={len(group['params'])}")
    else:
        optimizer_params = {
            "lr": args.learning_rate,
            "betas": (0.8, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-6
        }
        optimizer_params = [optimizer_params]  # Single parameter group
    
    optimizer = torch.optim.AdamW(model.parameters() if not args.use_layerwise_lr else optimizer_params, 
                                   **({} if args.use_layerwise_lr else optimizer_params[0]))
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_lambda(step, 200))
    
    if args.load_pretrain_ckpt:
        state_dict = torch.load(str(os.path.join(args.pretrain_checkpoint_path, args.pretrain_model_file)))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict['module'], strict=False)
        if missing_keys:
            print("Missing keys (not found in checkpoint):")
            for key in missing_keys:
                print(f"  {key}")
        if unexpected_keys:
            print("Unexpected keys (found in checkpoint but not in model):")
            for key in unexpected_keys:
                print(f"  {key}")

    finetune_config = {
        "pretrain_checkpoint_path": args.pretrain_checkpoint_path,
        "pretrain_model_name": args.pretrain_model_file,
        "pretrain_config_file": args.pretrain_config_file,
        "loss_evaluate": 20,
        "val_evaluate": 20,
        "log_path": log_path,
        "epoch": args.epoch,
    }
    cell_type_finetune(
        model,
        finetune_config,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device,
        finetune_logger
    )


if __name__ == '__main__':
    main_finetune()
