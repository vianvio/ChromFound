"""
Advanced fine-tuning script for ChromFound with gradient accumulation and dynamic learning rate scheduling
"""

import argparse
import json
import math
import os
import pickle
import random

import scanpy as sc
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader

from src.data.dataset_ds import DatasetMultiPad
from src.models.chromfd_mixer import PretrainModelMambaLM
from src.utils.model_utils import ModelUtils
from src.utils.tb_utils import setup_logging


def warmup_lambda(current_step, warmup_steps=1000):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between
    the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly
    between 0 and the initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_exponential_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, decay_rate=0.95, min_lr_ratio=0.1, last_epoch=-1):
    """
    Create a schedule with a learning rate that decays exponentially after a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Exponential decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, decay_rate ** (progress * num_training_steps))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, power=1.0, min_lr_ratio=0.0, last_epoch=-1):
    """
    Create a schedule with a learning rate that decays polynomially after a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Polynomial decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, (1.0 - progress) ** power)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


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


class AdaptiveGradientClipping(torch.nn.Module):
    """
    Adaptive gradient clipping to prevent exploding gradients
    """
    def __init__(self, clip_factor=0.01, eps=1e-3):
        super().__init__()
        self.clip_factor = clip_factor
        self.eps = eps

    def forward(self, parameters):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        
        for p in parameters:
            param_norm = torch.norm(p, p=2, dtype=torch.float32)
            grad_norm = torch.norm(p.grad, p=2, dtype=torch.float32)
            
            if param_norm > 0 and grad_norm > 0:
                max_norm = self.clip_factor * param_norm + self.eps
                clip_coef = max_norm / (grad_norm + self.eps)
                
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)


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


def evaluate_finetune_model(model, val_dataloader, criterion, device):
    model.eval()
    eval_loss = 0
    accuracy = 0
    data_shape = 0
    eval_steps = 0
    eval_f1_score = 0
    cell_type_label_list = []
    cell_type_pred_list = []
    cell_type_prob_list = []  # Store probabilities for ROC calculation
    
    with torch.no_grad():
        for val_batch in val_dataloader:
            value, chromosome, pos_start, pos_end, cell_type = val_batch
            value = value.to(device)
            chromosome = chromosome.to(device)
            cell_type_gpu = cell_type.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)
            cell_type_output = model(value, chromosome, pos_start, pos_end)
            
            # Apply softmax to get probabilities
            cell_type_probs = F.softmax(cell_type_output, dim=-1)
            
            tmp_loss_cell_type_prediction = criterion(cell_type_output, cell_type_gpu)
            cell_type_pred = torch.argmax(cell_type_output, dim=-1)

            cell_type_label_list.extend(cell_type.detach().cpu().numpy().tolist())
            cell_type_pred_list.extend(cell_type_pred.detach().cpu().numpy().tolist())
            cell_type_prob_list.extend(cell_type_probs.detach().cpu().numpy())

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

    # Calculate ROC-AUC if we have binary classification or can compute it
    try:
        # For multiclass, we compute ROC-AUC using one-vs-rest approach
        import numpy as np
        cell_type_probs_np = np.array(cell_type_prob_list)
        unique_labels = list(set(cell_type_label_list))
        
        if len(unique_labels) == 2:  # Binary classification
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(cell_type_label_list, classes=sorted(unique_labels))[:, 1]  # Get positive class
            roc_auc = roc_auc_score(y_true_bin, cell_type_probs_np[:, 1])
        elif len(unique_labels) > 2:  # Multiclass
            y_true_bin = label_binarize(cell_type_label_list, classes=sorted(unique_labels))
            roc_auc = roc_auc_score(y_true_bin, cell_type_probs_np, average='macro', multi_class='ovr')
        else:
            roc_auc = 0.0  # Single class
    except Exception as e:
        print(f"Could not compute ROC-AUC: {str(e)}")
        roc_auc = 0.0

    return eval_loss, eval_f1_score, accuracy, roc_auc, cell_type_label_list, cell_type_pred_list


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
    model = model.to(device)
    cell_type_criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')

    # Gradient accumulation parameters
    grad_accum_steps = finetune_args.get("grad_accum_steps", 1)  # Number of steps to accumulate gradients
    total_steps = finetune_args.get("total_steps", len(train_dataloader) * finetune_args.get("epoch"))

    # Initialize step counter
    step = 0
    global_step = 0
    best_f1_score = 0.0
    best_accuracy = 0.0
    best_roc_auc = 0.0

    # Initialize adaptive gradient clipping
    adaptive_clip = AdaptiveGradientClipping(clip_factor=0.01)

    for eph in range(finetune_args.get("epoch")):
        optimizer.zero_grad()  # Initialize gradients

        for batch_idx, batch in enumerate(train_dataloader):
            model.train()
            value, chromosome, pos_start, pos_end, cell_type = batch
            value = value.to(device)
            chromosome = chromosome.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)
            cell_type = cell_type.to(device)

            cell_type_output = model(value, chromosome, pos_start, pos_end)
            # Compute Focal Loss
            loss_cell_type_prediction = cell_type_criterion(cell_type_output, cell_type)

            # Normalize the loss to account for gradient accumulation
            loss_cell_type_prediction = loss_cell_type_prediction / grad_accum_steps

            # Backward pass
            loss_cell_type_prediction.backward()

            # Perform optimizer step after accumulating gradients
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                # Apply adaptive gradient clipping
                adaptive_clip(model.parameters())

                optimizer.step()
                optimizer.zero_grad()  # Reset gradients after step
                lr_scheduler.step()  # Update learning rate

                global_step += 1

                if global_step % finetune_args.get("loss_evaluate", 10) == 0:
                    accuracy = torch.sum(torch.argmax(cell_type_output, dim=-1) == cell_type).item() / cell_type.size(0)
                    logger.info(
                        f"[Train] loss at epoch {eph} step {global_step}: {loss_cell_type_prediction.item() * grad_accum_steps}, "
                        f"accuracy: {accuracy:.4f}, lr: {optimizer.param_groups[0]['lr']:.8f}"
                    )

                if global_step % finetune_args.get("val_evaluate", 10) == 0:
                    eval_loss, eval_f1_score, eval_accuracy, eval_roc_auc, eval_cell_type_label_list, eval_cell_type_pred_list = \
                        evaluate_finetune_model(model, val_dataloader, cell_type_criterion, device)
                    test_loss, test_f1_score, test_accuracy, test_roc_auc, eval_cell_type_label_list, eval_cell_type_pred_list = \
                        evaluate_finetune_model(model, test_dataloader, cell_type_criterion, device)
                    
                    logger.info(
                        f"[Evaluate] loss at epoch {eph} step {global_step}: {eval_loss}, "
                        f"cell type accuracy: {eval_accuracy:.4f}, f1 score: {eval_f1_score:.4f}, roc auc: {eval_roc_auc:.4f}, "
                        f"lr: {optimizer.param_groups[0]['lr']:.8f}"
                    )
                    logger.info(
                        f"[Test] loss at epoch {eph} step {global_step}: {test_loss}, "
                        f"cell type accuracy: {test_accuracy:.4f}, f1 score: {test_f1_score:.4f}, roc auc: {test_roc_auc:.4f}, "
                        f"lr: {optimizer.param_groups[0]['lr']:.8f}"
                    )
                    
                    # Save best model based on F1 score
                    if eval_f1_score > best_f1_score:
                        best_f1_score = eval_f1_score
                        with open(os.path.join(
                                finetune_args["log_path"], f"cell_type_label_pred.pkl"), "wb") as f:
                            pickle.dump((eval_cell_type_label_list, eval_cell_type_pred_list), f)
                        logger.info(
                            f"[Test] best validation f1_score: {best_f1_score:.4f} at epoch {eph} step {global_step}, "
                            f"test accuracy: {test_accuracy:.4f}, f1_score: {test_f1_score:.4f}, roc_auc: {test_roc_auc:.4f}"
                        )
                        torch.save(model.state_dict(), os.path.join(finetune_args["log_path"], "best_model.pt"))
                        
                        # Save metrics
                        metrics = {
                            'best_f1_score': best_f1_score,
                            'best_accuracy': test_accuracy,
                            'best_roc_auc': test_roc_auc,
                            'final_test_f1_score': test_f1_score,
                            'final_test_accuracy': test_accuracy,
                            'final_test_roc_auc': test_roc_auc
                        }
                        with open(os.path.join(finetune_args["log_path"], "metrics.json"), "w") as f:
                            json.dump(metrics, f)
            else:
                # Still accumulating gradients, don't update optimizer yet
                pass

            step += 1
            
        # Save model at the end of each epoch
        torch.save(model.state_dict(), os.path.join(finetune_args["log_path"], f"epoch_{eph}.pt"))
        
        # Early stopping check (optional)
        if finetune_args.get("early_stopping", False):
            patience = finetune_args.get("patience", 3)
            if eph > patience and best_f1_score <= max([best_f1_score]):
                logger.info(f"Early stopping triggered at epoch {eph}")
                break


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
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="number of gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="warmup_cosine", choices=["warmup_linear", "warmup_cosine", "warmup_exponential", "warmup_polynomial"], help="type of learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=200, help="number of warmup steps")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="minimum learning rate ratio for scheduling")
    parser.add_argument("--decay_rate", type=float, default=0.95, help="decay rate for exponential scheduling")
    parser.add_argument("--poly_power", type=float, default=1.0, help="power for polynomial decay scheduling")
    parser.add_argument("--early_stopping", action="store_true", default=False, help="enable early stopping")
    parser.add_argument("--patience", type=int, default=3, help="patience for early stopping")
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

    # Determine max_length from the original datasets
    max_length = adata_train_val.shape[1]

    # Combine cell types from both datasets for mapping
    cell_type = list(set(adata_train_val.obs[args.cell_type_col].unique().tolist() + adata_test.obs[
        args.cell_type_col].unique().tolist()))
    cell_type_map = {cell_type: idx for idx, cell_type in enumerate(sorted(cell_type))}

    # Split train_val into actual train and validation sets
    idx_list = [i for i in range(adata_train_val.X.shape[0])]
    random.shuffle(idx_list)
    split_idx = int(len(idx_list) * 0.9)
    train_idx = idx_list[:split_idx]
    val_idx = idx_list[split_idx:]
    adata_train = adata_train_val[train_idx]
    adata_val = adata_train_val[val_idx]

    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    os.system(f"cp {os.path.join(args.pretrain_checkpoint_path, args.pretrain_config_file)} {log_path}")
    os.system(f"cp {os.path.join(args.pretrain_checkpoint_path, 'chromosome_vocab.yaml')} {log_path}")

    log_file_path = os.path.join(log_path, "finetune.log")

    finetune_logger = setup_logging(log_file_path)

    finetune_logger.info('PretrainLogger is configured and ready.')
    finetune_logger.info(f"args from parser: {args}")
    finetune_logger.info(f"max length for cell type finetune: {max_length}")

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

    model = FinetuneModelMambaCellType(**pretrain_model_args)
    model = model.to(device)
    finetune_logger.info(f'Model parameters: {model}')
    optimizer_params = {
        "lr": args.learning_rate,
        "betas": (0.8, 0.999),
        "eps": 1e-8,
        "weight_decay": 1e-6
    }
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)

    # Calculate total training steps for scheduler
    total_steps = len(train_dataloader) * args.epoch

    # Select learning rate scheduler based on the specified type
    if args.lr_scheduler_type == "warmup_linear":
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_lambda(step, args.warmup_steps))
    elif args.lr_scheduler_type == "warmup_cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=args.min_lr_ratio
        )
    elif args.lr_scheduler_type == "warmup_exponential":
        lr_scheduler = get_exponential_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
            decay_rate=args.decay_rate,
            min_lr_ratio=args.min_lr_ratio
        )
    elif args.lr_scheduler_type == "warmup_polynomial":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
            power=args.poly_power,
            min_lr_ratio=args.min_lr_ratio
        )
    else:
        # Default to the original warmup lambda
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_lambda(step, args.warmup_steps))

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
        "grad_accum_steps": args.grad_accum_steps,
        "total_steps": len(train_dataloader) * args.epoch,
        "early_stopping": args.early_stopping,
        "patience": args.patience
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