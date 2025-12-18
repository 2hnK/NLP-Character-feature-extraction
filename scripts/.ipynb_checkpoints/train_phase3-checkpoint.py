"""
Phase 3 Training Script: Matching Prediction
Uses Siamese Network with Qwen3-VL Backbone and Cosine Embedding Loss.
Enhanced with Hard Negative Mining, Early Stopping, and Collapse Prevention.
"""

import sys
import os
import argparse
import random
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

# Suppress tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.qwen_backbone import Qwen3VLWithTextFeatureExtractor
from src.data.matching_dataset import MatchingDataset

# Logging Setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/train_phase3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """
    Cosine learning rate scheduler with linear warmup.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class HardNegativeMiningLoss(nn.Module):
    """
    Hard Negative Mining with InfoNCE-style loss.
    Combines CosineEmbeddingLoss with in-batch hard negative selection.
    """
    def __init__(self, margin=0.5, temperature=0.07, mining_strategy="semi-hard"):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.mining_strategy = mining_strategy
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, emb_a, emb_b, labels):
        """
        Args:
            emb_a: Embeddings from branch A [B, D]
            emb_b: Embeddings from branch B [B, D]
            labels: 1 for positive, -1 for negative [B]
        """
        batch_size = emb_a.size(0)

        # Normalize embeddings
        emb_a_norm = F.normalize(emb_a, p=2, dim=1)
        emb_b_norm = F.normalize(emb_b, p=2, dim=1)

        # Compute all pairwise similarities within the batch
        # sim_matrix[i, j] = similarity between emb_a[i] and emb_b[j]
        sim_matrix = torch.mm(emb_a_norm, emb_b_norm.t()) / self.temperature

        # Diagonal elements are the original pairs
        pos_mask = (labels == 1)
        neg_mask = (labels == -1)

        # === Part 1: Standard Cosine Embedding Loss ===
        base_loss = self.cosine_loss(emb_a, emb_b, labels)

        # === Part 2: Hard Negative Mining Loss ===
        hard_neg_loss = torch.tensor(0.0, device=emb_a.device)

        if pos_mask.sum() > 0 and batch_size > 1:
            # For each positive pair (a_i, b_i), find hard negatives
            pos_indices = torch.where(pos_mask)[0]

            for idx in pos_indices:
                # Positive similarity
                pos_sim = sim_matrix[idx, idx]

                # All other b's are potential negatives for this a
                neg_sims = sim_matrix[idx].clone()
                neg_sims[idx] = float('-inf')  # Exclude the positive

                if self.mining_strategy == "hard":
                    # Hardest negative: highest similarity among negatives
                    hard_neg_sim = neg_sims.max()
                elif self.mining_strategy == "semi-hard":
                    # Semi-hard: negatives that are closer than positive but still negative
                    # i.e., pos_sim - margin < neg_sim < pos_sim
                    semi_hard_mask = (neg_sims > pos_sim - self.margin) & (neg_sims < pos_sim)
                    if semi_hard_mask.sum() > 0:
                        hard_neg_sim = neg_sims[semi_hard_mask].max()
                    else:
                        # Fallback to hardest negative
                        hard_neg_sim = neg_sims.max()
                else:
                    hard_neg_sim = neg_sims.max()

                # Triplet-style margin loss: want pos_sim > hard_neg_sim + margin
                triplet_loss = F.relu(hard_neg_sim - pos_sim + self.margin * self.temperature)
                hard_neg_loss = hard_neg_loss + triplet_loss

            hard_neg_loss = hard_neg_loss / pos_mask.sum()

        # === Part 3: Diversity Regularization (prevent collapse) ===
        # Encourage embeddings to be diverse
        diversity_loss = torch.tensor(0.0, device=emb_a.device)

        if batch_size > 1:
            # Off-diagonal similarities should be low
            eye_mask = torch.eye(batch_size, device=emb_a.device, dtype=torch.bool)
            off_diag_sim = sim_matrix[~eye_mask]
            # Penalize high off-diagonal similarities
            diversity_loss = F.relu(off_diag_sim - 0.5 / self.temperature).mean()

        # Combined loss
        total_loss = base_loss + 0.5 * hard_neg_loss + 0.1 * diversity_loss

        return total_loss, {
            'base_loss': base_loss.item(),
            'hard_neg_loss': hard_neg_loss.item() if isinstance(hard_neg_loss, torch.Tensor) else hard_neg_loss,
            'diversity_loss': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss
        }


class EarlyStopping:
    """Early stopping based on validation metric."""
    def __init__(self, patience=7, min_delta=0.5, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def compute_embedding_stats(embeddings):
    """Compute statistics to detect embedding collapse."""
    if embeddings.size(0) < 2:
        return {'std': 0.0, 'mean_pairwise_sim': 0.0}

    # Per-dimension standard deviation
    std = embeddings.std(dim=0).mean().item()

    # Mean pairwise similarity
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(emb_norm, emb_norm.t())
    n = sim_matrix.size(0)
    off_diag_mask = ~torch.eye(n, device=embeddings.device, dtype=torch.bool)
    mean_sim = sim_matrix[off_diag_mask].mean().item()

    return {'std': std, 'mean_pairwise_sim': mean_sim}


def evaluate(model, val_loader, criterion, device, compute_stats=True):
    model.eval()
    total_loss = 0.0

    pos_emb_a = []
    pos_emb_b = []
    all_emb_a = []
    all_emb_b = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            if batch is None:
                continue

            img_a, text_a, img_b, text_b, labels = batch
            labels = labels.to(device)

            emb_a = model.forward_with_text(img_a, text_a)
            emb_b = model.forward_with_text(img_b, text_b)

            if emb_a.size(0) != labels.size(0) or emb_b.size(0) != labels.size(0):
                continue

            # Loss
            if hasattr(criterion, 'cosine_loss'):
                loss, _ = criterion(emb_a, emb_b, labels)
            else:
                loss = criterion(emb_a, emb_b, labels)
            total_loss += loss.item()

            # Collect embeddings
            all_emb_a.append(emb_a.cpu())
            all_emb_b.append(emb_b.cpu())

            pos_mask = (labels == 1)
            if pos_mask.sum() > 0:
                pos_emb_a.append(emb_a[pos_mask].cpu())
                pos_emb_b.append(emb_b[pos_mask].cpu())

    avg_loss = total_loss / max(len(val_loader), 1)

    # Embedding statistics (collapse detection)
    emb_stats = {'std': 0.0, 'mean_pairwise_sim': 0.0}
    if compute_stats and len(all_emb_a) > 0:
        all_emb = torch.cat(all_emb_a + all_emb_b, dim=0)
        emb_stats = compute_embedding_stats(all_emb)

    # Retrieval metrics
    logger.info(f"Number of valid positive pairs for evaluation: {sum(e.size(0) for e in pos_emb_a)}")

    if len(pos_emb_a) > 0:
        query_embs = torch.cat(pos_emb_a, dim=0)
        gallery_embs = torch.cat(pos_emb_b, dim=0)

        query_embs = F.normalize(query_embs, p=2, dim=1)
        gallery_embs = F.normalize(gallery_embs, p=2, dim=1)

        sim_matrix = torch.mm(query_embs, gallery_embs.t())

        ks = [1, 5, 10, 20, 50]
        recalls = {k: 0.0 for k in ks}
        mrr = 0.0
        num_queries = query_embs.size(0)

        for i in range(num_queries):
            target_sim = sim_matrix[i, i].item()
            higher_sim_count = (sim_matrix[i] > target_sim).sum().item()
            rank = higher_sim_count + 1

            mrr += 1.0 / rank

            for k in ks:
                if rank <= k:
                    recalls[k] += 1

        for k in ks:
            recalls[k] = (recalls[k] / num_queries) * 100.0

        mrr = mrr / num_queries
        recalls['mrr'] = mrr
        recalls['emb_std'] = emb_stats['std']
        recalls['mean_pairwise_sim'] = emb_stats['mean_pairwise_sim']

        return avg_loss, recalls
    else:
        return avg_loss, {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0, 'mrr': 0.0,
                         'emb_std': emb_stats['std'], 'mean_pairwise_sim': emb_stats['mean_pairwise_sim']}


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Log hyperparameters
    logger.info("=" * 50)
    logger.info("Hyperparameters:")
    logger.info(f"  epochs: {args.epochs}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  learning_rate: {args.learning_rate}")
    logger.info(f"  weight_decay: {args.weight_decay}")
    logger.info(f"  margin: {args.margin}")
    logger.info(f"  temperature: {args.temperature}")
    logger.info(f"  mining_strategy: {args.mining_strategy}")
    logger.info(f"  warmup_epochs: {args.warmup_epochs}")
    logger.info(f"  early_stopping_patience: {args.patience}")
    logger.info("=" * 50)

    # Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
    ])

    logger.info("Initializing Datasets...")
    train_dataset = MatchingDataset(
        jsonl_path=args.dataset_jsonl,
        image_root=args.image_root,
        transform=transform,
        mode='train',
        split_ratio=args.split_ratio,
        seed=args.seed
    )

    val_dataset = MatchingDataset(
        jsonl_path=args.dataset_jsonl,
        image_root=args.image_root,
        transform=transform,
        mode='valid',
        split_ratio=args.split_ratio,
        seed=args.seed
    )

    test_dataset = MatchingDataset(
        jsonl_path=args.dataset_jsonl,
        image_root=args.image_root,
        transform=transform,
        mode='test',
        split_ratio=args.split_ratio,
        seed=args.seed
    )

    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        imgs_a = [item[0] for item in batch]
        texts_a = [item[1] for item in batch]
        imgs_b = [item[2] for item in batch]
        texts_b = [item[3] for item in batch]
        labels = torch.stack([item[4] for item in batch])
        return imgs_a, texts_a, imgs_b, texts_b, labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True  # Ensure consistent batch size for hard negative mining
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    logger.info(f"Train Size: {len(train_dataset)}, Valid Size: {len(val_dataset)}, Test Size: {len(test_dataset)}")

    # Model
    logger.info(f"Loading Model: {args.model_name}")
    model = Qwen3VLWithTextFeatureExtractor(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        device=device
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning Rate Scheduler
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr / args.learning_rate
    )

    # Loss function with Hard Negative Mining
    criterion = HardNegativeMiningLoss(
        margin=args.margin,
        temperature=args.temperature,
        mining_strategy=args.mining_strategy
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='max'
    )

    # Training setup
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))

    best_r1 = 0.0
    best_epoch = 0

    logger.info("Starting Training...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_base_loss = 0.0
        total_hard_neg_loss = 0.0
        total_diversity_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, batch in enumerate(pbar):
            if batch is None:
                continue

            img_a, text_a, img_b, text_b, labels = batch
            labels = labels.to(device)

            optimizer.zero_grad()

            emb_a = model.forward_with_text(img_a, text_a)
            emb_b = model.forward_with_text(img_b, text_b)

            if emb_a.size(0) != labels.size(0) or emb_b.size(0) != labels.size(0):
                continue

            if torch.isnan(emb_a).any() or torch.isnan(emb_b).any():
                logger.error("NaN detected in embeddings!")
                continue

            loss, loss_components = criterion(emb_a, emb_b, labels)

            if epoch == 0 and i == 0:
                n_pos = (labels == 1).sum().item()
                n_neg = (labels == -1).sum().item()
                logger.info(f"Batch 0 Stats: Pos={n_pos}, Neg={n_neg}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_base_loss += loss_components['base_loss']
            total_hard_neg_loss += loss_components['hard_neg_loss']
            total_diversity_loss += loss_components['diversity_loss']

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })

            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/BaseLoss", loss_components['base_loss'], global_step)
            writer.add_scalar("Train/HardNegLoss", loss_components['hard_neg_loss'], global_step)
            writer.add_scalar("Train/DiversityLoss", loss_components['diversity_loss'], global_step)
            writer.add_scalar("Train/LR", current_lr, global_step)

        num_batches = max(len(train_loader), 1)
        avg_train_loss = total_loss / num_batches
        avg_base_loss = total_base_loss / num_batches
        avg_hard_neg_loss = total_hard_neg_loss / num_batches
        avg_diversity_loss = total_diversity_loss / num_batches

        logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} "
                   f"(base: {avg_base_loss:.4f}, hard_neg: {avg_hard_neg_loss:.4f}, div: {avg_diversity_loss:.4f})")

        # Validation
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        r1 = val_metrics[1]
        r5 = val_metrics[5]
        r20 = val_metrics[20]
        mrr = val_metrics['mrr']
        emb_std = val_metrics.get('emb_std', 0.0)
        mean_sim = val_metrics.get('mean_pairwise_sim', 0.0)

        logger.info(f"Epoch {epoch+1} Valid Loss: {val_loss:.4f}, R@1: {r1:.2f}%, R@5: {r5:.2f}%, R@20: {r20:.2f}%, MRR: {mrr:.4f}")
        logger.info(f"Epoch {epoch+1} Embedding Stats: std={emb_std:.6f}, mean_pairwise_sim={mean_sim:.4f}")

        # Collapse warning
        if emb_std < args.collapse_threshold:
            logger.warning(f"WARNING: Embedding std ({emb_std:.6f}) below threshold ({args.collapse_threshold}). Possible collapse!")

        # TensorBoard logging
        writer.add_scalar("Valid/Loss", val_loss, epoch)
        writer.add_scalar("Valid/Recall@1", r1, epoch)
        writer.add_scalar("Valid/Recall@5", r5, epoch)
        writer.add_scalar("Valid/Recall@20", r20, epoch)
        writer.add_scalar("Valid/MRR", mrr, epoch)
        writer.add_scalar("Valid/EmbeddingStd", emb_std, epoch)
        writer.add_scalar("Valid/MeanPairwiseSim", mean_sim, epoch)

        # Save best model
        if r1 > best_r1:
            best_r1 = r1
            best_epoch = epoch + 1
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"New Best Model Saved! (R@1: {best_r1:.2f}%)")

        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"))

        # Early stopping check
        if early_stopping(r1):
            logger.info(f"Early stopping triggered at epoch {epoch+1}. Best R@1: {best_r1:.2f}% at epoch {best_epoch}")
            break

    logger.info(f"Training Complete. Best R@1: {best_r1:.2f}% at epoch {best_epoch}")

    # Final Test Evaluation
    logger.info("Starting Evaluation on Test Set...")
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    logger.info(f"Test Loss: {test_loss:.4f}")
    recall_str = ", ".join([f"R@{k}: {test_metrics[k]:.2f}%" for k in [1, 5, 10, 20, 50]])
    logger.info(f"Test Metrics: {recall_str}, MRR: {test_metrics.get('mrr', 0.0):.4f}")
    logger.info(f"Test Embedding Stats: std={test_metrics.get('emb_std', 0.0):.6f}, mean_sim={test_metrics.get('mean_pairwise_sim', 0.0):.4f}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument("--dataset_jsonl", type=str, default="/home/sagemaker-user/data/dataset.jsonl")
    parser.add_argument("--image_root", type=str, default="/home/sagemaker-user/data/images")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_phase3_v2")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--embedding_dim", type=int, default=256)

    # Training - Updated values
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_epochs", type=int, default=3)

    # Loss function
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mining_strategy", type=str, default="semi-hard", choices=["hard", "semi-hard"])

    # Early stopping
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min_delta", type=float, default=0.5)

    # Collapse detection
    parser.add_argument("--collapse_threshold", type=float, default=0.05)

    # Data
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    train(args)
