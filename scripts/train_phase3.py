"""
Phase 3 Training Script: Matching Prediction
Uses Siamese Network with Qwen3-VL Backbone and Cosine Embedding Loss.
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer # Optional if needed explicitly

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

def evaluate(model, val_loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    
    # Containers for Retrieval Metric (Recall@K)
    # We only care about Positive pairs for Retrieval Evaluation
    # Query: User A from positive pair
    # Gallery: User B from positive pair (Candidate Pool)
    pos_emb_a = []
    pos_emb_b = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            if batch is None:
                continue
                
            img_a, text_a, img_b, text_b, labels = batch
            labels = labels.to(device)
            
            # Forward
            emb_a = model.forward_with_text(img_a, text_a)
            emb_b = model.forward_with_text(img_b, text_b)
            
            # Safety Check for Batch Size Mismatch
            if emb_a.size(0) != labels.size(0) or emb_b.size(0) != labels.size(0):
               continue
            
            # Loss (Compute on ALL pairs, both pos and neg)
            loss = criterion(emb_a, emb_b, labels)
            total_loss += loss.item()
            
            # Collect Positive Pairs for Retrieval Evaluation
            # labels: 1 (Pos), -1 (Neg)
            pos_mask = (labels == 1)
            
            if pos_mask.sum() > 0:
                pos_emb_a.append(emb_a[pos_mask].cpu())
                pos_emb_b.append(emb_b[pos_mask].cpu())
            
    avg_loss = total_loss / len(val_loader)
    
    # --- Retrieval Evaluation (Recall@K & MRR) ---
    logger.info(f"Number of valid positive pairs for evaluation: {len(pos_emb_a)}")
    
    if len(pos_emb_a) > 0:
        # Concatenate all positive embeddings
        query_embs = torch.cat(pos_emb_a, dim=0)   # (N_pos, D)
        gallery_embs = torch.cat(pos_emb_b, dim=0) # (N_pos, D)
        
        # Normalize (already done in model, but safe to redo or ensure)
        query_embs = nn.functional.normalize(query_embs, p=2, dim=1)
        gallery_embs = nn.functional.normalize(gallery_embs, p=2, dim=1)
        
        # Similarity Matrix: (N_pos, N_pos)
        # sim[i, j] = similarity between Query i and Gallery j
        sim_matrix = torch.matmul(query_embs, gallery_embs.t())
        
        # Metric Calculation
        # For each query i, the ground truth match is gallery i.
        # We calculate the rank of diag[i] among row i.
        
        ks = [1, 5, 10, 20, 50]
        recalls = {k: 0.0 for k in ks}
        mrr = 0.0
        num_queries = query_embs.size(0)
        
        for i in range(num_queries):
            target_sim = sim_matrix[i, i].item()
            # Count how many gallery items have higher similarity than the target
            # Note: We subtract 1 because the target itself is in the row check? 
            # No, higher_sim_count = number of items strictly greater than target.
            # Rank = higher_sim_count + 1.
            higher_sim_count = (sim_matrix[i] > target_sim).sum().item()
            rank = higher_sim_count + 1
            
            mrr += 1.0 / rank
            
            for k in ks:
                if rank <= k:
                    recalls[k] += 1
        
        # Average
        for k in ks:
            recalls[k] = (recalls[k] / num_queries) * 100.0 # Percent
        
        mrr = mrr / num_queries
        recalls['mrr'] = mrr
            
        return avg_loss, recalls
    else:
        # No positive pairs in validation set?
        return avg_loss, {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0, 'mrr': 0.0}

def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. Dataset & DataLoader
    # Qwen Processor expects PIL images, so we only Resize.
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
    
    # Collate fn for list of PIL images and strings
    def collate_fn(batch):
        # batch is list of tuples: (img_a, text_a, img_b, text_b, label) or None
        # Filter None items
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
            
        imgs_a = [item[0] for item in batch]
        texts_a = [item[1] for item in batch]
        imgs_b = [item[2] for item in batch]
        texts_b = [item[3] for item in batch]
        labels = torch.stack([item[4] for item in batch])
        return imgs_a, texts_a, imgs_b, texts_b, labels
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    logger.info(f"Train Size: {len(train_dataset)}, Valid Size: {len(val_dataset)}")
    
    # 2. Model
    logger.info(f"Loading Model: {args.model_name}")
    # Initialize with text support
    model = Qwen3VLWithTextFeatureExtractor(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        device=device
    )
    
    # 3. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Cosine Embedding Loss
    # Inputs: x1, x2, target (1 or -1)
    # Loss = 1 - cos(x1, x2) if target=1
    # Loss = max(0, cos(x1, x2) - margin) if target=-1
    criterion = nn.CosineEmbeddingLoss(margin=args.margin)
    
    # 4. Training Loop
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))
    
    best_acc = 0.0
    
    logger.info("Starting Training...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, batch in enumerate(pbar):
            img_a, text_a, img_b, text_b, labels = batch
            
            # To Device (Images & Labels only, text is list)
            labels = labels.to(device)
            pass
            
            optimizer.zero_grad()
            
            # Forward
            emb_a = model.forward_with_text(img_a, text_a)
            emb_b = model.forward_with_text(img_b, text_b)
            
            # Safety Check for Batch Size Mismatch
            if emb_a.size(0) != labels.size(0) or emb_b.size(0) != labels.size(0):
               continue
            
            # Debug: Check for NaNs
            if torch.isnan(emb_a).any() or torch.isnan(emb_b).any():
                logger.error("NaN detected in embeddings!")
                logger.error(f"Labels: {labels}")
                continue
                
            loss = criterion(emb_a, emb_b, labels)
            
            # Debug: Check first batch labels
            if epoch == 0 and i == 0:
                n_pos = (labels == 1).sum().item()
                n_neg = (labels == -1).sum().item()
                logger.info(f"Batch 0 Stats: Pos={n_pos}, Neg={n_neg}")
            
            loss.backward()
            
            # Gradient Clipping to prevent NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        r1 = val_metrics[1]
        r5 = val_metrics[5]
        r20 = val_metrics[20]
        mrr = val_metrics['mrr']
        
        logger.info(f"Epoch {epoch+1} Valid Loss: {val_loss:.4f}, R@1: {r1:.2f}%, R@5: {r5:.2f}%, R@20: {r20:.2f}%, MRR: {mrr:.4f}")
        
        writer.add_scalar("Valid/Loss", val_loss, epoch)
        writer.add_scalar("Valid/Recall@1", r1, epoch)
        writer.add_scalar("Valid/Recall@5", r5, epoch)
        writer.add_scalar("Valid/Recall@20", r20, epoch)
        writer.add_scalar("Valid/MRR", mrr, epoch)
        
        # Save Best (Based on Recall@1)
        if r1 > best_acc:
            best_acc = r1
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"New Best Model Saved! (R@1: {best_acc:.2f}%)")
            
        # Periodic Save
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    logger.info("Training Complete.")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", type=str, default="/home/sagemaker-user/data/dataset.jsonl")
    parser.add_argument("--image_root", type=str, default="/home/sagemaker-user/data/images")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_phase3")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8) # Small batch size due to VLM memory
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=2048) # Qwen output dim? 
    # Wait, in qwen_backbone.py, default embedding_dim=512 for projection head.
    # We should match that. Or is it 256 per plan? Plan said 256.
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--image_size", type=int, default=448) # Qwen inputs
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0) # safe for windows
    
    args = parser.parse_args()
    
    # Override embedding_dim to match plan (256)
    # But wait, Qwen3VLWithTextFeatureExtractor init takes embedding_dim argument.
    # It constructs a projection head to that dim.
    # So we should pass 256 here.
    args.embedding_dim = 256 
    
    # Fix Transform to return PIL images (because model expects PIL for processor)
    # We override the transform creation inside train() with a custom one or just pass None/Resize
    # Ideally, we define a wrapper class for transform that keeps PIL
    # or we handle it in Dataset.
    # For simplicity, we'll redefine the transform usage in train function.
    
    # Actually, let's just modify the train function to use Resize only.
    # BUT, transforms.Resize returns PIL if input is PIL.
    # DataLoader default collate fails on PIL.
    # We defined a custom collate_fn that stacks images? 
    # torch.stack expects tensors.
    
    # Solution: Custom Collate for PIL
    # imgs_a = [item[0] for item in batch] (List of PIL)
    # This matches what Qwen3VLWithTextFeatureExtractor.forward_with_text expects (List of PIL)!
    # Great.
    
    # So we must REMOVE ToTensor from the transform.
    
    train(args)
