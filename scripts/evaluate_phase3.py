
import sys
import os
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

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
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, val_loader, device):
    model.eval()
    
    # Containers for Retrieval Metric (Recall@K)
    pos_emb_a = []
    pos_emb_b = []
    
    logger.info("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            img_a, text_a, img_b, text_b, labels = batch
            labels = labels.to(device)
            
            # Forward
            emb_a = model.forward_with_text(img_a, text_a)
            emb_b = model.forward_with_text(img_b, text_b)
            
            # Safety Check
            if emb_a.size(0) != labels.size(0) or emb_b.size(0) != labels.size(0):
               continue

            # Collect Positive Pairs
            pos_mask = (labels == 1)
            if pos_mask.sum() > 0:
                pos_emb_a.append(emb_a[pos_mask].cpu())
                pos_emb_b.append(emb_b[pos_mask].cpu())
    
    # --- Retrieval Evaluation (Recall@K) ---
    if len(pos_emb_a) > 0:
        query_embs = torch.cat(pos_emb_a, dim=0)   # (N_pos, D)
        gallery_embs = torch.cat(pos_emb_b, dim=0) # (N_pos, D)
        
        # Normalize
        query_embs = nn.functional.normalize(query_embs, p=2, dim=1)
        gallery_embs = nn.functional.normalize(gallery_embs, p=2, dim=1)
        
        logger.info(f"Calculating similarity matrix for {query_embs.size(0)} pairs...")
        # Similarity Matrix
        sim_matrix = torch.matmul(query_embs, gallery_embs.t())
        
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
        
        # Print Results
        print("\n" + "="*30)
        print("   Evaluation Results   ")
        print("="*30)
        for k in ks:
            recalls[k] = (recalls[k] / num_queries) * 100.0
            print(f"Recall@{k}: {recalls[k]:.2f}%")
        print(f"MRR: {mrr / num_queries:.4f}")
        print("="*30 + "\n")
        
        recalls['mrr'] = mrr / num_queries
        return recalls
    else:
        logger.warning("No positive pairs found in dataset.")
        return {k: 0.0 for k in [1, 5, 10, 20, 50] + ['mrr']}

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. Dataset
    # Use Resize only, return PIL
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
    ])
    
    logger.info("Initializing Dataset...")
    val_dataset = MatchingDataset(
        jsonl_path=args.dataset_jsonl,
        image_root=args.image_root,
        transform=transform,
        mode='valid', # Evaluate on validation set
        split_ratio=args.split_ratio,
        seed=args.seed
    )
    
    def collate_fn(batch):
        # Return list of PIL images
        imgs_a = [item[0] for item in batch]
        texts_a = [item[1] for item in batch]
        imgs_b = [item[2] for item in batch]
        texts_b = [item[3] for item in batch]
        labels = torch.stack([item[4] for item in batch])
        return imgs_a, texts_a, imgs_b, texts_b, labels
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    # 2. Model
    logger.info(f"Loading Model: {args.model_name}")
    model = Qwen3VLWithTextFeatureExtractor(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        device=device
    )
    
    # Load Checkpoint
    checkpoint_path = args.checkpoint_path
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    # 3. Evaluate
    evaluate(model, val_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--dataset_jsonl", type=str, default="/home/sagemaker-user/data/dataset.jsonl")
    parser.add_argument("--image_root", type=str, default="/home/sagemaker-user/data/images")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    main(args)
