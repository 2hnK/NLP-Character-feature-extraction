"""
Main training script for profile feature extraction
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.backbone import ProfileFeatureExtractor
from src.models.losses import TripletLoss, OnlineTripletLoss
from src.data.dataset import create_dataloaders


class Trainer:
    """Training manager"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['hardware']['device'])

        # Initialize model
        self.model = self._build_model()

        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Initialize loss
        self.criterion = self._build_loss()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Initialize logging
        if config['logging']['wandb']['enabled']:
            wandb.init(
                project=config['logging']['wandb']['project'],
                config=config,
                tags=config['logging']['wandb'].get('tags', [])
            )

    def _build_model(self):
        """Build model"""
        model = ProfileFeatureExtractor(
            backbone_name=self.config['model']['backbone'],
            embedding_dim=self.config['model']['embedding_dim'],
            pretrained=self.config['model']['pretrained'],
            dropout=self.config['model']['dropout'],
            normalize=self.config['model']['normalize_embeddings']
        )

        model = model.to(self.device)

        # Multi-GPU
        if len(self.config['hardware']['gpu_ids']) > 1:
            model = nn.DataParallel(model, device_ids=self.config['hardware']['gpu_ids'])

        return model

    def _build_optimizer(self):
        """Build optimizer"""
        optimizer_name = self.config['training']['optimizer'].lower()

        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=self.config['training']['momentum'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_config = self.config['training']['scheduler']
        scheduler_type = scheduler_config['type']

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=scheduler_config['min_lr']
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor']
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        return scheduler

    def _build_loss(self):
        """Build loss function"""
        loss_config = self.config['training']['loss']
        loss_type = loss_config['type']

        if loss_type == 'triplet':
            criterion = TripletLoss(margin=loss_config['margin'])
        elif loss_type == 'online_triplet':
            criterion = OnlineTripletLoss(
                margin=loss_config['margin'],
                mining_strategy=self.config['training']['triplet']['strategy']
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return criterion

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            # Get data
            if 'anchor' in batch:
                # Triplet dataset
                anchors = batch['anchor'].to(self.device)
                positives = batch['positive'].to(self.device)
                negatives = batch['negative'].to(self.device)

                # Forward pass
                anchor_emb = self.model(anchors)
                positive_emb = self.model(positives)
                negative_emb = self.model(negatives)

                # Compute loss
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)

            else:
                # Online triplet dataset
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                embeddings = self.model(images)

                # Compute loss
                loss = self.criterion(embeddings, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config['training']['gradient_clip_val'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_val']
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Get data
                if 'anchor' in batch:
                    anchors = batch['anchor'].to(self.device)
                    positives = batch['positive'].to(self.device)
                    negatives = batch['negative'].to(self.device)

                    anchor_emb = self.model(anchors)
                    positive_emb = self.model(positives)
                    negative_emb = self.model(negatives)

                    loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                else:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)

                    embeddings = self.model(images)
                    loss = self.criterion(embeddings, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"

        # Get model (handle DataParallel)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        model_to_save.save_checkpoint(
            path=str(checkpoint_path),
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            config=self.config
        )

        # Save best model
        if is_best:
            best_path = checkpoint_dir.parent / 'saved_models' / 'best_model.pth'
            best_path.parent.mkdir(parents=True, exist_ok=True)
            model_to_save.save_checkpoint(
                path=str(best_path),
                optimizer=self.optimizer,
                epoch=self.current_epoch,
                config=self.config
            )

        # Save last checkpoint
        last_path = checkpoint_dir / 'last.pth'
        model_to_save.save_checkpoint(
            path=str(last_path),
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            config=self.config
        )

    def train(self, train_loader, val_loader):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.config['model']['backbone']}")
        print(f"Embedding dim: {self.config['model']['embedding_dim']}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)

            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Log metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if self.config['logging']['wandb']['enabled']:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.4f}")

            if (epoch + 1) % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint(is_best=is_best)

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train profile feature extractor')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    # Set random seed
    torch.manual_seed(config['reproducibility']['seed'])

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_root=config['paths']['processed_data'],
        metadata_csv=os.path.join(config['paths']['processed_data'], 'metadata.csv'),
        interactions_csv=config['paths'].get('interactions'),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        dataset_type='online_triplet'  # Use online triplet mining
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create trainer
    trainer = Trainer(config)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
