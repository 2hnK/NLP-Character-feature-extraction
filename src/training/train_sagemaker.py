"""
SageMaker Training Script for Qwen3-VL Profile Feature Extraction
"""

import os
import sys
import json
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# SageMaker imports
import sagemaker
from sagemaker.experiments import Run

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.qwen_backbone import Qwen3VLFeatureExtractor
from src.models.losses import OnlineTripletLoss
from src.data.dataset import create_dataloaders


class SageMakerTrainer:
    """Training manager for SageMaker environment"""

    def __init__(self, config, sm_training_env=None):
        """
        Args:
            config: Training configuration dict
            sm_training_env: SageMaker training environment dict
        """
        self.config = config
        self.sm_training_env = sm_training_env or {}

        # Setup paths
        self.setup_paths()

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

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
        self.global_step = 0

        # SageMaker Experiments tracking
        self.run = None
        if config['logging'].get('sagemaker_experiments', {}).get('enabled', False):
            try:
                experiment_name = config['logging']['sagemaker_experiments']['experiment_name']
                trial_name = config['logging']['sagemaker_experiments'].get('trial_name')
                self.run = Run(experiment_name=experiment_name, run_name=trial_name)
            except Exception as e:
                print(f"Could not initialize SageMaker Experiments: {e}")

    def setup_paths(self):
        """Setup SageMaker paths"""
        # Check if running in SageMaker training job
        if 'SM_MODEL_DIR' in os.environ:
            self.model_dir = os.environ['SM_MODEL_DIR']
            self.output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
            self.checkpoint_dir = os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints')
            self.data_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
        else:
            # Running locally in SageMaker Studio
            base_dir = Path.home() / "SageMaker" / "dating-matcher"
            self.model_dir = str(base_dir / "models")
            self.output_dir = str(base_dir / "output")
            self.checkpoint_dir = str(base_dir / "checkpoints")
            self.data_dir = self.config['paths']['processed_data']

        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f"Model dir: {self.model_dir}")
        print(f"Data dir: {self.data_dir}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")

    def _build_model(self):
        """Build Qwen3-VL model"""
        model = Qwen3VLFeatureExtractor(
            model_name=self.config['model']['model_name'],
            embedding_dim=self.config['model']['embedding_dim'],
            freeze_vision_encoder=self.config['model']['freeze_vision_encoder'],
            use_projection_head=self.config['model']['use_projection_head'],
            device=self.device
        )

        return model

    def _build_optimizer(self):
        """Build optimizer"""
        optimizer_name = self.config['training']['optimizer'].lower()

        # Different learning rates for vision encoder and projection head
        if self.config['model']['freeze_vision_encoder']:
            # Only train projection head
            params = self.model.projection_head.parameters()
        else:
            # Train both with different learning rates
            params = [
                {'params': self.model.model.parameters(), 'lr': self.config['training']['learning_rate'] * 0.1},
                {'params': self.model.projection_head.parameters(), 'lr': self.config['training']['learning_rate']}
            ]

        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                params,
                lr=self.config['training']['learning_rate'],
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
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        return scheduler

    def _build_loss(self):
        """Build loss function"""
        loss_config = self.config['training']['loss']

        criterion = OnlineTripletLoss(
            margin=loss_config['margin'],
            mining_strategy=self.config['training']['triplet']['strategy']
        )

        return criterion

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Gradient accumulation
        accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config['training']['mixed_precision']):
                embeddings = self.model(images)
                loss = self.criterion(embeddings, labels)

                # Scale loss for gradient accumulation
                loss = loss / accum_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accum_steps == 0:
                # Gradient clipping
                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_val']
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Update metrics
            total_loss += loss.item() * accum_steps
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item() * accum_steps})

            # Log to SageMaker Experiments
            if self.run and batch_idx % 10 == 0:
                self.run.log_metric(
                    name="train_loss_step",
                    value=loss.item() * accum_steps,
                    step=self.global_step
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.config['training']['mixed_precision']):
                    embeddings = self.model(images)
                    loss = self.criterion(embeddings, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{self.current_epoch}.pth"

        self.model.save_checkpoint(
            path=str(checkpoint_path),
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            config=self.config
        )

        # Save best model
        if is_best:
            best_path = Path(self.model_dir) / 'model.pth'
            self.model.save_checkpoint(
                path=str(best_path),
                optimizer=self.optimizer,
                epoch=self.current_epoch,
                config=self.config
            )
            print(f"Saved best model to {best_path}")

        # Save last checkpoint
        last_path = Path(self.checkpoint_dir) / 'last.pth'
        self.model.save_checkpoint(
            path=str(last_path),
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            config=self.config
        )

    def train(self, train_loader, val_loader):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']

        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model: {self.config['model']['model_name']}")
        print(f"Embedding dim: {self.config['model']['embedding_dim']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"LR: {current_lr:.6f}")

            # SageMaker Experiments logging
            if self.run:
                self.run.log_metric(name="train_loss", value=train_loss, step=epoch)
                self.run.log_metric(name="val_loss", value=val_loss, step=epoch)
                self.run.log_metric(name="learning_rate", value=current_lr, step=epoch)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"✓ New best validation loss: {val_loss:.4f}")

            if (epoch + 1) % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint(is_best=is_best)

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

        # Save final model to SageMaker model directory
        final_model_path = Path(self.model_dir) / 'model.pth'
        self.model.save_checkpoint(
            path=str(final_model_path),
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            config=self.config
        )
        print(f"Final model saved to: {final_model_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SageMaker training for profile matcher')

    # SageMaker parameters
    parser.add_argument('--config', type=str, default='configs/config_sagemaker.yaml',
                       help='Path to config file')

    # Hyperparameters (can be overridden by SageMaker)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--embedding-dim', type=int, default=None)

    # Parse known args (SageMaker may pass additional args)
    args, _ = parser.parse_known_args()

    return args


def main():
    """Main training function"""
    args = parse_args()

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        # Try absolute path
        config_path = os.path.join('/opt/ml/code', args.config)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override with command line arguments
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.embedding_dim:
        config['model']['embedding_dim'] = args.embedding_dim

    print("Configuration:")
    print(json.dumps(config, indent=2))

    # Set random seed
    torch.manual_seed(config['reproducibility']['seed'])

    # Get SageMaker training environment
    sm_training_env = json.loads(os.environ.get('SM_TRAINING_ENV', '{}'))

    # Create trainer
    trainer = SageMakerTrainer(config, sm_training_env)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_root=trainer.data_dir,
        metadata_csv=os.path.join(trainer.data_dir, 'metadata.csv'),
        interactions_csv=config['paths'].get('interactions'),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        dataset_type='online_triplet'
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
