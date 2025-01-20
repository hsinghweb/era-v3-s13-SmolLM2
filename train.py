import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from model import create_model, MODEL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        learning_rate: float = 3e-3,
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 2000,  # Save every 2000 steps
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.device = device
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Move model to device
        self.model.to(device)

    def save_checkpoint(self, is_best: bool = False):
        """Save a checkpoint of the training state"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': MODEL_CONFIG,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save latest checkpoint (for easy resume)
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint with validation loss: {self.best_val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path: str = None):
        """Load a training checkpoint"""
        if checkpoint_path is None:
            # Try to load latest checkpoint
            checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
            if not checkpoint_path.exists():
                logger.info("No checkpoint found. Starting training from scratch.")
                return False
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Verify model configuration matches
        if checkpoint['config'] != MODEL_CONFIG:
            raise ValueError("Checkpoint model configuration does not match current model configuration")
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Resumed training from epoch {self.current_epoch}, step {self.global_step}")
        return True

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            loss = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Log progress
            if self.global_step % 100 == 0:
                logger.info(f"Step {self.global_step}: loss = {loss.item():.4f}")
            
            # Save checkpoint
            if self.global_step % self.checkpoint_interval == 0:
                self.save_checkpoint()
                
                # Evaluate if we have a validation dataloader
                if self.val_dataloader is not None:
                    val_loss = self.evaluate()
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
        
        return total_loss / len(self.train_dataloader)

    def evaluate(self):
        """Evaluate the model on the validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        logger.info(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, num_epochs: int):
        """Train the model for the specified number of epochs"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            train_loss = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} completed. Average training loss: {train_loss:.4f}")
            
            # Save checkpoint at the end of each epoch
            self.save_checkpoint()

def main():
    # Create model
    model = create_model(seed=42)
    
    # Create your dataloaders here
    train_dataloader = None  # Replace with your actual train dataloader
    val_dataloader = None    # Replace with your actual validation dataloader
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=3e-3,
        checkpoint_dir="checkpoints",
        checkpoint_interval=2000
    )
    
    # Try to resume from checkpoint
    trainer.load_checkpoint()
    
    # Start training
    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main() 