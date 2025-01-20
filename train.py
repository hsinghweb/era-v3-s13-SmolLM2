import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
import glob
from model import create_model, MODEL_CONFIG
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading text from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        # Split text into chunks of max_length tokens
        tokens = self.tokenizer.encode(self.text)
        self.chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        logger.info(f"Created {len(self.chunks)} text chunks")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Ensure the chunk is exactly max_length by padding
        if len(chunk) < self.max_length:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))
        
        # Convert to tensor
        input_ids = torch.tensor(chunk)
        labels = input_ids.clone()
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

class ModelTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        input_file: str,
        batch_size: int = 4,
        max_length: int = 512,
        learning_rate: float = 3e-3,
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 100,
        max_checkpoints: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset and dataloader
        dataset = TextDataset(input_file, tokenizer, max_length)
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
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
        
        # Move model to device
        self.model.to(device)

    def cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints"""
        checkpoint_files = sorted(
            glob.glob(str(self.checkpoint_dir / 'checkpoint_step_*.pt')),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        
        # Remove older checkpoints if we have more than max_checkpoints
        while len(checkpoint_files) > self.max_checkpoints:
            oldest_checkpoint = checkpoint_files.pop(0)
            try:
                os.remove(oldest_checkpoint)
                logger.info(f"Removed old checkpoint: {oldest_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {oldest_checkpoint}: {e}")

    def save_checkpoint(self):
        """Save a checkpoint of the training state"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': MODEL_CONFIG,
        }
        
        # Save checkpoint with step number
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()

    def load_latest_checkpoint(self):
        """Load the latest available checkpoint"""
        checkpoint_files = sorted(
            glob.glob(str(self.checkpoint_dir / 'checkpoint_step_*.pt')),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        
        if not checkpoint_files:
            logger.info("No checkpoint found. Starting training from scratch.")
            return False
            
        latest_checkpoint = checkpoint_files[-1]
        logger.info(f"Loading latest checkpoint: {latest_checkpoint}")
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            
            # Verify model configuration matches
            if checkpoint['config'] != MODEL_CONFIG:
                raise ValueError("Checkpoint model configuration does not match current model configuration")
            
            # Load model and optimizer states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training state
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            
            logger.info(f"Resumed training from epoch {self.current_epoch}, step {self.global_step}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        steps_this_epoch = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
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
            steps_this_epoch += 1
            
            # Log progress
            if self.global_step % 100 == 0:
                logger.info(f"Step {self.global_step} (epoch step {steps_this_epoch}): loss = {loss.item():.4f}")
                self.save_checkpoint()
        
        logger.info(f"Epoch completed with {steps_this_epoch} steps")
        return total_loss / steps_this_epoch

    def train(self, num_epochs: int):
        """Train the model for the specified number of epochs"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            train_loss = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} completed. Average training loss: {train_loss:.4f}")

def main():
    # Create model and tokenizer
    model = create_model(seed=42)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        input_file="input.txt",  # Your input text file
        batch_size=4,
        max_length=512,
        learning_rate=3e-3,
        checkpoint_dir="checkpoints",
        checkpoint_interval=100,
        max_checkpoints=3
    )
    
    # Try to resume from latest checkpoint
    trainer.load_latest_checkpoint()
    
    # Start training
    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main() 