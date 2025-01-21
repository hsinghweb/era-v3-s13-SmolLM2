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
        checkpoint_interval: int = 50,
        max_checkpoints: int = 3,
        target_steps: int = 5000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.target_steps = target_steps
        
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
        
        # Training state - start from step 1
        self.global_step = 1  # Changed from 0 to 1
        
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
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {oldest_checkpoint}: {e}")

    def save_checkpoint(self):
        """Save a checkpoint of the training state"""
        checkpoint = {
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
            self.global_step = checkpoint['global_step']
            
            logger.info(f"Resumed training from step {self.global_step}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False

    def generate_sample_text(self, prompt="First Citizen:", max_length=100):
        """Generate sample text to show model's current capabilities"""
        self.model.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"\nGenerated text at step {self.global_step}:")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated_text}\n")
        self.model.train()

    def train_epoch(self, target_steps=None):
        """Train until target steps"""
        self.model.train()
        total_loss = 0
        steps_this_batch = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Check if we've reached target steps
            if target_steps and self.global_step > target_steps:
                break
            
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
            
            # Log progress and save checkpoint
            if self.global_step % 50 == 0:
                logger.info(f"Step {self.global_step}: loss = {loss.item():.4f}")
                self.save_checkpoint()
            
            if self.global_step % 500 == 0:
                self.generate_sample_text()
            
            # Increment step
            self.global_step += 1
            steps_this_batch += 1
        
        return total_loss / steps_this_batch if steps_this_batch > 0 else 0

    def train(self, target_steps: int = None):
        """Train the model until target steps"""
        logger.info(f"Starting training until step {target_steps}")
        
        while True:
            # Check if we've reached target steps
            if target_steps and self.global_step > target_steps:
                logger.info(f"Reached target steps ({target_steps}). Stopping training.")
                break
            
            train_loss = self.train_epoch(target_steps)

def main():
    # Load reference architecture
    with open('model_architecture_smollm2_135M.txt', 'r') as f:
        reference_architecture = f.read().strip()
    
    # Create model and tokenizer
    model = create_model(seed=42)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    # Get current model architecture as string
    current_architecture = str(model).strip()
    
    # Print and compare architectures
    logger.info("Checking Model Architecture...")
    logger.info("-------------------")
    logger.info("Current Model Architecture:")
    logger.info(current_architecture)
    logger.info("-------------------")
    
    # Compare architectures
    if current_architecture == reference_architecture:
        logger.info("✓ Model architecture exactly matches reference architecture")
    else:
        logger.error("✗ Model architecture differs from reference!")
        # Find differences
        current_lines = current_architecture.split('\n')
        reference_lines = reference_architecture.split('\n')
        
        for i, (curr, ref) in enumerate(zip(current_lines, reference_lines)):
            if curr != ref:
                logger.error(f"Line {i+1} differs:")
                logger.error(f"Expected: {ref}")
                logger.error(f"Got:      {curr}")
        
        raise ValueError("Model architecture does not match reference architecture")
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target_params = 134_515_008  # From model_architecture_smollm2_135M.txt
    
    logger.info("Parameter Counts:")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Target Parameters: {target_params:,}")
    
    if total_params != target_params:
        raise ValueError(f"Parameter count mismatch! Expected {target_params:,} but got {total_params:,}")
    
    logger.info("✓ Parameter count matches reference")
    logger.info("-------------------")
    
    # Continue with training setup...
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        input_file="input.txt",
        batch_size=4,
        max_length=512,
        learning_rate=3e-3,
        checkpoint_dir="checkpoints",
        checkpoint_interval=50,
        max_checkpoints=3,
        target_steps=5000
    )
    
    # Check if we're continuing training
    if trainer.load_latest_checkpoint():
        if trainer.global_step > 5000:
            logger.info("Continuing training for 50 more steps")
            trainer.train(target_steps=5050)
        else:
            logger.info("Continuing training until step 5000")
            trainer.train(target_steps=5000)
    else:
        logger.info("Starting fresh training to 5000 steps")
        trainer.train(target_steps=5000)

if __name__ == "__main__":
    main() 