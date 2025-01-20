import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_and_load_model():
    """
    Downloads and loads the SmolLM2-135M model, then prints its architecture
    """
    try:
        # Model checkpoint
        checkpoint = "HuggingFaceTB/SmolLM2-135M"
        
        logger.info(f"Downloading model from {checkpoint}...")
        
        # Download and load the model
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float32,  # Full precision
            trust_remote_code=True
        )
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        logger.info("Model and tokenizer downloaded successfully!")
        
        # Print model architecture
        logger.info("\nModel Architecture:")
        logger.info("=" * 50)
        logger.info(model)
        logger.info("=" * 50)
        
        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"\nTotal parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model, tokenizer = download_and_load_model()
        logger.info("Script completed successfully!")
    except Exception as e:
        logger.error("Script failed to complete.")
        raise
