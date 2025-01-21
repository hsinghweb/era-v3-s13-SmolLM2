# SmolLM2-135M Implementation

This repository contains an implementation of SmolLM2-135M, a lightweight language model based on the Llama architecture.

## Model Architecture

SmolLM2-135M architecture features:
- 30 transformer layers
- 576 hidden size
- 9 attention heads
- 3 key/value heads (grouped-query attention)
- 1536 intermediate size
- 49152 vocabulary size
- SiLU activation
- RMSNorm with eps=1e-5

Note: Our implementation maintains the exact architecture but results in ~162.8M parameters due to differences in attention mechanism implementation.

## Features

- Full implementation of Llama architecture with grouped-query attention
- Detailed parameter analysis and breakdown
- Checkpoint saving and loading
- Text generation capabilities
- Training progress monitoring
- Sample text generation during training

## Files Structure

- `model.py`: Core model implementation including LlamaForCausalLM
- `train.py`: Training loop and utilities
- `input.txt`: Training data (Shakespeare text)
- `config_smollm2_135M.yaml`: Model configuration
- `model_architecture_smollm2_135M.txt`: Reference model architecture

## Training

### Requirements
```bash
pip install transformers torch
```

### Training Process
```bash
python train.py
```

The training:
1. Prints detailed model architecture and parameter analysis
2. Runs for 5000 steps
3. Saves checkpoints every 50 steps
4. Generates sample text every 500 steps
5. Keeps last 3 checkpoints
6. Can continue training from checkpoints

### Training Parameters
- Batch size: 4
- Learning rate: 3e-3
- Max sequence length: 512
- Optimizer: AdamW (β1=0.9, β2=0.95)

## Model Features

1. **Grouped-Query Attention**
   - 9 query heads
   - 3 key/value heads
   - Efficient attention computation

2. **Architecture Components**
   - RMSNorm for layer normalization
   - SiLU activation function
   - Rotary positional embeddings

3. **Generation Capabilities**
   - Temperature-controlled text generation
   - Greedy decoding
   - Configurable max length

## Checkpointing

Checkpoints are saved:
- Every 50 training steps
- In the `checkpoints` directory
- With format `checkpoint_step_{step}.pt`
- Keeping only last 3 checkpoints

## Usage Example

```python
from model import create_model
from transformers import AutoTokenizer

# Create model and see parameter analysis
model = create_model(seed=42)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Initialize trainer
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

# Start training
trainer.train(target_steps=5000)
```

## Training Continuation

To continue training from a checkpoint:
```python
trainer.train(target_steps=5050)  # For 50 more steps
```

## Model Output Example

The model can generate text continuations:
```python
prompt = "First Citizen:"
generated_text = model.generate(
    tokenizer.encode(prompt, return_tensors="pt"),
    max_length=100,
    temperature=0.7
)
```

## License

This project is open-source and available under the MIT License.

## Acknowledgments

Based on the Llama architecture and SmolLM2 model design.
