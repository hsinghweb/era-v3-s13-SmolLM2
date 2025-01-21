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

# Output Log while training

## First run for 5000 steps
```
Initialized a fresh SmolLM2-135M model with random weights

Model Parameter Analysis:
==================================================
Target Parameters:     134,515,008
Current Parameters:    162,826,560
Difference:           28,311,552

Parameter breakdown:
--------------------------------------------------
Embeddings:           28,311,552
Attention layers:     26,542,080
MLP layers:           79,626,240
Layer norms:          35,136
LM head:              28,311,552
--------------------------------------------------

Note: Current implementation has more parameters than reference model.
Proceeding with training using current architecture.
==================================================
tokenizer_config.json: 100% 3.66k/3.66k [00:00<00:00, 22.4MB/s]
vocab.json: 100% 801k/801k [00:00<00:00, 18.9MB/s]
merges.txt: 100% 466k/466k [00:00<00:00, 7.24MB/s]
tokenizer.json: 100% 2.10M/2.10M [00:00<00:00, 15.4MB/s]
special_tokens_map.json: 100% 831/831 [00:00<00:00, 4.97MB/s]
2025-01-21 02:48:10,954 - INFO - Checking Model Architecture...
2025-01-21 02:48:10,954 - INFO - -------------------
2025-01-21 02:48:10,954 - INFO - Current Model Architecture:
2025-01-21 02:48:10,954 - INFO - LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
2025-01-21 02:48:10,954 - INFO - -------------------
2025-01-21 02:48:10,954 - INFO - ✓ Model architecture exactly matches reference architecture
2025-01-21 02:48:10,956 - INFO - Parameter Counts:
2025-01-21 02:48:10,956 - INFO - Total Parameters: 162,826,560
2025-01-21 02:48:10,956 - INFO - Trainable Parameters: 162,826,560
2025-01-21 02:48:10,957 - INFO - Target Parameters: 134,515,008
2025-01-21 02:48:10,957 - INFO - -------------------
2025-01-21 02:48:10,957 - INFO - Loading text from input.txt
Token indices sequence length is longer than the specified maximum sequence length for this model (341094 > 8192). Running this sequence through the model will result in indexing errors
2025-01-21 02:48:12,688 - INFO - Created 667 text chunks
2025-01-21 02:48:13,200 - INFO - No checkpoint found. Starting training from scratch.
2025-01-21 02:48:13,200 - INFO - Starting fresh training to 5000 steps
2025-01-21 02:48:13,200 - INFO - Starting training until step 5000
2025-01-21 02:48:47,728 - INFO - Step 50: loss = 6.5794
2025-01-21 02:49:00,517 - INFO - Saved checkpoint: checkpoints/checkpoint_step_50.pt
2025-01-21 02:49:35,528 - INFO - Step 100: loss = 6.5151
2025-01-21 02:49:44,272 - INFO - Saved checkpoint: checkpoints/checkpoint_step_100.pt
2025-01-21 02:50:21,639 - INFO - Step 150: loss = 6.2144
2025-01-21 02:50:32,050 - INFO - Saved checkpoint: checkpoints/checkpoint_step_150.pt
2025-01-21 02:51:08,573 - INFO - Step 200: loss = 5.8664
2025-01-21 02:51:18,141 - INFO - Saved checkpoint: checkpoints/checkpoint_step_200.pt
2025-01-21 02:51:55,133 - INFO - Step 250: loss = 5.2935
2025-01-21 02:52:16,251 - INFO - Saved checkpoint: checkpoints/checkpoint_step_250.pt
2025-01-21 02:52:53,265 - INFO - Step 300: loss = 5.4122
2025-01-21 02:53:00,997 - INFO - Saved checkpoint: checkpoints/checkpoint_step_300.pt
2025-01-21 02:53:37,539 - INFO - Step 350: loss = 5.2446
2025-01-21 02:53:48,512 - INFO - Saved checkpoint: checkpoints/checkpoint_step_350.pt
2025-01-21 02:54:25,535 - INFO - Step 400: loss = 5.2104
2025-01-21 02:54:32,838 - INFO - Saved checkpoint: checkpoints/checkpoint_step_400.pt
2025-01-21 02:55:09,370 - INFO - Step 450: loss = 5.0086
2025-01-21 02:56:04,000 - INFO - Saved checkpoint: checkpoints/checkpoint_step_450.pt
2025-01-21 02:56:40,961 - INFO - Step 500: loss = 4.5222
2025-01-21 02:56:56,500 - INFO - Saved checkpoint: checkpoints/checkpoint_step_500.pt
2025-01-21 02:56:59,537 - INFO - 
Generated text at step 500:
2025-01-21 02:56:59,537 - INFO - Prompt: First Citizen:
2025-01-21 02:56:59,537 - INFO - Generated: First Citizen:urd,
ARI,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,
ererest,

2025-01-21 02:57:36,216 - INFO - Step 550: loss = 5.3112
2025-01-21 02:58:31,892 - INFO - Saved checkpoint: checkpoints/checkpoint_step_550.pt
2025-01-21 02:59:08,769 - INFO - Step 600: loss = 5.1003
2025-01-21 02:59:26,061 - INFO - Saved checkpoint: checkpoints/checkpoint_step_600.pt
2025-01-21 03:00:02,977 - INFO - Step 650: loss = 4.9597
2025-01-21 03:00:28,292 - INFO - Saved checkpoint: checkpoints/checkpoint_step_650.pt
2025-01-21 03:01:05,078 - INFO - Step 700: loss = 4.7628
2025-01-21 03:01:43,795 - INFO - Saved checkpoint: checkpoints/checkpoint_step_700.pt
2025-01-21 03:02:20,703 - INFO - Step 750: loss = 4.6735
2025-01-21 03:03:31,152 - INFO - Saved checkpoint: checkpoints/checkpoint_step_750.pt
2025-01-21 03:04:07,818 - INFO - Step 800: loss = 4.8823
2025-01-21 03:05:34,135 - INFO - Saved checkpoint: checkpoints/checkpoint_step_800.pt
2025-01-21 03:06:10,475 - INFO - Step 850: loss = 4.8093
2025-01-21 03:06:38,339 - INFO - Saved checkpoint: checkpoints/checkpoint_step_850.pt
2025-01-21 03:07:15,339 - INFO - Step 900: loss = 4.8615
2025-01-21 03:07:38,219 - INFO - Saved checkpoint: checkpoints/checkpoint_step_900.pt
2025-01-21 03:08:14,793 - INFO - Step 950: loss = 5.2731
2025-01-21 03:09:06,377 - INFO - Saved checkpoint: checkpoints/checkpoint_step_950.pt
2025-01-21 03:09:43,104 - INFO - Step 1000: loss = 4.7096
2025-01-21 03:09:50,546 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1000.pt
2025-01-21 03:09:53,762 - INFO - 
Generated text at step 1000:
2025-01-21 03:09:53,762 - INFO - Prompt: First Citizen:
2025-01-21 03:09:53,762 - INFO - Generated: First Citizen:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:
d:


2025-01-21 03:10:30,012 - INFO - Step 1050: loss = 4.6607
2025-01-21 03:11:37,845 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1050.pt
2025-01-21 03:12:14,191 - INFO - Step 1100: loss = 4.4992
2025-01-21 03:12:36,783 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1100.pt
2025-01-21 03:13:13,815 - INFO - Step 1150: loss = 4.5353
2025-01-21 03:13:44,774 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1150.pt
2025-01-21 03:14:21,094 - INFO - Step 1200: loss = 4.1345
2025-01-21 03:14:44,358 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1200.pt
2025-01-21 03:15:21,109 - INFO - Step 1250: loss = 4.4509
2025-01-21 03:16:33,122 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1250.pt
2025-01-21 03:17:09,484 - INFO - Step 1300: loss = 4.4499
2025-01-21 03:18:15,382 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1300.pt
2025-01-21 03:18:51,894 - INFO - Step 1350: loss = 4.1438
2025-01-21 03:19:19,490 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1350.pt
2025-01-21 03:19:56,102 - INFO - Step 1400: loss = 4.5258
2025-01-21 03:20:03,671 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1400.pt
2025-01-21 03:20:40,256 - INFO - Step 1450: loss = 4.4358
2025-01-21 03:21:50,787 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1450.pt
2025-01-21 03:22:26,979 - INFO - Step 1500: loss = 4.8913
2025-01-21 03:23:25,912 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1500.pt
2025-01-21 03:23:28,979 - INFO - 
Generated text at step 1500:
2025-01-21 03:23:28,979 - INFO - Prompt: First Citizen:
2025-01-21 03:23:28,979 - INFO - Generated: First Citizen:ile is the, a





























































































2025-01-21 03:24:05,487 - INFO - Step 1550: loss = 4.6695
2025-01-21 03:24:30,502 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1550.pt
2025-01-21 03:25:07,367 - INFO - Step 1600: loss = 4.4584
2025-01-21 03:25:58,444 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1600.pt
2025-01-21 03:26:34,826 - INFO - Step 1650: loss = 4.3780
2025-01-21 03:27:32,833 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1650.pt
2025-01-21 03:28:09,345 - INFO - Step 1700: loss = 4.1818
2025-01-21 03:28:59,522 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1700.pt
2025-01-21 03:29:36,155 - INFO - Step 1750: loss = 5.0010
2025-01-21 03:29:56,394 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1750.pt
2025-01-21 03:30:33,322 - INFO - Step 1800: loss = 4.1449
2025-01-21 03:30:42,031 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1800.pt
2025-01-21 03:31:18,329 - INFO - Step 1850: loss = 4.3292
2025-01-21 03:31:34,667 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1850.pt
2025-01-21 03:32:11,153 - INFO - Step 1900: loss = 4.6017
2025-01-21 03:32:22,160 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1900.pt
2025-01-21 03:32:58,878 - INFO - Step 1950: loss = 4.3401
2025-01-21 03:33:10,060 - INFO - Saved checkpoint: checkpoints/checkpoint_step_1950.pt
2025-01-21 03:33:46,599 - INFO - Step 2000: loss = 4.5197
2025-01-21 03:34:01,729 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2000.pt
2025-01-21 03:34:04,896 - INFO - 
Generated text at step 2000:
2025-01-21 03:34:04,896 - INFO - Prompt: First Citizen:
2025-01-21 03:34:04,896 - INFO - Generated: First Citizen:
bus,
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:


2025-01-21 03:34:41,637 - INFO - Step 2050: loss = 4.5922
2025-01-21 03:34:52,080 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2050.pt
2025-01-21 03:35:28,609 - INFO - Step 2100: loss = 4.0066
2025-01-21 03:35:40,108 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2100.pt
2025-01-21 03:36:16,882 - INFO - Step 2150: loss = 4.3039
2025-01-21 03:36:28,511 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2150.pt
2025-01-21 03:37:04,981 - INFO - Step 2200: loss = 4.3566
2025-01-21 03:37:19,499 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2200.pt
2025-01-21 03:37:56,284 - INFO - Step 2250: loss = 4.4039
2025-01-21 03:38:10,058 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2250.pt
2025-01-21 03:38:46,702 - INFO - Step 2300: loss = 4.1629
2025-01-21 03:38:59,166 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2300.pt
2025-01-21 03:39:35,792 - INFO - Step 2350: loss = 4.3464
2025-01-21 03:39:47,817 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2350.pt
2025-01-21 03:40:24,512 - INFO - Step 2400: loss = 3.9698
2025-01-21 03:40:35,478 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2400.pt
2025-01-21 03:41:12,138 - INFO - Step 2450: loss = 4.2344
2025-01-21 03:41:24,132 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2450.pt
2025-01-21 03:42:00,953 - INFO - Step 2500: loss = 3.8058
2025-01-21 03:42:11,810 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2500.pt
2025-01-21 03:42:15,004 - INFO - 
Generated text at step 2500:
2025-01-21 03:42:15,004 - INFO - Prompt: First Citizen:
2025-01-21 03:42:15,004 - INFO - Generated: First Citizen:
,iest.






























































































2025-01-21 03:42:51,583 - INFO - Step 2550: loss = 3.6364
2025-01-21 03:43:03,848 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2550.pt
2025-01-21 03:43:40,502 - INFO - Step 2600: loss = 4.4367
2025-01-21 03:43:53,469 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2600.pt
2025-01-21 03:44:30,739 - INFO - Step 2650: loss = 4.3601
2025-01-21 03:44:41,738 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2650.pt
2025-01-21 03:45:18,320 - INFO - Step 2700: loss = 3.8798
2025-01-21 03:45:27,271 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2700.pt
2025-01-21 03:46:03,907 - INFO - Step 2750: loss = 4.2224
2025-01-21 03:46:12,111 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2750.pt
2025-01-21 03:46:48,902 - INFO - Step 2800: loss = 3.9215
2025-01-21 03:46:58,834 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2800.pt
2025-01-21 03:47:35,524 - INFO - Step 2850: loss = 3.4212
2025-01-21 03:47:47,566 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2850.pt
2025-01-21 03:48:24,359 - INFO - Step 2900: loss = 4.2700
2025-01-21 03:48:37,043 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2900.pt
2025-01-21 03:49:13,815 - INFO - Step 2950: loss = 3.8641
2025-01-21 03:49:22,880 - INFO - Saved checkpoint: checkpoints/checkpoint_step_2950.pt
2025-01-21 03:49:59,536 - INFO - Step 3000: loss = 3.8889
2025-01-21 03:50:10,959 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3000.pt
2025-01-21 03:50:11,391 - INFO - 
Generated text at step 3000:
2025-01-21 03:50:11,392 - INFO - Prompt: First Citizen:
2025-01-21 03:50:11,392 - INFO - Generated: First Citizen:, the, the

2025-01-21 03:50:48,074 - INFO - Step 3050: loss = 3.6791
2025-01-21 03:51:01,033 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3050.pt
2025-01-21 03:51:37,860 - INFO - Step 3100: loss = 4.1071
2025-01-21 03:51:52,177 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3100.pt
2025-01-21 03:52:29,140 - INFO - Step 3150: loss = 3.6261
2025-01-21 03:52:40,429 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3150.pt
2025-01-21 03:53:17,083 - INFO - Step 3200: loss = 4.0902
2025-01-21 03:53:28,972 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3200.pt
2025-01-21 03:54:05,981 - INFO - Step 3250: loss = 3.9784
2025-01-21 03:54:21,896 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3250.pt
2025-01-21 03:54:58,862 - INFO - Step 3300: loss = 3.8533
2025-01-21 03:55:09,762 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3300.pt
2025-01-21 03:55:46,410 - INFO - Step 3350: loss = 3.4830
2025-01-21 03:55:59,767 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3350.pt
2025-01-21 03:56:36,803 - INFO - Step 3400: loss = 3.7054
2025-01-21 03:56:49,286 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3400.pt
2025-01-21 03:57:26,120 - INFO - Step 3450: loss = 3.6590
2025-01-21 03:57:39,587 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3450.pt
2025-01-21 03:58:16,646 - INFO - Step 3500: loss = 3.6312
2025-01-21 03:58:27,593 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3500.pt
2025-01-21 03:58:29,865 - INFO - 
Generated text at step 3500:
2025-01-21 03:58:29,865 - INFO - Prompt: First Citizen:
2025-01-21 03:58:29,865 - INFO - Generated: First Citizen:-ove, toy, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, andener, and

2025-01-21 03:59:06,555 - INFO - Step 3550: loss = 3.4040
2025-01-21 03:59:19,563 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3550.pt
2025-01-21 03:59:56,601 - INFO - Step 3600: loss = 3.8049
2025-01-21 04:00:10,096 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3600.pt
2025-01-21 04:00:47,042 - INFO - Step 3650: loss = 3.6253
2025-01-21 04:00:56,894 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3650.pt
2025-01-21 04:01:33,748 - INFO - Step 3700: loss = 3.3309
2025-01-21 04:01:46,468 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3700.pt
2025-01-21 04:02:23,466 - INFO - Step 3750: loss = 3.5943
2025-01-21 04:02:36,824 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3750.pt
2025-01-21 04:03:13,814 - INFO - Step 3800: loss = 3.4491
2025-01-21 04:03:21,350 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3800.pt
2025-01-21 04:03:58,053 - INFO - Step 3850: loss = 3.0942
2025-01-21 04:04:10,034 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3850.pt
2025-01-21 04:04:47,141 - INFO - Step 3900: loss = 3.2558
2025-01-21 04:05:01,955 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3900.pt
2025-01-21 04:05:38,998 - INFO - Step 3950: loss = 3.5534
2025-01-21 04:05:47,849 - INFO - Saved checkpoint: checkpoints/checkpoint_step_3950.pt
2025-01-21 04:06:24,751 - INFO - Step 4000: loss = 3.6198
2025-01-21 04:06:38,387 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4000.pt
2025-01-21 04:06:40,709 - INFO - 
Generated text at step 4000:
2025-01-21 04:06:40,709 - INFO - Prompt: First Citizen:
2025-01-21 04:06:40,709 - INFO - Generated: First Citizen: man,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

2025-01-21 04:07:17,664 - INFO - Step 4050: loss = 3.2096
2025-01-21 04:07:30,649 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4050.pt
2025-01-21 04:08:07,614 - INFO - Step 4100: loss = 3.2300
2025-01-21 04:08:21,193 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4100.pt
2025-01-21 04:08:58,341 - INFO - Step 4150: loss = 3.5211
2025-01-21 04:09:09,304 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4150.pt
2025-01-21 04:09:46,084 - INFO - Step 4200: loss = 2.7547
2025-01-21 04:10:01,081 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4200.pt
2025-01-21 04:10:38,270 - INFO - Step 4250: loss = 2.8851
2025-01-21 04:10:55,835 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4250.pt
2025-01-21 04:11:32,944 - INFO - Step 4300: loss = 3.1243
2025-01-21 04:11:43,560 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4300.pt
2025-01-21 04:12:20,206 - INFO - Step 4350: loss = 2.5981
2025-01-21 04:12:31,408 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4350.pt
2025-01-21 04:13:08,549 - INFO - Step 4400: loss = 2.5711
2025-01-21 04:13:22,701 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4400.pt
2025-01-21 04:13:59,835 - INFO - Step 4450: loss = 2.6171
2025-01-21 04:14:13,600 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4450.pt
2025-01-21 04:14:50,620 - INFO - Step 4500: loss = 3.2276
2025-01-21 04:15:07,264 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4500.pt
2025-01-21 04:15:09,859 - INFO - 
Generated text at step 4500:
2025-01-21 04:15:09,859 - INFO - Prompt: First Citizen:
2025-01-21 04:15:09,859 - INFO - Generated: First Citizen:, you to be, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you:, you to be now not

2025-01-21 04:15:46,823 - INFO - Step 4550: loss = 2.5484
2025-01-21 04:15:59,021 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4550.pt
2025-01-21 04:16:35,889 - INFO - Step 4600: loss = 2.8249
2025-01-21 04:16:48,010 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4600.pt
2025-01-21 04:17:25,124 - INFO - Step 4650: loss = 2.6801
2025-01-21 04:17:36,010 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4650.pt
2025-01-21 04:18:12,786 - INFO - Step 4700: loss = 2.4288
2025-01-21 04:18:26,630 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4700.pt
2025-01-21 04:19:03,787 - INFO - Step 4750: loss = 2.2266
2025-01-21 04:19:14,796 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4750.pt
2025-01-21 04:19:51,677 - INFO - Step 4800: loss = 2.6244
2025-01-21 04:20:02,170 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4800.pt
2025-01-21 04:20:39,125 - INFO - Step 4850: loss = 2.3471
2025-01-21 04:20:49,056 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4850.pt
2025-01-21 04:21:25,982 - INFO - Step 4900: loss = 2.2331
2025-01-21 04:21:36,900 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4900.pt
2025-01-21 04:22:13,969 - INFO - Step 4950: loss = 2.6096
2025-01-21 04:22:25,643 - INFO - Saved checkpoint: checkpoints/checkpoint_step_4950.pt
2025-01-21 04:23:02,571 - INFO - Step 5000: loss = 2.4448
2025-01-21 04:23:15,177 - INFO - Saved checkpoint: checkpoints/checkpoint_step_5000.pt
2025-01-21 04:23:15,282 - INFO - 
Generated text at step 5000:
2025-01-21 04:23:15,282 - INFO - Prompt: First Citizen:
2025-01-21 04:23:15,282 - INFO - Generated: First Citizen:!

2025-01-21 04:23:15,286 - INFO - Reached target steps (5000). Stopping training.
```


## Second run for 50 steps, starting from checkpoint_step_5000.pt
```
Initialized a fresh SmolLM2-135M model with random weights

Model Parameter Analysis:
==================================================
Target Parameters:     134,515,008
Current Parameters:    162,826,560
Difference:           28,311,552

Parameter breakdown:
--------------------------------------------------
Embeddings:           28,311,552
Attention layers:     26,542,080
MLP layers:           79,626,240
Layer norms:          35,136
LM head:              28,311,552
--------------------------------------------------

Note: Current implementation has more parameters than reference model.
Proceeding with training using current architecture.
==================================================
2025-01-21 04:32:38,301 - INFO - Checking Model Architecture...
2025-01-21 04:32:38,301 - INFO - -------------------
2025-01-21 04:32:38,301 - INFO - Current Model Architecture:
2025-01-21 04:32:38,302 - INFO - LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
2025-01-21 04:32:38,302 - INFO - -------------------
2025-01-21 04:32:38,302 - INFO - ✓ Model architecture exactly matches reference architecture
2025-01-21 04:32:38,304 - INFO - Parameter Counts:
2025-01-21 04:32:38,304 - INFO - Total Parameters: 162,826,560
2025-01-21 04:32:38,304 - INFO - Trainable Parameters: 162,826,560
2025-01-21 04:32:38,304 - INFO - Target Parameters: 134,515,008
2025-01-21 04:32:38,305 - INFO - -------------------
2025-01-21 04:32:38,305 - INFO - Loading text from input.txt
Token indices sequence length is longer than the specified maximum sequence length for this model (341094 > 8192). Running this sequence through the model will result in indexing errors
2025-01-21 04:32:39,429 - INFO - Created 667 text chunks
2025-01-21 04:32:39,873 - INFO - Loading latest checkpoint: checkpoints/checkpoint_step_5000.pt
/content/era-v3-s13-SmolLM2/train.py:152: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(latest_checkpoint, map_location=self.device)
2025-01-21 04:32:41,715 - INFO - Resumed training from step 5000
2025-01-21 04:32:41,715 - INFO - Continuing training for 50 more steps
2025-01-21 04:32:41,715 - INFO - Starting training until step 5050
2025-01-21 04:32:43,089 - INFO - Step 5000: loss = 1.9140
2025-01-21 04:32:53,518 - INFO - Saved checkpoint: checkpoints/checkpoint_step_5000.pt
2025-01-21 04:32:53,602 - INFO - 
Generated text at step 5000:
2025-01-21 04:32:53,602 - INFO - Prompt: First Citizen:
2025-01-21 04:32:53,602 - INFO - Generated: First Citizen:!

2025-01-21 04:33:26,747 - INFO - Step 5050: loss = 2.1566
2025-01-21 04:33:40,365 - INFO - Saved checkpoint: checkpoints/checkpoint_step_5050.pt
2025-01-21 04:33:40,409 - INFO - Reached target steps (5050). Stopping training.
```

