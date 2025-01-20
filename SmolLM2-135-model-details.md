# SmolLM2

## Model Summary
SmolLM2 is a family of compact language models available in three sizes: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device.

SmolLM2 demonstrates significant advances over its predecessor SmolLM1, particularly in:
- Instruction following
- Knowledge
- Reasoning

The 135M model was trained on 2 trillion tokens using a diverse dataset combination:
- FineWeb-Edu
- DCLM
- The Stack
- New filtered datasets (to be released soon)

We developed the instruct version through supervised fine-tuning (SFT) using a combination of public datasets and our own curated datasets. We then applied Direct Preference Optimization (DPO) using UltraFeedback.

The instruct model additionally supports tasks such as:
- Text rewriting
- Summarization
- Function calling (for the 1.7B model)

You can find the:
- SFT dataset: [HuggingFace Dataset](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
- Finetuning code: [GitHub Repository](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2)

## How to Use

### Installation
```bash
pip install transformers
```

### Running the Model on CPU/GPU/Multi GPU

#### Using Full Precision
```python
pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "HuggingFaceTB/SmolLM2-135M"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
for multiple GPUs install accelerate and do:
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
inputs = tokenizer.encode("Gravity is", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

#### Using BFloat16
```python
pip install accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
checkpoint = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
for fp16 use torch_dtype=torch.float16 instead
model = AutoModelForCausalLM.from_pretrained(
checkpoint,
device_map="auto",
torch_dtype=torch.bfloat16
)
inputs = tokenizer.encode("Gravity is", return_tensors="pt").to("cpu")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```





