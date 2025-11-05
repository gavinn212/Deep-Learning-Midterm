## Checkpoint Link
https://drive.google.com/file/d/1t5UfR19FVlRpREVHY3IuC5KcW6RNi2AX/view?usp=sharing
## Project Overview

This repository contains a PEFT/LoRA adapter checkpoint and tokenizer configuration for causal language modeling (text generation), built on the base model `unsloth/meta-llama-3.1-8b-unsloth-bnb-4bit`. It is suitable for running inference or continuing fine-tuning on a 4-bit quantized Llama 3.1 8B base.

### Directory Structure

- `adapter_config.json`: LoRA adapter configuration (r=32, alpha=64, dropout=0), targeting `q_proj/k_proj/v_proj/o_proj` and `gate_proj/up_proj/down_proj`.
- `tokenizer.json`: Tokenizer model and special token definitions (`PreTrainedTokenizerFast`).
- `tokenizer_config.json`: Tokenizer loading settings and the list of added special tokens.
- `special_tokens_map.json`: Special token mapping for BOS/EOS/PAD.
- `README.md`: This document.

## Base Model & Adapter

- Base model (fetched online): `unsloth/meta-llama-3.1-8b-unsloth-bnb-4bit`
- This folder: LoRA adapter configuration and tokenizer files (if adapter weights such as `.safetensors` are present here, you can load them directly; otherwise, place them in this directory or pass the path explicitly).

## Environment & Installation

Recommended core dependencies:

- `transformers`
- `peft`
- `accelerate`
- `bitsandbytes` (for 4-bit quantized weights)

Example (for reference):

```bash
pip install -U transformers peft accelerate bitsandbytes
```

## Quickstart (Inference)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "unsloth/meta-llama-3.1-8b-unsloth-bnb-4bit"
adapter_path = "."  # path to this repository or where adapter weights are stored

# Prefer the tokenizer in this folder (includes special tokens and right-side padding settings)
tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

prompt = "Briefly explain the Pythagorean theorem."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Tokenizer Notes

- Max sequence length: `model_max_length = 2048`
- Right-side padding (`padding_side = "right"`)
- Key special tokens: `<|begin_of_text|>`, `<|end_of_text|>`, `<|finetune_right_pad_id|>`, plus chat-format reserved tokens such as `<|start_header_id|>`, `<|end_header_id|>`, `<|eom_id|>`, etc.

## Training Configuration (from DL_midterm0_81.ipynb)

- Base loading (Unsloth):
  - `FastLanguageModel.from_pretrained(model_name="unsloth/Meta-Llama-3.1-8B", load_in_4bit=True, max_seq_length=2048, dtype=auto)`
- LoRA setup (`FastLanguageModel.get_peft_model`):
  - `r = 32`
  - `lora_alpha = 64`
  - `lora_dropout = 0`
  - `bias = "none"`
  - `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - `use_gradient_checkpointing = "unsloth"`
  - `random_state = 42`
- Dataset sampling:
  - Source: `ad6398/nyu-dl-teach-maths-comp`
  - Train = 10,000, Validation = 1,000 (shuffled with seed 42)
- SFTTrainer (TRL) key arguments:
  - `per_device_train_batch_size = 32`
  - `per_device_eval_batch_size = 48`
  - `gradient_accumulation_steps = 2`
  - `num_train_epochs = 3`
  - `learning_rate = 2e-4`
  - `warmup_steps = 50`
  - `fp16 = not torch.cuda.is_bf16_supported()`; `bf16 = torch.cuda.is_bf16_supported()`
  - Evaluation: `eval_strategy = "steps"`, `eval_steps = 50`
  - Checkpoints: `save_strategy = "steps"`, `save_steps = 50`, `save_total_limit = 3`, `load_best_model_at_end = True`, `metric_for_best_model = "eval_loss"`
  - Logging: `logging_steps = 10`, `logging_first_step = True`
  - Optimizer & schedule: `optim = "adamw_torch_fused"`, `lr_scheduler_type = "cosine"`, `weight_decay = 0.01`
  - Dataloader: `dataloader_num_workers = 16`, `dataloader_pin_memory = True`, `group_by_length = True`
- Inference settings used for validation/testing:
  - `max_new_tokens = 8`, `do_sample = False`, `temperature = 0.0`, `use_cache = True`
  - Output parsing expects a single token: `True` or `False`

Note: The adapter config in this folder targets the 4-bit base `unsloth/meta-llama-3.1-8b-unsloth-bnb-4bit`. The notebook demonstrates training with Unsloth on `unsloth/Meta-Llama-3.1-8B` loaded in 4-bit; both are compatible with the listed LoRA settings (r=32, alpha=64, dropout=0) and target modules.

## Notes

- This adapter is for causal language modeling (CAUSAL_LM).
- If releasing publicly on Hugging Face, please complete the model card with training data, evaluation metrics, intended uses and limitations.

## Acknowledgements

- Base model and 4-bit loading approach come from the `unsloth` ecosystem.
- Inference and adapter loading are powered by `transformers` and `peft`.


