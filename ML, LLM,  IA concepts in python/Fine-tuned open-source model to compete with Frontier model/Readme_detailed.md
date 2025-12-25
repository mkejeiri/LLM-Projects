# QLoRA: Efficient Fine-Tuning for Large Language Models

![LoRA Target Modules](./LoRA.jpg)

---

## 🎯 Learning Objectives
- Understand the fundamental concepts of LoRA (Low-Rank Adaptation)
- Learn how QLoRA combines quantization with LoRA for efficient training
- Master the practical implementation of QLoRA for fine-tuning LLMs
- Apply QLoRA to fine-tune Llama 3.2 on custom datasets

---

## 📚 Module Overview

**QLoRA = Q + LoRA**

QLoRA combines two powerful techniques:
1. **LoRA** (Low-Rank Adaptation) - Efficient parameter training
2. **Q** (Quantization) - Memory-efficient model loading

---

## Part 1: Understanding LoRA

### The Problem: Training Large Models is Expensive

**Llama 3.2 Specifications:**
- **3 billion parameters**
- **13GB GPU memory** (just to load!)
- **28 decoder layers** stacked together

Each decoder layer contains:
- Self-attention layers (learn which input parts matter most)
- Multi-layer perceptron (MLP) layers
- Activation functions (SiLU)
- Layer normalization

**Traditional Training Requirements:**
```
Forward Pass → Calculate Loss → Backward Pass → Update Parameters
```

For 3 billion parameters, this is computationally prohibitive on consumer hardware.

---

### The LoRA Solution: Train Smaller, Adapt Larger

#### Core Concept

Instead of training all 3 billion parameters:
1. **Freeze** all original model weights
2. **Select** target modules (most impactful layers)
3. **Create** small low-rank adapter matrices
4. **Train** only these adapters
5. **Add** adapters to frozen weights during inference

#### Why "Low-Rank"?

**Rank** = dimensionality of a matrix

LoRA uses matrices with **fewer dimensions** than the original model layers, dramatically reducing trainable parameters.

---

### LoRA Architecture Details

#### Target Modules


Not all layers are equally important. LoRA focuses on:
- Query/Key/Value projection matrices in attention layers
- Output projection layers
- Feed-forward network layers

These are the **target modules** - the layers we'll adapt.

#### The LoRA Trick: Matrix Decomposition

Instead of one large adapter matrix, LoRA uses **two smaller matrices**:

```
LoRA_A × LoRA_B = Adapter Matrix
```

**Mathematical Formula:**
```
Output = Original_Weight × Input + α × (LoRA_A × LoRA_B) × Input
```

Where:
- `Original_Weight`: Frozen pretrained weights
- `LoRA_A`: First low-rank matrix (shape: d × r)
- `LoRA_B`: Second low-rank matrix (shape: r × k)
- `α`: Scaling factor (alpha)
- `r`: Rank (typically 8, 16, 32, or 64)

#### Why Two Matrices?

**Dimension Compatibility:**
- Original layer: `d × k` (e.g., 4096 × 4096)
- LoRA_A: `d × r` (e.g., 4096 × 16)
- LoRA_B: `r × k` (e.g., 16 × 4096)
- Result: `d × k` ✓ (matches original!)

**Parameter Reduction:**
- Original: 4096 × 4096 = **16,777,216 parameters**
- LoRA (r=16): (4096 × 16) + (16 × 4096) = **131,072 parameters**
- **Reduction: 99.2%!**

---

### LoRA Hyperparameters

![LoRA Hyperparameters](./Lora-fine-tune.jpg)

#### 1. Rank (r)
- **Lower rank** (4-8): Fewer parameters, faster training, less expressive
- **Higher rank** (32-64): More parameters, slower training, more expressive
- **Typical choice**: 16 or 32

#### 2. Alpha (α)
- Scaling factor for adapter contribution
- **Common practice**: α = 2 × r
- Controls how much the adapters influence the model

#### 3. Target Modules
- **Attention only**: `["q_proj", "v_proj"]`
- **All linear layers**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- More modules = more parameters but better adaptation

---

## Part 2: Adding the "Q" - Quantization

### What is Quantization?

**Quantization** reduces memory by using lower precision numbers:

- **FP32** (Full Precision): 32 bits per parameter → 13GB for Llama 3.2
- **FP16** (Half Precision): 16 bits per parameter → 6.5GB
- **INT8** (8-bit): 8 bits per parameter → 3.25GB
- **INT4** (4-bit): 4 bits per parameter → **1.6GB** ✓

### QLoRA = 4-bit Quantization + LoRA

**Key Innovation:**
1. Load base model in 4-bit precision (saves memory)
2. Train LoRA adapters in higher precision (maintains quality)
3. Combine for inference

**Result:** Fine-tune 3B parameter models on consumer GPUs!

---

## Part 3: Practical Implementation

### Setup Requirements

```python
# Required libraries
!pip install -q --upgrade bitsandbytes #pip install transformers peft bitsandbytes accelerate datasets
```

### Basic QLoRA Configuration

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# LoRA config
lora_config = LoraConfig(
    r=16,                              # Rank
    lora_alpha=32,                     # Alpha (2 × r)
    target_modules=["q_proj", "v_proj"], # Target modules
    lora_dropout=0.05,                 # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 8,388,608 || all params: 3,008,388,608 || trainable%: 0.28%
```

---

## Part 4: Training Workflow

### Step 1: Prepare Dataset

```python
from datasets import load_dataset

dataset = load_dataset("your_dataset")

def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

dataset = dataset.map(format_prompt)
```

### Step 2: Configure Training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./qlora-llama-3.2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)
```

### Step 3: Train

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
```

### Step 4: Save and Load

```python
# Save only LoRA adapters (small!)
model.save_pretrained("./qlora-adapters")

# Load later
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
model = PeftModel.from_pretrained(base_model, "./qlora-adapters")
```

---

## 🎓 Key Takeaways

1. **LoRA freezes base model weights** and trains small adapter matrices
2. **Low-rank decomposition** (A × B) enables massive parameter reduction
3. **QLoRA adds 4-bit quantization** for memory efficiency
4. **Target modules** determine which layers to adapt
5. **Rank and alpha** control adapter capacity and influence
6. **Training only adapters** means fast iteration and small checkpoints

---

## 💡 Practical Tips

### Choosing Rank
- Start with **r=16** for most tasks
- Increase to **r=32** if underfitting
- Decrease to **r=8** for very specific tasks

### Choosing Target Modules
- **Minimum**: `["q_proj", "v_proj"]` (attention only)
- **Recommended**: Add `["k_proj", "o_proj"]` (full attention)
- **Maximum**: All linear layers (most parameters)

### Memory Optimization
- Use **gradient_accumulation_steps** to simulate larger batches
- Enable **gradient_checkpointing** for even lower memory
- Use **bfloat16** instead of fp16 on supported hardware

---

## 🔬 Hands-On Exercise

### Challenge: Fine-tune Llama 3.2 for Your Domain

**Task:** Fine-tune Llama 3.2-3B on a domain-specific dataset using QLoRA

**Steps:**
1. Choose a dataset (e.g., medical, legal, code, customer support)
2. Configure QLoRA with appropriate hyperparameters
3. Train for 3 epochs
4. Evaluate on held-out test set
5. Compare with base model performance

**Deliverables:**
- Training script
- LoRA adapter weights
- Performance comparison report

---

## 📖 Further Reading

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Original LoRA research
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - QLoRA methodology
- [PEFT Documentation](https://huggingface.co/docs/peft) - Hugging Face PEFT library
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B) - Model details

---

## 🚀 Next Steps

After mastering QLoRA, explore:
- **Multi-adapter inference** - Use different adapters for different tasks
- **Adapter merging** - Combine multiple trained adapters
- **DoRA** - Weight-decomposed LoRA for better performance
- **Full fine-tuning** - When you have the compute budget

---

## ❓ Common Questions

**Q: Can I use LoRA with any model?**  
A: Yes! LoRA works with any transformer-based model (GPT, BERT, T5, etc.)

**Q: How much GPU memory do I need?**  
A: With QLoRA, you can fine-tune Llama 3.2-3B on a 16GB GPU (e.g., RTX 4080)

**Q: Are LoRA adapters portable?**  
A: Yes! Adapters are tiny (MBs) and can be shared/loaded independently

**Q: Does LoRA hurt performance?**  
A: Minimal impact! Often matches full fine-tuning with <1% of parameters

**Q: Can I train multiple adapters?**  
A: Absolutely! Train different adapters for different tasks, swap at inference

---

## 🎯 Success Criteria

You've mastered QLoRA when you can:
- ✅ Explain why LoRA uses matrix decomposition
- ✅ Configure QLoRA for different model sizes
- ✅ Fine-tune a model on custom data
- ✅ Optimize hyperparameters for your use case
- ✅ Debug memory and training issues

---




# Llama 3.2 3B Architecture - LoRA Target Modules

## Model Overview

**Llama 3.2 3B**: 3 billion parameters, 28 decoder layers
- **Memory**: 13GB (FP32)
- **Model dimension**: 3,072

---

## Decoder Layer Structure (28 layers)

Each layer contains:

### 1. Self-Attention (Target Modules for LoRA)

```python
(self_attn): LlamaAttention(
  (q_proj): Linear(3072 → 3072)  # Query projection - PRIMARY LoRA TARGET
  (k_proj): Linear(3072 → 1024)  # Key projection - LoRA TARGET
  (v_proj): Linear(3072 → 1024)  # Value projection - PRIMARY LoRA TARGET
  (o_proj): Linear(3072 → 3072)  # Output projection - LoRA TARGET
)
```

**These attention layers are the primary target modules for LoRA.**



Typical LoRA targeting:
- **Start with**: `q_proj`, `v_proj` (most common)
- **Add if needed**: `k_proj`, `o_proj` (full attention)

---

### 2. MLP Layers (Optional LoRA Targets)

```python
(mlp): LlamaMLP(
  (gate_proj): Linear(3072 → 8192)  # Optional LoRA target
  (up_proj): Linear(3072 → 8192)    # Optional LoRA target
  (down_proj): Linear(8192 → 3072)  # Optional LoRA target
  (act_fn): SiLUActivation()
)
```

**Add MLP layers to LoRA targets for:**
- More flexibility in fine-tuning
- Better absorption of training data
- Slightly better results (but slower training)


#### SiLU Activation Function

**SiLU** (Sigmoid Linear Unit), also called **Swish**

**Formula**: `SiLU(x) = x × sigmoid(x) = x × (1 / (1 + e^(-x)))`

**Formula breakdown**:
- `sigmoid(x) = 1 / (1 + e^(-x))` - Sigmoid function outputs values between 0 and 1
- `SiLU(x) = x × sigmoid(x)` - Multiply input by its sigmoid
- When x is large positive: sigmoid(x) ≈ 1, so SiLU(x) ≈ x (nearly linear)
- When x is large negative: sigmoid(x) ≈ 0, so SiLU(x) ≈ 0 (smoothly suppressed)
- When x = 0: sigmoid(0) = 0.5, so SiLU(0) = 0

**Example values**:
- SiLU(-2) = -2 × 0.12 = -0.24
- SiLU(-1) = -1 × 0.27 = -0.27
- SiLU(0) = 0 × 0.5 = 0
- SiLU(1) = 1 × 0.73 = 0.73
- SiLU(2) = 2 × 0.88 = 1.76

**Behavior**:
- Negative values: Smoothly approaches 0 (not hard cutoff like ReLU)
- Positive values: Nearly linear, slightly curved upward
- At x=0: Output is 0

**Why use SiLU?**
- Smooth, non-monotonic function
- Better gradient flow than ReLU
- Performs well in deep networks like Llama

---

## LoRA Hyperparameters

### 1. Rank (r)

**Common values**: 8, 16, 32 (powers of 2)
- **r = 8**: Fewer parameters, faster training
- **r = 16**: Balanced (recommended starting point)
- **r = 32**: More parameters, more expressive

**Note**: Powers of 2 are traditional but not required. You could use r = 11, though it feels unconventional.

### 2. Alpha (α)

**Rule of thumb**: α = 2 × r
- If r = 32, then α = 64
- This is the standard practice
- Other values work but typically perform slightly worse

### 3. Target Modules

**Typical progression**:
1. **Start with**: Attention heads only (`q_proj`, `v_proj`)
2. **Add if needed**: Full attention (`k_proj`, `o_proj`)
3. **Add if needed**: MLP layers (`gate_proj`, `up_proj`, `down_proj`)

**Trade-off**: More target modules = more parameters = longer training but potentially better results

---

## QLoRA: Quantization

### What Gets Quantized

**The base model is quantized** (not the LoRA adapters)
- 32-bit (FP32) → 4-bit quantization
- 13GB → ~3.25GB memory

### How Quantization Works

Each parameter is like a dimmer switch:
- **32-bit**: Extremely fine-grained control
- **4-bit**: Only 16 possible positions (coarse-grained)

**Key insight**: 4-bit quantization reduces performance only slightly, not proportionally to the precision loss.

### Quantization Details

1. **Not integers**: 16 positions map to floating-point values using normal distribution
2. **Base model only**: LoRA matrices remain higher precision
3. **Memory savings**: 75% reduction (32-bit → 4-bit)
4. **Performance impact**: Minimal (like MP3 vs WAV)

---

## LoRA Formula

**For each target module**:

```
Output = Frozen_Weight × Input + α × (LoRA_A × LoRA_B) × Input
```

Where:
- **Frozen_Weight**: Original 3B parameters (frozen, quantized to 4-bit)
- **LoRA_A**: Small matrix (d × r)
- **LoRA_B**: Small matrix (r × k)
- **α**: Scaling factor (typically 2 × r)

---

## Summary for LoRA Implementation

**Target modules to adapt**:
- Primary: `q_proj`, `v_proj` (attention)
- Secondary: `k_proj`, `o_proj` (full attention)
- Optional: `gate_proj`, `up_proj`, `down_proj` (MLP)

**Hyperparameters**:
- **r**: Start with 16
- **α**: Use 2 × r
- **Target modules**: Start with attention, add MLP if needed

----
## 1. Environment Setup

### 1.1 Google Colab Configuration

**Select T4 GPU Runtime**:
- Top right → Change runtime type → T4 GPU
- Free tier with Google account
- 15GB GPU memory
- Verify: View resources menu

**Check your GPU**:
```python
# You should see T4 GPU with 15GB in View Resources
```

### 1.2 Critical Error Handling

**Common Error**: "CUDA is required but not available for bitsandbytes"

**This is NOT a package problem!** Google swapped your hardware.

**Solution**:
1. Runtime >> Disconnect and delete runtime
2. Edit >> Clear All Outputs  
3. Connect to new T4
4. Rerun from top (including pip installs)

### 1.3 Installation & Imports

```python
# ALWAYS run this first (use -q for quiet, --upgrade for latest)
!pip install -q --upgrade bitsandbytes

# Core imports
import os
import re
import math
from tqdm import tqdm
from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    set_seed
)
from peft import LoraConfig, PeftModel
from datetime import datetime
```

### 1.4 HuggingFace Authentication

**Setup**:
- Click key icon (left sidebar) in Colab
- Add secret: `HF_TOKEN` with your HuggingFace token
- Get token from https://huggingface.co

```python
# Login to HuggingFace
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
```

---

## 2. Constants & Configuration

```python
# Base model selection
BASE_MODEL = "meta-llama/Llama-3.2-3B"

# Alternative models (similar size):
# - "google/gemma-2b"
# - "microsoft/phi-2"
# - "deepseek-ai/deepseek-coder-1.3b"
# - "Qwen/Qwen-1.8B" (Alibaba, no approval needed)

# Project configuration
PROJECT_NAME = "price"
RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"

# Training mode
LITE_MODE = False  # True for quick training, False for full dataset

# Dataset selection
DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

# Pre-trained model for inspection
FINETUNED_MODEL = f"ed-donner/price-2025-11-30_15.10.55-lite"
```

**Key Points**:
- Llama 3.2 requires approval 
- Llama is industry standard (Meta was first with open source)
- ~20,000 samples is good starting point for training data

---

## 3. Understanding Llama 3.2 Architecture

### 3.1 Load Base Model (No Quantization)

```python
# Load without quantization to inspect architecture
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    device_map="auto"
)

# Check memory usage
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")
# Output: 12.9 GB

# Inspect architecture
print(base_model)
```

**What you'll see**:
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 3072)
    (layers): ModuleList(
      (0-27): 28 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(3072, 3072, bias=False)
          (k_proj): Linear(3072, 1024, bias=False)
          (v_proj): Linear(3072, 1024, bias=False)
          (o_proj): Linear(3072, 3072, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(3072, 8192, bias=False)
          (up_proj): Linear(3072, 8192, bias=False)
          (down_proj): Linear(8192, 3072, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((3072,))
        (post_attention_layernorm): LlamaRMSNorm((3072,))
      )
    )
    (norm): LlamaRMSNorm((3072,))
  )
  (lm_head): Linear(3072, 128256, bias=False)
)
```

### 3.2 Architecture Breakdown

**Embedding Layer**:
```python
(embed_tokens): Embedding(128256, 3072)
```
- **Input**: 128,256 dimensions (one-hot encoded token)
- **Output**: 3,072 dimensions (embedding vector)
- **Function**: Converts sparse token representation to dense embedding

**Token Encoding Explained**:
- Vocabulary: 128,256 possible tokens
- One-hot vector: All zeros except one position
- Example: Token "and" (index 1) = [1, 0, 0, ..., 0] (128,256 long)
- Example: Token "yes" (index 2) = [0, 1, 0, ..., 0] (128,256 long)
- Embedding compresses this to 3,072 dense values

**28 Decoder Layers** (stacked):

Each layer has:

1. **Self-Attention** (learns which input parts matter most):
   - `q_proj`: 3,072 → 3,072 (Query)
   - `k_proj`: 3,072 → 1,024 (Key)
   - `v_proj`: 3,072 → 1,024 (Value)
   - `o_proj`: 3,072 → 3,072 (Output)

2. **MLP** (Multi-Layer Perceptron - feed-forward network):
   - `gate_proj`: 3,072 → 8,192
   - `up_proj`: 3,072 → 8,192
   - `down_proj`: 8,192 → 3,072
   - `act_fn`: SiLU activation

3. **Normalization**:
   - `input_layernorm`: RMSNorm before attention
   - `post_attention_layernorm`: RMSNorm before MLP

**Output Head**:
```python
(lm_head): Linear(3072, 128256, bias=False)
```
- **Input**: 3,072 dimensions (final hidden state)
- **Output**: 128,256 dimensions (probability for each token)
- **Function**: Predicts next token probabilities

### 3.3 How Neural Networks Work (Quick Recap)

**Layers = Matrix Calculations**:
- Each neuron: weighted sum (linear regression) + non-linearity
- Implementation: Matrix multiplies (efficient on GPU)
- GPU advantage: Designed for parallel matrix operations (originally for 3D graphics)

**Dimensions**:
- Input dimensions: How many numbers go in
- Output dimensions: How many numbers come out
- Matrix shape: (input_dim, output_dim)

**After inspecting, restart session**:
```python
# Runtime >> Restart session
# This clears GPU memory
```

---

## 4. Quantization: The "Q" in QLoRA

### 4.1 8-bit Quantization

```python
# After restarting: rerun pip install, imports, constants, login

# Configure 8-bit quantization
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")
# Output: 3.6 GB (was 12.9 GB)

# Inspect layers
print(base_model)
```

**What changed**:
- Layers now show `Linear8bit` instead of `Linear`
- 256 possible values per parameter (8 bits = 2^8)
- Memory reduced by ~72%

### 4.2 4-bit Quantization (QLoRA Standard)

```python
# After restarting: rerun pip install, imports, constants, login

# Configure 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # Enable 4-bit
    bnb_4bit_use_double_quant=True,             # Extra quantization pass
    bnb_4bit_compute_dtype=torch.bfloat16,      # Compute dtype (fast on T4)
    bnb_4bit_quant_type="nf4"                   # NormalFloat4 quantization
)

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.2f} GB")
# Output: 2.2 GB (was 12.9 GB)

# Inspect layers
print(base_model)
```

**What changed**:
- Layers now show `Linear4bit`
- 16 possible values per parameter (4 bits = 2^4)
- Memory reduced by ~83%

### 4.3 Quantization Parameters Explained

**`load_in_4bit=True`**:
- Reduces each parameter from 32-bit to 4-bit
- Like a dimmer switch: 16 positions instead of billions

**`bnb_4bit_use_double_quant=True`**:
- Applies quantization twice for extra memory savings
- Standard practice, always use

**`bnb_4bit_compute_dtype=torch.bfloat16`**:
- Data type used during computation
- `bfloat16` is fast on T4 GPUs
- Adjust based on your GPU

**`bnb_4bit_quant_type="nf4"`**:
- **NF4** = NormalFloat4
- Maps 16 positions to floating-point values (NOT integers)
- Uses normal distribution to intelligently place values
- Most popular and effective method

### 4.4 Key Quantization Facts

1. **What gets quantized**: Base model ONLY (not LoRA adapters)
2. **Not integers**: 16 positions map to floating-point values
3. **Performance impact**: Minimal (like MP3 vs WAV)
4. **LoRA adapters**: Stay at full 32-bit precision
5. **Why it works**: Neural networks have more detail than needed; quantization is efficient compression

---

## 5. LoRA: Low-Rank Adaptation

### 5.1 Load Fine-tuned Model (Base + LoRA)

```python
# Load 4-bit base model first (see section 4.2)

# Load LoRA adapters on top
fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e9:,.2f} GB")
# Output: 2.27 GB (base 2.2 GB + adapters ~70 MB)

# Inspect architecture
print(fine_tuned_model)
```

**What you'll see** (zoomed out view):
```
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 3072)
        (layers): ModuleList(
          (0-27): 28 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): Linear4bit(
                (lora_dropout): ModuleDict(...)
                (lora_A): ModuleDict(
                  (default): Linear(3072, 32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(32, 3072, bias=False)
                )
              )
              (k_proj): Linear4bit(
                (lora_A): Linear(3072, 32, bias=False)
                (lora_B): Linear(32, 1024, bias=False)
              )
              (v_proj): Linear4bit(
                (lora_A): Linear(3072, 32, bias=False)
                (lora_B): Linear(32, 1024, bias=False)
              )
              (o_proj): Linear4bit(
                (lora_A): Linear(3072, 32, bias=False)
                (lora_B): Linear(32, 3072, bias=False)
              )
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear4bit(...)
              (up_proj): Linear4bit(...)
              (down_proj): Linear4bit(...)
            )
          )
        )
      )
    )
  )
)
```

### 5.2 Understanding LoRA Structure

**Each target module now has**:
- Original `Linear4bit` layer (frozen, quantized base weights)
- `lora_A`: Small matrix (input_dim × r)
- `lora_B`: Small matrix (r × output_dim)

**Example: q_proj with r=32**:
- Base: 3,072 → 3,072 (frozen)
- lora_A: 3,072 → 32
- lora_B: 32 → 3,072
- When multiplied: lora_A × lora_B = 3,072 × 3,072 (matches base!)

**Formula**:
```
Output = Frozen_Base_Weight × Input + α × (lora_A × lora_B) × Input
```

**Why two matrices?**:
- Can't add small matrix to large matrix (dimension mismatch)
- Two small matrices multiply to create large matrix
- This large result can be added to base weights
- Mathematical trick for parameter efficiency

---

## 6. LoRA Parameter Calculations

### 6.1 Lite Mode Configuration (r=32, Attention Only)

```python
# Configuration
r = 32  # Rank (inner dimension)

# Attention layer dimensions (see architecture above)
lora_q_proj = 3072 * r + 3072 * r  # = 196,608
lora_k_proj = 3072 * r + 1024 * r  # = 131,072
lora_v_proj = 3072 * r + 1024 * r  # = 131,072
lora_o_proj = 3072 * r + 3072 * r  # = 196,608

# Total per layer
lora_layer = lora_q_proj + lora_k_proj + lora_v_proj + lora_o_proj
# = 655,360 parameters per layer

# 28 layers total
params = lora_layer * 28
# = 18,350,080 parameters

# Size in MB (4 bytes per parameter for FP32)
size = (params * 4) / 1_000_000
# = 73.4 MB

print(f"Total number of params: {params:,} and size {size:,.1f}MB")
```

**Lite Mode Summary**:
- **r**: 32
- **α**: 64 (2 × r)
- **Target modules**: q_proj, k_proj, v_proj, o_proj (attention only)
- **Training data**: 20,000 samples
- **Parameters**: 18.4M
- **Size**: 73 MB

### 6.2 Full Mode Configuration (r=256, Attention + MLP)

```python
# Configuration
r = 256  # Much larger rank

# Attention layers
lora_q_proj = 3072 * r + 3072 * r
lora_k_proj = 3072 * r + 1024 * r
lora_v_proj = 3072 * r + 1024 * r
lora_o_proj = 3072 * r + 3072 * r

# MLP layers (additional targets)
lora_gate_proj = 3072 * r + 8192 * r  # = 2,883,584
lora_up_proj = 3072 * r + 8192 * r    # = 2,883,584
lora_down_proj = 3072 * r + 8192 * r  # = 2,883,584

# Total per layer (attention + MLP)
lora_layer = (lora_q_proj + lora_k_proj + lora_v_proj + lora_o_proj + 
              lora_gate_proj + lora_up_proj + lora_down_proj)

# 28 layers total
params = lora_layer * 28

# Size in MB
size = (params * 4) / 1_000_000

print(f"Total number of params: {params:,} and size {size:,.1f}MB")
```

**Full Mode Summary**:
- **r**: 256 (extreme setup for large dataset)
- **α**: 512 (2 × r)
- **Target modules**: All 7 (q, k, v, o, gate, up, down)
- **Training data**: Full dataset (much larger)
- **Parameters**: Much more than lite mode
- **Size**: Significantly larger

---

## 7. Practical Workflow Summary

### 7.1 Complete Setup Sequence

```python
# 1. Install packages
!pip install -q --upgrade bitsandbytes

# 2. Imports
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from google.colab import userdata
from huggingface_hub import login

# 3. Constants
BASE_MODEL = "meta-llama/Llama-3.2-3B"
LITE_MODE = True

# 4. Login
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# 5. Configure quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# 6. Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto"
)

# 7. Load LoRA adapters (if using pre-trained)
fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

# 8. Check memory
print(f"Memory: {fine_tuned_model.get_memory_footprint() / 1e9:,.2f} GB")
```

### 7.2 Memory Management

**Always restart between model loads**:
```python
# Runtime >> Restart session
# Then rerun: pip install, imports, constants, login
```

**Why?**:
- Clears GPU memory
- Prevents out-of-memory errors
- Ensures clean state

### 7.3 Inspection Tips

```python
# Print full architecture
print(model)

# Check memory usage
print(f"Memory: {model.get_memory_footprint() / 1e9:,.2f} GB")

# For deeper understanding, paste architecture into ChatGPT:
# "Explain this Llama architecture in detail"
```

---

## 8. Key Concepts for LLM Engineers

### 8.1 Quantization
- **Purpose**: Reduce memory by using lower precision
- **4-bit**: 16 possible values (not integers, mapped to floats)
- **Impact**: Minimal performance loss, massive memory savings
- **What**: Base model only (NOT LoRA adapters)

### 8.2 LoRA
- **Purpose**: Train small adapters instead of full model
- **How**: Two matrices (lora_A × lora_B) added to frozen base
- **Size**: ~70MB for lite mode vs 13GB for full model
- **Formula**: `Output = Base × Input + α × (lora_A × lora_B) × Input`

### 8.3 Target Modules
- **Start**: Attention only (q_proj, v_proj minimum)
- **Expand**: Full attention (add k_proj, o_proj)
- **Maximum**: Add MLP (gate_proj, up_proj, down_proj)
- **Trade-off**: More modules = more parameters = better adaptation but slower

### 8.4 Hyperparameters
- **r (rank)**: 8, 16, 32, 64 (powers of 2 traditional but not required)
- **α (alpha)**: 2 × r (standard practice)
- **Training data**: ~20,000 samples good starting point

### 8.5 Architecture Understanding
- **28 layers**: Each with attention + MLP
- **3,072 dimensions**: Inner model dimension
- **128,256 tokens**: Vocabulary size
- **Matrix operations**: Core of neural network computation
- **GPU efficiency**: Parallel matrix calculations

---


## 9. Critical Reminders

✅ **Always use T4 GPU** in Colab  
✅ **Restart session** between model loads  
✅ **CUDA error** = hardware swap, not package issue  
✅ **HuggingFace token** in Colab secrets  
✅ **Quantize base model** only (not LoRA)  
✅ **LoRA adapters** are tiny (MBs not GBs)  
✅ **Start with r=32** for attention layers  
✅ **Use α = 2 × r** as default  
✅ **Inspect architecture** with print(model)  
✅ **Monitor memory** with get_memory_footprint()



----------------

#  **Fine-Tuning LLMs**
Building a model that estimates product prices from descriptions using fine-tuned Llama 3.2-3B.
##  **Key Concepts & Workflow**
###  **1. Data Preparation Strategy**

**Token Truncation (Critical Decision)**
-  **Cutoff at 110 tokens** for input summaries (affects 5.7% of data)
-  **Rationale**: GPU memory is proportional to max sequence length
- Padding unused tokens wastes memory and slows training
-  **Final max sequence length: 128 tokens** (power of 2 - industry best practice)
- Includes: question (16 tokens) + summary (110 tokens) + prefix
  

**Prompt Engineering**

```python

# Structure

Question:  "What does this cost to the nearest dollar?"

Summary:  [Title, Category, Brand, Description, Details]

Prefix:  "Price is $"

Completion:  [rounded price for training, exact for testing]

``` 

**Why Round Training Prices?**

- Focus model on predicting **dollars, not cents**

- LLMs treat each token equally during loss calculation

- Predicting cents would waste training effort on less important digits

-  **Llama advantage**: All 3-digit numbers = 1 token (0-999)

- Test data keeps exact prices for fair evaluation

  

---

  

###  **2. Model Architecture & Quantization**

  

**Base Model**: `meta-llama/Llama-3.2-3B`

-  **4-bit quantization** using BitsAndBytes

- Reduces from ~12GB to **2.2GB** memory footprint

- Configuration:

```python

BitsAndBytesConfig(

load_in_4bit=True,

bnb_4bit_use_double_quant=True,

bnb_4bit_compute_dtype=torch.bfloat16,

bnb_4bit_quant_type="nf4"

)

```

  

**Tokenizer Setup**

```python

tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side =  "right"

```

  

---

  

###  **3. Dataset Structure**

  

**HuggingFace Format**

-  **Columns**: `prompt` and `completion` (expected by HF Trainer)

-  **Splits**:

- Light: 20K train, 1K val, 1K test

- Full: 800K train, 10K val, 10K test

  

**Data Distribution**

- Average tokens: ~86-101

- Longest sequence: 126 tokens (before padding to 128)

- 5.7% of data truncated at 110 tokens

  

---

  

###  **4. Evaluation Framework**

  

**util.py Components**

```python

class  Tester:

- post_process(): Extract numeric predictions

- color_for(): Green (<$40 error), Orange (<$80), Red (>$80)

- run_datapoint(): Single prediction evaluation

- chart(): Scatter plot (predicted vs actual)

- error_trend_chart(): Running average with  95% CI

- report(): MSE, R², average error

```

  

**Metrics**

-  **Mean Absolute Error** (primary)

-  **MSE** (Mean Squared Error)

-  **R²** (coefficient of determination)

-  **Color-coded accuracy**: Visual feedback system

  

---

  

###  **5. Base Model Performance**

  

**Baseline Test** (before fine-tuning)

```python

def  model_predict(item):

inputs = tokenizer(item["prompt"],  return_tensors="pt").to("cuda")

with torch.no_grad():

output_ids = base_model.generate(**inputs,  max_new_tokens=8)

return tokenizer.decode(generated_ids)

```

  

**Example Result**:

- ***Input: V2 Distortion Pedal description***

- Actual: $219

- Prediction: $349.99 (off by $130)

---

###  **6. Critical Engineering Decisions**

**Hyperparameters to Experiment With**:

1.  **Token cutoff** (110) - Most impactful for performance

2.  **Max sequence length** (128) - Memory/speed tradeoff

3.  **Rounding strategy** - Focus training on important digits

4.  **Data quality** - More important than exotic hyperparameters (R, Alpha, learning rates)

  

**Why These Matter**:

- Changing dataset = biggest performance impact

- Less glamorous than tuning R/Alpha but more effective

- Tight data constraints = more training iterations in same time

  

---

  

###  **7. Technical Insights**

  

**Classification vs Regression**

- LLMs do **classification** (next token prediction)

- 128K possible tokens = 128K buckets

- Price prediction = classifying into price buckets

- Not true regression despite numeric output

  

**Llama Tokenization Advantage**

- All 3-digit numbers → 1 token

- Perfect for price prediction (0-999 range)

- Other models (Qwen, Phi) split numbers differently

  

**Memory Management**

- GPU RAM usage = linear with max_sequence_length

- Padding to power of 2 (128) = efficient memory allocation

- Quantization reduces model size by ~75%

**Workflow**:

1. Load preprocessed data from HuggingFace Hub

2. Tokenize and analyze distribution

3. Create prompts with truncation

4. Push to Hub as `items_prompts_lite/full`

5. Load quantized base model

6. Evaluate baseline performance

  

---

  

###  **9. Best Practices**

  

✅ **Always use powers of 2** for sequence lengths

✅ **Truncate strategically** - analyze what gets cut

✅ **Round training targets** when precision doesn't matter

✅ **Test with exact values** for fair comparison

✅ **Monitor token distributions** before setting limits

✅ **Use 4-bit quantization** for consumer GPUs

✅ **Validate on small dataset first** (lite mode)
 

##  **Key Takeaways**
1.  **Data preparation is 80% of the work** - get it right before training

2.  **Memory constraints drive architecture decisions** - quantization is essential

3.  **Domain-specific optimizations matter** - rounding prices, token analysis

4.  **Evaluation framework first** - know how you'll measure success

5.  **Start small, scale up** - lite mode validates approach quickly

  --- 

##  **Complete Code Walkthrough**

###  **Part 1: Item Class (items.py)**
The `Item` class is the core data structure using Pydantic for validation.
```python

from pydantic import BaseModel

from datasets import Dataset, DatasetDict, load_dataset

from typing import Optional, Self

  

PREFIX =  "Price is $"

QUESTION =  "What does this cost to the nearest dollar?"

  

class  Item(BaseModel):

"""

An Item is a data-point of a Product with a Price

"""

title:  str

category:  str

price:  float

full: Optional[str]  =  None

weight: Optional[float]  =  None

summary: Optional[str]  =  None

prompt: Optional[str]  =  None

completion: Optional[str]  =  None

id: Optional[int]  =  None

```

  

**Explanation**:

- Uses Pydantic for automatic validation and serialization

-  `Optional` fields allow flexibility for different pipeline stages

-  `summary` contains the product description

-  `prompt` and `completion` are generated for training

  

---

  

####  **Method: count_tokens()**

```python

def  count_tokens(self,  tokenizer):

"""Count tokens in the summary"""

return  len(tokenizer.encode(self.summary,  add_special_tokens=False))

```

  

**Explanation**:

- Converts text to tokens using the model's tokenizer

-  `add_special_tokens=False` excludes BOS/EOS tokens

- Used to analyze token distribution and set truncation limits

  

---

  

####  **Method: make_prompts()**

```python

def  make_prompts(self,  tokenizer,  max_tokens,  do_round):

"""Make prompts and completions"""

# Tokenize the summary

tokens = tokenizer.encode(self.summary,  add_special_tokens=False)

# Truncate if exceeds max_tokens

if  len(tokens)  > max_tokens:

summary = tokenizer.decode(tokens[:max_tokens]).rstrip()

else:

summary =  self.summary

# Build prompt (question + summary + prefix)

self.prompt =  f"{QUESTION}\n\n{summary}\n\n{PREFIX}"

# Build completion (rounded for training, exact for testing)

self.completion =  f"{round(self.price)}.00"  if do_round else  str(self.price)

```

  

**Explanation**:

-  **Truncation**: Cuts summary at `max_tokens` (110) to save GPU memory

-  **Prompt structure**: Question → Summary → "Price is $"

-  **Rounding logic**:

- Training: Round to nearest dollar (e.g., "64.00")

- Testing: Keep exact price (e.g., "219.0")

-  **Why round?** Focus model on predicting dollars, not cents

  

---

  

####  **Method: push_prompts_to_hub()**

```python

@staticmethod

def  push_prompts_to_hub(

dataset_name:  str,  train: list[Self],  val: list[Self],  test: list[Self]

):

"""Push Item lists to HuggingFace Hub in prompt-completion format."""

DatasetDict(

{

"train": Dataset.from_list([item.to_datapoint()  for item in train]),

"val": Dataset.from_list([item.to_datapoint()  for item in val]),

"test": Dataset.from_list([item.to_datapoint()  for item in test]),

}

).push_to_hub(dataset_name)

  

def  to_datapoint(self)  ->  dict:

return  {"prompt":  self.prompt,  "completion":  self.completion}

```

  

**Explanation**:

- Converts Item objects to HuggingFace Dataset format

- Only includes `prompt` and `completion` columns (required by HF Trainer)

- Uploads to HuggingFace Hub for easy access in Colab

  

---

  

###  **Data Preparation Pipeline**

  

####  **Critical Insight: Why This Matters**

  

> "Changing your dataset is one of the most powerful ways to impact performance. It's not the most glamorous of hyperparameters. People like to explore R and Alpha and all sorts of other things. Learning rates, lots of things that people explore. This kind of data related hyperparameter might feel less attractive. It often makes an enormous difference."

  

**Key Point**: Data-related decisions (truncation, rounding) have MORE impact than fancy hyperparameters like R, Alpha, or learning rates.

  

####  **Step 1: Load Data**

```python

import os

from dotenv import load_dotenv

from huggingface_hub import login

from pricer.items import Item

from transformers import AutoTokenizer

import matplotlib.pyplot as plt

  

# Configuration

LITE_MODE =  False  # True = 20K samples, False = 800K samples

  

# Login to HuggingFace

load_dotenv(override=True)

hf_token = os.environ['HF_TOKEN']

login(hf_token,  add_to_git_credential=True)

  

# Load preprocessed data
username =  "Your username"
dataset =  f"{username}/items_lite"  if LITE_MODE else  f"{username}/items_full"
train, val, test = Item.from_hub(dataset)
items = train + val + test
print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")

```

  

**Explanation**:

-  `LITE_MODE`: Quick validation with 20K samples before full 800K run

- Loads preprocessed data from previous week (already has summaries)

- Combines all splits for token analysis
---

####  **Step 2: Analyze Token Distribution**

```python

# Load tokenizer

BASE_MODEL =  "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Count tokens for each item

from tqdm.notebook import tqdm

token_counts =  [item.count_tokens(tokenizer)  for item in tqdm(items)]

# Visualize distribution

plt.figure(figsize=(15,  6))

plt.title(f"Tokens in Summary: Avg {sum(token_counts)/len(token_counts):,.1f} and highest {max(token_counts):,}")

plt.xlabel('Number of tokens in summary')

plt.ylabel('Count')

plt.hist(token_counts,  rwidth=0.7,  color="skyblue",  bins=range(0,  200,  10))

plt.show()

```

  

**Output**:

- Average: ~86 tokens

- Highest: 4582 tokens (outlier with product codes)

- Most items: <100 tokens

  

**Explanation**:

- Histogram reveals long tail distribution

- Helps decide truncation point (110 tokens)

- Visualizes impact of truncation decision

  

---

  

####  **Step 3: Determine Truncation Point**

```python

CUTOFF =  110

cut =  len([count for count in token_counts if count > CUTOFF])

print(f"With this CUTOFF, we will truncate {cut:,} items which is {cut/len(items):.1%}")

# Output: "With this CUTOFF, we will truncate 47,083 items which is 5.7%"

```

  

**Explanation**:

-  **110 tokens** chosen to balance:

- Memory efficiency (less padding)

- Data retention (only 5.7% truncated)

-  **Trade-off**: More tokens = richer data but slower training

-  **Experimentation**: This is a hyperparameter to tune

  

---

####  **Step 4: Generate Prompts**

```python
# Training/validation: Round prices
for item in tqdm(train+val):
	item.make_prompts(tokenizer,  CUTOFF=110,  do_round=True)
	# Test: Keep exact prices
for item in tqdm(test):
	item.make_prompts(tokenizer,  CUTOFF=110,  do_round=False)
	# Inspect results

print("PROMPT:")
print(test[0].prompt)
print("COMPLETION:")
print(test[0].completion)

```

  

**Output Example**:

```
PROMPT:
What does this cost to the nearest dollar?
Title: Excess V2 Distortion/Modulation Pedal
Category: Music Pedals
Brand: Old Blood Noise
Description: A versatile pedal offering distortion...
Price is $
COMPLETION:
219.0
```

**Explanation**:

- Prompt ends with "Price is $" (model completes with price)

- Training data: "64.00" (rounded)

- Test data: "219.0" (exact)
---
####  **Step 5: Verify Final Token Count**
```python
# Count tokens in full prompt + completion
prompt_token_counts =  [item.count_prompt_tokens(tokenizer)  for item in tqdm(items)]
plt.figure(figsize=(15,  6))
plt.title(f"Tokens: Avg {sum(prompt_token_counts)/len(prompt_token_counts):,.1f} and highest {max(prompt_token_counts):,}")
plt.xlabel('Number of tokens in prompt and the completion')
plt.ylabel('Count')
plt.hist(prompt_token_counts,  rwidth=0.7,  color="gold",  bins=range(0,  200,  10))
plt.show()
```
**Output**:

- Average: 101 tokens
- Highest: 126 tokens
-  **Max sequence length: 128** (power of 2, with buffer)

**Explanation**:
- 110 (summary) + 16 (question + prefix) = 126 tokens
- Pad to 128 for efficient GPU memory allocation
- Powers of 2 are industry best practice
---

  

####  **Step 6: Upload to HuggingFace**

```python

username =  "Your username"
dataset =  f"{username}/items_prompts_lite"  if LITE_MODE else  f"{username}/items_prompts_full"
Item.push_prompts_to_hub(dataset, train, val, test)
```
**Explanation**:

- Creates dataset with only `prompt` and `completion` columns

- Uploads to HuggingFace Hub for Colab access

- Separate lite/full datasets for experimentation

  

---

  

###  **Part 3: Base Model Evaluation**
`evaluation.py`
  

####  **Step 1: Setup Colab Environment**

```python
# Install quantization library
!pip install -q --upgrade bitsandbytes
# Download evaluation utilities
!wget -q https://raw.githubusercontent.com/ed-donner/llm_engineering/main/week7/util.py -O util.py
# Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from util import evaluate
from google.colab import userdata
from huggingface_hub import login
  
# Constants
BASE_MODEL =  "meta-llama/Llama-3.2-3B"
PROJECT_NAME =  "price"
LITE_MODE =  True
DATA_USER =  "Your User"
DATASET_NAME =  f"{DATA_USER}/items_prompts_lite"  if LITE_MODE else  f" {DATA_USER}/items_prompts_full"
```

**Explanation**:

-  `bitsandbytes`: Required for 4-bit quantization
-  `util.py`: Custom evaluation framework
- Colab provides free T4 GPU (16GB VRAM)

---

####  **Step 2: Load Data**
```python
# Login to HuggingFace
hf_token = userdata.get('HF_TOKEN')
login(hf_token,  add_to_git_credential=True)
 

# Load dataset
dataset = load_dataset(DATASET_NAME)
train = dataset['train']
val = dataset['val']
test = dataset['test']
# Inspect first item
print(train[0])
# Output: {'prompt': '...', 'completion': '64.00'}
```

**Explanation**:
- Loads from HuggingFace Hub (uploaded in previous step)
- Dataset has only 2 columns: `prompt` and `completion`
- Ready for supervised fine-tuning (SFT)

  

---

  

####  **Step 3: Configure Quantization**

```python
QUANT_4_BIT =  True
if QUANT_4_BIT:
	quant_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_use_double_quant=True,  # Nested quantization
	bnb_4bit_compute_dtype=torch.bfloat16,  # Computation precision
	bnb_4bit_quant_type="nf4"  # NormalFloat4 quantization
)

else:
	quant_config = BitsAndBytesConfig(
	load_in_8bit=True,
	bnb_8bit_compute_dtype=torch.bfloat16
)
```

**Explanation**:
-  **4-bit quantization**: Reduces model from ~12GB to ~2.2GB
-  **Double quantization**: Further compression with minimal quality loss
-  **bfloat16**: Faster computation on modern GPUs
-  **NF4**: Optimal quantization method for LLMs
---

  

###  **Step 4: Load Model and Tokenizer**

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,  trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Use EOS as padding
tokenizer.padding_side =  "right"  # Pad on the right side

# Load quantized model
base_model = AutoModelForCausalLM.from_pretrained(
							BASE_MODEL,
							quantization_config=quant_config,
							device_map="auto",  # Automatically distribute across GPUs
							)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
print(f"Memory footprint: {base_model.get_memory_footprint()  /  1e9:.1f} GB")
# Output: "Memory footprint: 2.2 GB"
```
**Explanation**:
-  **pad_token = eos_token**: Standard practice for decoder-only models
-  **padding_side = "right"**: Prevents warnings during training
-  **device_map = "auto"**: Handles multi-GPU setups automatically
-  **2.2GB**: Fits comfortably on T4 GPU (16GB VRAM)
---
####  **Step 5: Define Prediction Function**

```python
def  model_predict(item):
"""Generate price prediction for a single item"""
	# Tokenize prompt
	inputs = tokenizer(item["prompt"],  return_tensors="pt").to("cuda")
	# Generate completion (no gradient computation)
	with torch.no_grad():
		output_ids = base_model.generate(**inputs,  max_new_tokens=8)

	# Extract only the generated tokens (not the prompt)
	prompt_len = inputs["input_ids"].shape[1]
	generated_ids = output_ids[0, prompt_len:]
	# Decode to text
	return tokenizer.decode(generated_ids)
```
**Explanation**:

-  **torch.no_grad()**: Disables gradient tracking (inference only)
-  **max_new_tokens=8**: Enough for price (e.g., "349.99")
-  **Slicing**: Removes prompt from output, returns only prediction
-  **Returns**: String like "349.99 what is the price" (model continues)
---
####  **Step 6: Test Single Prediction**

```python
# Inspect test item
print(test[0])
# Output: {'prompt': '...V2 Distortion Pedal...Price is $', 'completion': '219.0'}

# Get prediction
prediction = model_predict(test[0])
print(prediction)
# Output: "349.99 what is the price"
```

  

**Explanation**:
-  **Actual price**: $219
-  **Predicted**: $349.99
-  **Error**: $130 (baseline before fine-tuning)
- Model has no domain knowledge yet
---

####  **Step 7: Full Evaluation**
```python
evaluate(model_predict, test)
```
**What evaluate() does**:
1. Runs `model_predict()` on all test items
2. Extracts numeric predictions using regex
3. Calculates errors (predicted - actual)
4. Color codes: Green (<$40), Orange (<$80), Red (>$80)
5. Plots:
	- Scatter plot (predicted vs actual)
	- Error trend chart (running average with 95% CI)
6. Reports:
	- Mean Absolute Error
	- MSE (Mean Squared Error)
	- R² (coefficient of determination)

  

**Explanation**:

- Baseline performance establishes improvement target
- Visual feedback helps diagnose model behavior
- Metrics comparable to frontier models (GPT-5: $44.74 error)
---
###  **Part 4: Evaluation Utilities (util.py)**
####  **Tester Class Overview**
```python
import re
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm

class  Tester:
	def  __init__(self,  predictor,  data,  title=None,  size=200):
	self.predictor = predictor # Prediction function
	self.data = data # Test dataset
	self.title = title or  self.make_title(predictor)
	self.size = size # Number of samples to evaluate
	self.titles =  []  # Product titles
	self.guesses =  []  # Predictions
	self.truths =  []  # Actual prices
	self.errors =  []  # Absolute errors
	self.colors =  []  # Color codes
```
---

####  **Method: post_process()**
```python
@staticmethod
def  post_process(value):
	"""Extract numeric value from model output"""
	if  isinstance(value,  str):
		value = value.replace("$",  "").replace(",",  "")
		match = re.search(r"[-+]?\d*\.\d+|\d+", value)
		return  float(match.group())  if match else  0
	else:
		return value
```

**Explanation**:
- Handles various formats: "$349.99", "349", "349.99 dollars"
- Regex extracts first number found
- Returns 0 if no number found (rare edge case)


---
####  **Method: color_for()**
```python
def  color_for(self,  error,  truth):
	"""Assign color based on error magnitude"""
	if error <  40  or error / truth <  0.2:  # <$40 or <20%
	return  "green"
elif error <  80  or error / truth <  0.4:  # <$80 or <40%
	return  "orange"
else:
	return  "red"
```

**Explanation**:
-  **Green**: Excellent prediction
-  **Orange**: Acceptable prediction
-  **Red**: Poor prediction
- Uses both absolute ($) and relative (%) thresholds
---
####  **Method: run()**
```python
def  run(self):
"""Run evaluation on all test samples"""
for i in tqdm(range(self.size)):
	datapoint =  self.data[i]

	# Get prediction
	value =  self.predictor(datapoint)
	guess =  self.post_process(value)
	# Get ground truth
	truth =  float(datapoint["completion"])

	# Calculate error
	error =  abs(guess - truth)
	color =  self.color_for(error, truth)

	# Extract title
	pieces = datapoint["prompt"].split("Title: ")
	title = pieces[1].split("\n")[0]  if  len(pieces)  >  1  else pieces[0]
	title = title if  len(title)  <=  40  else title[:40]  +  "..."

# Store results
self.titles.append(title)
self.guesses.append(guess)
self.truths.append(truth)
self.errors.append(error)
self.colors.append(color)
# Print progress
print(f"{COLOR_MAP[color]}${error:.0f} ",  end="")
clear_output(wait=True)
self.report()
```

  

**Explanation**:

- Progress bar shows evaluation status
- Color-coded errors printed in real-time
- Clears output and shows final report

  
---
####  **Method: chart()**
```python
def  chart(self,  title):
"""Create scatter plot of predictions vs actuals"""
df = pd.DataFrame({
"truth":  self.truths,
"guess":  self.guesses,
"title":  self.titles,
"error":  self.errors,
"color":  self.colors,
})

# Create scatter plot
fig = px.scatter(
df,
x="truth",
y="guess",
color="color",
color_discrete_map={"green":  "green",  "orange":  "orange",  "red":  "red"},
title=title,
labels={"truth":  "Actual Price",  "guess":  "Predicted Price"},
)

# Add y=x reference line
max_val =  float(max(df["truth"].max(), df["guess"].max()))
fig.add_trace(go.Scatter(
x=[0, max_val],
y=[0, max_val],
mode="lines",
line=dict(width=2,  dash="dash",  color="deepskyblue"),
name="y = x",
))
fig.show()
```

**Explanation**:

-  **y=x line**: Perfect predictions would fall on this line
-  **Color coding**: Visual assessment of accuracy
-  **Interactive**: Hover to see product details
---

####  **Method: error_trend_chart()**

```python

def  error_trend_chart(self):
"""Plot running average error with confidence interval"""
n =  len(self.errors)
# Calculate running mean
running_sums =  list(accumulate(self.errors))
x =  list(range(1, n +  1))
running_means =  [s / i for s, i in  zip(running_sums, x)]
# Calculate running std and 95% CI
running_squares =  list(accumulate(e * e for e in  self.errors))
running_stds =  [
math.sqrt((sq_sum / i)  -  (mean**2))  if i >  1  else  0
for i, sq_sum, mean in  zip(x, running_squares, running_means)
]
ci =  [1.96  *  (sd / math.sqrt(i))  if i >  1  else  0  for i, sd in  zip(x, running_stds)]
# Plot with confidence band
fig = go.Figure()
fig.add_trace(go.Scatter(
x=x + x[::-1],
y=[m + c for m, c in  zip(running_means, ci)]  +
[m - c for m, c in  zip(running_means, ci)][::-1],
fill="toself",
fillcolor="rgba(128,128,128,0.2)",
line=dict(color="rgba(255,255,255,0)"),
name="95% CI",
))
fig.show()
```

  

**Explanation**:

- Shows how error stabilizes as more samples evaluated
-  **95% CI**: Statistical confidence in the estimate
- Helps determine if sample size is sufficient
---

  

####  **Method: report()**
```python
def  report(self):
"""Generate final evaluation report"""
average_error =  sum(self.errors)  /  self.size
mse = mean_squared_error(self.truths,  self.guesses)
r2 = r2_score(self.truths,  self.guesses)  *  100
title =  f"{self.title} results<br><b>Error:</b> ${average_error:,.2f} <b>MSE:</b> {mse:,.0f} <b>r²:</b> {r2:.1f}%"
self.error_trend_chart()
self.chart(title)
```
**Explanation**:
-  **Average Error**: Primary metric (dollars off)
-  **MSE**: Penalizes large errors more heavily
-  **R²**: Percentage of variance explained (0-100%)
---
 

###  **Part 5: Complete Workflow Summary**
```python

# 1. LOAD DATA
train, val, test = Item.from_hub("ed-donner/items_full")

# 2. ANALYZE TOKENS
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
token_counts =  [item.count_tokens(tokenizer)  for item in train+val+test]

# 3. GENERATE PROMPTS
for item in train+val:
	item.make_prompts(tokenizer,  max_tokens=110,  do_round=True)
for item in test:
	item.make_prompts(tokenizer,  max_tokens=110,  do_round=False)

# 4. UPLOAD TO HUB
Item.push_prompts_to_hub("[Your user]/items_prompts_full", train, val, test)

# 5. LOAD QUANTIZED MODEL
quant_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_compute_dtype=torch.bfloat16,
bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
"meta-llama/Llama-3.2-3B",
quantization_config=quant_config,
device_map="auto"
)


# 6. DEFINE PREDICTION
def  model_predict(item):
inputs = tokenizer(item["prompt"],  return_tensors="pt").to("cuda")
with torch.no_grad():
	output_ids = base_model.generate(**inputs,  max_new_tokens=8)
	prompt_len = inputs["input_ids"].shape[1]
	generated_ids = output_ids[0, prompt_len:]
return tokenizer.decode(generated_ids)
  

# 7. EVALUATE

from util import evaluate
evaluate(model_predict, test)
```
**This workflow**:
1. Prepares data with optimal truncation
2. Creates training-ready prompts
3. Loads memory-efficient quantized model
4. Establishes baseline performance
5. Sets stage for fine-tuning
---
  
##  **Important Constants**
```python
BASE_MODEL =  "meta-llama/Llama-3.2-3B"
PROJECT_NAME =  "price"
CUTOFF =  110  # Token truncation
MAX_SEQ_LENGTH =  128  # Power of 2
LITE_MODE =  True  # 20K vs 800K training samples
```
---
##  **Performance Targets**

-  **Baseline (untrained Llama 3.2)**: ~$130 error
-  **GPT-5 (frontier model)**: $44.74 error
-  **Goal**: Beat frontier models with fine-tuned 3B model
---

This approach demonstrates production-level ML engineering: balancing performance, efficiency, and practical constraints while maintaining scientific rigor.

---------------


#  **Fine-Tuning Training**

##  **Overview: The Training Phase**
**What We're Doing**: Fine-tuning Llama 3.2-3B using QLoRA for price prediction

**Hardware Requirements**:
-  **LITE_MODE=True**: Free T4 GPU (16GB VRAM)
-  **LITE_MODE=False**: Paid A100 GPU (40GB+ VRAM)

**Training Scale**:
-  **Lite**: 20K samples, 1 epoch, ~30 minutes
-  **Full**: 800K samples, 3 epochs, ~24 hours
---
##  **Complete Training Script Walkthrough**
###  **Part 1: Environment Setup**
####  **Install Dependencies**
```python
!pip install -q --upgrade bitsandbytes==0.48.2 trl==0.25.1
!wget -q https://raw.githubusercontent.com/ed-donner/llm_engineering/main/week7/util.py -O util.py
```

**Critical Versions**:
-  `bitsandbytes==0.48.2`: Quantization library (version matters!)
-  `trl==0.25.1`: Transformer Reinforcement Learning (SFT trainer)

**Why These Versions?**:
- Newer versions may have breaking changes
- These are tested and stable
- Compatibility with Colab environment
---
####  **Imports**

```python
import os
import re
import math
from tqdm import tqdm
from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import  (
		AutoModelForCausalLM,
		AutoTokenizer,
		TrainingArguments,
		set_seed,
		BitsAndBytesConfig
		)

from datasets import load_dataset, Dataset, DatasetDict
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datetime import datetime
import matplotlib.pyplot as plt
```

  

**Key Libraries**:
-  `transformers`: HuggingFace model loading
-  `peft`: Parameter-Efficient Fine-Tuning (LoRA)
-  `trl`: Supervised Fine-Tuning trainer
-  `wandb`: Experiment tracking (Weight and Biases)
-  `bitsandbytes`: Quantization (imported implicitly)
---
##  **Part 2: Configuration Constants**
####  **Model and Project Setup**
```python
BASE_MODEL =  "meta-llama/Llama-3.2-3B"
PROJECT_NAME =  "price"
HF_USER =  "[Your username]" # CHANGE THIS to your HuggingFace username  
LITE_MODE =  True  # False for full training  
DATA_USER =  "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"
 

# Generate unique run name with timestamp
RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
if LITE_MODE:
RUN_NAME +=  "-lite"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"
```
**Explanation**:

-  `RUN_NAME`: Timestamp ensures unique model names
-  `HUB_MODEL_NAME`: Where model will be saved on HuggingFace
-  `LITE_MODE`: Controls dataset size and hyperparameters

*Example Run Name*: `price-2025-01-15_14.30.45-lite`

---

  

####  **Overall Hyperparameters**

```python
EPOCHS =  1  if LITE_MODE else  3
BATCH_SIZE =  32  if LITE_MODE else  256
MAX_SEQUENCE_LENGTH =  128
GRADIENT_ACCUMULATION_STEPS =  1
```
 **Explanation**:
**EPOCHS**:
-  **Lite**: 1 epoch (quick validation)
-  **Full**: 3 epochs (full training)
- More epochs = more training, risk of overfitting

**BATCH_SIZE**:
-  **Lite**: 32 (fits in T4 memory)
-  **Full**: 256 (requires A100)
- Larger batch = more stable gradients, faster training
 

**MAX_SEQUENCE_LENGTH**: 128
- Must match data preparation 
- Power of 2 for efficiency
- Includes prompt + completion
 

**GRADIENT_ACCUMULATION_STEPS**: 1
- Accumulate gradients over N steps before updating
- Simulates larger batch size
- Set to 1 = no accumulation (batch size is sufficient)
---

  

####  **QLoRA Hyperparameters**

  

```python

QUANT_4_BIT =  True
# LoRA rank
LORA_R =  32  if LITE_MODE else  256
LORA_ALPHA = LORA_R *  2  # 64 (lite) or 512 (full)

# Target modules
ATTENTION_LAYERS =  ["q_proj",  "v_proj",  "k_proj",  "o_proj"]
MLP_LAYERS =  ["gate_proj",  "up_proj",  "down_proj"]
TARGET_MODULES = ATTENTION_LAYERS if LITE_MODE else ATTENTION_LAYERS + MLP_LAYERS
LORA_DROPOUT =  0.1
```
**Deep Dive on LoRA Parameters**:
**LORA_R (Rank)**:
-  **Lite**: 32 (fewer trainable parameters)
-  **Full**: 256 (more capacity)
- Higher rank = more expressiveness, more memory
- Typical range: 8-256
  

**LORA_ALPHA**:
- Scaling factor for LoRA updates
- Rule of thumb: `ALPHA = R * 2`
- Controls magnitude of LoRA contribution
- Higher alpha = stronger LoRA influence


**TARGET_MODULES**:
-  **Attention layers**: Query, Key, Value, Output projections
-  **MLP layers**: Gate, Up, Down projections
-  **Lite**: Attention only (faster, less memory)
-  **Full**: Attention + MLP (better performance)

  

**Why These Choices?**:
- Attention layers: Most important for understanding context
- MLP layers: Additional capacity for complex patterns
- Trade-off: Performance vs. speed/memory

  

**LORA_DROPOUT**: 0.1
- Regularization to prevent overfitting
- 10% of LoRA weights randomly dropped during training
- Standard value, rarely needs tuning
---
####  **Training Hyperparameters**
```python
LEARNING_RATE =  1e-4  # 0.0001
WARMUP_RATIO =  0.01  # 1% of training
LR_SCHEDULER_TYPE =  'cosine'
WEIGHT_DECAY =  0.001
OPTIMIZER =  "paged_adamw_32bit"
```
 
**Explanation**:

**LEARNING_RATE**: 1e-4
- How fast model learns
- Too high: Unstable training, divergence
- Too low: Slow convergence
- 1e-4 is standard for fine-tuning

  
**WARMUP_RATIO**: 0.01
- Gradually increase LR for first 1% of training
- Prevents early instability
- Helps model settle into good optimization path
  
**LR_SCHEDULER_TYPE**: *'cosine'*
- Learning rate decreases following cosine curve
- Smooth decay from peak to near-zero
- Better than linear decay for fine-tuning

**WEIGHT_DECAY**: 0.001
- L2 regularization
- Prevents weights from growing too large
- Reduces overfitting

**OPTIMIZER**: "paged_adamw_32bit"
-  **AdamW**: Adam with Weight Decay
-  **Paged**: Offloads optimizer states to CPU when needed
-  **32bit**: Full precision for optimizer (not quantized)
- Saves GPU memory while maintaining training quality
---
####  **GPU Capability Detection**
```python
capability = torch.cuda.get_device_capability()
use_bf16 = capability[0]  >=  8
```

**Explanation**:
-  **Capability >= 8**: A100, H100 (supports bfloat16)
-  **Capability < 8**: T4, V100 (use float16)
-  **bfloat16**: Better numerical stability than float16
-  **float16**: Older GPUs, slightly less stable

**Why This Matters**:
- bfloat16 has larger dynamic range
- Reduces risk of overflow/underflow
- Automatically selects best precision for your GPU

---

  

####  **Tracking Configuration**

```python
VAL_SIZE =  500  if LITE_MODE else  1000
LOG_STEPS =  5  if LITE_MODE else  10
SAVE_STEPS =  100  if LITE_MODE else  200
LOG_TO_WANDB =  True
```


**Explanation**:

**VAL_SIZE**:
- Number of validation samples to evaluate
-  **Lite**: 500 (faster evaluation)

-  **Full**: 1000 (more reliable metrics)

- Subset of full validation set for speed


**LOG_STEPS**:
- Log metrics every N training steps
-  **Lite**: Every 5 steps (frequent feedback)
-  **Full**: Every 10 steps (less overhead)

- Appears in Weights & Biases dashboard


**SAVE_STEPS**:
- Save checkpoint every N steps
-  **Lite**: Every 100 steps
-  **Full**: Every 200 steps

- Allows resuming if training interrupted

  **LOG_TO_WANDB**: True
- Enable Weights & Biases tracking
- Set to False for local-only training
- Provides real-time training visualization
---
  

###  **Part 3: Authentication**
 
####  **HuggingFace Login**


```python
hf_token = userdata.get('HF_TOKEN')
login(hf_token,  add_to_git_credential=True)
```

**Why Needed**:
- Download gated models (Llama requires agreement)
- Upload fine-tuned model to your account
- Access private datasets
---
####  **Weights & Biases Login**
```python
wandb_api_key = userdata.get('WANDB_API_KEY')
os.environ["WANDB_API_KEY"]  = wandb_api_key
wandb.login()
# Configure W&B
os.environ["WANDB_PROJECT"]  = PROJECT_NAME
os.environ["WANDB_LOG_MODEL"]  =  "false"  # Don't upload model to W&B
os.environ["WANDB_WATCH"]  =  "false"  # Don't watch gradients
```

**Setup Steps**:

1. Create account at https://wandb.ai
2. Get API key: Settings → API Keys
3. Add to Colab secrets: `WANDB_API_KEY`
**Configuration**:
-  `WANDB_PROJECT`: Groups runs together
-  `WANDB_LOG_MODEL`: "false" (model goes to HuggingFace, not W&B)
-  `WANDB_WATCH`: "false" (saves memory, we don't need gradient tracking)

**Why Use W&B?**:
- Real-time training metrics
- Loss curves, learning rate schedules
- Compare multiple runs
- Share results with team
---


###  **Part 4: Load Data**
```python
dataset = load_dataset(DATASET_NAME)
train = dataset['train']
val = dataset['val'].select(range(VAL_SIZE))
test = dataset['test']
# Optional: Reduce training set for quick experiments
# train = train.select(range(10000))
if LOG_TO_WANDB:
wandb.init(project=PROJECT_NAME,  name=RUN_NAME)
```

**Explanation**:
- Loads from HuggingFace Hub 
-  `val.select(range(VAL_SIZE))`: Use subset for faster evaluation
-  `test`: Full test set (not used during training)
-  `wandb.init()`: Start tracking run

**Optional Reduction**:
- Uncomment to train on 10K samples instead of full dataset
- Useful for hyperparameter tuning
- Faster iteration cycles
---
###  **Part 5: Quantization Configuration**

```python
if QUANT_4_BIT:
	quant_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_use_double_quant=True,
	bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
	bnb_4bit_quant_type="nf4"
	)
else:
	quant_config = BitsAndBytesConfig(
	load_in_8bit=True,
	bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
	)

```

**4-bit Quantization Parameters**:

**load_in_4bit**: True
- Quantize weights to 4 bits
- Reduces memory by ~75%
- Minimal quality loss

**bnb_4bit_use_double_quant**: True
- Quantize the quantization constants
- Additional memory savings
- "Nested quantization"
 
**bnb_4bit_compute_dtype**: bfloat16 or float16
- Precision for computations (not storage)
- Dequantize to this precision during forward pass
- Matches GPU capability

**bnb_4bit_quant_type**: "nf4"
- NormalFloat4: Optimal for normally distributed weights
- Better than uniform quantization
- Designed specifically for neural networks

**8-bit Alternative**:
- Less aggressive compression
- Slightly better quality
- More memory usage
- Rarely needed for fine-tuning
---
###  **Part 6: Load Model and Tokenizer**

```python
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,  trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side =  "right"
base_model = AutoModelForCausalLM.from_pretrained(
BASE_MODEL,
quantization_config=quant_config,
device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
print(f"Memory footprint: {base_model.get_memory_footprint()  /  1e6:.1f} MB")
```
**Tokenizer Setup** :
-  `pad_token = eos_token`: Standard for decoder-only models
-  `padding_side = "right"`: Pad after sequence
- Prevents warnings during training

**Model Loading**:
-  `quantization_config`: Apply 4-bit quantization
-  `device_map="auto"`: Automatically distribute across GPUs
- Handles multi-GPU setups transparently

**Memory Check**:
- Expected: ~2200 MB (2.2 GB)
- Confirms quantization worked
- Base model before LoRA adapters
---

###  **Part 7: LoRA Configuration**

```python
lora_parameters = LoraConfig(
		lora_alpha=LORA_ALPHA,
		lora_dropout=LORA_DROPOUT,
		r=LORA_R,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules=TARGET_MODULES,
		)
```
**Parameter Breakdown**:

**lora_alpha**: 64 (lite) or 512 (full)
- Scaling factor for LoRA updates
- Formula: `alpha / r` determines update magnitude
- Higher alpha = stronger LoRA influence

**lora_dropout**: 0.1
- Dropout rate for LoRA layers
- Regularization technique
- Prevents overfitting

 **r**: 32 (lite) or 256 (full)
- Rank of LoRA matrices
- Number of trainable parameters per layer
- Higher rank = more capacity

  

**bias**: "none"
- Don't train bias terms
- Reduces parameters
- Standard for LoRA

  

**task_type**: "CAUSAL_LM"
- Causal Language Modeling
- Autoregressive generation
- Predicts next token

**target_modules**: Attention (lite) or Attention+MLP (full)
- Which layers get LoRA adapters
- More modules = better performance, more memory
  

**LoRA Math**:
- Original weight: `W` (frozen)
- LoRA update: `ΔW = B @ A` where `A` is `r × d`, `B` is `d × r`
- Final weight: `W + (alpha/r) * ΔW`
- Trainable params: `2 * r * d` per layer (much less than `d * d`)
---
###  **Part 8: Training Configuration**
```python
train_parameters = SFTConfig(
output_dir=PROJECT_RUN_NAME,
num_train_epochs=EPOCHS,
per_device_train_batch_size=BATCH_SIZE,
per_device_eval_batch_size=1,
gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
optim=OPTIMIZER,
save_steps=SAVE_STEPS,
save_total_limit=10,
logging_steps=LOG_STEPS,
learning_rate=LEARNING_RATE,
weight_decay=WEIGHT_DECAY,
fp16=not use_bf16,
bf16=use_bf16,
max_grad_norm=0.3,
max_steps=-1,
warmup_ratio=WARMUP_RATIO,
group_by_length=True,
lr_scheduler_type=LR_SCHEDULER_TYPE,
report_to="wandb"  if LOG_TO_WANDB else  None,
run_name=RUN_NAME,
max_length=MAX_SEQUENCE_LENGTH,
save_strategy="steps",
hub_strategy="every_save",
push_to_hub=True,
hub_model_id=HUB_MODEL_NAME,
hub_private_repo=True,
eval_strategy="steps",
eval_steps=SAVE_STEPS
)
```

  

**Critical Parameters Explained**:

**output_dir**: Local checkpoint directory
- Saves checkpoints during training
- Can resume from here if interrupted

**per_device_eval_batch_size**: 1
- Evaluate one sample at a time
- Saves memory during evaluation
- Training batch size can be larger

**save_total_limit**: 10
- Keep only last 10 checkpoints
- Prevents disk space issues
- Older checkpoints automatically deleted

**max_grad_norm**: 0.3
- Gradient clipping threshold
- Prevents exploding gradients
- Stabilizes training

**max_steps**: -1
- Train for full epochs (not fixed steps)
- -1 means "use num_train_epochs"
  

**group_by_length**: True
- Group similar-length sequences in batches
- Reduces padding waste
- Faster training

**save_strategy**: "steps"
- Save based on step count (not epochs)
- More granular control

**hub_strategy**: "every_save"
- Upload to HuggingFace every time we save
- Automatic backup
- Can resume from hub if Colab crashes

**push_to_hub**: True
- Enable automatic uploads
- Model appears in your HuggingFace account

**hub_private_repo**: True

- Model is private (not public)
- Change to False to share publicly

**eval_strategy**: "steps"

- Evaluate every `eval_steps` steps
- Tracks validation loss during training
---
###  **Part 9: Create Trainer**
```python
fine_tuning = SFTTrainer(
model=base_model,
train_dataset=train,
eval_dataset=val,
peft_config=lora_parameters,
args=train_parameters
)
```

**SFTTrainer** (Supervised Fine-Tuning Trainer):
- Combines model, data, and configs
- Handles training loop automatically
- Integrates LoRA, quantization, and logging
- From `trl` library (Transformer Reinforcement Learning)

**What It Does**:
1. Adds LoRA adapters to frozen base model
2. Sets up optimizer and scheduler
3. Handles batching and gradient accumulation
4. Logs metrics to W&B
5. Saves checkpoints
6. Uploads to HuggingFace Hub
---

###  **Part 10: Training Execution**
```python
# Fine-tune!
fine_tuning.train()

# Push our fine-tuned model to Hugging Face
fine_tuning.model.push_to_hub(PROJECT_RUN_NAME,  private=True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")

if LOG_TO_WANDB:
	wandb.finish()
```

**Training Process**:
1.  **Initialization**: Sets up LoRA adapters
2.  **Training loop**: Iterates through batches
3.  **Forward pass**: Compute predictions
4.  **Loss calculation**: Compare with targets
5.  **Backward pass**: Compute gradients
6.  **Optimizer step**: Update LoRA weights
7.  **Logging**: Send metrics to W&B
8.  **Checkpointing**: Save every SAVE_STEPS
9.  **Evaluation**: Run on validation set every EVAL_STEPS

**What You'll See**:
```
Step 5: loss=2.345, lr=0.0001
Step 10: loss=2.123, lr=0.00009
...
Saving checkpoint at step 100
Uploading to HuggingFace...
```

**Final Upload**:
-  `push_to_hub()`: Ensures final model is saved
- Even if training completes, explicitly push
-  `wandb.finish()`: Closes W&B run cleanly
---

##  **Training Interruption & Resumption**
###  **If Colab Stops Your Training**
**Why It Happens**:
- Free tier: Google reclaims resources when busy
- Paid tier: 24-hour limit
- Network issues
- Browser closed

**How to Resume**:

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
BASE_MODEL,
quantization_config=quant_config,
device_map="auto",
)

# Load LoRA adapters from last checkpoint
fine_tuned_model = PeftModel.from_pretrained(
base_model,
f"{HF_USER}/{PROJECT_RUN_NAME}", # Your HuggingFace model
is_trainable=True  # CRITICAL: Allow continued training
)
  
# Create new trainer with resumed model
fine_tuning = SFTTrainer(
model=fine_tuned_model,
train_dataset=train,
eval_dataset=val,
peft_config=lora_parameters,
args=train_parameters
)

# Continue training
fine_tuning.train(resume_from_checkpoint=True)
```
 
**Key Point**: `is_trainable=True`

- Without this, LoRA adapters are frozen
- Training will fail or do nothing
- Easy to forget, critical to remember
---

##  **Monitoring Training**
###  **Weights & Biases Dashboard**
**Key Metrics to Watch**:
1.  **Training Loss**:
- Should decrease steadily
- Jagged line is normal (batch-to-batch variation)
- Smoothed trend should go down

  

2.  **Validation Loss**:

- Should decrease with training loss
- If increases while training loss decreases: **overfitting**
- Gap between train/val loss indicates generalization

  
3.  **Learning Rate**:
- Starts low (warmup)
- Increases to peak
- Decreases following cosine schedule
- Should reach near-zero by end

  

4.  **Gradient Norm**:
- Should be stable
- Spikes indicate instability
- Clipped at max_grad_norm (0.3)

  **Healthy Training**:
```
Step 100: train_loss=2.1, val_loss=2.3, lr=0.0001
Step 200: train_loss=1.8, val_loss=2.0, lr=0.00009
Step 300: train_loss=1.6, val_loss=1.8, lr=0.00008
```

**Overfitting**:
```
Step 100: train_loss=2.1, val_loss=2.3
Step 200: train_loss=1.5, val_loss=2.4 ← val loss increased!
Step 300: train_loss=1.2, val_loss=2.6 ← getting worse
```

**Solution to Overfitting**:
- Increase dropout
- Reduce epochs
- Add more training data
- Increase weight decay
---

##  **Hyperparameter Tuning Guide**
###  **What to Tune First**
 

**Priority 1: Data-Related**
1.  **Token cutoff** (110): Try 90, 110, 130
2.  **Rounding strategy**: With/without
3.  **Training set size**: 10K, 20K, full

**Priority 2: LoRA**
1.  **LORA_R**: 16, 32, 64, 128, 256
2.  **TARGET_MODULES**: Attention only vs. Attention+MLP
3.  **LORA_ALPHA**: Try `R`, `R*2`, `R*4`
 

**Priority 3: Training**
1.  **LEARNING_RATE**: 5e-5, 1e-4, 2e-4
2.  **BATCH_SIZE**: 16, 32, 64 (memory permitting)
3.  **EPOCHS**: 1, 2, 3, 5

**Priority 4: Regularization**

1.  **LORA_DROPOUT**: 0.05, 0.1, 0.2
2.  **WEIGHT_DECAY**: 0.0001, 0.001, 0.01
 

###  **Systematic Approach**

1.  **Baseline**: Run with default settings
2.  **One at a time**: Change one parameter
3.  **Compare**: Use W&B to compare runs
4.  **Iterate**: Keep improvements, discard regressions
5.  **Document**: Note what works and why
 ---

##  **Memory Optimization Tricks**
###  **If You Run Out of Memory**
**Reduce Batch Size**:
```python
BATCH_SIZE =  16  # Instead of 32
GRADIENT_ACCUMULATION_STEPS =  2  # Simulate batch_size=32
```

  

**Reduce LoRA Rank**:
```python
LORA_R =  16  # Instead of 32
```
**Target Fewer Modules**:
```python
TARGET_MODULES =  ["q_proj",  "v_proj"]  # Only Q and V
```
**Reduce Sequence Length**:
```python
MAX_SEQUENCE_LENGTH =  64  # Instead of 128
# Must re-prepare data with new cutoff
```
**Use 8-bit Instead of 4-bit**:
```python
QUANT_4_BIT =  False  # Uses more memory but might be more stable
```
---

##  **Common Errors and Solutions**
###  **Error: "CUDA out of memory"**

**Solution**:
1. Reduce BATCH_SIZE
2. Increase GRADIENT_ACCUMULATION_STEPS
3. Reduce LORA_R
4. Use fewer TARGET_MODULES
5. Restart runtime and clear cache


###  **Error: "Token not found"**
**Solution**:
- Check HF_TOKEN is set correctly
- Regenerate token with write permissions
- Accept Llama license agreement on HuggingFace

###  **Error: "Dataset not found"**

**Solution**:
- Verify dataset name matches 
- Check DATA_USER is correct
- Ensure dataset is public or you're logged in

###  **Error: "Loss is NaN"**

**Solution**:
- Reduce LEARNING_RATE (try 5e-5)
- Check data quality (no NaN values)
- Increase WARMUP_RATIO (try 0.05)
- Reduce max_grad_norm (try 0.1)
---
##  **Complete Training Checklist**
###  **Before Starting**:

- [x] HF_TOKEN configured
- [x] WANDB_API_KEY configured
- [x] Llama license accepted on HuggingFace
- [x] Dataset uploaded 
- [x] GPU selected (T4 for lite, A100 for full)
- [x] LITE_MODE set correctly
- [x] HF_USER changed to your username
 
###  **During Training**:
- [x] W&B dashboard open
- [x] Training loss decreasing
- [x] Validation loss decreasing
- [x] No CUDA OOM errors
- [x] Checkpoints saving to HuggingFace
- [x] Learning rate following schedule

  
###  **After Training**:
- [x] Final model pushed to hub
- [x] W&B run finished cleanly
- [x] Model appears in HuggingFace account
---

##  **Key Takeaways**
1.  **QLoRA = Quantization + LoRA**: Efficient fine-tuning for large models
2.  **Hyperparameters matter**: But data quality matters more
3.  **Start small**: Lite mode validates approach quickly
4.  **Monitor closely**: W&B dashboard is your friend
5.  **Save often**: Colab can stop anytime
6.  **Experiment systematically**: One change at a time
7.  **Document everything**: You'll forget what worked
---
##  **Next Steps**
  **Evaluation**: 
- Load fine-tuned model
- Run on test set
- Compare with base model
- Compare with GPT-5 ($44.74 error)
- Analyze results
- Iterate if needed

**Success Criteria**:
- Beat base model significantly
- Approach or beat GPT-5 performance
- Model generalizes to test set
- Training was stable


-------------------
# Testing Fine-Tuned LLM Models 
---

## 1. Core Concepts

### 1.1 The Four-Step Training Loop

During training (NOT testing), LLMs undergo a continuous cycle of four critical steps. Understanding this loop is essential for comprehending what happens during fine-tuning and why testing is fundamentally different.

#### Step 1: Forward Pass

**What Actually Happens**:
- **Common Misconception**: "Forward pass predicts the next token"
- **Reality**: Forward pass outputs a **probability distribution over ALL 128,000 tokens in vocabulary**
![Forward pass](./img/forward_pass.jpg)

**Technical Details**:
```
Input: Prompt tokens → Neural Network Layers → LM Head (final layer)
Output: 128,000-dimensional vector where each element = probability of that token being next
```

**Key Properties**:
- All probabilities sum to 1.0 (normalized distribution)
- Each token gets assigned a probability (0.0 to 1.0)
- Model doesn't "pick" a token yet—it just calculates probabilities

**Example**:
```
Prompt: "The price is"
Model Output (simplified):
  Token 99 ("$45"): 15% probability ← Highest
  Token 89 ("$39"): 8% probability  ← Actual correct answer
  Token 42 ("high"): 3% probability
  ... (128,000 total tokens)
```

**Temperature Sampling** (how we extract prediction):
- **Temperature = 0**: Pick token with maximum probability (deterministic)
  - In example above: Always pick token 99 (15%)
- **Temperature > 0**: Sample from probability distribution (stochastic)
  - Higher probability tokens more likely, but not guaranteed

---

#### Step 2: Loss Calculation (Cross-Entropy Loss)

**The "Aha Moment"**:
![Forward pass](./img/los_calculation.jpg)

> "We actually don't consider that 99 token at all. It doesn't have any bearing in the loss calculation... Rather we say what probability did the model give to the token that we actually wanted?"


**Formula**: `Loss = -log(P(correct_token))`
![Forward pass](./img/los_calculation_log.jpg)

**How It Works**:
1. Ignore the predicted token (token 99 with 15%)
2. Look ONLY at probability assigned to **actual correct token** (token 89)
3. Calculate: `-log(P(token_89))`

**Mathematical Properties**:
- If P(correct_token) = 100% (1.0) → Loss = -log(1.0) = 0 ✓ Perfect!
- If P(correct_token) = 50% (0.5) → Loss = -log(0.5) = 0.69
- If P(correct_token) = 10% (0.1) → Loss = -log(0.1) = 2.30
- If P(correct_token) = 1% (0.01) → Loss = -log(0.01) = 4.61

**Why Negative Log?** :
- "It just sort of works out really nicely with the maths"
- Elegant for backpropagation calculations (chain rule)
- Meets all required properties:
  - Perfect prediction (100%) → Zero loss
  - Lower probability → Higher loss
  - Differentiable (smooth gradient)

**Why Not Simple Difference?**
- LLMs predict tokens, not numbers
- Cross-entropy works for ANY output: prices, text, code, etc.
- "Standard way of calculating loss for a classification problem"
- We're classifying the most likely next token from 128,000 options

**Concrete Example**:
```
Scenario: Predicting product price
Prompt: "Title: Laptop... Predict price:"

Forward Pass Output:
  Token 5234 ("$999"): 12% ← Model's top pick
  Token 5189 ("$899"): 10% ← Actual correct price
  Token 5301 ("$1099"): 8%
  ... (rest of 128,000 tokens)

Loss Calculation:
  Loss = -log(0.10) = 2.30
  
Note: Token 5234 (12%) is IGNORED despite being highest probability!
```

---

#### Step 3: Backpropagation (Backward Pass)

**Purpose**: Calculate how to adjust model parameters to reduce loss

**The Chain Rule Magic**:
- Works backwards from output to input layers
- Calculates gradients as a function of gradients that came before
- "Repeatedly applying the chain rule working backwards"

![Forward pass](./img/backprop.jpg)

**Technical Process**:
```
1. Start at output: ∂Loss/∂output
2. Layer N: ∂Loss/∂weights_N = ∂Loss/∂output_N × ∂output_N/∂weights_N
3. Layer N-1: ∂Loss/∂weights_(N-1) = ∂Loss/∂output_(N-1) × ∂output_(N-1)/∂weights_(N-1)
4. Continue backwards through all layers to input
```

**Why It's Revolutionary**:
- **Efficiency**: "Operation that would have been very time consuming can happen rapidly"
- **Parallelization**: Calculations happen efficiently in parallel on GPUs
- **Scalability**: Enables training billion-parameter models

**Historical Context**:
- Algorithm invented in 1970s
- **1986 Paper**: Geoff Hinton ("godfather of modern AI") popularized for neural networks
- "One of the secrets that has made training neural networks so effective"

**What We Get**:
- Gradient for every trainable parameter (billions of gradients)
- Each gradient indicates:
  - Direction to adjust parameter (positive/negative)
  - Magnitude of adjustment needed

**Example**:
```
Parameter: Weight in layer 15, position [234, 567]
Gradient: -0.0023

Interpretation:
- Negative gradient → Increase this weight to reduce loss
- Magnitude 0.0023 → Small adjustment needed
```

---

#### Step 4: Optimization

**Purpose**: Update model parameters using calculated gradients

**Update Formula** (simplified):
```
new_weight = old_weight - (learning_rate × gradient)
```
![Forward pass](./img/Optimization.jpg)

**Key Concepts**:

1. **Learning Rate**: Controls step size
   - Too large: Overshoot optimal values (unstable training)
   - Too small: Training too slow (may never converge)
   - Typical: 1e-4 to 1e-5 for fine-tuning

2. **LoRA/PEFT Context**:
   - Only adapter weights updated (1-2% of parameters)
   - Base model frozen (saves computation)
   - "All of the adapters" = LoRA matrices

**Concrete Example**:
```
Before Optimization:
  Weight_A = 0.5234
  Gradient = -0.0023
  Learning_rate = 0.0001

Calculation:
  new_weight = 0.5234 - (0.0001 × -0.0023)
  new_weight = 0.5234 + 0.00000023
  new_weight = 0.52340023

Result: Weight slightly increased to reduce loss next iteration
```

**The Loop Continues**:
```
Iteration 1: Forward → Loss=2.30 → Backprop → Optimize
Iteration 2: Forward → Loss=2.15 → Backprop → Optimize
Iteration 3: Forward → Loss=2.01 → Backprop → Optimize
...
Iteration 1000: Forward → Loss=0.45 → Backprop → Optimize
```

---

### 1.2 Training vs Testing: Critical Distinction

**Training Loop** (4 steps):
```
1. Forward Pass → Probability distribution
2. Loss Calculation → -log(P(correct_token))
3. Backpropagation → Calculate gradients
4. Optimization → Update weights
↓
Repeat thousands of times
```

**Testing/Inference** (1 step only):
```
1. Forward Pass → Probability distribution → Sample token
↓
Done! No loss, no backprop, no optimization
```

**Why Testing is Different**:
- **No ground truth needed**: We don't know correct answer yet
- **No gradients**: `torch.no_grad()` disables gradient tracking
- **No weight updates**: Model frozen in final state
- **Memory efficient**: No gradient storage (saves ~2x memory)
- **Faster**: Skip backprop and optimization steps

**Code Manifestation**:
```python
# TRAINING (not in our test code)
for batch in train_data:
    outputs = model(batch.input_ids)  # Step 1: Forward
    loss = criterion(outputs, batch.labels)  # Step 2: Loss
    loss.backward()  # Step 3: Backprop
    optimizer.step()  # Step 4: Optimize

# TESTING (our code)
with torch.no_grad():  # Disable gradients
    outputs = model(batch.input_ids)  # Step 1 ONLY: Forward
    prediction = outputs.argmax()  # Extract prediction
```

---

## 2. Testing Pipeline Architecture

### 2.1 Why Cross-Entropy Loss Isn't Used in Testing

**Summary of Key Properties**:
- Formula: `-log(P(correct_token))`
- Perfect prediction (P=1.0) → Loss = 0
- Worse prediction (P→0) → Loss → ∞
- Universal: Works for any token prediction task
- Efficient: Mathematically elegant for gradient calculations

**Why This Matters for Testing**:
- During testing, we DON'T calculate cross-entropy loss
- We use different metrics: MAE, MSE, R² (see Section 5.7)
- Cross-entropy is training-only metric
- Understanding it helps interpret training logs and convergence

**No Gradient Computation**: Uses `torch.no_grad()` context manager to disable gradient tracking (saves memory and computation)

---

## 3. Code Analysis

### 3.1 Configuration Constants

```python
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "ed-donner"  # Replace with your HuggingFace username
LITE_MODE = False
```

**Purpose**: Define model identifiers and operational mode
- `BASE_MODEL`: Foundation model before fine-tuning (3 billion parameters)
- `LITE_MODE`: Toggle between lite dataset (faster testing) vs full dataset (production evaluation)

**Dataset Selection Logic**:
```python
DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"
```
- Lite: Smaller dataset for rapid iteration
- Full: Complete dataset for final evaluation

**Model Versioning**:
```python
if LITE_MODE:
  RUN_NAME = "2025-11-30_15.10.55-lite"
  REVISION = None
else:
  RUN_NAME = "2025-11-28_18.47.07"
  REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"
```
- `RUN_NAME`: Timestamp-based identifier for training run
- `REVISION`: Git commit hash for reproducibility (specific model checkpoint)
- `HUB_MODEL_NAME`: Full path to HuggingFace Hub model (`{HF_USER}/{PROJECT_NAME}-{RUN_NAME}`)

### 3.2 Quantization Configuration (QLoRA)

```python
QUANT_4_BIT = True
capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8
```

**GPU Capability Detection**:
- Checks CUDA compute capability (e.g., 8.0 for A100, 8.6 for RTX 3090)
- `use_bf16 = True` if capability ≥ 8 (modern GPUs support bfloat16)
- bfloat16: Better numerical stability than float16 for LLMs

**4-bit Quantization Setup**:
```python
if QUANT_4_BIT:
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Quantize quantization constants
    bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    bnb_4bit_quant_type="nf4"  # NormalFloat4 - optimized for normally distributed weights
  )
```

**Technical Details**:
- **4-bit quantization**: Reduces memory by 75% vs 16-bit (4/16 = 0.25)
- **Double quantization**: Further compresses quantization constants themselves
- **NF4 (NormalFloat4)**: Specialized 4-bit format for neural network weights (assumes normal distribution)
- **Compute dtype**: Actual computation happens in bfloat16/float16 (dequantized on-the-fly)

**8-bit Fallback**:
```python
else:
  quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
  )
```
- Used if 4-bit causes accuracy issues (rare)
- 50% memory reduction vs 16-bit

### 3.3 Tokenizer Loading

```python
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

**Purpose**: Convert text to token IDs and vice versa
- `trust_remote_code=True`: Allow custom tokenizer code from HuggingFace
- `pad_token = eos_token`: Use end-of-sequence token for padding (Llama models lack dedicated pad token)
- `padding_side = "right"`: Add padding tokens to right side of sequence (standard for causal LMs)

### 3.4 Base Model Loading

```python
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",  # Automatically distribute layers across available GPUs
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
```

**Key Points**:
- `AutoModelForCausalLM`: Loads decoder-only transformer (GPT-style architecture)
- `quantization_config`: Applies QLoRA quantization during loading
- `device_map="auto"`: Intelligent layer distribution (handles multi-GPU, CPU offloading)
- Synchronizes pad token ID between tokenizer and model generation config

### 3.5 PEFT Adapter Loading (Critical Step)

```python
if REVISION:
  fine_tuned_model = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME, revision=REVISION)
else:
  fine_tuned_model = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME)
```

**PEFT (Parameter-Efficient Fine-Tuning) Integration**:
- Loads LoRA adapters on top of frozen base model
- `base_model`: Frozen 3B parameter foundation (quantized)
- `HUB_MODEL_NAME`: Path to fine-tuned adapter weights (~1-2% of base model size)
- `revision`: Specific checkpoint for reproducibility

**Memory Efficiency**:
```python
print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")
```
- Typical footprint: ~3-4 GB for 3B model with 4-bit quantization
- Without quantization: ~12 GB (4x larger)

---

## 4. Inference Function

### 4.1 Model Prediction Logic

```python
def model_predict(item):
    inputs = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = fine_tuned_model.generate(**inputs, max_new_tokens=8)
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids)
```

**Step-by-Step Breakdown**:

1. **Tokenization**:
   ```python
   inputs = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
   ```
   - Converts text prompt to token IDs
   - `return_tensors="pt"`: Returns PyTorch tensors
   - `.to("cuda")`: Moves tensors to GPU
   - Output: `{"input_ids": tensor([[token1, token2, ...]]), "attention_mask": tensor([[1, 1, ...]])}`

2. **Generation (No Gradient Tracking)**:
   ```python
   with torch.no_grad():
       output_ids = fine_tuned_model.generate(**inputs, max_new_tokens=8)
   ```
   - `torch.no_grad()`: Disables gradient computation (inference mode)
   - `max_new_tokens=8`: Generate up to 8 new tokens (sufficient for price like "$123.45")
   - `**inputs`: Unpacks input_ids and attention_mask
   - Output: Full sequence (prompt + generated tokens)

3. **Extract Generated Tokens**:
   ```python
   prompt_len = inputs["input_ids"].shape[1]
   generated_ids = output_ids[0, prompt_len:]
   ```
   - `prompt_len`: Number of tokens in original prompt
   - Slice to get only newly generated tokens (exclude prompt echo)
   - `[0, ...]`: First (and only) batch element

4. **Decode to Text**:
   ```python
   return tokenizer.decode(generated_ids)
   ```
   - Converts token IDs back to human-readable text
   - Returns string like "$45.99" or "45.99"

**Why max_new_tokens=8?**
- Price format: "$" (1 token) + digits (2-4 tokens) + "." (1 token) + cents (1-2 tokens)
- Provides buffer for model verbosity (e.g., "$45.99\n" or "The price is $45.99")

### 4.2 Evaluation Execution

```python
set_seed(42)
evaluate(model_predict, test)
```

**Purpose**: Run systematic evaluation on test dataset
- `set_seed(42)`: Ensures reproducible sampling (if temperature > 0)
- `evaluate()`: Utility function from `util.py` (see Section 5)
- `test`: Test split from HuggingFace dataset

---

## 5. Code Analysis: util.py (Evaluation Framework)

### 5.1 Tester Class Architecture

```python
class Tester:
    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE):
        self.predictor = predictor  # Function that takes datapoint, returns prediction
        self.data = data            # HuggingFace dataset
        self.title = title or self.make_title(predictor)
        self.size = size            # Number of test samples (default 200)
        self.titles = []            # Product titles
        self.guesses = []           # Model predictions
        self.truths = []            # Ground truth prices
        self.errors = []            # Absolute errors
        self.colors = []            # Color codes for visualization
```

**Design Pattern**: Encapsulates evaluation logic with state tracking

### 5.2 Post-Processing Predictions

```python
@staticmethod
def post_process(value):
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        return float(match.group()) if match else 0
    else:
        return value
```

**Purpose**: Extract numeric price from model output
- **Input**: Raw model output (e.g., "$45.99", "The price is 45.99", "45.99\n")
- **Process**:
  1. Remove currency symbols and commas
  2. Regex pattern: `[-+]?\d*\.\d+|\d+` matches floats or integers
  3. Extract first numeric match
- **Output**: Float (e.g., 45.99)
- **Fallback**: Returns 0 if no number found (handles model failures gracefully)

**Regex Breakdown**:
- `[-+]?`: Optional sign
- `\d*\.\d+`: Decimal number (e.g., 45.99, .99)
- `|`: OR
- `\d+`: Integer (e.g., 45)

### 5.3 Error Categorization

```python
def color_for(self, error, truth):
    if error < 40 or error / truth < 0.2:
        return "green"   # Good: <$40 or <20% error
    elif error < 80 or error / truth < 0.4:
        return "orange"  # Acceptable: <$80 or <40% error
    else:
        return "red"     # Poor: ≥$80 or ≥40% error
```

**Purpose**: Visual categorization of prediction quality
- **Absolute threshold**: $40 (green), $80 (orange)
- **Relative threshold**: 20% (green), 40% (orange)
- **Logic**: Uses OR (either condition triggers color)
- **Example**: $10 error on $20 item = 50% error → RED (fails both thresholds)

### 5.4 Single Datapoint Evaluation

```python
def run_datapoint(self, i):
    datapoint = self.data[i]
    value = self.predictor(datapoint)  # Call model_predict()
    guess = self.post_process(value)
    truth = float(datapoint["completion"])
    error = abs(guess - truth)
    color = self.color_for(error, truth)
    pieces = datapoint["prompt"].split("Title: ")
    title = pieces[1].split("\n")[0] if len(pieces) > 1 else pieces[0]
    title = title if len(title) <= 40 else title[:40] + "..."
    return title, guess, truth, error, color
```

**Workflow**:
1. Fetch datapoint from dataset
2. Get model prediction (calls `model_predict()`)
3. Convert prediction to float
4. Extract ground truth from `completion` field
5. Calculate absolute error
6. Assign color category
7. Extract product title for display (truncate to 40 chars)

**Dataset Structure Assumption**:
```python
{
  "prompt": "Title: Product Name\nDescription: ...\nPredict price:",
  "completion": "45.99"
}
```

### 5.5 Visualization: Scatter Plot

```python
def chart(self, title):
    df = pd.DataFrame({
        "truth": self.truths,
        "guess": self.guesses,
        "title": self.titles,
        "error": self.errors,
        "color": self.colors,
    })
    
    df["hover"] = [
        f"{t}\nGuess=${g:,.2f} Actual=${y:,.2f}"
        for t, g, y in zip(df["title"], df["guess"], df["truth"])
    ]
    
    max_val = float(max(df["truth"].max(), df["guess"].max()))
    
    fig = px.scatter(
        df, x="truth", y="guess", color="color",
        color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
        title=title,
        labels={"truth": "Actual Price", "guess": "Predicted Price"},
        width=800, height=600,
    )
```

**Key Features**:
- **X-axis**: Ground truth prices
- **Y-axis**: Model predictions
- **Color coding**: Green/orange/red by error magnitude
- **Hover text**: Product title + predicted/actual prices
- **Reference line**: y=x diagonal (perfect predictions)

**Interpretation**:
- Points on diagonal: Perfect predictions
- Points above diagonal: Overestimation
- Points below diagonal: Underestimation
- Scatter spread: Model uncertainty

### 5.6 Visualization: Error Trend Chart

```python
def error_trend_chart(self):
    n = len(self.errors)
    
    # Running mean
    running_sums = list(accumulate(self.errors))
    x = list(range(1, n + 1))
    running_means = [s / i for s, i in zip(running_sums, x)]
    
    # Running standard deviation
    running_squares = list(accumulate(e * e for e in self.errors))
    running_stds = [
        math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0
        for i, sq_sum, mean in zip(x, running_squares, running_means)
    ]
    
    # 95% confidence interval
    ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, running_stds)]
    upper = [m + c for m, c in zip(running_means, ci)]
    lower = [m - c for m, c in zip(running_means, ci)]
```

**Statistical Calculations**:

1. **Running Mean**: 
   - Formula: `μ_n = (Σ errors) / n`
   - Shows average error as more samples evaluated

2. **Running Standard Deviation**:
   - Formula: `σ_n = sqrt(E[X²] - E[X]²)`
   - Measures error variability

3. **95% Confidence Interval**:
   - Formula: `CI = 1.96 × (σ / sqrt(n))`
   - 1.96 = z-score for 95% confidence (normal distribution)
   - Narrows as n increases (more samples = more confidence)

**Visualization**:
- Main line: Running mean error
- Shaded band: 95% confidence interval
- Title: Final mean ± CI (e.g., "Error: $62.51 ± $3.45")

**Interpretation**:
- Converging mean: Model performance stabilizes
- Narrowing CI: Increasing confidence in estimate
- Flat trend: Consistent performance across dataset

### 5.7 Metrics Reporting

```python
def report(self):
    average_error = sum(self.errors) / self.size
    mse = mean_squared_error(self.truths, self.guesses)
    r2 = r2_score(self.truths, self.guesses) * 100
    title = f"{self.title} results<br><b>Error:</b> ${average_error:,.2f} <b>MSE:</b> {mse:,.0f} <b>r²:</b> {r2:.1f}%"
    self.error_trend_chart()
    self.chart(title)
```

**Metrics Explained**:

1. **Average Error (MAE - Mean Absolute Error)**:
   - Formula: `(Σ |predicted - actual|) / n`
   - Interpretation: Average dollar amount off
   - Example: $62.51 means predictions off by ~$62 on average

2. **MSE (Mean Squared Error)**:
   - Formula: `(Σ (predicted - actual)²) / n`
   - Penalizes large errors more heavily (squared term)
   - Example: MSE=5000 means typical error ~$70 (sqrt(5000))

3. **R² Score (Coefficient of Determination)**:
   - Formula: `1 - (SS_res / SS_tot)` where:
     - `SS_res = Σ(actual - predicted)²` (residual sum of squares)
     - `SS_tot = Σ(actual - mean(actual))²` (total sum of squares)
   - Range: -∞ to 100%
   - Interpretation:
     - 100%: Perfect predictions
     - 0%: Model no better than predicting mean
     - <0%: Model worse than predicting mean
   - Example: 75% means model explains 75% of price variance

**Benchmark Comparison** (from code comments):
- Human performance: $87.62 average error
- GPT-4o-mini: $62.51 average error
- Goal: Match or beat GPT-4o-mini

### 5.8 Main Execution Loop

```python
def run(self):
    for i in tqdm(range(self.size)):
        title, guess, truth, error, color = self.run_datapoint(i)
        self.titles.append(title)
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}${error:.0f} ", end="")
    clear_output(wait=True)
    self.report()
```

**Execution Flow**:
1. Iterate through `size` test samples (default 200)
2. Evaluate each datapoint
3. Store results in lists
4. Print colored error in real-time (visual feedback)
5. Clear output after completion
6. Generate comprehensive report with charts

**Progress Tracking**:
- `tqdm`: Progress bar showing completion percentage
- Colored error output: Quick visual assessment during run
- `clear_output()`: Clean final display (removes intermediate prints)

### 5.9 Public API

```python
def evaluate(function, data, size=DEFAULT_SIZE):
    Tester(function, data, size=size).run()
```

**Purpose**: Simple entry point for evaluation
- **Input**: 
  - `function`: Prediction function (e.g., `model_predict`)
  - `data`: HuggingFace dataset
  - `size`: Number of samples to evaluate
- **Output**: Displays charts and metrics (no return value)

---

## 6. Complete Workflow Mapping

### 6.1 End-to-End Pipeline

```
1. SETUP PHASE
   ├─ Load quantization config (4-bit/8-bit)
   ├─ Load tokenizer (text ↔ tokens)
   ├─ Load base model (frozen, quantized)
   └─ Load PEFT adapters (fine-tuned weights)

2. INFERENCE PHASE (per sample)
   ├─ Tokenize prompt → token IDs
   ├─ Generate tokens (no gradients)
   ├─ Extract new tokens (remove prompt)
   └─ Decode to text → price string

3. EVALUATION PHASE
   ├─ Post-process prediction → float
   ├─ Calculate error vs ground truth
   ├─ Categorize error (green/orange/red)
   └─ Accumulate statistics

4. REPORTING PHASE
   ├─ Calculate metrics (MAE, MSE, R²)
   ├─ Generate error trend chart
   └─ Generate scatter plot
```

### 6.2 Course Concepts → Code Mapping

| Course Concept | Code Implementation | Location |
|----------------|---------------------|----------|
| Cross-Entropy Loss | NOT used (training only) | N/A (testing phase) |
| Probability Distribution | `model.generate()` samples from distribution | `model_predict()` |
| Temperature Sampling | Controlled by `set_seed(42)` | Main execution |
| Forward Pass | `fine_tuned_model.generate()` | `model_predict()` |
| Tokenization | `tokenizer()` and `tokenizer.decode()` | `model_predict()` |
| Quantization (QLoRA) | `BitsAndBytesConfig` | Model loading |
| PEFT/LoRA | `PeftModel.from_pretrained()` | Adapter loading |
| Inference (No Backprop) | `torch.no_grad()` context | `model_predict()` |
| Evaluation Metrics | MAE, MSE, R² calculations | `util.py::report()` |

---

## 7. Critical Technical Details

### 7.1 Memory Optimization
- **4-bit quantization**: 75% memory reduction (12GB → 3GB for 3B model)
- **PEFT adapters**: Only load ~1-2% additional parameters
- **No gradient storage**: `torch.no_grad()` prevents gradient accumulation

### 7.2 Reproducibility
- `set_seed(42)`: Ensures consistent sampling
- `revision` parameter: Pins exact model checkpoint
- `temperature=0` (implicit): Deterministic generation

### 7.3 Error Handling
- `post_process()` returns 0 if no number found (graceful degradation)
- Regex extraction handles various output formats
- Color categorization uses OR logic (flexible thresholds)

### 7.4 Performance Benchmarks
- **Target**: Beat human ($87.62) and approach GPT-4o-mini ($62.51)
- **Evaluation size**: 200 samples (balance between speed and statistical significance)
- **Confidence interval**: 95% CI narrows with more samples

---

## 8. Key Takeaways for LLM Engineers

1. **Testing ≠ Training**: No loss calculation, backpropagation, or optimization during inference
2. **Quantization is Essential**: Enables running 3B models on consumer GPUs
3. **PEFT Efficiency**: Fine-tuned adapters add minimal memory overhead
4. **Post-Processing Matters**: Robust extraction handles model output variability
5. **Multiple Metrics**: MAE (interpretability), MSE (penalizes outliers), R² (variance explained)
6. **Statistical Rigor**: Confidence intervals quantify estimate uncertainty
7. **Visualization**: Scatter plots reveal systematic biases (over/underestimation patterns)
8. **Reproducibility**: Seed setting + version pinning ensures consistent results

---

## 9. Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| OOM (Out of Memory) | Use 4-bit quantization, reduce batch size |
| Inconsistent results | Set seed, pin model revision |
| Poor post-processing | Use robust regex, handle edge cases |
| Misleading metrics | Report multiple metrics (MAE, MSE, R²) |
| Insufficient samples | Use ≥200 samples for stable CI |
| Ignoring variance | Always report confidence intervals |

---

## 10. Production Considerations

1. **Latency**: `max_new_tokens=8` minimizes generation time
2. **Throughput**: Batch inference for multiple items (not shown in code)
3. **Monitoring**: Track error distribution over time (drift detection)
4. **Fallbacks**: Handle `post_process()` returning 0 (model failure)
5. **A/B Testing**: Compare multiple model versions using same evaluation framework
6. **Cost**: Quantization enables deployment on cheaper hardware

