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
- Llama 3.2 requires approval (done in Week 3)
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



