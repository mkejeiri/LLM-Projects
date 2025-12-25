# **AdamW Optimizer - Deep Dive**

---

## **What is AdamW?**

**AdamW** = **Adam** with **Weight Decay** (decoupled)

**Full Name**: Adaptive Moment Estimation with Weight Decay

**Paper**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)

---

## **The Evolution: SGD → Adam → AdamW**

### **1. Stochastic Gradient Descent (SGD)**

```python
# Basic SGD update
weight = weight - learning_rate * gradient
```

**Problems**:
- Same learning rate for all parameters
- Slow convergence
- Sensitive to learning rate choice
- Gets stuck in saddle points

---

### **2. Adam (Adaptive Moment Estimation)**

**Key Innovation**: Adaptive learning rates per parameter

```python
# Adam maintains two moving averages:
m_t = β1 * m_{t-1} + (1 - β1) * gradient        # First moment (mean)
v_t = β2 * v_{t-1} + (1 - β2) * gradient²       # Second moment (variance)

# Bias correction (important for early steps)
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)

# Update rule
weight = weight - learning_rate * m_hat / (sqrt(v_hat) + ε)
```

**What This Means**:
- **m_t**: Running average of gradients (momentum)
- **v_t**: Running average of squared gradients (adaptive learning rate)
- **β1**: Typically 0.9 (momentum decay)
- **β2**: Typically 0.999 (variance decay)
- **ε**: Small constant (1e-8) to prevent division by zero

**Advantages**:
- Adapts learning rate per parameter
- Works well with sparse gradients
- Requires minimal tuning
- Fast convergence

**Problem with Adam**:
- Weight decay implementation was coupled with gradient
- Led to suboptimal regularization

---

### **3. AdamW (Adam with Decoupled Weight Decay)**

**The Problem with Adam's Weight Decay**:

```python
# Adam with L2 regularization (WRONG way)
gradient = gradient + weight_decay * weight  # Add to gradient
# Then apply Adam update
```

**Why This is Bad**:
- Weight decay gets scaled by adaptive learning rate
- Inconsistent regularization across parameters
- Parameters with large gradients get less regularization
- Parameters with small gradients get more regularization

**AdamW Solution (CORRECT way)**:

```python
# AdamW: Decouple weight decay from gradient
# Step 1: Apply Adam update (without weight decay in gradient)
m_t = β1 * m_{t-1} + (1 - β1) * gradient
v_t = β2 * v_{t-1} + (1 - β2) * gradient²
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)
weight_update = learning_rate * m_hat / (sqrt(v_hat) + ε)

# Step 2: Apply weight decay SEPARATELY
weight = weight - weight_update - learning_rate * weight_decay * weight
```

**Key Difference**:
- Weight decay applied **directly** to weights
- **Not** added to gradient
- Consistent regularization regardless of gradient magnitude

---

## **AdamW in Practice**

### **Hyperparameters**

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,              # Learning rate
    betas=(0.9, 0.999),   # (β1, β2) for momentum and variance
    eps=1e-8,             # Epsilon for numerical stability
    weight_decay=0.001    # Weight decay coefficient
)
```

**Typical Values**:
- **lr**: 1e-4 to 1e-3 for fine-tuning, 1e-3 to 1e-2 for training from scratch
- **β1**: 0.9 (rarely changed)
- **β2**: 0.999 for transformers, 0.99 for CNNs
- **weight_decay**: 0.001 to 0.01

---

## **Why AdamW for LLM Fine-Tuning?**

### **1. Memory Efficiency**

**Adam State**:
- **First moment (m)**: Same size as model parameters
- **Second moment (v)**: Same size as model parameters
- **Total**: 2× parameter memory

**Example**:
- Llama 3.2-3B: 3 billion parameters
- FP32: 3B × 4 bytes = 12 GB (model)
- Adam states: 12 GB × 2 = 24 GB (optimizer)
- **Total**: 36 GB just for model + optimizer!

**Solution: Paged AdamW**:
```python
OPTIMIZER = "paged_adamw_32bit"
```
- Offloads optimizer states to CPU RAM when not needed
- Brings back to GPU only during optimizer step
- Saves precious GPU memory
- Minimal performance impact

---

### **2. Better Generalization**

**Weight Decay Effect**:
- Prevents weights from growing too large
- Encourages simpler solutions
- Reduces overfitting
- Especially important for fine-tuning (limited data)

**Comparison**:
```
Adam (with L2):     Test Error = 15.2%
AdamW:              Test Error = 12.8%  ← Better!
```

---

### **3. Stable Training**

**Adaptive Learning Rates**:
- Different parameters learn at different rates
- Attention weights might need smaller updates
- MLP weights might need larger updates
- AdamW handles this automatically

**Momentum Benefits**:
- Smooths out noisy gradients
- Helps escape local minima
- Accelerates convergence in consistent directions

---

## **AdamW vs Other Optimizers**

### **Comparison Table**

| Optimizer | Memory | Speed | Generalization | Tuning Difficulty |
|-----------|--------|-------|----------------|-------------------|
| SGD       | Low    | Fast  | Good           | Hard              |
| Adam      | High   | Fast  | Okay           | Easy              |
| AdamW     | High   | Fast  | **Best**       | Easy              |
| Adafactor | Low    | Slow  | Good           | Medium            |

### **When to Use Each**

**AdamW** (Default choice):
- Fine-tuning LLMs
- Training transformers
- When you have enough GPU memory
- When you want best results with minimal tuning

**SGD with Momentum**:
- Training CNNs (ResNet, etc.)
- When memory is extremely limited
- When you have time to tune learning rate carefully

**Adafactor**:
- Training very large models (>10B parameters)
- When GPU memory is severely constrained
- Willing to sacrifice some speed

---

## **The Math Behind AdamW**

### **Complete Algorithm**

```python
# Initialize
m_0 = 0  # First moment
v_0 = 0  # Second moment
t = 0    # Time step

# Hyperparameters
α = 1e-4      # Learning rate
β1 = 0.9      # Exponential decay rate for first moment
β2 = 0.999    # Exponential decay rate for second moment
ε = 1e-8      # Small constant
λ = 0.001     # Weight decay

# Training loop
for each batch:
    t = t + 1
    
    # Compute gradient
    g_t = ∇L(θ_{t-1})
    
    # Update biased first moment estimate
    m_t = β1 * m_{t-1} + (1 - β1) * g_t
    
    # Update biased second moment estimate
    v_t = β2 * v_{t-1} + (1 - β2) * g_t²
    
    # Compute bias-corrected first moment
    m̂_t = m_t / (1 - β1^t)
    
    # Compute bias-corrected second moment
    v̂_t = v_t / (1 - β2^t)
    
    # Update parameters (AdamW)
    θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
                    ↑                          ↑
                Adam update              Weight decay
```

---

## **Practical Example: Training Step**

### **Scenario**: Fine-tuning Llama 3.2-3B

```python
# Initial setup
learning_rate = 1e-4
weight_decay = 0.001
β1, β2 = 0.9, 0.999

# Example parameter (one weight in attention layer)
weight = 0.5
gradient = 0.02  # From backpropagation

# Step 1: Update first moment (momentum)
m = 0.9 * m_prev + 0.1 * 0.02
m = 0.9 * 0.0 + 0.002 = 0.002

# Step 2: Update second moment (variance)
v = 0.999 * v_prev + 0.001 * (0.02)²
v = 0.999 * 0.0 + 0.0000004 = 0.0000004

# Step 3: Bias correction (important for early steps)
m_hat = 0.002 / (1 - 0.9^1) = 0.002 / 0.1 = 0.02
v_hat = 0.0000004 / (1 - 0.999^1) = 0.0000004 / 0.001 = 0.0004

# Step 4: Compute Adam update
adam_update = 0.0001 * 0.02 / (sqrt(0.0004) + 1e-8)
adam_update = 0.0001 * 0.02 / 0.02 = 0.0001

# Step 5: Apply weight decay
weight_decay_term = 0.0001 * 0.001 * 0.5 = 0.00000005

# Step 6: Final update
weight_new = 0.5 - 0.0001 - 0.00000005 = 0.4999
```

---

## **Why "Paged" AdamW?**

### **The Memory Problem**

```python
# Standard AdamW memory usage
Model weights:     3B params × 4 bytes = 12 GB
Gradients:         3B params × 4 bytes = 12 GB
Adam m states:     3B params × 4 bytes = 12 GB
Adam v states:     3B params × 4 bytes = 12 GB
Total:             48 GB  ← Doesn't fit in T4 (16GB)!
```

### **Paged AdamW Solution**

```python
OPTIMIZER = "paged_adamw_32bit"
```

**How It Works**:
1. **Store optimizer states in CPU RAM** (much larger than GPU VRAM)
2. **Page in** only the states needed for current batch
3. **Update** on GPU
4. **Page out** back to CPU
5. Repeat for next batch

**Benefits**:
- Fits large models in small GPUs
- ~10% slower than regular AdamW
- Much better than not being able to train at all!

**Memory Savings**:
```
Regular AdamW:  48 GB GPU required
Paged AdamW:    16 GB GPU + 32 GB CPU RAM
```

---

## **AdamW Variants in the Wild**

### **1. 8-bit AdamW**

```python
OPTIMIZER = "paged_adamw_8bit"
```

- Quantize optimizer states to 8-bit
- Further memory savings
- Minimal quality loss
- Good for extremely large models

### **2. AdamW with Fused Operations**

```python
optimizer = torch.optim.AdamW(model.parameters(), fused=True)
```

- Combines multiple operations into single GPU kernel
- ~20% faster
- Requires newer PyTorch and CUDA

### **3. Distributed AdamW**

```python
# For multi-GPU training
optimizer = torch.distributed.optim.ZeroRedundancyOptimizer(
    model.parameters(),
    optimizer_class=torch.optim.AdamW,
    lr=1e-4
)
```

- Shards optimizer states across GPUs
- Each GPU stores only part of states
- Reduces memory per GPU

---

## **Common Mistakes with AdamW**

### **Mistake 1: Using Adam Instead of AdamW**

```python
# WRONG
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)

# CORRECT
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
```

**Why**: Adam applies weight decay incorrectly (coupled with gradient)

---

### **Mistake 2: Too High Learning Rate**

```python
# WRONG for fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)  # Too high!

# CORRECT
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Just right
```

**Symptoms**: Loss explodes, NaN values, unstable training

---

### **Mistake 3: No Weight Decay**

```python
# WRONG
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

# CORRECT
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
```

**Result**: Overfitting, poor generalization

---

### **Mistake 4: Wrong Beta Values**

```python
# WRONG for transformers
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.99))  # β2 too low

# CORRECT
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999))  # Standard
```

**Why**: Transformers need longer memory of past gradients (higher β2)

---

## **Tuning AdamW for Your Task**

### **Learning Rate**

**Start with**: 1e-4

**If loss doesn't decrease**: Try 1e-3 (higher)

**If loss is unstable**: Try 5e-5 (lower)

**Rule of thumb**: 
- Larger models → smaller LR
- More data → can use higher LR
- Fine-tuning → smaller LR than training from scratch

---

### **Weight Decay**

**Start with**: 0.001

**If overfitting**: Increase to 0.01

**If underfitting**: Decrease to 0.0001

**Rule of thumb**:
- Small dataset → higher weight decay
- Large dataset → lower weight decay
- Simple task → lower weight decay
- Complex task → higher weight decay

---

### **Betas**

**Almost never change these!**

**Default**: (0.9, 0.999)

**Only change if**:
- Training is very unstable → Try (0.9, 0.9999)
- Need faster adaptation → Try (0.8, 0.999)

---

## **AdamW in HuggingFace Transformers**

### **Automatic Setup**

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    optim="adamw_torch",           # Use PyTorch's AdamW
    learning_rate=1e-4,
    weight_decay=0.001,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
)
```

### **Available Optimizers**

```python
optim="adamw_torch"        # Standard PyTorch AdamW
optim="adamw_hf"           # HuggingFace implementation
optim="paged_adamw_32bit"  # Paged version (saves memory)
optim="paged_adamw_8bit"   # 8-bit paged (even more memory savings)
optim="adafactor"          # Alternative for very large models
```

---

## **Monitoring AdamW During Training**

### **Key Metrics in W&B**

**1. Learning Rate**:
- Should follow scheduler (warmup → decay)
- Check it's in expected range (1e-5 to 1e-3)

**2. Gradient Norm**:
- Should be stable (not exploding)
- Typical range: 0.1 to 10.0
- If >100: Learning rate too high

**3. Weight Norm**:
- Should grow slowly or stay stable
- If growing rapidly: Weight decay too low
- If shrinking: Weight decay too high

**4. Loss**:
- Should decrease smoothly
- Jagged is okay (batch-to-batch variation)
- If flat: Learning rate too low
- If exploding: Learning rate too high

---

## **Summary: Why AdamW is the Default**

✅ **Adaptive learning rates** - Works well out of the box  
✅ **Momentum** - Smooth, fast convergence  
✅ **Proper regularization** - Better generalization than Adam  
✅ **Memory efficient** - Paged variants fit large models  
✅ **Minimal tuning** - Standard hyperparameters work well  
✅ **Industry standard** - Used by all major LLM trainers  

**Bottom Line**: Unless you have a specific reason to use something else, use AdamW.

---

## **Quick Reference**

```python
# Standard fine-tuning setup
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,              # Learning rate
    betas=(0.9, 0.999),   # Momentum parameters
    eps=1e-8,             # Numerical stability
    weight_decay=0.001    # Regularization
)

# For memory-constrained GPUs
OPTIMIZER = "paged_adamw_32bit"

# For extremely large models
OPTIMIZER = "paged_adamw_8bit"
```

---

**Key Takeaway**: AdamW = Adam + Proper Weight Decay = Better Results
