# Llama 3.2 3B Model Architecture Breakdown

## Overview

This is a **Llama 3.2 3B** model (3 billion parameters) - a smaller, more efficient variant compared to the 8B model. The architecture follows the same decoder-only transformer design but with reduced dimensions.

---

## Model Specifications Comparison

| Specification | Llama 3.2 3B (Your Model) | Llama 3.1 8B (Reference) |
|---------------|---------------------------|--------------------------|
| **Model Dimension** (d_model) | 3,072 | 4,096 |
| **Vocabulary Size** | 128,256 | 128,256 |
| **Number of Layers** | 28 | 32 |
| **MLP Hidden Dimension** | 8,192 | 14,336 |
| **MLP Expansion Ratio** | 2.67× | 3.5× |
| **Query Heads** | 24 (3072/128) | 32 (4096/128) |
| **KV Heads** | 8 (1024/128) | 8 (1024/128) |
| **GQA Ratio** | 3:1 | 4:1 |
| **Total Parameters** | ~3B | ~8B |

---

## Architecture Breakdown

### 1. Top-Level Structure

```python
LlamaForCausalLM(
  (model): LlamaModel(...)      # Core transformer
  (lm_head): Linear(...)         # Output projection
)
```

**LlamaForCausalLM**: Wrapper for causal (autoregressive) language modeling
- **model**: The actual transformer layers
- **lm_head**: Projects hidden states to vocabulary logits

---

### 2. Embedding Layer

```python
(embed_tokens): Embedding(128256, 3072)
```

**Function**: Converts token IDs to dense vectors

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_embeddings` | 128,256 | Vocabulary size (same as 8B model) |
| `embedding_dim` | 3,072 | Model dimension (25% smaller than 8B) |

**Memory**: 128,256 × 3,072 × 4 bytes (FP32) = ~1.6 GB

**Key Point**: Same vocabulary as larger models, enabling consistent tokenization across Llama family.

---

### 3. Decoder Layers

```python
(layers): ModuleList(
  (0-27): 28 x LlamaDecoderLayer(...)
)
```

**28 layers** instead of 32 (8B model)
- **Trade-off**: Fewer layers = less depth, but faster inference
- **Each layer**: Refines representations through attention + MLP

---

### 4. Self-Attention Mechanism (Grouped-Query Attention)

```python
(self_attn): LlamaAttention(
  (q_proj): Linear(in_features=3072, out_features=3072, bias=False)
  (k_proj): Linear(in_features=3072, out_features=1024, bias=False)
  (v_proj): Linear(in_features=3072, out_features=1024, bias=False)
  (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
)
```

### Attention Head Configuration

**Query Projection**: 3072 → 3072
- **Number of heads**: 3072 / 128 = **24 query heads**
- **Head dimension**: 128 (standard)

**Key/Value Projections**: 3072 → 1024
- **Number of KV heads**: 1024 / 128 = **8 KV heads**
- **Shared across**: 24 / 8 = **3 query heads per KV head**

### Grouped-Query Attention (GQA) Visualization

```
Query Heads:  [Q1 Q2 Q3] [Q4 Q5 Q6] [Q7 Q8 Q9] ... [Q22 Q23 Q24]
                    ↓           ↓           ↓              ↓
KV Heads:          K1/V1       K2/V2       K3/V3   ...   K8/V8
```

**Each KV head is shared by 3 query heads** (3:1 ratio)

### Why GQA?

| Metric | Multi-Head Attention | Grouped-Query Attention |
|--------|---------------------|-------------------------|
| **KV Cache Size** | 24 heads × 128 = 3,072 | 8 heads × 128 = 1,024 |
| **Memory Reduction** | Baseline | **67% smaller** |
| **Inference Speed** | Baseline | **Faster** (less KV cache) |
| **Quality** | Baseline | **~98% of MHA quality** |

**KV Cache per Token**:
```
2 (K+V) × 28 layers × 8 heads × 128 dim × 2 bytes (FP16)
= 114,688 bytes ≈ 112 KB per token
```

For 8K context: 8,192 × 112 KB = **896 MB KV cache**

---

### 5. MLP (Feed-Forward Network)

```python
(mlp): LlamaMLP(
  (gate_proj): Linear(in_features=3072, out_features=8192, bias=False)
  (up_proj): Linear(in_features=3072, out_features=8192, bias=False)
  (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
  (act_fn): SiLUActivation()
)
```

### SwiGLU Architecture

**Expansion Ratio**: 8,192 / 3,072 = **2.67×** (vs 3.5× in 8B model)

**Computation**:
```python
gate = SiLU(gate_proj(x))    # [B, L, 8192]
up = up_proj(x)               # [B, L, 8192]
hidden = gate * up            # Element-wise multiplication
output = down_proj(hidden)    # [B, L, 3072]
```

**Why Smaller Expansion?**
- **Parameter efficiency**: Reduces total model size
- **Speed**: Faster forward pass
- **Trade-off**: Slightly less representational capacity per layer

**Parameters per MLP**:
- gate_proj: 3,072 × 8,192 = 25.2M
- up_proj: 3,072 × 8,192 = 25.2M
- down_proj: 8,192 × 3,072 = 25.2M
- **Total**: ~75M parameters per layer

---

### 6. Normalization Layers

```python
(input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
(post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
```

**RMSNorm** (Root Mean Square Normalization)

**Formula**:
```
RMSNorm(x) = x / sqrt(mean(x²) + 1e-05) * γ
```

**Applied**:
1. **Before attention**: Stabilizes attention computation
2. **Before MLP**: Stabilizes feed-forward computation

**Pre-Norm Architecture**:
```
x → RMSNorm → Attention → Residual Add
  ↓
  → RMSNorm → MLP → Residual Add
```

---

### 7. Rotary Position Embeddings

```python
(rotary_emb): LlamaRotaryEmbedding()
```

**RoPE** (Rotary Position Embedding)

**Purpose**: Encodes position information through rotation matrices

**Applied to**: Query and Key projections (not Value)

**Benefits**:
- ✅ Relative position awareness
- ✅ Extrapolation to longer sequences
- ✅ No learned parameters (computed on-the-fly)

---

### 8. Final Normalization

```python
(norm): LlamaRMSNorm((3072,), eps=1e-05)
```

**Final RMSNorm** before LM head
- Normalizes output of last decoder layer
- Stabilizes logit magnitudes

---

### 9. Language Modeling Head

```python
(lm_head): Linear(in_features=3072, out_features=128256, bias=False)
```

**Function**: Projects hidden states to vocabulary logits

**Transformation**: 3,072 → 128,256

**Output**: Logits for each token in vocabulary

**Inference**:
```python
hidden_states = model(input_ids)        # [B, L, 3072]
logits = lm_head(norm(hidden_states))   # [B, L, 128256]
next_token = argmax(logits[:, -1, :])   # Greedy decoding
```

**Weight Tying**: Often shares weights with `embed_tokens` to save parameters

---

## Parameter Count Breakdown

| Component | Parameters | Percentage |
|-----------|------------|------------|
| **Embeddings** | 393M | 13.1% |
| **28 × Attention** | 672M | 22.4% |
| **28 × MLP** | 2,100M | 70.0% |
| **Normalization** | ~0.3M | <0.1% |
| **LM Head** (if not tied) | 393M | - |
| **Total (with tied weights)** | **~3B** | **100%** |

---

## Key Differences from 8B Model

### Dimension Reduction Strategy

| Aspect | 3B Model | 8B Model | Reduction |
|--------|----------|----------|-----------|
| **d_model** | 3,072 | 4,096 | 25% |
| **Layers** | 28 | 32 | 12.5% |
| **MLP Hidden** | 8,192 | 14,336 | 43% |
| **Query Heads** | 24 | 32 | 25% |
| **KV Heads** | 8 | 8 | 0% |

**Design Philosophy**:
- **Preserve KV heads**: Maintains attention quality
- **Reduce width**: Smaller d_model (3072 vs 4096)
- **Reduce depth**: Fewer layers (28 vs 32)
- **Reduce MLP**: Smaller expansion ratio (2.67× vs 3.5×)

---

## Performance Characteristics

### Advantages of 3B Model

1. **Memory Efficiency**
   - Model size: ~3GB (FP32) or ~1.5GB (FP16)
   - KV cache: 67% smaller than equivalent MHA
   - Fits on consumer GPUs (RTX 3090, 4090)

2. **Inference Speed**
   - Faster forward pass (fewer parameters)
   - Lower latency for real-time applications
   - Better throughput on limited hardware

3. **Fine-tuning Friendly**
   - Requires less VRAM for training
   - Faster iteration cycles
   - Suitable for QLoRA on consumer hardware

### Trade-offs

1. **Capacity**
   - Less representational power than 8B
   - May struggle with very complex tasks
   - Shorter "reasoning chains"

2. **Performance**
   - Typically 5-10% lower on benchmarks
   - Less world knowledge
   - More prone to hallucination

---

## Computational Analysis

### FLOPs per Token (Generation)

**Attention** (per layer):
```
2 × 3072 × (3072 + 1024 + 1024 + 3072) ≈ 50M FLOPs
```

**MLP** (per layer):
```
2 × 3072 × 8192 × 3 ≈ 151M FLOPs
```

**Total per layer**: ~201M FLOPs
**Total for 28 layers**: ~5.6 GFLOPs per token

**Comparison**: 8B model uses ~14 TFLOPs per token (2.5× more)

---

## Use Cases for 3B Model

### Ideal For:
- ✅ **Edge deployment** (mobile, embedded devices)
- ✅ **Real-time applications** (chatbots, assistants)
- ✅ **Fine-tuning experiments** (fast iteration)
- ✅ **Cost-sensitive production** (lower inference costs)
- ✅ **Multi-model ensembles** (run multiple models)

### Not Ideal For:
- ❌ Complex reasoning tasks
- ❌ Extensive world knowledge queries
- ❌ Long-form content generation
- ❌ Highly specialized domains (without fine-tuning)

---

## Summary

This **Llama 3.2 3B** model represents an efficient, production-ready architecture:

- **28 decoder layers** with **3,072 dimensions**
- **Grouped-Query Attention** (24 Q heads, 8 KV heads, 3:1 ratio)
- **SwiGLU MLP** with 2.67× expansion
- **RMSNorm** for efficient normalization
- **RoPE** for position encoding
- **~3 billion parameters** total

**Key Innovation**: Maintains 8 KV heads (same as 8B model) while reducing other dimensions, preserving attention quality while dramatically reducing memory and compute requirements.

**Perfect for**: Fine-tuning on consumer hardware, edge deployment, and applications requiring fast inference with acceptable quality trade-offs.
