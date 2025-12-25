# Neural Networks and LLM Fine-Tuning: Complete Guide
---
# "THE PRICE IS RIGHT" 

build a model that predicts how much something costs from a description, based on a scrape of Amazon data

A model that can estimate how much something costs, from its description.

# Order of play

## Neural Networks and LLMs

from Traditional ML to Neural Networks to Large Language Models

```python
# imports

import os
from dotenv import load_dotenv
from huggingface_hub import login
from pricer.evaluator import evaluate
from litellm import completion
from pricer.items import Item
import numpy as np
from tqdm.notebook import tqdm
import csv
from sklearn.feature_extraction.text import HashingVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
```

```python
LITE_MODE = False

load_dotenv(override=True)
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)
```

```python
username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")
```

# Before we look at the Artificial Neural Networks

## There is a different kind of Neural Network we could consider

```python
# Write the test set to a CSV

with open('human_in.csv', 'w', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for t in test[:100]:
        writer.writerow([t.summary, 0])
```

```python
# Read it back in

human_predictions = []
with open('human_out.csv', 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        human_predictions.append(float(row[1]))
```

```python
def human_pricer(item):
    idx = test.index(item)
    return human_predictions[idx]
```

```python
human = human_pricer(test[0])
actual = test[0].price
print(f"Human predicted {human} for an item that actually costs {actual}")
```

```python
evaluate(human_pricer, test, size=100)
```

# vanilla Neural Network

get deeper into how Neural Networks work, and how to train a neural network using Pytorch.

```python
# Prepare documents and prices

y = np.array([float(item.price) for item in train])
documents = [item.summary for item in train]
```

```python
# Use the HashingVectorizer for a Bag of Words model
# Using binary=True with the CountVectorizer makes "one-hot vectors"

np.random.seed(42)
vectorizer = HashingVectorizer(n_features=5000, stop_words='english', binary=True)
X = vectorizer.fit_transform(documents)
```

```python
# Define the neural network - here is Pytorch code to create a 8 layer neural network

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 64)
        self.layer6 = nn.Linear(64, 64)
        self.layer7 = nn.Linear(64, 64)
        self.layer8 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output1 = self.relu(self.layer1(x))
        output2 = self.relu(self.layer2(output1))
        output3 = self.relu(self.layer3(output2))
        output4 = self.relu(self.layer4(output3))
        output5 = self.relu(self.layer5(output4))
        output6 = self.relu(self.layer6(output5))
        output7 = self.relu(self.layer7(output6))
        output8 = self.layer8(output7)
        return output8
```

```python
# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X.toarray())
y_train_tensor = torch.FloatTensor(y).unsqueeze(1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_tensor, y_train_tensor, test_size=0.01, random_state=42)

# Create the loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model
input_size = X_train_tensor.shape[1]
model = NeuralNetwork(input_size)
```

```python
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {trainable_params:,}")
```

```python
# Define loss function and optimizer

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# We will do 2 complete runs through the data

EPOCHS = 2

for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in tqdm(train_loader):
        optimizer.zero_grad()

        # The next 4 lines are the 4 stages of training: forward pass, loss calculation, backward pass, optimize
        
        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_function(val_outputs, y_val)

    print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {loss.item():.3f}, Val Loss: {val_loss.item():.3f}')
```

```python
def neural_network(item):
    model.eval()
    with torch.no_grad():
        vector = vectorizer.transform([item.summary])
        vector = torch.FloatTensor(vector.toarray())
        result = model(vector)[0].item()
    return max(0, result)
```

```python
evaluate(neural_network, test)
```

# using a frontier model!

Prediction from Frontier models do out of the box; no training, just inference based on their world knowledge.

```python
def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [{"role": "user", "content": message}]
```

```python
print(test[0].summary)
```

```python
messages_for(test[0])
```

```python
# The function for gpt-4.1-nano

def gpt_4__1_nano(item):
    response = completion(model="openai/gpt-4.1-nano", messages=messages_for(item))
    return response.choices[0].message.content
```

```python
gpt_4__1_nano(test[0])
```

```python
test[0].price
```

```python
evaluate(gpt_4__1_nano, test)
```

```python
def claude_opus_4_5(item):
    response = completion(model="anthropic/claude-opus-4-5", messages=messages_for(item))
    return response.choices[0].message.content
```

```python
evaluate(claude_opus_4_5, test)
```

```python
def gemini_3_pro_preview(item):
    response = completion(model="gemini/gemini-3-pro-preview", messages=messages_for(item), reasoning_effort='low')
    return response.choices[0].message.content
```

```python
evaluate(gemini_3_pro_preview, test, size=50, workers=2)
```

```python
def gemini_2__5_flash_lite(item):
    response = completion(model="gemini/gemini-2.5-flash-lite", messages=messages_for(item))
    return response.choices[0].message.content
```

```python
evaluate(gemini_2__5_flash_lite, test)
```

```python

def grok_4__1_fast(item):
    response = completion(model="xai/grok-4-1-fast-non-reasoning", messages=messages_for(item), seed=42)
    return response.choices[0].message.content
```

```python
evaluate(grok_4__1_fast, test)
```

```python
# The function for gpt-5.1

def gpt_5__1(item):
    response = completion(model="gpt-5.1", messages=messages_for(item), reasoning_effort='high', seed=42)
    return response.choices[0].message.content
```

```python
evaluate(gpt_5__1, test)
```

---
## Table of Contents
1. [Neural Network Fundamentals](#neural-network-fundamentals)
2. [Data Preparation and Training](#data-preparation-and-training)
3. [OpenAI Fine-Tuning Methods](#openai-fine-tuning-methods)
4. [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
5. [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
6. [Reinforcement Learning from Human Feedback (RLHF)](#reinforcement-learning-from-human-feedback-rlhf)
7. [Parameter-Efficient Fine-Tuning (LoRA)](#parameter-efficient-fine-tuning-lora)
8. [Practical Implementation Examples](#practical-implementation-examples)

---

## Neural Network Fundamentals

### Architecture Overview

```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 64)
        self.layer6 = nn.Linear(64, 64)
        self.layer7 = nn.Linear(64, 64)
        self.layer8 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        return self.layer8(x)
```

**Key Components:**
- **8 layers**: Progressive compression (5000 → 128 → 64 → 1)
- **ReLU activation**: Non-linearity for complex pattern learning
- **669,249 parameters**: Trainable weights and biases
- **Feedforward architecture**: Data flows in one direction

---

## Data Preparation and Training

### Text Vectorization

```python
# Convert text to numerical features
vectorizer = HashingVectorizer(n_features=5000, stop_words='english', binary=True)
X = vectorizer.fit_transform(documents)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X.toarray())
y_train_tensor = torch.FloatTensor(y).unsqueeze(1)
```

### Training Loop

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

for epoch in range(EPOCHS):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

---

## OpenAI Fine-Tuning Methods

### 1. OpenAI API Fine-Tuning

```python
from openai import OpenAI

client = OpenAI()

# Prepare training data in JSONL format
def create_training_data(items):
    jsonl_data = []
    for item in items:
        message = {
            "messages": [
                {"role": "user", "content": f"Estimate the price: {item.summary}"},
                {"role": "assistant", "content": f"${item.price:.2f}"}
            ]
        }
        jsonl_data.append(message)
    return jsonl_data

# Upload training file
with open("training_data.jsonl", "rb") as f:
    training_file = client.files.create(file=f, purpose="fine-tune")

# Create fine-tuning job
fine_tune_job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 1,
        "learning_rate_multiplier": 0.1
    }
)

# Monitor training
job_status = client.fine_tuning.jobs.retrieve(fine_tune_job.id)
print(f"Status: {job_status.status}")

# Use fine-tuned model
response = client.chat.completions.create(
    model=fine_tune_job.fine_tuned_model,
    messages=[{"role": "user", "content": "Estimate price: Gaming laptop"}]
)
```

---

## Supervised Fine-Tuning (SFT)

### Overview
SFT trains models on input-output pairs to learn specific tasks or behaviors.

### Implementation with Transformers

```python
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from datasets import Dataset

class SFTTrainer:
    def __init__(self, model_name, tokenizer_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, data):
        def tokenize_function(examples):
            # Format: "Input: {input}\nOutput: {output}"
            texts = [f"Input: {inp}\nOutput: {out}" for inp, out in zip(examples['input'], examples['output'])]
            return self.tokenizer(texts, truncation=True, padding=True, max_length=512)
        
        dataset = Dataset.from_dict(data)
        return dataset.map(tokenize_function, batched=True)
    
    def train(self, train_dataset, eval_dataset=None):
        training_args = TrainingArguments(
            output_dir="./sft_model",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        return trainer

# Example usage
sft_trainer = SFTTrainer("microsoft/DialoGPT-medium", "microsoft/DialoGPT-medium")

# Prepare price estimation data
train_data = {
    'input': ["Gaming laptop with RTX 4080", "Wireless headphones", "Smartphone 128GB"],
    'output': ["$1,299.99", "$89.99", "$699.99"]
}

train_dataset = sft_trainer.prepare_dataset(train_data)
trainer = sft_trainer.train(train_dataset)
```

### Advanced SFT with Custom Loss

```python
import torch.nn.functional as F

class CustomSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Shift labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Custom loss with label smoothing
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=0.1
        )
        
        return (loss, outputs) if return_outputs else loss
```

---

## Direct Preference Optimization (DPO)

### Overview
DPO trains models to prefer certain outputs over others without requiring a reward model.

### Mathematical Foundation

```
L_DPO = -E[(x,y_w,y_l)~D][log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))]

Where:
- y_w: preferred (winning) response
- y_l: less preferred (losing) response  
- π_θ: policy being trained
- π_ref: reference policy
- β: temperature parameter
```

### Implementation

```python
from trl import DPOTrainer
import torch

class DPOFineTuner:
    def __init__(self, model_name, ref_model_name=None):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Reference model (often same as initial model)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_name or model_name
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_preference_dataset(self, preference_data):
        """
        preference_data format:
        [
            {
                "prompt": "Estimate the price of this laptop:",
                "chosen": "$1,299 - This gaming laptop with RTX 4080 is reasonably priced.",
                "rejected": "$5,000 - This laptop is extremely expensive."
            }
        ]
        """
        def format_example(example):
            return {
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"]
            }
        
        return [format_example(ex) for ex in preference_data]
    
    def train(self, preference_dataset):
        training_args = TrainingArguments(
            output_dir="./dpo_model",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-7,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            remove_unused_columns=False,
        )
        
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=preference_dataset,
            tokenizer=self.tokenizer,
            beta=0.1,  # Temperature parameter
            max_length=512,
            max_prompt_length=256,
        )
        
        dpo_trainer.train()
        return dpo_trainer

# Example usage
dpo_trainer = DPOFineTuner("microsoft/DialoGPT-medium")

preference_data = [
    {
        "prompt": "Estimate the price of a gaming laptop with RTX 4080:",
        "chosen": "$1,299 - This is a competitive price for this specification.",
        "rejected": "$500 - This price is unrealistically low for an RTX 4080 laptop."
    },
    {
        "prompt": "Price for wireless earbuds:",
        "chosen": "$89 - Good quality wireless earbuds typically cost around this amount.",
        "rejected": "$10 - Quality wireless earbuds cannot be this cheap."
    }
]

formatted_dataset = dpo_trainer.prepare_preference_dataset(preference_data)
trainer = dpo_trainer.train(formatted_dataset)
```

### Custom DPO Loss Implementation

```python
def dpo_loss(policy_logps_chosen, policy_logps_rejected, 
             reference_logps_chosen, reference_logps_rejected, beta=0.1):
    """
    Compute DPO loss
    """
    # Compute log ratios
    chosen_logratios = policy_logps_chosen - reference_logps_chosen
    rejected_logratios = policy_logps_rejected - reference_logps_rejected
    
    # DPO loss
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    # Metrics
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    
    return loss, chosen_rewards, rejected_rewards
```

---

## Reinforcement Learning from Human Feedback (RLHF)

### Overview
RLHF uses human preferences to train a reward model, then optimizes the policy using reinforcement learning.

### Three-Stage Process

#### Stage 1: Supervised Fine-Tuning
```python
# Already covered in SFT section
sft_model = train_sft_model(base_model, demonstration_data)
```

#### Stage 2: Reward Model Training

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        # Use last token's hidden state
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden_state)
        return reward

class RewardModelTrainer:
    def __init__(self, model_name):
        self.base_model = AutoModel.from_pretrained(model_name)
        self.reward_model = RewardModel(self.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def prepare_comparison_data(self, comparisons):
        """
        comparisons format:
        [
            {
                "prompt": "Estimate laptop price:",
                "response_a": "$1,299 for RTX 4080 laptop",
                "response_b": "$5,000 for RTX 4080 laptop", 
                "preference": "a"  # or "b"
            }
        ]
        """
        dataset = []
        for comp in comparisons:
            prompt = comp["prompt"]
            resp_a = comp["response_a"]
            resp_b = comp["response_b"]
            
            # Tokenize
            text_a = prompt + resp_a
            text_b = prompt + resp_b
            
            tokens_a = self.tokenizer(text_a, return_tensors="pt", truncation=True, max_length=512)
            tokens_b = self.tokenizer(text_b, return_tensors="pt", truncation=True, max_length=512)
            
            # Label: 1 if A preferred, 0 if B preferred
            label = 1 if comp["preference"] == "a" else 0
            
            dataset.append({
                "tokens_a": tokens_a,
                "tokens_b": tokens_b,
                "label": label
            })
        
        return dataset
    
    def train_reward_model(self, comparison_dataset):
        optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-5)
        
        for epoch in range(3):
            total_loss = 0
            for batch in comparison_dataset:
                # Get rewards for both responses
                reward_a = self.reward_model(
                    batch["tokens_a"]["input_ids"],
                    batch["tokens_a"]["attention_mask"]
                )
                reward_b = self.reward_model(
                    batch["tokens_b"]["input_ids"], 
                    batch["tokens_b"]["attention_mask"]
                )
                
                # Preference loss (Bradley-Terry model)
                logits = reward_a - reward_b
                label = torch.tensor([batch["label"]], dtype=torch.float)
                loss = F.binary_cross_entropy_with_logits(logits, label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch}, Loss: {total_loss/len(comparison_dataset)}")
        
        return self.reward_model
```

#### Stage 3: PPO Training

```python
from trl import PPOTrainer, PPOConfig

class RLHFTrainer:
    def __init__(self, sft_model, reward_model, tokenizer):
        self.model = sft_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        
        # PPO configuration
        self.ppo_config = PPOConfig(
            model_name="price_estimator",
            learning_rate=1.41e-5,
            batch_size=16,
            mini_batch_size=4,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
        )
        
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
    
    def generate_responses(self, prompts):
        """Generate responses for given prompts"""
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + 50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def compute_rewards(self, prompts, responses):
        """Compute rewards using trained reward model"""
        rewards = []
        for prompt, response in zip(prompts, responses):
            text = prompt + response
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                reward = self.reward_model(
                    tokens["input_ids"],
                    tokens["attention_mask"]
                ).item()
            
            rewards.append(reward)
        
        return rewards
    
    def train_with_ppo(self, prompts, num_epochs=100):
        for epoch in range(num_epochs):
            # Generate responses
            responses = self.generate_responses(prompts)
            
            # Compute rewards
            rewards = self.compute_rewards(prompts, responses)
            
            # Convert to tensors
            query_tensors = [self.tokenizer(p, return_tensors="pt")["input_ids"][0] for p in prompts]
            response_tensors = [self.tokenizer(r, return_tensors="pt")["input_ids"][0] for r in responses]
            reward_tensors = [torch.tensor(r) for r in rewards]
            
            # PPO step
            stats = self.ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Mean reward: {np.mean(rewards):.3f}")

# Example usage
comparison_data = [
    {
        "prompt": "Estimate the price of a gaming laptop:",
        "response_a": "$1,299 - Reasonable price for RTX 4080",
        "response_b": "$10,000 - Extremely overpriced",
        "preference": "a"
    }
]

# Train reward model
reward_trainer = RewardModelTrainer("microsoft/DialoGPT-medium")
comparison_dataset = reward_trainer.prepare_comparison_data(comparison_data)
reward_model = reward_trainer.train_reward_model(comparison_dataset)

# RLHF training
rlhf_trainer = RLHFTrainer(sft_model, reward_model, tokenizer)
prompts = ["Estimate laptop price:", "Price wireless headphones:"]
rlhf_trainer.train_with_ppo(prompts)
```

---

## Parameter-Efficient Fine-Tuning (LoRA)

### LoRA Implementation

```python
from peft import LoraConfig, get_peft_model, TaskType

class LoRAFineTuner:
    def __init__(self, model_name, task_type=TaskType.CAUSAL_LM):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            task_type=task_type,
            r=8,                    # Rank
            lora_alpha=32,          # Scaling factor
            lora_dropout=0.1,       # Dropout
            target_modules=[        # Target attention layers
                "q_proj", "k_proj", "v_proj", "o_proj"
            ],
            bias="none",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, self.lora_config)
    
    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
    
    def train(self, train_dataset, eval_dataset=None):
        training_args = TrainingArguments(
            output_dir="./lora_model",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,     # Higher LR for LoRA
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        return trainer
    
    def save_lora_adapters(self, path):
        self.model.save_pretrained(path)
    
    def load_lora_adapters(self, path):
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, path)
    
    def merge_and_save(self, path):
        # Merge LoRA weights into base model
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

# Example usage
lora_trainer = LoRAFineTuner("microsoft/DialoGPT-medium")
lora_trainer.print_trainable_parameters()

# Train with LoRA
trainer = lora_trainer.train(train_dataset)

# Save only LoRA adapters (small file)
lora_trainer.save_lora_adapters("./price_lora")

# Or merge and save full model
lora_trainer.merge_and_save("./price_model_merged")
```

### QLoRA (Quantized LoRA)

```python
from transformers import BitsAndBytesConfig

def create_qlora_model(model_name):
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to quantized model
    model = get_peft_model(model, lora_config)
    return model

# Can fine-tune 70B model on single consumer GPU!
qlora_model = create_qlora_model("meta-llama/Llama-2-70b-hf")
```

---

## Practical Implementation Examples

### Complete Price Estimation Pipeline

```python
class PriceEstimationPipeline:
    def __init__(self):
        self.models = {}
        self.tokenizer = None
    
    def train_sft_model(self, training_data):
        """Stage 1: Supervised Fine-Tuning"""
        print("Training SFT model...")
        
        sft_trainer = SFTTrainer("microsoft/DialoGPT-medium", "microsoft/DialoGPT-medium")
        train_dataset = sft_trainer.prepare_dataset(training_data)
        trainer = sft_trainer.train(train_dataset)
        
        self.models['sft'] = trainer.model
        self.tokenizer = sft_trainer.tokenizer
        return trainer.model
    
    def train_dpo_model(self, preference_data, sft_model):
        """Stage 2: Direct Preference Optimization"""
        print("Training DPO model...")
        
        dpo_trainer = DPOFineTuner("microsoft/DialoGPT-medium")
        dpo_trainer.model = sft_model  # Start from SFT model
        
        formatted_dataset = dpo_trainer.prepare_preference_dataset(preference_data)
        trainer = dpo_trainer.train(formatted_dataset)
        
        self.models['dpo'] = trainer.model
        return trainer.model
    
    def train_lora_model(self, training_data):
        """Alternative: LoRA Fine-Tuning"""
        print("Training LoRA model...")
        
        lora_trainer = LoRAFineTuner("microsoft/DialoGPT-medium")
        
        # Prepare dataset for LoRA
        def tokenize_function(examples):
            texts = [f"Price: {inp} -> {out}" for inp, out in zip(examples['input'], examples['output'])]
            return lora_trainer.tokenizer(texts, truncation=True, padding=True, max_length=512)
        
        dataset = Dataset.from_dict(training_data)
        train_dataset = dataset.map(tokenize_function, batched=True)
        
        trainer = lora_trainer.train(train_dataset)
        self.models['lora'] = trainer.model
        return trainer.model
    
    def evaluate_models(self, test_data):
        """Evaluate all trained models"""
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name} model...")
            
            predictions = []
            for item in test_data:
                prompt = f"Estimate price: {item['input']}"
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=inputs["input_ids"].shape[1] + 20,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                predictions.append(response)
            
            results[model_name] = predictions
        
        return results

# Example usage
pipeline = PriceEstimationPipeline()

# Training data
sft_data = {
    'input': [
        "Gaming laptop RTX 4080 16GB RAM",
        "Wireless earbuds noise cancelling", 
        "Smartphone 256GB 5G"
    ],
    'output': [
        "$1,299.99",
        "$149.99", 
        "$799.99"
    ]
}

preference_data = [
    {
        "prompt": "Estimate laptop price:",
        "chosen": "$1,299 - Competitive price for RTX 4080",
        "rejected": "$500 - Too low for this specification"
    }
]

test_data = [
    {"input": "Gaming desktop RTX 4090"},
    {"input": "Tablet 128GB WiFi"}
]

# Train models
sft_model = pipeline.train_sft_model(sft_data)
dpo_model = pipeline.train_dpo_model(preference_data, sft_model)
lora_model = pipeline.train_lora_model(sft_data)

# Evaluate
results = pipeline.evaluate_models(test_data)
for model_name, predictions in results.items():
    print(f"{model_name}: {predictions}")
```

### OpenAI API Integration

```python
class OpenAIFineTuner:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def prepare_jsonl_data(self, training_examples):
        """Convert training data to JSONL format"""
        jsonl_lines = []
        for example in training_examples:
            message = {
                "messages": [
                    {"role": "user", "content": f"Estimate the price: {example['input']}"},
                    {"role": "assistant", "content": example['output']}
                ]
            }
            jsonl_lines.append(json.dumps(message))
        
        return "\n".join(jsonl_lines)
    
    def create_fine_tune_job(self, training_data, model="gpt-4o-mini-2024-07-18"):
        # Prepare data
        jsonl_data = self.prepare_jsonl_data(training_data)
        
        # Save to file
        with open("training_data.jsonl", "w") as f:
            f.write(jsonl_data)
        
        # Upload file
        with open("training_data.jsonl", "rb") as f:
            training_file = self.client.files.create(file=f, purpose="fine-tune")
        
        # Create fine-tuning job
        job = self.client.fine_tuning.jobs.create(
            training_file=training_file.id,
            model=model,
            hyperparameters={
                "n_epochs": 3,
                "batch_size": 1,
                "learning_rate_multiplier": 0.1
            },
            suffix="price-estimator"
        )
        
        return job
    
    def monitor_job(self, job_id):
        """Monitor fine-tuning progress"""
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            print(f"Status: {job.status}")
            
            if job.status == "succeeded":
                print(f"Fine-tuned model: {job.fine_tuned_model}")
                return job.fine_tuned_model
            elif job.status == "failed":
                print(f"Job failed: {job.error}")
                return None
            
            time.sleep(30)  # Check every 30 seconds
    
    def use_fine_tuned_model(self, model_id, prompt):
        """Use the fine-tuned model for inference"""
        response = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1
        )
        
        return response.choices[0].message.content

# Example usage
openai_trainer = OpenAIFineTuner("your-api-key")

training_examples = [
    {"input": "Gaming laptop RTX 4080", "output": "$1,299.99"},
    {"input": "Wireless headphones", "output": "$89.99"},
    {"input": "Smartphone 128GB", "output": "$699.99"}
]

# Create and monitor fine-tuning job
job = openai_trainer.create_fine_tune_job(training_examples)
fine_tuned_model = openai_trainer.monitor_job(job.id)

# Use fine-tuned model
if fine_tuned_model:
    result = openai_trainer.use_fine_tuned_model(
        fine_tuned_model, 
        "Estimate the price: Gaming desktop RTX 4090"
    )
    print(f"Prediction: {result}")
```

---

## Key Takeaways

### Method Comparison

| Method | Parameters Updated | Memory Usage | Training Time | Performance | Use Case |
|--------|-------------------|--------------|---------------|-------------|----------|
| Full Fine-tuning | 100% | Very High | Long | Best | Large datasets, unlimited resources |
| LoRA | 0.1-1% | Low | Fast | 95-100% | Limited resources, multiple tasks |
| SFT | 100% | High | Medium | Good | Task-specific adaptation |
| DPO | 100% | High | Medium | Better alignment | Preference learning |
| RLHF | 100% | Very High | Very Long | Best alignment | Human preference optimization |
| OpenAI API | 0% (external) | None | Fast | Good | Quick prototyping, no infrastructure |

### Best Practices

1. **Start Simple**: Begin with OpenAI API fine-tuning or LoRA
2. **Data Quality**: High-quality training data is more important than quantity
3. **Evaluation**: Always evaluate on held-out test data
4. **Hyperparameter Tuning**: Start with recommended values, then optimize
5. **Resource Management**: Use gradient checkpointing and mixed precision for memory efficiency
6. **Monitoring**: Track training metrics and validation performance

### Choosing the Right Method

- **OpenAI API**: Quick prototyping, no infrastructure needed
- **LoRA**: Limited GPU memory, multiple tasks, fast iteration
- **SFT**: Task-specific adaptation with good data
- **DPO**: When you have preference data, want better alignment
- **RLHF**: Maximum alignment quality, have human feedback resources
- **Full Fine-tuning**: Unlimited resources, maximum performance needed

This comprehensive guide provides both theoretical understanding and practical implementation examples for modern neural network and LLM fine-tuning techniques.