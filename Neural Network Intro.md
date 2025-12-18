# Neural Network Deep Dive 
---

## Table of Contents
1. [Neural Network Architecture](#neural-network-architecture)
2. [Data Preparation](#data-preparation)
3. [Stochastic Gradient Descent](#stochastic-gradient-descent)
4. [Complete Code Walkthrough](#complete-code-walkthrough)

---

## Neural Network Architecture

### The NeuralNetwork Class

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

### Class Inheritance

**`class NeuralNetwork(nn.Module)`**

The class inherits from `nn.Module`, which is PyTorch's base class for all neural network modules. This inheritance provides:
- Automatic parameter tracking and management
- GPU/CPU device management
- Training/evaluation mode switching
- Built-in methods for saving and loading models

### Constructor Breakdown

**`def __init__(self, input_size):`**

The constructor initializes all the layers of the neural network.

**`super(NeuralNetwork, self).__init__()`**
- Calls the parent class (`nn.Module`) constructor
- Essential for proper initialization of PyTorch functionality

### The 8 Layers Explained

**Layer 1: Input Compression**
```python
self.layer1 = nn.Linear(input_size, 128)
```
- **Input**: 5000 features (from HashingVectorizer)
- **Output**: 128 neurons
- **Purpose**: Compresses high-dimensional text features into a more manageable representation
- **Math**: `output = input × weights + bias`
- **Parameters**: 5000 × 128 + 128 = 640,128 parameters

**Layers 2-7: Hidden Processing Layers**
```python
self.layer2 = nn.Linear(128, 64)
self.layer3 = nn.Linear(64, 64)
self.layer4 = nn.Linear(64, 64)
self.layer5 = nn.Linear(64, 64)
self.layer6 = nn.Linear(64, 64)
self.layer7 = nn.Linear(64, 64)
```
- **Purpose**: Process and transform information through multiple levels of abstraction
- **Layer 2**: Compresses from 128 to 64 neurons (128 × 64 + 64 = 8,256 parameters)
- **Layers 3-7**: Maintain 64 neurons each (64 × 64 + 64 = 4,160 parameters each)
- **Why multiple layers?**: Each layer can learn increasingly complex patterns
  - Early layers: Simple patterns (e.g., "luxury", "cheap")
  - Middle layers: Combinations (e.g., "luxury electronics")
  - Deep layers: Complex price indicators

**Layer 8: Output Layer**
```python
self.layer8 = nn.Linear(64, 1)
```
- **Input**: 64 neurons
- **Output**: 1 value (the predicted price)
- **Parameters**: 64 × 1 + 1 = 65 parameters
- **No activation function**: Allows any real number output for price prediction

### Activation Function

**`self.relu = nn.ReLU()`**

ReLU (Rectified Linear Unit) is defined as: **f(x) = max(0, x)**

**Why ReLU?**
- **Non-linearity**: Without activation functions, multiple layers would collapse into a single linear transformation
- **Simplicity**: Computationally efficient (just a comparison and selection)
- **Gradient flow**: Helps prevent vanishing gradient problem in deep networks
- **Sparsity**: Produces sparse activations (many zeros), which can be beneficial

**Example:**
```
Input: [-2, -1, 0, 1, 2]
ReLU Output: [0, 0, 0, 1, 2]
```

### Forward Pass

The `forward` method defines how data flows through the network:

```python
def forward(self, x):
    output1 = self.relu(self.layer1(x))      # 5000 → 128
    output2 = self.relu(self.layer2(output1)) # 128 → 64
    output3 = self.relu(self.layer3(output2)) # 64 → 64
    output4 = self.relu(self.layer4(output3)) # 64 → 64
    output5 = self.relu(self.layer5(output4)) # 64 → 64
    output6 = self.relu(self.layer6(output5)) # 64 → 64
    output7 = self.relu(self.layer7(output6)) # 64 → 64
    output8 = self.layer8(output7)            # 64 → 1 (no ReLU!)
    return output8
```

**Data Flow Visualization:**
```
Input (5000 features)
    ↓ [Linear + ReLU]
128 neurons
    ↓ [Linear + ReLU]
64 neurons
    ↓ [Linear + ReLU]
64 neurons (repeated 5 times)
    ↓ [Linear only]
1 output (predicted price)
```

**Why no ReLU on the final layer?**
- Prices can theoretically be any value
- ReLU would prevent negative predictions
- In log-space, negative values are valid (representing prices < $1)

### Network Characteristics

**Architecture Type**: Feedforward Neural Network (FNN)
- Data flows in one direction: input → hidden layers → output
- No loops or cycles

**Depth**: 8 layers (deep network)
- Allows learning hierarchical representations
- Each layer builds on previous layer's features

**Width**: Varies from 5000 → 128 → 64 → 1
- Funnel architecture: progressively compresses information
- Forces network to learn most important features

**Total Parameters**: ~665,000 trainable parameters
- These are the "knobs" that training adjusts
- Each parameter is a weight or bias value

---

## Data Preparation

### Converting to PyTorch Tensors

```python
X_train_tensor = torch.FloatTensor(X.toarray())
y_train_tensor = torch.FloatTensor(y).unsqueeze(1)
```

#### Understanding X_train_tensor

**`X.toarray()`**
- `X` is a sparse matrix from HashingVectorizer
- **Sparse matrix**: Only stores non-zero values (memory efficient)
  - Example: [0, 0, 1, 0, 0, 0, 1, 0] stored as {2: 1, 6: 1}
- **Dense array**: Stores all values including zeros
  - Example: [0, 0, 1, 0, 0, 0, 1, 0] stored as-is
- Conversion necessary because PyTorch tensors are dense

**`torch.FloatTensor()`**
- Converts NumPy array to PyTorch tensor
- Uses 32-bit floating point precision (float32)
- **Result shape**: (800000, 5000)
  - 800,000 training examples
  - 5,000 features per example

#### Understanding y_train_tensor

**`torch.FloatTensor(y)`**
- Converts price array to PyTorch tensor
- Initial shape: (800000,) - a 1D array

**`.unsqueeze(1)`**
- Adds a dimension at position 1
- Changes shape from (800000,) to (800000, 1)
- **Why?** Neural networks expect 2D output: [batch_size, output_features]

**Visualization:**
```
Before unsqueeze: [10.5, 25.3, 15.7, ...]  # Shape: (800000,)
After unsqueeze:  [[10.5],                  # Shape: (800000, 1)
                   [25.3],
                   [15.7],
                   ...]
```

### Splitting the Data

```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train_tensor, 
    y_train_tensor, 
    test_size=0.01, 
    random_state=42
)
```

**Purpose**: Create separate training and validation sets

**Parameters:**
- **`test_size=0.01`**: Use 1% for validation
  - Validation: 8,000 items (1% of 800,000)
  - Training: 792,000 items (99% of 800,000)
- **`random_state=42`**: Ensures reproducible splits
  - Same random seed = same split every time
  - Important for comparing experiments

**Why Validate?**
- **Training set**: Used to update model weights
- **Validation set**: Used to monitor generalization
- **Prevents overfitting**: If validation error increases while training error decreases, the model is memorizing rather than learning

**Result:**
```
X_train: (792000, 5000) - training features
y_train: (792000, 1)    - training prices
X_val:   (8000, 5000)   - validation features
y_val:   (8000, 1)      - validation prices
```

### Creating the DataLoader

```python
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

#### TensorDataset

**`TensorDataset(X_train, y_train)`**
- Wraps tensors into a dataset object
- Pairs each feature vector with its corresponding price
- Allows indexing: `train_dataset[0]` returns `(features, price)` for first item

**Example:**
```python
train_dataset[0]  # Returns: (tensor([0., 1., 0., ...]), tensor([25.99]))
```

#### DataLoader

**`DataLoader(train_dataset, batch_size=64, shuffle=True)`**

Creates an iterator that feeds data to the model in batches.

**Key Parameters:**

**`batch_size=64`**
- Processes 64 examples at once (not all 792,000)
- **Why batches?**
  1. **Memory efficiency**: Can't fit 792k examples in GPU memory
  2. **Faster training**: GPUs process batches in parallel
  3. **Better gradients**: Batch average provides good gradient estimate
  4. **Regularization**: Noise in batch gradients helps generalization

**`shuffle=True`**
- Randomizes order each epoch
- Prevents learning order-dependent patterns
- Essential for good training

**How it works:**
```python
for batch_X, batch_y in train_loader:
    # batch_X shape: (64, 5000) - 64 examples, 5000 features each
    # batch_y shape: (64, 1)    - 64 prices
    # Process this batch...
```

**Batch Calculation:**
- Total examples: 792,000
- Batch size: 64
- Number of batches per epoch: 792,000 ÷ 64 = 12,375 batches

### Model Initialization

```python
input_size = X_train_tensor.shape[1]  # Gets 5000
model = NeuralNetwork(input_size)
```

**`X_train_tensor.shape[1]`**
- Gets the second dimension (number of features)
- Shape is (800000, 5000), so shape[1] = 5000

**`NeuralNetwork(input_size)`**
- Creates the model with correct input dimensions
- Initializes all weights randomly (typically using Xavier or He initialization)
- Model is ready but untrained - needs gradient descent!

---

## Stochastic Gradient Descent

### The Core Concept

Imagine you're lost in foggy mountains trying to reach the valley (lowest point). You can only see a few feet around you. Gradient descent is like:

1. **Feel the slope** under your feet (compute gradient)
2. **Take a step** downhill (update weights)
3. **Repeat** until you reach the bottom (minimum loss)

In neural networks:
- **Mountain** = Loss landscape (error surface)
- **Valley** = Minimum error (optimal weights)
- **Your position** = Current weights
- **Slope** = Gradient (direction of steepest increase)

### Three Variants of Gradient Descent

#### 1. Batch Gradient Descent (Original)

```python
# Use ALL 792,000 examples to calculate gradient
for epoch in range(num_epochs):
    predictions = model(X_train)  # All data at once
    loss = calculate_loss(predictions, y_train)
    loss.backward()  # Calculate gradient using ALL data
    optimizer.step()  # Update weights once per epoch
```

**Characteristics:**
- **Pros**: 
  - Most accurate gradient
  - Smooth, stable convergence
  - Deterministic (same result every time)
- **Cons**: 
  - VERY SLOW (792k examples per update!)
  - Huge memory requirements
  - Can get stuck in local minima
- **Usage**: Rarely used in modern deep learning

#### 2. Stochastic Gradient Descent (True SGD)

```python
# Use ONE example at a time
for epoch in range(num_epochs):
    for i in range(len(X_train)):
        prediction = model(X_train[i])  # Single example
        loss = calculate_loss(prediction, y_train[i])
        loss.backward()  # Gradient from 1 example
        optimizer.step()  # Update weights immediately
```

**Characteristics:**
- **Pros**: 
  - Very fast updates (792k updates per epoch!)
  - Can escape local minima (noisy gradients)
  - Low memory usage
- **Cons**: 
  - Very noisy, unstable training
  - Inefficient on GPUs (no parallelization)
  - May never converge precisely
- **Usage**: Rarely used in practice

#### 3. Mini-Batch SGD (Standard Practice)

```python
# Use SMALL BATCHES (e.g., 64 examples)
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:  # 64 examples
        predictions = model(batch_X)
        loss = calculate_loss(predictions, batch_y)
        loss.backward()  # Gradient from 64 examples
        optimizer.step()  # Update weights
```

**Characteristics:**
- **Pros**: 
  - Fast and stable
  - GPU-efficient (parallel processing)
  - Good gradient estimates
  - Best of both worlds
- **Cons**: 
  - None really - this is the standard
- **Usage**: This is what everyone uses (confusingly still called "SGD")

### How SGD Works Step-by-Step

Using the code with `batch_size=64`:

```python
# Setup
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
for epoch in range(10):
    for batch_X, batch_y in train_loader:  # 12,375 batches per epoch
        
        # STEP 1: FORWARD PASS
        predictions = model(batch_X)  # Shape: (64, 1)
        # Pass 64 examples through the network
        
        # STEP 2: CALCULATE LOSS
        loss = criterion(predictions, batch_y)  # Single number
        # Measure how wrong the predictions are
        # MSE = mean((predictions - actual)²)
        
        # STEP 3: BACKWARD PASS (compute gradients)
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Calculate ∂loss/∂weight for each weight
        # Uses backpropagation algorithm
        
        # STEP 4: UPDATE WEIGHTS
        optimizer.step()  # weight = weight - learning_rate × gradient
        # Move weights in direction that reduces loss
```

### The Mathematics

For each weight in the network:

```
new_weight = old_weight - learning_rate × gradient
```

**Where:**
- **Gradient** = ∂loss/∂weight (rate of change of loss with respect to weight)
- **Negative gradient** = direction to decrease loss
- **Learning rate** = step size (e.g., 0.01)

**Example:**
```
Current weight: 0.5
Gradient: 2.0 (loss increases as weight increases)
Learning rate: 0.01

New weight = 0.5 - (0.01 × 2.0) = 0.5 - 0.02 = 0.48
```

The weight decreased because the gradient was positive (loss was increasing).

### Why "Stochastic"?

"Stochastic" means **random** or **probabilistic**. Instead of using all data (deterministic), we:

1. **Randomly shuffle** data each epoch
2. **Use random subsets** (batches)
3. **Introduce randomness** in gradient estimates

**Benefits:**
- Helps escape bad local minima
- Provides regularization effect
- Faster convergence in practice

### Visualizing One Epoch

```
Epoch 1: (792,000 examples ÷ 64 = 12,375 batches)
  
  Batch 1:    [64 items] → forward → loss=5.2 → backward → update weights
  Batch 2:    [64 items] → forward → loss=5.1 → backward → update weights
  Batch 3:    [64 items] → forward → loss=5.0 → backward → update weights
  ...
  Batch 12,375: [64 items] → forward → loss=2.3 → backward → update weights
  
Epoch 2: (shuffle data and repeat)
  Batch 1:    [64 items] → forward → loss=2.2 → backward → update weights
  ...
```

### Key Hyperparameters

#### 1. Learning Rate (lr=0.01)

**What it controls**: Size of each step in weight space

**Too large (e.g., 1.0):**
- Overshoots minimum
- Unstable, divergent training
- Loss oscillates or increases

**Too small (e.g., 0.0001):**
- Very slow convergence
- May get stuck in local minima
- Requires many epochs

**Just right (e.g., 0.01):**
- Steady decrease in loss
- Converges in reasonable time
- Finds good solution

**Visualization:**
```
Large LR:  ↓↑↓↑↓↑  (oscillating)
Small LR:  ↓.↓.↓.  (slow but steady)
Good LR:   ↓↓↓.↓.  (fast then steady)
```

#### 2. Batch Size (batch_size=64)

**What it controls**: Number of examples per gradient update

**Larger batches (e.g., 512):**
- More stable gradients
- Better GPU utilization
- More memory required
- Fewer updates per epoch
- May generalize worse

**Smaller batches (e.g., 16):**
- Noisier gradients (can be good!)
- Less memory required
- More updates per epoch
- May generalize better
- Less GPU efficient

**Common choices**: 32, 64, 128, 256

#### 3. Number of Epochs

**What it controls**: How many times to see all training data

**Too few (e.g., 1-2):**
- Underfitting
- Model hasn't learned enough
- High training and validation error

**Too many (e.g., 100+):**
- Overfitting
- Model memorizes training data
- Low training error, high validation error

**Just right (e.g., 10-20):**
- Good balance
- Monitor validation loss to decide

### Why Mini-Batch SGD Works

Despite using only 64 examples at a time (not all 792k), the gradient is a **good enough estimate** of the true direction to minimize loss.

**Key insight**: 
- The average gradient over a batch approximates the true gradient
- Over many batches, errors average out
- The randomness actually helps escape local minima

**Mathematical justification:**
```
True gradient ≈ Average of batch gradients
E[gradient_batch] = gradient_full_dataset
```

By the law of large numbers, the average of many batch gradients converges to the true gradient.

---

## Complete Code Walkthrough

### Full Training Example

```python
# 1. PREPARE DATA
y = np.array([float(item.price) for item in train])
documents = [item.summary for item in train]

# 2. VECTORIZE TEXT
vectorizer = HashingVectorizer(n_features=5000, stop_words='english', binary=True)
X = vectorizer.fit_transform(documents)

# 3. CONVERT TO TENSORS
X_train_tensor = torch.FloatTensor(X.toarray())
y_train_tensor = torch.FloatTensor(y).unsqueeze(1)

# 4. SPLIT DATA
X_train, X_val, y_train, y_val = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.01, random_state=42
)

# 5. CREATE DATALOADER
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 6. INITIALIZE MODEL
input_size = X_train_tensor.shape[1]
model = NeuralNetwork(input_size)

# 7. SETUP TRAINING
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 8. TRAINING LOOP
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, y_val)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"  Val Loss: {val_loss.item():.4f}")

# 9. MAKE PREDICTIONS
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
```

### Key Concepts Summary

**Neural Network**: A function approximator with learnable parameters (weights and biases)

**Forward Pass**: Data flows through layers to produce predictions

**Loss Function**: Measures how wrong predictions are (MSE for regression)

**Backward Pass**: Computes gradients using backpropagation

**Optimizer**: Updates weights using gradients (SGD, Adam, etc.)

**Batch Processing**: Process multiple examples simultaneously for efficiency

**Epochs**: Complete passes through the training data

**Validation**: Monitor performance on unseen data to prevent overfitting

---

## Conclusion

This neural network approach to price prediction demonstrates:

1. **Deep learning fundamentals**: Layers, activations, forward/backward passes
2. **Practical PyTorch**: Tensors, DataLoaders, training loops
3. **Optimization**: Stochastic gradient descent and its variants
4. **Best practices**: Train/validation splits, batch processing, monitoring

The 8-layer architecture progressively compresses text features (5000 → 128 → 64 → 1) while learning which patterns correlate with price. Through iterative gradient descent, the network discovers complex relationships between product descriptions and prices that would be difficult to encode manually.

----------------------
## Neural Networks and LLMs for Price Prediction

---

## Table of Contents
1. [Import Statements - Deep Dive](#import-statements)
2. [Environment Setup](#environment-setup)
3. [Data Loading](#data-loading)
4. [Human Neural Network Experiment](#human-neural-network)
5. [Artificial Neural Network](#artificial-neural-network)
6. [Training Process](#training-process)
7. [Frontier Models](#frontier-models)

---

## Import Statements - Deep Dive

### Standard Library Imports

```python
import os
```
**Purpose**: Operating system interface
- Access environment variables: `os.environ['HF_TOKEN']`
- File path operations
- System-level operations

```python
import csv
```
**Purpose**: CSV file reading/writing
- `csv.writer()`: Write data to CSV files
- `csv.reader()`: Read data from CSV files
- Handles proper escaping and formatting

### Environment Management

```python
from dotenv import load_dotenv
```
**What it does**: Loads environment variables from `.env` file
**Why needed**: Keeps API keys and secrets out of code
**Usage**:
```python
load_dotenv(override=True)  # Loads .env file, overrides existing vars
```

### Hugging Face Integration

```python
from huggingface_hub import login
```
**Purpose**: Authenticate with Hugging Face Hub
**What it does**: 
- Stores authentication token
- Enables access to private models/datasets
- Required for downloading datasets

**Usage**:
```python
login(token, add_to_git_credential=True)
```

### Custom Project Modules

```python
from pricer.evaluator import evaluate
```
**Purpose**: Custom evaluation function for price predictions
**What it does**:
- Takes a prediction function and test data
- Calculates error metrics (MAE, MSE, R²)
- Generates visualization plots
- Shows color-coded results (red=bad, green=good, orange=okay)

```python
from pricer.items import Item
```
**Purpose**: Custom class for product items
**Attributes**:
- `item.summary`: Product description text
- `item.price`: Actual price
**Methods**:
- `Item.from_hub(dataset)`: Load from Hugging Face

```python
from litellm import completion
```
**Purpose**: Unified interface for multiple LLM APIs
**Supports**: OpenAI, Anthropic, Google, xAI, etc.
**Usage**:
```python
response = completion(
    model="openai/gpt-4.1-nano",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Data Science Libraries

```python
import numpy as np
```
**Purpose**: Numerical computing library
**Key features**:
- Fast array operations
- Mathematical functions
- Random number generation
**Usage in code**:
```python
y = np.array([float(item.price) for item in train])
np.random.seed(42)  # Reproducible randomness
```

```python
from tqdm.notebook import tqdm
```
**Purpose**: Progress bars for Jupyter notebooks
**What it does**: Shows visual progress during loops
**Usage**:
```python
for batch in tqdm(train_loader):
    # Shows: [████████] 100% 12375/12375
```

### Text Processing

```python
from sklearn.feature_extraction.text import HashingVectorizer
```
**Purpose**: Convert text to numerical features
**How it works**:
1. Takes text documents
2. Splits into words (tokens)
3. Hashes each word to a fixed position (0-4999)
4. Creates sparse binary vector

**Parameters**:
- `n_features=5000`: Output vector size
- `stop_words='english'`: Remove common words (the, a, is)
- `binary=True`: 1 if word present, 0 if absent

**Example**:
```python
vectorizer = HashingVectorizer(n_features=5000, stop_words='english', binary=True)
X = vectorizer.fit_transform(["laptop computer", "gaming laptop"])
# Result: Two 5000-dimensional sparse vectors
```

### PyTorch Deep Learning

```python
import torch
```
**Purpose**: Main PyTorch library
**What it provides**:
- Tensor operations (like NumPy but GPU-enabled)
- Automatic differentiation
- Neural network building blocks

**Key concepts**:
- **Tensor**: Multi-dimensional array (like NumPy array)
- **GPU support**: `.cuda()` moves tensors to GPU
- **Autograd**: Automatic gradient computation

```python
import torch.nn as nn
```
**Purpose**: Neural network modules
**Key classes**:
- `nn.Module`: Base class for all neural networks
- `nn.Linear`: Fully connected layer
- `nn.ReLU`: Activation function
- `nn.MSELoss`: Mean Squared Error loss

```python
import torch.optim as optim
```
**Purpose**: Optimization algorithms
**Key optimizers**:
- `optim.SGD`: Stochastic Gradient Descent
- `optim.Adam`: Adaptive Moment Estimation (usually better than SGD)
- `optim.AdamW`: Adam with weight decay

```python
from torch.utils.data import DataLoader, TensorDataset
```
**Purpose**: Data loading utilities

**TensorDataset**:
- Wraps tensors into a dataset
- Pairs inputs with targets
```python
dataset = TensorDataset(X_train, y_train)
dataset[0]  # Returns (features, label) tuple
```

**DataLoader**:
- Batches data
- Shuffles data
- Parallel loading
```python
loader = DataLoader(dataset, batch_size=64, shuffle=True)
for batch_X, batch_y in loader:
    # Process 64 examples at a time
```

### Machine Learning Utilities

```python
from sklearn.model_selection import train_test_split
```
**Purpose**: Split data into train/validation/test sets
**Parameters**:
- `test_size=0.01`: Use 1% for validation
- `random_state=42`: Reproducible splits
**Returns**: X_train, X_val, y_train, y_val

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
```
**Purpose**: Learning rate scheduling
**What it does**: Gradually reduces learning rate following cosine curve
**Why useful**: 
- Start with large steps (fast learning)
- End with small steps (fine-tuning)
- Helps avoid overshooting minimum

**Usage**:
```python
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
for epoch in range(num_epochs):
    train()
    scheduler.step()  # Adjust learning rate
```

---

## Environment Setup

```python
LITE_MODE = False

load_dotenv(override=True)
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)
```

### Line-by-Line Explanation

**`LITE_MODE = False`**
- Controls dataset size
- `False`: Full dataset (800k items)
- `True`: Lite dataset (smaller, faster)

**`load_dotenv(override=True)`**
- Reads `.env` file in project root
- Loads variables into `os.environ`
- `override=True`: Replaces existing environment variables

**`hf_token = os.environ['HF_TOKEN']`**
- Retrieves Hugging Face token from environment
- Token format: `hf_xxxxxxxxxxxxxxxxxxxxx`
- Required for private datasets

**`login(hf_token, add_to_git_credential=True)`**
- Authenticates with Hugging Face
- `add_to_git_credential=True`: Attempts to save to git credentials
- May show warning if git credential helper not configured

---

## Data Loading

```python
username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")
```

### Detailed Breakdown

**Dataset Selection**:
- Lite: `ed-donner/items_lite` (smaller dataset)
- Full: `ed-donner/items_full` (800k training items)

**`Item.from_hub(dataset)`**:
- Custom method that downloads from Hugging Face
- Returns three lists: train, val, test
- Each item has `.summary` (text) and `.price` (float)

**Output**: `Loaded 800,000 training items, 10,000 validation items, 10,000 test items`

**Data Split**:
- **Training (800k)**: Used to train the model
- **Validation (10k)**: Monitor performance during training
- **Test (10k)**: Final evaluation (never seen during training)

---

## Human Neural Network Experiment

This section demonstrates that humans can be thought of as a "neural network" for price prediction.

### Writing Test Data

```python
with open('human_in.csv', 'w', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for t in test[:100]:
        writer.writerow([t.summary, 0])
```

**Purpose**: Create CSV for human to fill in price guesses

**Process**:
1. Opens file in write mode
2. Creates CSV writer
3. Writes first 100 test items
4. Format: `[description, 0]` (0 is placeholder for human guess)

**Example CSV content**:
```
"Old Blood Noise Excess V2 Distortion Cho...",0
"telpo Headlight Assembly Fit For 15 Camr...",0
```

### Reading Human Predictions

```python
human_predictions = []
with open('human_out.csv', 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        human_predictions.append(float(row[1]))
```

**Purpose**: Load human's price guesses

**Process**:
1. Human manually edits CSV, replacing 0 with price guess
2. Code reads second column (index 1)
3. Converts to float
4. Stores in list

### Human Pricer Function

```python
def human_pricer(item):
    idx = test.index(item)
    return human_predictions[idx]
```

**How it works**:
1. Find item's position in test list
2. Return corresponding human prediction
3. Matches evaluation function signature

### Evaluation

```python
evaluate(human_pricer, test, size=100)
```

**Results shown**:
- Average error: ~$87.62
- Color-coded predictions (red/green/orange)
- Scatter plot comparing predictions vs actual
- Confidence intervals

**Key insight**: Humans are reasonably good at price estimation but not perfect!

---

## Artificial Neural Network

### Data Preparation

```python
y = np.array([float(item.price) for item in train])
documents = [item.summary for item in train]
```

**`y = np.array([...])`**:
- Extracts all prices from training data
- Converts to NumPy array
- Shape: (800000,)
- Example: [219.0, 115.99, 144.29, ...]

**`documents = [...]`**:
- Extracts all product descriptions
- Python list of strings
- Length: 800,000
- Example: ["Old Blood Noise Excess...", "telpo Headlight..."]

### Text Vectorization

```python
np.random.seed(42)
vectorizer = HashingVectorizer(n_features=5000, stop_words='english', binary=True)
X = vectorizer.fit_transform(documents)
```

**`np.random.seed(42)`**:
- Sets random seed for reproducibility
- HashingVectorizer uses randomness internally
- Same seed = same results every time

**HashingVectorizer Process**:

1. **Tokenization**: Split text into words
   ```
   "Laptop Computer 15 inch" → ["Laptop", "Computer", "15", "inch"]
   ```

2. **Stop word removal**: Remove common words
   ```
   ["Laptop", "Computer", "15", "inch"] → ["Laptop", "Computer", "15", "inch"]
   # "the", "a", "is" would be removed
   ```

3. **Hashing**: Map each word to position 0-4999
   ```
   "Laptop" → hash → position 2847
   "Computer" → hash → position 193
   ```

4. **Binary vector**: Set positions to 1
   ```
   [0, 0, ..., 1, ..., 1, ..., 0]  # 5000 elements
   ```

**Result `X`**:
- Sparse matrix (800000, 5000)
- Each row = one product
- Each column = one hash bucket
- Values = 0 or 1

### Neural Network Definition

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

**Already explained in detail in previous response - see neural_network_explanation.md**

### Tensor Conversion

```python
X_train_tensor = torch.FloatTensor(X.toarray())
y_train_tensor = torch.FloatTensor(y).unsqueeze(1)
```

**Already explained in detail in previous response**

### Data Splitting

```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.01, random_state=42
)
```

**Already explained in detail in previous response**

### DataLoader Creation

```python
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

**Already explained in detail in previous response**

### Model Initialization

```python
input_size = X_train_tensor.shape[1]
model = NeuralNetwork(input_size)
```

**Already explained in detail in previous response**

### Parameter Count

```python
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params:,}")
```

**Output**: `Number of trainable parameters: 669,249`

**Breakdown**:
- Layer 1: 5000 × 128 + 128 = 640,128
- Layer 2: 128 × 64 + 64 = 8,256
- Layers 3-7: (64 × 64 + 64) × 5 = 20,800
- Layer 8: 64 × 1 + 1 = 65
- **Total**: 669,249 parameters

**What are parameters?**
- Weights and biases that the model learns
- Each connection between neurons has a weight
- Each neuron has a bias
- Training adjusts these to minimize error

---

## Training Process

```python
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 2

for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in tqdm(train_loader):
        optimizer.zero_grad()
        
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

### Loss Function

```python
loss_function = nn.MSELoss()
```

**MSE (Mean Squared Error)**:
```
MSE = (1/n) × Σ(predicted - actual)²
```

**Why squared?**
- Penalizes large errors more
- Always positive
- Differentiable (needed for gradients)

**Example**:
```
Predicted: [100, 200, 150]
Actual:    [120, 180, 160]
Errors:    [-20,  20, -10]
Squared:   [400, 400, 100]
MSE:       (400 + 400 + 100) / 3 = 300
```

### Optimizer

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Adam (Adaptive Moment Estimation)**:
- More sophisticated than SGD
- Adapts learning rate for each parameter
- Uses momentum (considers past gradients)
- Generally converges faster

**`lr=0.001`**: Learning rate
- Controls step size
- 0.001 is a common default
- Too large: unstable, overshoots
- Too small: slow convergence

### Training Loop Breakdown

**Outer loop**: `for epoch in range(EPOCHS):`
- One epoch = one pass through all training data
- 2 epochs = see all 792,000 examples twice

**`model.train()`**:
- Sets model to training mode
- Enables dropout, batch normalization (if present)
- Tracks gradients

**Inner loop**: `for batch_X, batch_y in tqdm(train_loader):`
- Processes 64 examples at a time
- 792,000 ÷ 64 = 12,375 iterations per epoch
- `tqdm` shows progress bar

**Four Training Steps**:

1. **`optimizer.zero_grad()`**
   - Clears old gradients
   - PyTorch accumulates gradients by default
   - Must clear before each batch

2. **`outputs = model(batch_X)`** - FORWARD PASS
   - Pass batch through network
   - Input: (64, 5000)
   - Output: (64, 1)
   - Calls `forward()` method

3. **`loss = loss_function(outputs, batch_y)`** - COMPUTE LOSS
   - Compare predictions to actual prices
   - Returns single number (average error for batch)
   - Example: loss = 12484.0

4. **`loss.backward()`** - BACKWARD PASS
   - Computes gradients using backpropagation
   - Calculates ∂loss/∂weight for every parameter
   - Automatic differentiation (autograd)

5. **`optimizer.step()`** - UPDATE WEIGHTS
   - Adjusts weights using gradients
   - Formula: `weight = weight - lr × gradient`
   - Adam uses more complex update rule

### Validation

```python
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    val_loss = loss_function(val_outputs, y_val)
```

**`model.eval()`**:
- Sets model to evaluation mode
- Disables dropout, batch normalization updates

**`with torch.no_grad():`**:
- Disables gradient computation
- Saves memory
- Faster inference
- Not training, just evaluating

**Validation purpose**:
- Check if model generalizes
- Detect overfitting
- Decide when to stop training

### Training Output

```
Epoch [1/2], Train Loss: 12484.000, Val Loss: 12042.084
Epoch [2/2], Train Loss: 6236.586, Val Loss: 10925.284
```

**Observations**:
- Train loss decreases (good!)
- Val loss decreases but less (some overfitting)
- Model is learning patterns

---

## Making Predictions

```python
def neural_network(item):
    model.eval()
    with torch.no_grad():
        vector = vectorizer.transform([item.summary])
        vector = torch.FloatTensor(vector.toarray())
        result = model(vector)[0].item()
    return max(0, result)
```

### Step-by-Step

1. **`model.eval()`**: Set to evaluation mode

2. **`vector = vectorizer.transform([item.summary])`**:
   - Convert text to 5000-dim vector
   - Same process as training
   - Must use same vectorizer!

3. **`vector = torch.FloatTensor(vector.toarray())`**:
   - Convert to PyTorch tensor
   - Shape: (1, 5000)

4. **`result = model(vector)[0].item()`**:
   - Pass through network
   - `[0]`: Get first (only) prediction
   - `.item()`: Convert tensor to Python float

5. **`return max(0, result)`**:
   - Ensure non-negative price
   - Model might predict negative values

### Evaluation

```python
evaluate(neural_network, test)
```

**Results**:
- Error: ~$63.42
- Much better than human ($87.62)!
- R² = 59.7% (explains 60% of variance)

---

## Frontier Models

### Message Formatting

```python
def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\\n\\n{item.summary}"
    return [{"role": "user", "content": message}]
```

**Purpose**: Format prompt for LLM APIs

**Structure**:
- Clear instruction
- Product description
- Request concise response

**Example**:
```python
messages_for(test[0])
# Returns:
[{
    "role": "user",
    "content": "Estimate the price of this product. Respond with the price, no explanation\n\nOld Blood Noise Excess V2 Distortion Cho..."
}]
```

### GPT-4.1-nano

```python
def gpt_4__1_nano(item):
    response = completion(model="openai/gpt-4.1-nano", messages=messages_for(item))
    return response.choices[0].message.content
```

**Model**: OpenAI's smallest GPT-4 variant
**Cost**: Very cheap
**Speed**: Fast
**Accuracy**: Good for simple tasks

### Claude Opus 4.5

```python
def claude_opus_4_5(item):
    response = completion(model="anthropic/claude-opus-4-5", messages=messages_for(item))
    return response.choices[0].message.content
```

**Model**: Anthropic's most capable model
**Cost**: Expensive
**Speed**: Slower
**Accuracy**: Excellent reasoning

### Gemini 3 Pro Preview

```python
def gemini_3_pro_preview(item):
    response = completion(
        model="gemini/gemini-3-pro-preview", 
        messages=messages_for(item), 
        reasoning_effort='low'
    )
    return response.choices[0].message.content
```

**Model**: Google's reasoning model
**`reasoning_effort='low'`**: Faster, less thorough
**Options**: 'low', 'medium', 'high'

### Gemini 2.5 Flash Lite

```python
def gemini_2__5_flash_lite(item):
    response = completion(model="gemini/gemini-2.5-flash-lite", messages=messages_for(item))
    return response.choices[0].message.content
```

**Model**: Google's fast, lightweight model
**Cost**: Very cheap
**Speed**: Very fast

### Grok 4.1 Fast

```python
def grok_4__1_fast(item):
    response = completion(
        model="xai/grok-4-1-fast-non-reasoning", 
        messages=messages_for(item), 
        seed=42
    )
    return response.choices[0].message.content
```

**Model**: xAI's fast model
**`seed=42`**: Reproducible outputs
**`non-reasoning`**: Direct answers, no chain-of-thought

### GPT-5.1

```python
def gpt_5__1(item):
    response = completion(
        model="gpt-5.1", 
        messages=messages_for(item), 
        reasoning_effort='high', 
        seed=42
    )
    return response.choices[0].message.content
```

**Model**: OpenAI's reasoning model
**`reasoning_effort='high'`**: Most thorough reasoning
**Cost**: Most expensive
**Accuracy**: Best performance

---

## Key Takeaways

1. **Text Vectorization**: Convert text to numbers using HashingVectorizer
2. **Neural Networks**: Stack of linear layers + activations
3. **Training**: Forward pass → loss → backward pass → update weights
4. **Batch Processing**: Process multiple examples simultaneously
5. **Validation**: Monitor generalization to prevent overfitting
6. **Frontier Models**: State-of-the-art LLMs can predict without training
7. **Trade-offs**: Speed vs accuracy vs cost

---





