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

---
