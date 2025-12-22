## **Building Advanced RAG with ChromaDB Vector Stores (No LangChain), EnsembleAgent (RAG with chatgpt 5+ llama Fine-Tuned + In house neuronal network)**

### **Overall System Design**
The project builds an autonomous deal-finding system with a hierarchical agent architecture:

```
┌─────────────────────────────────────────┐
│      PLANNING AGENT (Step 4)            │
│  Coordinates entire workflow            │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┬──────────────┐
    ▼                     ▼              ▼
┌─────────┐      ┌──────────────┐   ┌─────────┐
│ SCANNER │      │   ENSEMBLE   │   │MESSENGER│
│  AGENT  │      │    AGENT     │   │  AGENT  │
│ (Step 3)│      │   (Step 2)   │   │ (Step 3)│
└─────────┘      └──────┬───────┘   └─────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌──────────┐   ┌─────────┐
    │SPECIAL- │   │FRONTIER  │   │ NEURAL  │
    │  IST    │   │  AGENT   │   │ NETWORK │
    │ AGENT   │   │ (RAG)    │   │  AGENT  │
    └─────────┘   └──────────┘   └─────────┘
```

---

## **Step 2 FOCUS: ENSEMBLE PRICING WITH RAG**
#### **A. SPECIALIST AGENT** (Fine-tuned Model on Modal)
```python
class SpecialistAgent(Agent):
    name = "Specialist Agent"
    color = Agent.RED
    
    def __init__(self):
        # Connects to Modal cloud service
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()
    
    def price(self, description: str) -> float:
        # Remote call to fine-tuned Llama model
        result = self.pricer.price.remote(description)
        return result
```

**Key Points:**
- Uses fine-tuned Llama 3.2 (4-bit quantized)
- Deployed on Modal.com (serverless GPU)
- Best performance so far: **$39.85 error**
- Trained on 800K Amazon products

---

#### **B. FRONTIER AGENT** (RAG with GPT-5.1)
```python
class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE
    MODEL = "gpt-5.1"
    
    def __init__(self, collection):
        self.client = OpenAI()
        self.collection = collection  # ChromaDB
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def find_similars(self, description: str):
        # 1. Encode description to 384-dim vector
        vector = self.model.encode([description])
        
        # 2. Query ChromaDB for 5 similar products
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(),
            n_results=5
        )
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        return documents, prices
    
    def price(self, description: str) -> float:
        # 3. Get similar products
        documents, prices = self.find_similars(description)
        
        # 4. Build RAG prompt with context
        message = f"Estimate price of: {description}\n\n"
        message += "Context - similar products:\n"
        for doc, price in zip(documents, prices):
            message += f"{doc}\nPrice: ${price:.2f}\n\n"
        
        # 5. Call GPT-5.1 with reasoning_effort="none"
        response = self.client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": message}],
            reasoning_effort="none",
            seed=42
        )
        return self.get_price(response.choices[0].message.content)
```

**RAG Pipeline Breakdown:**
1. **Vectorization**: Text → 384-dim vector (all-MiniLM-L6-v2)
2. **Similarity Search**: Find 5 nearest neighbors in ChromaDB
3. **Context Injection**: Add similar products + prices to prompt
4. **Inference**: GPT-5.1 estimates with context
5. **Result**: **$30.19 error** (beats fine-tuning with LoRA on LLAMA 3.2!)

---

#### **C. NEURAL NETWORK AGENT** (Deep Residual Network)
```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.block(x) + x)  # Skip connection

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, num_layers=10, 
                 hidden_size=4096, dropout_prob=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_prob) 
            for _ in range(num_layers - 2)
        ])
        self.output_layer = nn.Linear(hidden_size, 1)
```

**Architecture:**
- Input: 5000 features (HashingVectorizer)
- 10 layers with residual connections
- 4096 hidden units per layer
- LayerNorm + Dropout for regularization
- Performance: **~$46 error**

---

### **2. ENSEMBLE AGENT** (Combining All Three)

```python
class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW
    
    def __init__(self, collection):
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.neural_network = NeuralNetworkAgent()
        self.preprocessor = Preprocessor()
    
    def price(self, description: str) -> float:
        # 1. Preprocess text
        rewrite = self.preprocessor.preprocess(description)
        
        # 2. Get predictions from all 3 models
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        neural_network = self.neural_network.price(rewrite)
        
        # 3. Weighted combination
        combined = (frontier * 0.8 + 
                   specialist * 0.1 + 
                   neural_network * 0.1)
        
        return combined
```

**Ensemble Weights:**
- **80%** Frontier (RAG) - strongest model
- **10%** Specialist (fine-tuned)
- **10%** Neural Network

**Final Result: $29.90 error** (87% R²)

---

### **3. CHROMADB VECTOR STORE SETUP**

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize
client = chromadb.PersistentClient(path="products_vectorstore")
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create collection
collection = client.create_collection("products")

# Populate with 800K products (takes ~30 min on GPU)
for i in range(0, len(train), 1000):
    documents = [item.summary for item in train[i:i+1000]]
    vectors = encoder.encode(documents).astype(float).tolist()
    metadatas = [{"category": item.category, "price": item.price} 
                 for item in train[i:i+1000]]
    ids = [f"doc_{j}" for j in range(i, i+1000)]
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas
    )
```

**Vector Store Details:**
- **Encoder**: all-MiniLM-L6-v2 (384 dimensions)
- **Database**: ChromaDB (open-source, persistent)
- **Size**: 800K products from Amazon
- **Query Time**: Fast similarity search via HNSW index

---

### **4. PREPROCESSOR** (Text Rewriting)

```python
class Preprocessor:
    SYSTEM_PROMPT = """Create a concise description of a product.
    Respond only in this format:
    Title: Rewritten short precise title
    Category: eg Electronics
    Brand: Brand name
    Description: 1 sentence description
    Details: 1 sentence on features"""
    
    def __init__(self, model_name="ollama/llama3.2"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434" if "ollama" in model_name else None
    
    def preprocess(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
        response = completion(
            messages=messages,
            model=self.model_name,
            api_base=self.base_url
        )
        return response.choices[0].message.content
```

**Purpose:**
- Standardizes product descriptions
- Improves consistency across models
- Uses local Ollama or cloud LLM

---

### **5. AGENT BASE CLASS** (Logging Infrastructure)

```python
class Agent:
    # ANSI color codes
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    BG_BLACK = '\033[40m'
    RESET = '\033[0m'
    
    name: str = ""
    color: str = '\033[37m'
    
    def log(self, message):
        color_code = self.BG_BLACK + self.color
        message = f"[{self.name}] {message}"
        logging.info(color_code + message + self.RESET)
```

**Benefits:**
- Color-coded logs for each agent
- Easy debugging of multi-agent workflows
- Consistent logging interface

---

## **KEY TECHNICAL CONCEPTS**

### **A. RAG (Retrieval Augmented Generation)**
**Definition**: Enhance LLM responses by retrieving relevant context from external knowledge base

**Pipeline:**
```
User Query → Vectorize → Similarity Search → Retrieve Context → 
Inject into Prompt → LLM Generation → Response
```

**Why It Works:**
- Provides specific examples (5 similar products with prices)
- Grounds LLM in real data
- No training required (inference-time technique)
- Beats fine-tuning: $30.19 vs $39.85

---

### **B. Ensemble Learning**
**Theory**: Combine multiple models to reduce variance and improve accuracy

**Example:**
```
Model A: Always off by +$10 or -$10
Model B: Always off by +$10 or -$10

If errors are uncorrelated:
- Sometimes both +$10 → Average = +$10
- Sometimes A=+$10, B=-$10 → Average = $0 ✓
- Average error < individual errors
```

**Linear Combination:**
```python
ensemble_price = w1*model1 + w2*model2 + w3*model3
# Weights determined by validation performance
```

---

### **C. Vector Embeddings**
**Purpose**: Convert text to numerical representation capturing semantic meaning

**all-MiniLM-L6-v2:**
- Input: Text string
- Output: 384-dimensional vector
- Property: Similar texts → similar vectors (cosine similarity)

**Visualization (t-SNE):**
- 384D → 2D/3D for plotting
- Products cluster by category
- Toy car parts near automotive parts (semantic similarity)

---

### **D. Modal.com Deployment**
**Serverless GPU Inference:**
```python
import modal

app = modal.App("pricer-service")

@app.cls(gpu="A10G", container_idle_timeout=300)
class Pricer:
    @modal.method()
    def price(self, description: str) -> float:
        # Run fine-tuned model
        return prediction
```

**Benefits:**
- Pay-per-use GPU
- Auto-scaling
- Cold start: ~30s, then fast
- Remote calls via `pricer.price.remote(text)`

---

## **PERFORMANCE COMPARISON**

| Model | Error | R² | Method |
|-------|-------|-----|--------|
| Baseline (ML) | $87.60 | - | Random Forest |
| GPT-4o | $44.74 | - | Zero-shot |
| Fine-tuned Llama | $39.85 | 82% | Training |
| **GPT-5.1 + RAG** | **$30.19** | **75%** | **Inference** |
| **Ensemble** | **$29.90** | **87%** | **Combined** |

**Key Insight**: RAG (inference-time) now beats fine-tuning (training-time)!

---

## **PRACTICAL IMPLEMENTATION NOTES**

### **1. ChromaDB Population**
```python
# Takes 30-60 minutes for 800K items
# Can use LITE_MODE = True for 20K items (faster)
# Remove check to force rebuild:
if collection_name not in existing_collection_names:
    # populate...
```

### **2. Neural Network Weights**
- Download `deep_neural_network.pth` (1GB+)
- Place in project directory
- Load with PyTorch

### **3. Ensemble Tuning**
```python
# Current weights (rough estimate)
combined = frontier * 0.8 + specialist * 0.1 + neural * 0.1

# Better approach: Linear regression on validation set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(predictions_val, actual_prices_val)
# Use lr.coef_ as weights
```

### **4. Cost Optimization**
- Use `reasoning_effort="none"` for GPT-5.1 (cheaper)
- Ollama for preprocessing (free, local)
- Modal cold starts: keep warm with periodic pings

---

## **AGENT WORKFLOW EXAMPLE**

```python
# User asks: "How much is this TV?"
description = "Samsung 55-inch 4K Smart TV"

# 1. Ensemble Agent receives request
ensemble = EnsembleAgent(collection)

# 2. Preprocess text
rewrite = preprocessor.preprocess(description)
# → "Title: Samsung 4K Smart TV\nCategory: Electronics..."

# 3. Specialist Agent (Modal)
specialist_price = specialist.price(rewrite)  # $450

# 4. Frontier Agent (RAG)
similars, prices = frontier.find_similars(rewrite)
# → ["Sony 55\" 4K TV - $480", "LG 55\" 4K TV - $420", ...]
frontier_price = frontier.price(rewrite)  # $445

# 5. Neural Network Agent
nn_price = neural_network.price(rewrite)  # $460

# 6. Combine
final = 0.8*445 + 0.1*450 + 0.1*460  # $447

# 7. Return to user
print(f"Estimated price: ${final:.2f}")
```

---

# CODE tested step by step and reassembled in code directory
### **Step 1: Import Required Libraries**
```python
# imports

import os
import logging
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from litellm import completion
from tqdm.notebook import tqdm
from agents.evaluator import evaluate
from agents.items import Item
```

### **Step 2: Environment Setup**
```python
# environment

load_dotenv(override=True)
DB = "products_vectorstore"
```

### **Step 3: HuggingFace Login**
```python
# Log in to HuggingFace
# If you don't have a HuggingFace account, you can set one up for free at www.huggingface.co
# And then add the HF_TOKEN to your .env file as explained in the project README

hf_token = os.environ['HF_TOKEN']
login(token=hf_token, add_to_git_credential=False)
```

### **Step 4: Set Lite Mode and Load Dataset**
```python
LITE_MODE = False

username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")
```

### **Step 5: Initialize ChromaDB Client**
```python
client = chromadb.PersistentClient(path=DB)
```

### **Step 6: Initialize SentenceTransformer Encoder**
```python
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

### **Step 7: Test Encoding (Example)**
```python
# Pass in a list of texts, get back a numpy array of vectors

vector = encoder.encode(["A proficient AI engineer who has almost reached the finale of AI Engineering Core Track!"])[0]
print(vector.shape)
vector
```

### **Step 8: Populate ChromaDB with Product Vectors**
```python
# Check if the collection exists; if not, create it

collection_name = "products"

existing_collection_names = [collection.name for collection in client.list_collections()]
#train = train[:10_000] #took 10000 instead of 800000 computer crashes!
#if collection_name in existing_collection_names:
#    client.delete_collection(collection_name)
if collection_name not in existing_collection_names:
    collection = client.create_collection(collection_name)
    for i in tqdm(range(0, len(train), 1000)):
        documents = [item.summary for item in train[i: i+1000]]
        vectors = encoder.encode(documents).astype(float).tolist()
        metadatas = [{"category": item.category, "price": item.price} for item in train[i: i+1000]]
        ids = [f"doc_{j}" for j in range(i, i+1000)]
        ids = ids[:len(documents)]
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

collection = client.get_or_create_collection(collection_name)
```

### **Step 9: Visualize Vectorized Data (Setup)**
```python
# It is very fun turning this up to 800_000 and seeing the full dataset visualized,
# but it almost crashes my box every time so do that at your own risk!! 10_000 is safe!

MAXIMUM_DATAPOINTS = 10_000

CATEGORIES = ['Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics','Musical_Instruments', 'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games']
COLORS = ['cyan', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'red']
```

### **Step 10: Prepare Data for Visualization**
```python
# Prework
result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=MAXIMUM_DATAPOINTS)
vectors = np.array(result['embeddings'])
documents = result['documents']
categories = [metadata['category'] for metadata in result['metadatas']]
colors = [COLORS[CATEGORIES.index(c)] for c in categories]
```

### **Step 11: Create 2D Visualization with TSNE**
```python
# Let's try a 2D chart
# TSNE stands for t-distributed Stochastic Neighbor Embedding - it's a common technique for reducing dimensionality of data

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=4, color=colors, opacity=0.7),
    text=[f"Category: {c}<br>Text: {d[:50]}..." for c, d in zip(categories, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vectorstore Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y'),
    width=1200,
    height=800,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()
```

### **Step 12: Create 3D Visualization with TSNE**
```python
# Let's try 3D!

tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=2, color=colors, opacity=0.7),
    text=[f"Category: {c}<br>Text: {d[:50]}..." for c, d in zip(categories, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=1200,
    height=800,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()
```

### **Step 13: Test Item and Vector Function**
```python
test[0]

def vector(item):
    return encoder.encode(item.summary)
```

### **Step 14: Find Similar Products Function**
```python
def find_similars(item):
    vec = vector(item)
    results = collection.query(query_embeddings=vec.astype(float).tolist(), n_results=5)
    documents = results['documents'][0][:]
    prices = [m['price'] for m in results['metadatas'][0][:]]
    return documents, prices

find_similars(test[0])
```

### **Step 15: Create Context for RAG**
```python
# We need to give some context to GPT-5.1 by selecting 5 products with similar descriptions

def make_context(similars, prices):
    message = "For context, here are some other items that might be similar to the item you need to estimate.\n\n"
    for similar, price in zip(similars, prices):
        message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
    return message

documents, prices = find_similars(test[0])
print(make_context(documents, prices))
```

### **Step 16: Create Messages for GPT-5.1**
```python
def messages_for(item, similars, prices):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}\n\n"
    message += make_context(similars, prices)
    return [{"role": "user", "content": message}]

documents, prices = find_similars(test[0])
print(messages_for(test[0], documents, prices)[0]['content'])
```

### **Step 17: GPT-5.1 RAG Function**
```python
# The function for gpt-5-mini

def gpt_5__1_rag(item):
    documents, prices = find_similars(item)
    response = completion(model="gpt-5.1", messages=messages_for(item, documents, prices), reasoning_effort="none", seed=42)
    return response.choices[0].message.content

# How much does our favorite distortion pedal cost?
test[0].price

# Let's do this!!
gpt_5__1_rag(test[0])
```

### **Step 18: Evaluate GPT-5.1 RAG Performance**
```python
evaluate(gpt_5__1_rag, test)
```

### **Step 19: Initialize Modal Specialist Agent**
```python
import modal
Pricer = modal.Cls.from_name("pricer-service", "Pricer")
pricer = Pricer()

def specialist(item):
    return pricer.price.remote(item.summary)
```

### **Step 20: Price Extraction Helper Function**
```python
def get_price(reply):
    reply = reply.replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", reply)
    return float(match.group()) if match else 0
```

### **Step 21: Deep Neural Network Setup (Optional)**
```python
# Download the Neural Network weights into this directory, older
# The file `deep_neural_network.pth` here: only work on mac MPS, otherwise create your own go to neuronal network folder Neural_Network_InHouse
# https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK?usp=drive_link

from agents.deep_neural_network import DeepNeuralNetworkInference

runner = DeepNeuralNetworkInference()
runner.setup()
runner.load("deep_neural_network.pth")

def deep_neural_network(item):
    return runner.inference(item.summary)
```

### **Step 22: Ensemble Function**
```python
def ensemble(item):
    price1 = get_price(gpt_5__1_rag(item))
    price2 = specialist(item)
    #price3 = deep_neural_network(item)
    return price1 * 0.8 + price2 * 0.1 + price3 * 0.1
```

### **Step 23: Evaluate Ensemble Performance**
```python
evaluate(ensemble, test)
```

---

## **CRITICAL TAKEAWAYS**

1. **RAG > Fine-tuning** (for this task): $30.19 vs $39.85
2. **Ensemble > Individual**: $29.90 beats all single models
3. **Vector stores are fast**: ChromaDB + HNSW for 800K items
4. **Inference techniques matter**: Context injection is powerful
5. **Agent architecture**: Modular, testable, production-ready
6. **Color-coded logging**: Essential for debugging multi-agent systems
7. **Modal deployment**: Serverless GPU for fine-tuned models
8. **Preprocessing**: Standardize inputs for better results

---

## **NEXT STEPS (Steps 3-5)**

- **Step 3**: Scanner Agent (RSS feeds), Messenger Agent (notifications)
- **Step 4**: Planning Agent (orchestrates all agents autonomously)
- **Step 5**: UI + Framework integration, complete autonomous system


