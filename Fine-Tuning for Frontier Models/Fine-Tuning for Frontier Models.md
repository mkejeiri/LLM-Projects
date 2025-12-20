#  Fine-tuning using GPT-4.1-nano

## Basic Concepts

### Fine-tuning
**Definition**: The process of taking a pre-trained model and continuing training on a specific dataset to adapt it to a particular task or domain. Unlike training from scratch, fine-tuning leverages the knowledge already encoded in the model's weights.

**Why Fine-tune?**
- Domain adaptation: Specializes general models for specific tasks
- Data efficiency: Requires far fewer examples than training from scratch
- Performance: Often outperforms zero-shot or few-shot prompting
- Consistency: Produces more reliable, formatted outputs

### JSONL (JSON Lines)
**Definition**: A text format where each line is a valid JSON object. Used for streaming large datasets and batch processing.

**Example**:
```
{"name": "item1", "price": 10.99}
{"name": "item2", "price": 25.50}
```

### Supervised Fine-tuning (SFT)
**Definition**: Training where each example consists of an input and the desired output. The model learns to map inputs to outputs by minimizing prediction error.

---

## Code Walkthrough

### Cell 1: Imports

```python
import os
import re
import json
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from pricer.items import Item
from pricer.evaluator import evaluate
```

**Library Breakdown**:

| Library | Purpose | Usage in This Notebook |
|---------|---------|------------------------|
| `os` | Operating system interface | Access environment variables for API keys |
| `re` | Regular expressions | Pattern matching (imported but not used here) |
| `json` | JSON serialization | Convert Python dicts to JSON strings for JSONL format |
| `dotenv` | Environment variable management | Load API credentials from `.env` file |
| `huggingface_hub` | HuggingFace API client | Authenticate and download datasets |
| `openai` | OpenAI API client | Create fine-tuning jobs, upload files, run inference |
| `pricer.items` | Custom module | `Item` dataclass representing products |
| `pricer.evaluator` | Custom module | `evaluate()` function for model assessment |

---

### Cell 2: Environment Setup

```python
LITE_MODE = False

load_dotenv(override=True)
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)
```

**Line-by-Line**:

1. **`LITE_MODE = False`**
   - Boolean flag to toggle between full dataset and lite version
   - `False` = use full dataset (`items_full`)
   - `True` = use lite dataset (`items_lite`) for faster experimentation

2. **`load_dotenv(override=True)`**
   - Loads environment variables from `.env` file in project root
   - `override=True`: Overwrites existing environment variables if present
   - Typical `.env` content: `HF_TOKEN=hf_xxxxx` and `OPENAI_API_KEY=sk-xxxxx`

3. **`hf_token = os.environ['HF_TOKEN']`**
   - Retrieves HuggingFace API token from environment
   - Raises `KeyError` if `HF_TOKEN` not found

4. **`login(hf_token, add_to_git_credential=True)`**
   - Authenticates with HuggingFace Hub
   - `add_to_git_credential=True`: Stores token in git credential manager for persistence
   - Enables downloading private/gated datasets

---

### Cell 3: Dataset Loading

```python
username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")
```

**Explanation**:

1. **`username**
   - HuggingFace username hosting the dataset

2. **`dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"`**
   - Conditional dataset selection using f-string
   - Result: `"ed-donner/items_full"` (since `LITE_MODE=False`)

3. **`train, val, test = Item.from_hub(dataset)`**
   - Class method that downloads dataset from HuggingFace Hub
   - Returns three lists of `Item` objects (train/validation/test splits)
   - `Item` is a dataclass with attributes like `summary` (text) and `price` (float)

4. **`print(f"Loaded {len(train):,} training items...")`**
   - `:,` format specifier adds thousand separators (e.g., "10,000")
   - **Expected Output**: `"Loaded 50,000 training items, 10,000 validation items, 10,000 test items"`

---

### Cell 4: OpenAI Client Initialization

```python
openai = OpenAI()
```

**What Happens**:
- Instantiates OpenAI API client
- Automatically reads `OPENAI_API_KEY` from environment variables
- Raises error if API key not found
- This client object provides access to:
  - `openai.chat.completions.create()` - Chat API
  - `openai.files.create()` - File uploads
  - `openai.fine_tuning.jobs.create()` - Fine-tuning jobs

---

### Cell 5: Data Subset Selection

```python
fine_tune_train = train[:100]
fine_tune_validation = val[:50]
```

**Rationale**:
- **Training**: 100 examples (0.2% of full dataset)
- **Validation**: 50 examples (0.5% of validation set)

**Why So Few?**
- OpenAI recommends 50-100 examples for small models
- Cost consideration: 100 examples ≈ $0.05, 20,000 examples ≈ $3.42
- Product descriptions are short (low token count per example)
- Single epoch prevents overfitting

**Trade-off**: Lower cost vs potentially reduced performance

---

### Cell 6: Check Training Size

```python
len(fine_tune_train)
```

**Output**: `100`

**Purpose**: Verification step to confirm subset size

---

## Step 1: Data Preparation

### Cell 7: Message Formatting Function

```python
def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"${item.price:.2f}"}
    ]
```

**Concept: ChatML Format**
- OpenAI's chat models expect messages in role-based format
- Roles: `system` (instructions), `user` (input), `assistant` (output)

**Function Breakdown**:

1. **`message = f"Estimate the price...{item.summary}"`**
   - Constructs user prompt with instruction + product description
   - `\n\n` adds two newlines for readability
   - Example: `"Estimate the price of this product. Respond with the price, no explanation\n\nApple iPhone 15 Pro Max 256GB"`

2. **`{"role": "user", "content": message}`**
   - User message dict representing the input

3. **`{"role": "assistant", "content": f"${item.price:.2f}"}`**
   - Assistant message dict representing the desired output
   - `:.2f` formats price to 2 decimal places (e.g., `$19.99`)
   - This is the **ground truth** the model learns to predict

**Return Value**: List of two dicts (user message + assistant response)

---

### Cell 8: Test Message Formatting

```python
messages_for(fine_tune_train[0])
```

**Expected Output**:
```python
[
    {
        "role": "user",
        "content": "Estimate the price of this product. Respond with the price, no explanation\n\nSamsung Galaxy S23 Ultra 512GB Phantom Black"
    },
    {
        "role": "assistant",
        "content": "$1199.99"
    }
]
```

**Purpose**: Visual inspection of formatted training example

---

### Cell 9: JSONL Conversion Function

```python
def make_jsonl(items):
    result = ""
    for item in items:
        messages = messages_for(item)
        messages_str = json.dumps(messages)
        result += '{"messages": ' + messages_str +'}\n'
    return result.strip()
```

**Step-by-Step**:

1. **`result = ""`**
   - Initialize empty string to accumulate JSONL lines

2. **`for item in items:`**
   - Iterate through each `Item` object

3. **`messages = messages_for(item)`**
   - Convert item to message list (user + assistant)

4. **`messages_str = json.dumps(messages)`**
   - Serialize message list to JSON string
   - Example: `'[{"role": "user", "content": "..."}, {"role": "assistant", "content": "$19.99"}]'`

5. **`result += '{"messages": ' + messages_str +'}\n'`**
   - Wrap messages in outer JSON object with `"messages"` key
   - Add newline to create JSONL format
   - Example line: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "$19.99"}]}`

6. **`return result.strip()`**
   - Remove trailing whitespace/newline

**Output Format**:
```
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "$19.99"}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "$25.50"}]}
```

---

### Cell 10: Test JSONL Generation

```python
print(make_jsonl(train[:3]))
```

**Expected Output** (3 lines of JSONL):
```
{"messages": [{"role": "user", "content": "Estimate the price of this product. Respond with the price, no explanation\n\nProduct 1 description"}, {"role": "assistant", "content": "$19.99"}]}
{"messages": [{"role": "user", "content": "Estimate the price of this product. Respond with the price, no explanation\n\nProduct 2 description"}, {"role": "assistant", "content": "$45.00"}]}
{"messages": [{"role": "user", "content": "Estimate the price of this product. Respond with the price, no explanation\n\nProduct 3 description"}, {"role": "assistant", "content": "$12.50"}]}
```

**Purpose**: Verify JSONL formatting before file creation

---

### Cell 11: File Writing Function

```python
def write_jsonl(items, filename):
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)
```

**Explanation**:

1. **`with open(filename, "w") as f:`**
   - Opens file in write mode (`"w"`)
   - Context manager ensures file is closed after block
   - Overwrites file if it exists

2. **`jsonl = make_jsonl(items)`**
   - Converts items to JSONL string

3. **`f.write(jsonl)`**
   - Writes entire JSONL string to file

---

### Cell 12-13: Write Training and Validation Files

```python
write_jsonl(fine_tune_train, "jsonl/fine_tune_train.jsonl")
write_jsonl(fine_tune_validation, "jsonl/fine_tune_validation.jsonl")
```

**Result**:
- Creates `jsonl/fine_tune_train.jsonl` with 100 examples
- Creates `jsonl/fine_tune_validation.jsonl` with 50 examples
- Files are ready for upload to OpenAI

---

### Cell 14: Upload Training File

```python
with open("jsonl/fine_tune_train.jsonl", "rb") as f:
    train_file = openai.files.create(file=f, purpose="fine-tune")
```

**Breakdown**:

1. **`open("jsonl/fine_tune_train.jsonl", "rb")`**
   - Opens file in binary read mode (`"rb"`)
   - Binary mode required for file upload

2. **`openai.files.create(file=f, purpose="fine-tune")`**
   - Uploads file to OpenAI's storage
   - `purpose="fine-tune"`: Tags file for fine-tuning pipeline
   - Returns `FileObject` with metadata

3. **`train_file`** contains:
   - `id`: Unique file identifier (e.g., `"file-abc123"`)
   - `bytes`: File size
   - `created_at`: Unix timestamp
   - `filename`: Original filename
   - `purpose`: `"fine-tune"`

---

### Cell 15: Inspect Training File Object

```python
train_file
```

**Expected Output**:
```python
FileObject(
    id='file-abc123xyz',
    bytes=45678,
    created_at=1704067200,
    filename='fine_tune_train.jsonl',
    object='file',
    purpose='fine-tune',
    status='processed',
    status_details=None
)
```

**Key Fields**:
- `id`: Used to reference file in fine-tuning job
- `status`: `"processed"` means file is ready for training

---

### Cell 16-17: Upload Validation File

```python
with open("jsonl/fine_tune_validation.jsonl", "rb") as f:
    validation_file = openai.files.create(file=f, purpose="fine-tune")

validation_file
```

**Same Process**: Uploads validation set and returns `FileObject`

---

## Step 2: Fine-tuning Job Creation

### Cell 18: Create Fine-tuning Job

```python
openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=validation_file.id,
    model="gpt-4.1-nano-2025-04-14",
    seed=42,
    hyperparameters={"n_epochs": 1, "batch_size": 1},
    suffix="pricer"
)
```

**Parameter Deep Dive**:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `training_file` | `train_file.id` | File ID from upload (e.g., `"file-abc123"`) |
| `validation_file` | `validation_file.id` | Validation file ID for monitoring overfitting |
| `model` | `"gpt-4.1-nano-2025-04-14"` | Base model to fine-tune (smallest GPT-4 variant) |
| `seed` | `42` | Random seed for reproducibility |
| `hyperparameters` | `{"n_epochs": 1, "batch_size": 1}` | Training configuration |
| `suffix` | `"pricer"` | Custom identifier appended to model name |

**Hyperparameters**:

1. **`n_epochs: 1`**
   - **Epoch**: One complete pass through the training dataset
   - 1 epoch = model sees each of 100 examples exactly once
   - Why 1? Prevents overfitting on small dataset

2. **`batch_size: 1`**
   - **Batch Size**: Number of examples processed before updating weights
   - 1 = update weights after every example (stochastic gradient descent)
   - Why 1? Maximizes gradient updates for small dataset

**Model Naming**: Final model will be named `gpt-4.1-nano-2025-04-14:pricer:abc123`

**Return Value**: `FineTuningJob` object with job metadata

---

### Cell 19: List Recent Jobs

```python
openai.fine_tuning.jobs.list(limit=1)
```

**Purpose**: Retrieve most recent fine-tuning job

**Output Structure**:
```python
SyncCursorPage(
    data=[
        FineTuningJob(
            id='ftjob-xyz789',
            created_at=1704067200,
            model='gpt-4.1-nano-2025-04-14',
            fine_tuned_model=None,  # Populated after completion
            status='queued',  # or 'running', 'succeeded', 'failed'
            ...
        )
    ],
    has_more=False
)
```

**Status Values**:
- `queued`: Waiting for compute resources
- `running`: Training in progress
- `succeeded`: Training completed successfully
- `failed`: Training encountered error

---

### Cell 20: Extract Job ID

```python
job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id
```

**Breakdown**:
1. `.list(limit=1)`: Returns `SyncCursorPage` with 1 job
2. `.data`: List of `FineTuningJob` objects
3. `[0]`: First (most recent) job
4. `.id`: Job identifier string (e.g., `"ftjob-xyz789"`)

**Purpose**: Store job ID for monitoring and retrieval

---

### Cell 21: Display Job ID

```python
job_id
```

**Output**: `'ftjob-xyz789'`

---

### Cell 22: Retrieve Job Details

```python
openai.fine_tuning.jobs.retrieve(job_id)
```

**Purpose**: Get current status and metadata of fine-tuning job

**Output** (while running):
```python
FineTuningJob(
    id='ftjob-xyz789',
    created_at=1704067200,
    finished_at=None,
    fine_tuned_model=None,
    model='gpt-4.1-nano-2025-04-14',
    status='running',
    trained_tokens=5000,
    training_file='file-abc123',
    validation_file='file-def456',
    hyperparameters={'n_epochs': 1, 'batch_size': 1},
    result_files=[],
    ...
)
```

**Output** (after completion):
```python
FineTuningJob(
    ...
    finished_at=1704067800,
    fine_tuned_model='ft:gpt-4.1-nano-2025-04-14:org:pricer:abc123',
    status='succeeded',
    trained_tokens=12500,
    result_files=['file-results123'],
    ...
)
```

**Key Fields**:
- `fine_tuned_model`: Custom model identifier (only after success)
- `trained_tokens`: Total tokens processed during training
- `result_files`: Contains training metrics (loss curves, etc.)

---

### Cell 23: List Training Events

```python
openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data
```

**Purpose**: Stream training logs and metrics

**Output** (list of events):
```python
[
    FineTuningJobEvent(
        id='ftevent-1',
        created_at=1704067210,
        level='info',
        message='Fine-tuning job started',
        ...
    ),
    FineTuningJobEvent(
        id='ftevent-2',
        created_at=1704067250,
        level='info',
        message='Step 10/100: training loss=0.523',
        ...
    ),
    FineTuningJobEvent(
        id='ftevent-3',
        created_at=1704067290,
        level='info',
        message='Step 20/100: training loss=0.412',
        ...
    ),
    ...
]
```

**Event Types**:
- Job lifecycle: started, completed, failed
- Training metrics: loss, learning rate, step count
- Validation metrics: validation loss (if validation file provided)

**Monitoring**: Can poll this endpoint to track progress

---

## Step 3: Inference with Fine-tuned Model

### Cell 24: Retrieve Fine-tuned Model Name

```python
fine_tuned_model_name = openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model
```

**Explanation**:
1. Retrieves job details
2. Extracts `fine_tuned_model` field
3. Stores custom model identifier

**Value**: `'ft:gpt-4.1-nano-2025-04-14:org:pricer:abc123'`

**Model Name Format**: `ft:{base_model}:{org}:{suffix}:{unique_id}`

---

### Cell 25: Display Model Name

```python
fine_tuned_model_name
```

**Output**: `'ft:gpt-4.1-nano-2025-04-14:org:pricer:abc123'`

---

### Cell 26: Test Message Function

```python
def test_messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [
        {"role": "user", "content": message},
    ]
```

**Difference from Training**:
- Only includes user message (no assistant response)
- Used for inference, not training
- Model generates the assistant response

---

### Cell 27: Test Message Formatting

```python
test_messages_for(test[0])
```

**Output**:
```python
[
    {
        "role": "user",
        "content": "Estimate the price of this product. Respond with the price, no explanation\n\nSony WH-1000XM5 Wireless Headphones"
    }
]
```

---

### Cell 28: Inference Function

```python
def gpt_4__1_nano_fine_tuned(item):
    response = openai.chat.completions.create(
        model=fine_tuned_model_name,
        messages=test_messages_for(item),
        max_tokens=7
    )
    return response.choices[0].message.content
```

**Breakdown**:

1. **`openai.chat.completions.create(...)`**
   - Calls Chat Completions API with fine-tuned model

2. **`model=fine_tuned_model_name`**
   - Uses custom fine-tuned model instead of base model

3. **`messages=test_messages_for(item)`**
   - Passes formatted user message

4. **`max_tokens=7`**
   - Limits response length to 7 tokens
   - Sufficient for `"$XXX.XX"` format (typically 3-5 tokens)
   - Reduces latency and cost

5. **`response.choices[0].message.content`**
   - Extracts generated text from response
   - `choices[0]`: First (and only) completion
   - `.message.content`: Assistant's response string

**Return Value**: String like `"$45.99"`

---

### Cell 29: Single Prediction Test

```python
print(test[0].price)
print(gpt_4__1_nano_fine_tuned(test[0]))
```

**Expected Output**:
```
299.99

```

**Predicted Output**:
```
$295.00
```

**Interpretation**:
- Ground truth: $299.99
- Prediction: $295.00
- Absolute error: $4.99

---

### Cell 30: Full Evaluation

```python
evaluate(gpt_4__1_nano_fine_tuned, test)
```

**What `evaluate()` Does**:
1. Iterates through test set
2. Calls inference function for each item
3. Parses predicted price from string (regex to extract number)
4. Computes Mean Absolute Error (MAE)
5. Returns MAE in dollars

**Expected Output**: `Mean Absolute Error: $85.23`

**Interpretation**: On average, predictions are off by $85.23

---

### Cell 31: Performance Comparison

```python
# 96.58 - mini 200
# 79.29 - mini 2000
# 82.26 - nano 2000
# 67.75 - nano 20,000
```

**Results Table**:

| Training Size | Model | MAE ($) | Cost |
|---------------|-------|---------|------|
| 200 | gpt-4.1-mini | 96.58 | ~$0.10 |
| 2,000 | gpt-4.1-mini | 79.29 | ~$1.00 |
| 2,000 | gpt-4.1-nano | 82.26 | ~$0.50 |
| 20,000 | gpt-4.1-nano | 67.75 | ~$3.42 |

**Insights**:
1. **Scaling**: 10x more data (200→2000) reduces MAE by ~17%
2. **Model size**: Mini slightly outperforms Nano at same data size
3. **Diminishing returns**: 2K→20K (10x data) only improves MAE by 18%
4. **Cost-performance**: 100 examples likely achieves ~$90 MAE at $0.05 cost

---

## Summary

This notebook demonstrates the complete fine-tuning pipeline:

1. **Data Preparation**: Format examples as ChatML, convert to JSONL
2. **File Upload**: Send training/validation data to OpenAI
3. **Job Creation**: Configure and launch fine-tuning job
4. **Monitoring**: Track training progress via API
5. **Inference**: Use fine-tuned model for predictions
6. **Evaluation**: Assess performance on held-out test set

**Key Takeaways**:
- Fine-tuning requires minimal data (50-100 examples)
- JSONL format enables streaming and batch processing
- Hyperparameters (epochs, batch size) must be tuned for dataset size
- Custom models are accessed via unique identifiers
- Cost scales linearly with training examples and tokens



-----
## Key Objectives of Fine-Tuning for Frontier Models
-----

1. **Setting style or tone**  
   Achieving a style or tone that cannot be reliably accomplished through prompting alone.

2. **Improving output reliability**  
   Increasing consistency in producing a specific type of output.

3. **Correcting prompt-following failures**  
   Addressing cases where the model struggles to follow complex prompts.

4. **Handling edge cases**  
   Improving performance on rare or difficult scenarios.

5. **Learning new skills or tasks**  
   Enabling capabilities that are hard to clearly articulate in a prompt.

---

## Why a Problem Like Ours Doesn’t Benefit Much from Fine-Tuning

- The problem and desired output style can be clearly specified in a prompt.
- The model can already leverage its extensive world knowledge from pre-training; adding a small amount of extra data (e.g., prices) does not materially improve performance.

---

## Key Takeaways

- **Failed experiments are a natural part of data science.**
- **Your mission:** Experiment with hyperparameters and see if you can achieve improvements.