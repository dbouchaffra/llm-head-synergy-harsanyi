import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from transformers import GPT2Model, AutoTokenizer
from datasets import load_dataset

# ------------------------------------------------------------
# 1. Helper functions (same as pruning script)
# ------------------------------------------------------------
def get_head_symbols(model, tokenizer, device, text, layer_idx=0, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    layer_attn = outputs.attentions[layer_idx].squeeze(0)          # (num_heads, seq_len, seq_len)
    num_heads = layer_attn.shape[0]
    symbols = []
    for h in range(num_heads):
        argmax_indices = torch.argmax(layer_attn[h], dim=-1).cpu().numpy()
        symbols.append(tuple(argmax_indices))
    return symbols

def collect_symbols_for_layer(model, tokenizer, device, dataset, layer_idx, max_examples=500):
    head_symbols = {h: [] for h in range(model.config.n_head)}
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        # Extract text – adapt to each dataset's format
        if 'question' in example:
            text = example['question']
        elif 'problem' in example:
            text = example['problem']
        elif 'text' in example:
            text = example['text']
        else:
            # Fallback for HumanEval: 'prompt' field
            text = example.get('prompt', '')
        symbols = get_head_symbols(model, tokenizer, device, text, layer_idx=layer_idx)
        for h, sym in enumerate(symbols):
            head_symbols[h].append(sym)
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{max_examples} examples")
    return head_symbols

def joint_entropy_from_symbols(symbols_lists):
    num_examples = len(symbols_lists[0])
    joint_symbols = [tuple(symbols_lists[h][i] for h in range(len(symbols_lists))) for i in range(num_examples)]
    probs = Counter(joint_symbols)
    total = len(joint_symbols)
    return -sum((cnt/total) * np.log(cnt/total) for cnt in probs.values())

def compute_pairwise_dividends(head_symbols):
    """Positive dividends: d({i}) = H_i, d({i,j}) = H_i + H_j - H_ij."""
    heads = list(head_symbols.keys())
    n = len(heads)
    singletons = {}
    for i in heads:
        H_i = joint_entropy_from_symbols([head_symbols[i]])
        singletons[i] = H_i
    pairs = {}
    for i in range(n):
        for j in range(i+1, n):
            H_ij = joint_entropy_from_symbols([head_symbols[i], head_symbols[j]])
            d_ij = singletons[i] + singletons[j] - H_ij
            pairs[(i,j)] = d_ij
    return singletons, pairs

# ------------------------------------------------------------
# 2. Load GPT-2 model once
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = GPT2Model.from_pretrained(
    "gpt2",
    output_attentions=True,
    torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------
# 3. Dataset loading with correct splits
# ------------------------------------------------------------
def load_dataset_task(task_name, max_examples=500):
    """Load a subset of the dataset (first 'max_examples' items) using the appropriate split."""
    if task_name == 'GSM8K':
        dataset = load_dataset("gsm8k", "main", split="test")
    elif task_name == 'MATH':
        # Community mirror of the MATH dataset (only 'train' split available)
        dataset = load_dataset("qwedsacf/competition_math", split="train")
    elif task_name == 'TriviaQA':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="test")
    elif task_name == 'SQuAD':
        dataset = load_dataset("squad", split="validation")   # 'validation' not 'test'
    elif task_name == 'HumanEval':
        dataset = load_dataset("openai_humaneval", split="test")
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Limit to max_examples
    dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset

# ------------------------------------------------------------
# 4. Compute or load dividends per task
# ------------------------------------------------------------
layer_idx = 0   # we analyse layer 0 for consistency
tasks = ['GSM8K', 'MATH', 'TriviaQA', 'SQuAD', 'HumanEval']

# Cache directory for dividends per task
CACHE_DIR_TASKS = "./task_dividends_cache"
os.makedirs(CACHE_DIR_TASKS, exist_ok=True)

task_dividends = {}   # dict: task_name -> (singletons, pairs)

for task in tasks:
    cache_file = os.path.join(CACHE_DIR_TASKS, f"{task}_layer_{layer_idx}_dividends.pkl")
    if os.path.exists(cache_file):
        print(f"Loading cached dividends for {task} (layer {layer_idx})")
        with open(cache_file, "rb") as f:
            singletons, pairs = pickle.load(f)
    else:
        print(f"\nComputing dividends for {task} on 500 examples (layer {layer_idx})...")
        dataset = load_dataset_task(task, max_examples=500)
        head_symbols = collect_symbols_for_layer(model, tokenizer, device, dataset, layer_idx, max_examples=500)
        singletons, pairs = compute_pairwise_dividends(head_symbols)
        with open(cache_file, "wb") as f:
            pickle.dump((singletons, pairs), f)
    task_dividends[task] = (singletons, pairs)

# ------------------------------------------------------------
# 5. Compute Jaccard similarity for positive pairwise coalitions
# ------------------------------------------------------------
def get_positive_pair_set(pairs, threshold=0.0):
    """Return set of frozenset({i,j}) for which d_ij > threshold."""
    return {frozenset([i, j]) for (i, j), val in pairs.items() if val > threshold}

n_tasks = len(tasks)
sim_matrix = np.zeros((n_tasks, n_tasks))

for i, task_i in enumerate(tasks):
    _, pairs_i = task_dividends[task_i]
    set_i = get_positive_pair_set(pairs_i, threshold=0.0)
    for j, task_j in enumerate(tasks):
        _, pairs_j = task_dividends[task_j]
        set_j = get_positive_pair_set(pairs_j, threshold=0.0)
        if len(set_i) == 0 and len(set_j) == 0:
            sim = 1.0
        elif len(set_i) == 0 or len(set_j) == 0:
            sim = 0.0
        else:
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            sim = intersection / union
        sim_matrix[i, j] = sim

# ------------------------------------------------------------
# 6. Plot heatmap
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            xticklabels=tasks, yticklabels=tasks,
            square=True, cbar_kws={"label": "Jaccard Similarity"})
plt.title(f"Task Similarity: Positive Pairwise Coalitions (GPT-2, Layer {layer_idx})")
plt.tight_layout()
plt.savefig("task_heatmap.png", dpi=300)
plt.show()

print("\nHeatmap saved as 'task_heatmap.png'.")
print("Similarity matrix:")
for i, task_i in enumerate(tasks):
    row = [f"{sim_matrix[i,j]:.3f}" for j in range(n_tasks)]
    print(f"{task_i:12s}: " + " ".join(row))