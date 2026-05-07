import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Model, AutoTokenizer
from datasets import load_dataset
from itertools import combinations, chain
from collections import Counter

# ------------------------------------------------------------
# Load model and tokenizer (same as before)
# ------------------------------------------------------------
def load_model(model_name="gpt2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2Model.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    return model, tokenizer, device

# ------------------------------------------------------------
# Extract attention outputs for a single text (symbols per head)
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
        argmax_indices = torch.argmax(layer_attn[h], dim=-1).cpu().numpy()   # (seq_len,)
        symbols.append(tuple(argmax_indices))
    return symbols

# ------------------------------------------------------------
# Collect symbols for a layer over a dataset
# ------------------------------------------------------------
def collect_symbols_for_layer(model, tokenizer, device, dataset, layer_idx=0, max_examples=500):
    head_symbols = {h: [] for h in range(12)}   # 12 heads in GPT-2
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        text = example['question']
        symbols = get_head_symbols(model, tokenizer, device, text, layer_idx=layer_idx)
        for h, sym in enumerate(symbols):
            head_symbols[h].append(sym)
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} examples")
    return head_symbols

# ------------------------------------------------------------
# Compute joint entropy from symbol lists
# ------------------------------------------------------------
def joint_entropy_from_symbols(symbols_lists):
    num_examples = len(symbols_lists[0])
    joint_symbols = [tuple(symbols_lists[h][i] for h in range(len(symbols_lists))) for i in range(num_examples)]
    probs = Counter(joint_symbols)
    total = len(joint_symbols)
    return -sum((cnt/total) * np.log(cnt/total) for cnt in probs.values())

# ------------------------------------------------------------
# Compute Harsanyi dividends for up to coalitions of size 3
# ------------------------------------------------------------
def compute_harsanyi_dividends_head_layer(head_symbols, max_coalition_size=3):
    heads = list(head_symbols.keys())
    H = {frozenset(): 0.0}
    for size in range(1, max_coalition_size+1):
        for comb in combinations(heads, size):
            sym_lists = [head_symbols[h] for h in comb]
            H[frozenset(comb)] = joint_entropy_from_symbols(sym_lists)
    dividends = {}
    for B in H:
        if B == frozenset():
            continue
        total = 0.0
        for k in range(len(B)+1):
            for A in combinations(B, k):
                A_set = frozenset(A)
                sign = (-1)**(len(B) - k)
                total += sign * H.get(A_set, 0.0)
        dividends[B] = -total
    return dividends

# ------------------------------------------------------------
# Plot pair heatmap
# ------------------------------------------------------------
def plot_pair_heatmap(dividends, n_heads=12, title="Pairwise Harsanyi Dividends (Layer 0)"):
    pair_matrix = np.zeros((n_heads, n_heads))
    for i in range(n_heads):
        for j in range(i+1, n_heads):
            val = dividends.get(frozenset({i,j}), 0.0)
            pair_matrix[i,j] = val
            pair_matrix[j,i] = val
    np.fill_diagonal(pair_matrix, np.nan)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(pair_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"label": "Harsanyi dividend"})
    plt.title(title)
    plt.xlabel("Head index")
    plt.ylabel("Head index")
    plt.tight_layout()
    plt.savefig("pair_dividends_heatmap.png", dpi=300)
    plt.show()

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    # Load model and dataset
    model, tokenizer, device = load_model("gpt2")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Collect symbols for layer 0 (500 examples)
    head_symbols = collect_symbols_for_layer(model, tokenizer, device, dataset, layer_idx=0, max_examples=500)
    
    # Compute dividends
    dividends = compute_harsanyi_dividends_head_layer(head_symbols, max_coalition_size=3)
    
    # Plot heatmap
    plot_pair_heatmap(dividends, n_heads=12, title="Pairwise Harsanyi Dividends (GPT-2 Layer 0, 500 GSM8K examples)")