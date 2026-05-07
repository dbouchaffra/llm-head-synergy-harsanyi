import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# CONFIGURATION
# ================================
CACHE_DIR = "./harsanyi_cache_positive"   # adjust if needed

PERPLEXITY = {5: 2189.02, 10: 2516.24, 20: 2148.43}

# ================================
# 1. Load data and compute necessary arrays
# ================================
def load_layer_dividends(cache_dir, layer_idx):
    cache_file = os.path.join(cache_dir, f"layer_{layer_idx}_dividends.pkl")
    with open(cache_file, "rb") as f:
        singletons, pairs = pickle.load(f)
    return singletons, pairs

def reconstruct_pairwise_matrix(singletons, pairs, n_heads=12):
    mat = np.zeros((n_heads, n_heads))
    for i in range(n_heads):
        mat[i,i] = singletons[i]
    for (i,j), val in pairs.items():
        mat[i,j] = val
        mat[j,i] = val
    return mat

def compute_head_scores_from_dividends(singletons, pairs, n_heads=12):
    scores = np.zeros(n_heads)
    for i in range(n_heads):
        scores[i] = singletons[i]
        for j in range(n_heads):
            if j != i:
                key = (min(i,j), max(i,j))
                if key in pairs:
                    scores[i] += 0.5 * pairs[key]
    return scores

# Load layer 0 for heatmap
singletons0, pairs0 = load_layer_dividends(CACHE_DIR, 0)
pair_matrix0 = reconstruct_pairwise_matrix(singletons0, pairs0)

# Compute head scores for all layers
num_layers = 12
n_heads = 12
all_scores = np.zeros((num_layers, n_heads))
for layer in range(num_layers):
    sing, pair = load_layer_dividends(CACHE_DIR, layer)
    all_scores[layer] = compute_head_scores_from_dividends(sing, pair, n_heads)

# Pruning masks
def get_prune_masks(scores, percentages=[5,10,20]):
    masks = {}
    for p in percentages:
        k = max(1, int(np.ceil(p / 100.0 * n_heads)))
        mask = np.zeros_like(scores, dtype=bool)
        for layer in range(num_layers):
            idx_sorted = np.argsort(scores[layer])
            worst = idx_sorted[:k]
            mask[layer, worst] = True
        masks[p] = mask
    return masks

masks = get_prune_masks(all_scores, [5,10,20])

# ================================
# 2. Figure 1: Heatmap
# ================================
plt.figure(figsize=(10,8))
sns.heatmap(pair_matrix0, annot=True, fmt=".3f", cmap="viridis", 
            square=True, cbar_kws={"label": "Pairwise Harsanyi dividend"})
plt.title("Pairwise Harsanyi Dividends (GPT-2 Layer 0, 500 GSM8K examples)")
plt.xlabel("Head index")
plt.ylabel("Head index")
plt.tight_layout()
plt.savefig("figure_pairwise_heatmap.png", dpi=300)
plt.show()

# ================================
# ================================
# ================================
# ================================
# 3. Figure 2: Head scores (6 rows, 2 columns) - compact for manuscript
# ================================
fig, axes = plt.subplots(6, 2, figsize=(6.5, 20))   # narrow width (6.5 inches)
axes = axes.flatten()
for layer in range(num_layers):
    heads = np.arange(n_heads)
    axes[layer].bar(heads, all_scores[layer])
    axes[layer].set_title(f"Layer {layer}", fontsize=9, pad=8)
    axes[layer].set_xlabel("Head index", fontsize=8, labelpad=4)
    axes[layer].set_ylabel("Score", fontsize=8)
    axes[layer].set_ylim([20, 45])
    axes[layer].set_xticks(heads)
    axes[layer].set_xticklabels(heads, rotation=45, fontsize=6)
    axes[layer].tick_params(axis='y', labelsize=6)
plt.suptitle("Head importance scores per layer (Shapley value from Harsanyi dividends)", 
             fontsize=11, y=0.95)
plt.subplots_adjust(hspace=0.6, wspace=0.1, top=0.92)   # minimal horizontal gap
plt.savefig("figure_head_scores_per_layer.png", dpi=300, bbox_inches='tight')
plt.show()
# ================================
# 4. Figure 3: Pruning masks (vertical stack) - fix colorbar overlap
# ================================
fig, axes = plt.subplots(3, 1, figsize=(7, 12), constrained_layout=True)  # constrained_layout auto-adjusts
for idx, pct in enumerate([5,10,20]):
    mask = masks[pct]
    im = axes[idx].imshow(mask, aspect='auto', cmap='Reds', interpolation='nearest')
    axes[idx].set_title(f"{pct}% pruning", fontsize=12)
    axes[idx].set_xlabel("Head index", fontsize=10)
    axes[idx].set_ylabel("Layer index", fontsize=10)
    axes[idx].set_xticks(range(n_heads))
    axes[idx].set_yticks(range(num_layers))
    axes[idx].tick_params(axis='both', labelsize=8)
    # Add number of pruned heads per layer as text
    for layer in range(num_layers):
        n_pruned = np.sum(mask[layer])
        if n_pruned > 0:
            axes[idx].text(n_heads-0.5, layer, f"{n_pruned}", ha='left', va='center', fontsize=8, color='white')
# Add a single colorbar for all subplots
cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02, shrink=0.6)
cbar.set_label("Pruned (True)", fontsize=10)
fig.suptitle("Pruning masks across layers and heads", fontsize=14)
plt.savefig("figure_pruning_masks.png", dpi=300)
plt.show()

# ================================
# 5. Figure 4: Performance curve
# ================================
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
from datasets import load_dataset

def get_baseline_perplexity(model_name="gpt2", num_examples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("gsm8k", "main", split="test")
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, example in enumerate(dataset.select(range(num_examples))):
            text = example['question']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
    avg_loss = total_loss / num_examples
    return np.exp(avg_loss)

print("Computing baseline perplexity (0% pruning) on 5 examples...")
baseline_ppl = get_baseline_perplexity()
print(f"Baseline perplexity: {baseline_ppl:.2f}")

PERPLEXITY[0] = baseline_ppl
prune_vals = sorted(PERPLEXITY.keys())
ppl_vals = [PERPLEXITY[p] for p in prune_vals]

plt.figure(figsize=(8,6))
plt.plot(prune_vals, ppl_vals, marker='o', linestyle='-', linewidth=2)
plt.xlabel("Pruning percentage")
plt.ylabel("Perplexity (lower is better)")
plt.title("Effect of head pruning on GPT-2 perplexity (GSM8K, 5 examples)")
plt.grid(True)
plt.xticks(prune_vals)
plt.savefig("figure_perplexity_vs_pruning.png", dpi=300)
plt.show()

print("All figures saved.")