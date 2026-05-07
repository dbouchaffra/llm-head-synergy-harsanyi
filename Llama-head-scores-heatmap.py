import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ------------------------------------------------------------
# Set larger fonts for better readability (valid rcParams only)
# ------------------------------------------------------------
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# ------------------------------------------------------------
# Helper functions (identical to earlier code)
# ------------------------------------------------------------
def get_head_symbols(model, tokenizer, device, text, layer_idx=0, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    layer_attn = outputs.attentions[layer_idx].squeeze(0)
    num_heads = layer_attn.shape[0]
    symbols = []
    for h in range(num_heads):
        argmax_indices = torch.argmax(layer_attn[h], dim=-1).cpu().numpy()
        symbols.append(tuple(argmax_indices))
    return symbols

def collect_symbols_for_layer(model, tokenizer, device, dataset, layer_idx, max_examples=500):
    head_symbols = {h: [] for h in range(model.config.num_attention_heads)}
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        text = example.get('question', '')
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

def head_importance_score(singletons, pairs, n_heads):
    scores = np.zeros(n_heads)
    for i in range(n_heads):
        scores[i] = singletons[i]
        for j in range(n_heads):
            if j != i:
                key = (min(i,j), max(i,j))
                if key in pairs:
                    scores[i] += 0.5 * pairs[key]
    return scores

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
def main():
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    CACHE_DIR = "./llama_dividends_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    MAX_EXAMPLES = 500
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_attentions=True,
        torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32
    ).to(DEVICE)
    model.eval()

    dataset = load_dataset("gsm8k", "main", split="test")

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"Running analysis on: {MODEL_NAME}")
    print(f"Layers: {num_layers}, Heads per layer: {num_heads}")

    all_scores = []
    for layer in range(num_layers):
        cache_file = os.path.join(CACHE_DIR, f"layer_{layer}_dividends.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached dividends for layer {layer}")
            with open(cache_file, "rb") as f:
                singletons, pairs = pickle.load(f)
        else:
            print(f"Computing symbols for layer {layer}...")
            head_symbols = collect_symbols_for_layer(model, tokenizer, DEVICE, dataset, layer, max_examples=MAX_EXAMPLES)
            print(f"Computing dividends for layer {layer}...")
            singletons, pairs = compute_pairwise_dividends(head_symbols)
            with open(cache_file, "wb") as f:
                pickle.dump((singletons, pairs), f)

        scores = head_importance_score(singletons, pairs, num_heads)
        all_scores.append(scores)
        print(f"Layer {layer} average score: {np.mean(scores):.2f}")

    scores_matrix = np.array(all_scores)

    # Plot heatmap with larger fonts
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(scores_matrix, cmap='viridis', cbar_kws={"label": "Head importance (Shapley score)", "shrink": 0.8})
    ax.set_title(f"Head importance scores, {MODEL_NAME} on GSM8K (500 examples)", fontsize=18)
    ax.set_xlabel("Head index", fontsize=16)
    ax.set_ylabel("Layer index", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Fix colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label("Head importance (Shapley score)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig("llama_head_scores_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("\nHeatmap saved as 'llama_head_scores_heatmap.png' with larger fonts.")

if __name__ == "__main__":
    main()