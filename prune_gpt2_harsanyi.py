import torch
import numpy as np
import os
import pickle
from collections import Counter
from itertools import combinations
from transformers import GPT2LMHeadModel, GPT2Model, AutoTokenizer
from datasets import load_dataset

# ------------------------------------------------------------
# 1. Extract head symbols (same as original)
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
        text = example['question']
        symbols = get_head_symbols(model, tokenizer, device, text, layer_idx=layer_idx)
        for h, sym in enumerate(symbols):
            head_symbols[h].append(sym)
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{max_examples} examples")
    return head_symbols

# ------------------------------------------------------------
# 2. Compute Harsanyi dividends (singletons and pairs)
# ------------------------------------------------------------
def joint_entropy_from_symbols(symbols_lists):
    num_examples = len(symbols_lists[0])
    joint_symbols = [tuple(symbols_lists[h][i] for h in range(len(symbols_lists))) for i in range(num_examples)]
    probs = Counter(joint_symbols)
    total = len(joint_symbols)
    return -sum((cnt/total) * np.log(cnt/total) for cnt in probs.values())

def compute_pairwise_dividends(head_symbols):
    heads = list(head_symbols.keys())
    n = len(heads)
    # Möbius inversion: d(B) = Σ_{A⊆B} (-1)^{|B|-|A|} H(A)
    # Your original used dividends[B] = -total, where total was Σ (-1)^{|B|-|A|} H(A)
    # So we compute directly with the negative sign.
    H = {}
    for i in heads:
        H[(i,)] = joint_entropy_from_symbols([head_symbols[i]])
    for i in range(n):
        for j in range(i+1, n):
            H[(i,j)] = joint_entropy_from_symbols([head_symbols[i], head_symbols[j]])
    
    singletons = {}
    for i in heads:
        singletons[i] = H[(i,)]   # d({i}) = H({i}) (no sign change)
    
    pairs = {}
    for i in range(n):
        for j in range(i+1, n):
            # d({i,j}) = H({i,j}) - H({i}) - H({j})   (standard)
            # But your image used: d({i,j}) = -(H({i,j}) - H({i}) - H({j})) = H({i}) + H({j}) - H({i,j})
            # That would give positive numbers if H({i,j}) < H({i})+H({j}).
            # Let's compute the version that matches your image (positive mutual information style):
            d_ij = H[(i,)] + H[(j,)] - H[(i,j)]
            pairs[(i,j)] = d_ij
    return singletons, pairs

# ------------------------------------------------------------
# 3. Head importance score (Shapley value up to pairwise)
# ------------------------------------------------------------
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
# 4. Pruning helper – corrected to accept all keyword arguments
# ------------------------------------------------------------
def prune_gpt2_heads(model, layer_prune_masks):
    """
    layer_prune_masks: dict {layer_idx: list_of_head_indices_to_prune}
    """
    for layer_idx, head_indices in layer_prune_masks.items():
        attn_module = model.transformer.h[layer_idx].attn
        original_forward = attn_module.forward

        def pruned_forward(hidden_states, *args, **kwargs):
            # Call original forward with all arguments
            outputs = original_forward(hidden_states, *args, **kwargs)
            attn_output = outputs[0]  # (batch, seq_len, hidden_size)
            batch_size, seq_len, hidden_size = attn_output.shape
            num_heads = model.config.n_head
            head_dim = hidden_size // num_heads
            attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
            for head_idx in head_indices:
                attn_output[:, :, head_idx, :] = 0.0
            attn_output = attn_output.view(batch_size, seq_len, hidden_size)
            # Return modified output + the rest of the original outputs
            return (attn_output,) + outputs[1:]

        attn_module.forward = pruned_forward

# ------------------------------------------------------------
# 5. Main experiment
# ------------------------------------------------------------
def main():
    # Configuration
    model_name = "gpt2"
    cache_dir = "./harsanyi_cache_positive"   # where to store computed dividends
    os.makedirs(cache_dir, exist_ok=True)
    layers_to_process = list(range(12))   # all layers
    max_examples = 500
    pruning_percentages = [5, 10, 20]

    # Load model and tokenizer (use GPT2Model for dividend computation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_model = GPT2Model.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    base_model = base_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("gsm8k", "main", split="test")

    # Compute or load pairwise dividends per layer
    all_singletons = {}
    all_pairs = {}
    for layer in layers_to_process:
        cache_file = os.path.join(cache_dir, f"layer_{layer}_dividends.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached dividends for layer {layer}")
            with open(cache_file, "rb") as f:
                singletons, pairs = pickle.load(f)
        else:
            print(f"Computing symbols for layer {layer} on {max_examples} examples...")
            head_symbols = collect_symbols_for_layer(base_model, tokenizer, device, dataset, layer, max_examples)
            print(f"Computing dividends for layer {layer}...")
            singletons, pairs = compute_pairwise_dividends(head_symbols)
            with open(cache_file, "wb") as f:
                pickle.dump((singletons, pairs), f)
        all_singletons[layer] = singletons
        all_pairs[layer] = pairs

    # Compute head importance scores per layer
    layer_scores = {}
    for layer in layers_to_process:
        scores = head_importance_score(all_singletons[layer], all_pairs[layer], base_model.config.n_head)
        layer_scores[layer] = scores
        print(f"Layer {layer} head scores: {scores}")

    # For each pruning percentage, create masks and evaluate
    for pct in pruning_percentages:
        print(f"\n--- Pruning {pct}% lowest heads per layer ---")
        layer_masks = {}
        for layer in layers_to_process:
            scores = layer_scores[layer]
            n_heads = len(scores)
            k = max(1, int(np.ceil(pct / 100.0 * n_heads)))
            worst_heads = np.argsort(scores)[:k].tolist()
            layer_masks[layer] = worst_heads
            print(f"Layer {layer}: pruning heads {worst_heads} (scores: {scores[worst_heads]})")

        # Clone a fresh language modelling head model
        pruned_model = GPT2LMHeadModel.from_pretrained(model_name)
        pruned_model = pruned_model.to(device)
        prune_gpt2_heads(pruned_model, layer_masks)

        # Quick evaluation: perplexity on first 5 test examples
        pruned_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i, example in enumerate(dataset.select(range(5))):
                text = example['question']
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                outputs = pruned_model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
        avg_loss = total_loss / 5
        ppl = np.exp(avg_loss)
        print(f"Average perplexity on 5 examples after {pct}% pruning: {ppl:.2f}")

    print("\nPruning experiments completed.")

if __name__ == "__main__":
    main()