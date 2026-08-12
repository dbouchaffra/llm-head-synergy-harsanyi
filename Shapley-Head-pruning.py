from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "openai-community/gpt2"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

# ------------------------------------------------------------
# Shapley estimation
# ------------------------------------------------------------

# Number of GSM8K TRAIN examples used to estimate head importance.
NUM_CALIBRATION_EXAMPLES = 32

# Number of random head permutations sampled per layer.
#
# 32  = relatively quick preliminary experiment
# 64  = recommended starting point
# 128 = stronger estimate but ~2x slower
NUM_PERMUTATIONS = 64

CALIBRATION_MAX_LENGTH = 256

# ------------------------------------------------------------
# GSM8K perplexity evaluation
# ------------------------------------------------------------

EVAL_MAX_LENGTH = 512

# Set to None to evaluate the complete GSM8K test set.
#
# For a quick debugging run, use e.g. 100.
NUM_EVAL_EXAMPLES = None

# ------------------------------------------------------------
# Pruning ratios
# GPT-2 has 12 heads/layer:
#
# 1 / 12 = 8.33%
# 2 / 12 = 16.67%
# 3 / 12 = 25%
# ------------------------------------------------------------

HEADS_TO_PRUNE = [0, 1, 2, 3]

OUTPUT_DIR = Path("shapley_gsm8k_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. Load model and tokenizer
# ============================================================

def load_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        clean_up_tokenization_spaces=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    # Eager attention is safest when using head_mask.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
    ).to(DEVICE)

    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return model, tokenizer


# ============================================================
# 4. GSM8K text representation
# ============================================================

def build_lm_text(example):
    """
    Represent a GSM8K question-answer pair as a language-modeling
    sequence.

    IMPORTANT:
    Use exactly the same representation for every pruning method.
    """

    return (
        f"Question: {example['question']}\n"
        f"Answer: {example['answer']}"
    )


# ============================================================
# 5. Loss for one text under a specific head mask
# ============================================================

@torch.no_grad()
def compute_text_loss(
    model,
    tokenizer,
    text: str,
    head_mask: torch.Tensor,
    max_length: int,
):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    outputs = model(
        **encoded,
        labels=encoded["input_ids"],
        head_mask=head_mask,
        use_cache=False,
    )

    return outputs.loss.item()


# ============================================================
# 6. Mean calibration loss for a coalition
# ============================================================

@torch.no_grad()
def coalition_loss(
    model,
    tokenizer,
    calibration_texts,
    target_layer: int,
    active_heads: set[int],
):
    """
    All heads remain active in all layers EXCEPT target_layer.

    Inside target_layer, only heads in active_heads are enabled.

    This defines the coalition game independently for each layer.
    """

    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    # All heads active initially.
    head_mask = torch.ones(
        num_layers,
        num_heads,
        dtype=torch.float32,
        device=DEVICE,
    )

    # Disable entire target layer.
    head_mask[target_layer, :] = 0.0

    # Re-enable heads belonging to the current coalition.
    for head in active_heads:
        head_mask[target_layer, head] = 1.0

    losses = []

    for text in calibration_texts:
        loss = compute_text_loss(
            model=model,
            tokenizer=tokenizer,
            text=text,
            head_mask=head_mask,
            max_length=CALIBRATION_MAX_LENGTH,
        )

        losses.append(loss)

    return sum(losses) / len(losses)


# ============================================================
# 7. Monte-Carlo Shapley estimation
# ============================================================

def estimate_layer_shapley(
    model,
    tokenizer,
    calibration_texts,
    layer_index: int,
    num_permutations: int,
):
    """
    Estimate Shapley values for all heads in one transformer layer.

    For a permutation:

        [4, 7, 2, ...]

    we begin with no active heads in the target layer.

    Heads are added one by one.

    Marginal contribution:

        v(S U {i}) - v(S)

    with

        v(S) = -Loss(S)

    therefore:

        marginal = Loss(S) - Loss(S U {i})

    Positive marginal contribution means that adding the head
    improves predictive performance.
    """

    num_heads = model.config.n_head

    shapley = torch.zeros(
        num_heads,
        dtype=torch.float64,
    )

    heads = list(range(num_heads))

    print(
        f"\nEstimating Shapley values for layer "
        f"{layer_index}/{model.config.n_layer - 1}"
    )

    for permutation_index in tqdm(
        range(num_permutations),
        desc=f"Layer {layer_index}",
    ):

        permutation = heads.copy()
        random.shuffle(permutation)

        active_heads = set()

        # ----------------------------------------------------
        # v(empty coalition)
        # ----------------------------------------------------

        previous_loss = coalition_loss(
            model=model,
            tokenizer=tokenizer,
            calibration_texts=calibration_texts,
            target_layer=layer_index,
            active_heads=active_heads,
        )

        # ----------------------------------------------------
        # Sequentially add heads according to permutation
        # ----------------------------------------------------

        for head in permutation:

            active_heads.add(head)

            new_loss = coalition_loss(
                model=model,
                tokenizer=tokenizer,
                calibration_texts=calibration_texts,
                target_layer=layer_index,
                active_heads=active_heads,
            )

            # v(S U i) - v(S)
            #
            # Since v = -Loss:
            #
            # (-new_loss) - (-previous_loss)
            # = previous_loss - new_loss
            marginal_contribution = (
                previous_loss - new_loss
            )

            shapley[head] += marginal_contribution

            previous_loss = new_loss

    shapley /= num_permutations

    return shapley


# ============================================================
# 8. Estimate Shapley values for ALL GPT-2 layers
# ============================================================

def estimate_all_shapley_values(
    model,
    tokenizer,
    calibration_texts,
):
    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    values = torch.zeros(
        num_layers,
        num_heads,
        dtype=torch.float64,
    )

    for layer in range(num_layers):

        values[layer] = estimate_layer_shapley(
            model=model,
            tokenizer=tokenizer,
            calibration_texts=calibration_texts,
            layer_index=layer,
            num_permutations=NUM_PERMUTATIONS,
        )

        print(
            f"\nLayer {layer} Shapley values:"
        )

        for head, value in enumerate(values[layer]):
            print(
                f"  Head {head:2d}: "
                f"{value.item():+.8f}"
            )

    return values


# ============================================================
# 9. Construct layer-wise Shapley rankings
# ============================================================

def get_shapley_rankings(shapley_values):
    """
    Lowest Shapley value = least useful head = prune first.
    """

    rankings = {}

    for layer in range(shapley_values.shape[0]):

        ranking = torch.argsort(
            shapley_values[layer],
            descending=False,
        ).tolist()

        rankings[layer] = ranking

    return rankings


# ============================================================
# 10. Build pruning mask
# ============================================================

def build_pruning_mask(
    model,
    rankings,
    heads_per_layer: int,
):
    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    mask = torch.ones(
        num_layers,
        num_heads,
        dtype=torch.float32,
        device=DEVICE,
    )

    if heads_per_layer == 0:
        return mask

    for layer in range(num_layers):

        heads_to_remove = rankings[layer][
            :heads_per_layer
        ]

        for head in heads_to_remove:
            mask[layer, head] = 0.0

    return mask


# ============================================================
# 11. Token-weighted GSM8K perplexity
# ============================================================

@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    examples,
    head_mask,
):
    """
    Computes token-weighted perplexity.

    This is preferable to simply averaging example-level
    perplexities because examples have different lengths.
    """

    total_nll = 0.0
    total_tokens = 0

    for example in tqdm(
        examples,
        desc="Evaluating PPL",
    ):

        text = build_lm_text(example)

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=EVAL_MAX_LENGTH,
        )

        input_ids = encoded["input_ids"].to(DEVICE)

        attention_mask = encoded["attention_mask"].to(
            DEVICE
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            head_mask=head_mask,
            use_cache=False,
        )

        # GPT-2 causal-LM loss predicts tokens 1...N
        num_tokens = int(
            attention_mask[:, 1:].sum().item()
        )

        total_nll += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    mean_nll = total_nll / total_tokens

    perplexity = math.exp(mean_nll)

    return perplexity


# ============================================================
# 12. Main
# ============================================================

def main():

    print("=" * 70)
    print("SHAPLEY HEAD PRUNING — GPT-2 / GSM8K")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(
        f"Calibration examples: "
        f"{NUM_CALIBRATION_EXAMPLES}"
    )
    print(
        f"Permutations/layer: "
        f"{NUM_PERMUTATIONS}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, tokenizer = load_model_and_tokenizer()

    print("\nModel loaded.")

    print(
        f"Layers: {model.config.n_layer}"
    )

    print(
        f"Heads/layer: {model.config.n_head}"
    )

    # --------------------------------------------------------
    # Load GSM8K
    # --------------------------------------------------------

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
    )

    train_set = dataset["train"]
    test_set = dataset["test"]

    # --------------------------------------------------------
    # Fixed calibration subset
    # --------------------------------------------------------

    generator = random.Random(SEED)

    calibration_indices = generator.sample(
        range(len(train_set)),
        NUM_CALIBRATION_EXAMPLES,
    )

    calibration_examples = [
        train_set[i]
        for i in calibration_indices
    ]

    calibration_texts = [
        build_lm_text(example)
        for example in calibration_examples
    ]

    print(
        f"\nCalibration examples loaded: "
        f"{len(calibration_texts)}"
    )

    # --------------------------------------------------------
    # Evaluation subset
    # --------------------------------------------------------

    if NUM_EVAL_EXAMPLES is None:

        evaluation_examples = list(test_set)

    else:

        evaluation_examples = list(
            test_set.select(
                range(NUM_EVAL_EXAMPLES)
            )
        )

    print(
        f"Evaluation examples: "
        f"{len(evaluation_examples)}"
    )

    # ========================================================
    # Shapley estimation
    # ========================================================

    shapley_values = estimate_all_shapley_values(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
    )

    # Save raw values.
    shapley_path = (
        OUTPUT_DIR /
        "gpt2_gsm8k_shapley_values.pt"
    )

    torch.save(
        shapley_values,
        shapley_path,
    )

    print(
        f"\nShapley values saved to:\n"
        f"{shapley_path}"
    )

    # ========================================================
    # Rankings
    # ========================================================

    rankings = get_shapley_rankings(
        shapley_values
    )

    print("\n" + "=" * 70)
    print("SHAPLEY PRUNING RANKINGS")
    print("=" * 70)

    for layer, ranking in rankings.items():

        print(
            f"Layer {layer:2d}: "
            f"{ranking}"
        )

    ranking_path = (
        OUTPUT_DIR /
        "shapley_rankings.json"
    )

    with open(
        ranking_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            rankings,
            f,
            indent=2,
        )

    # ========================================================
    # GSM8K perplexity
    # ========================================================

    results = {}

    print("\n" + "=" * 70)
    print("GSM8K PERPLEXITY")
    print("=" * 70)

    for heads_per_layer in HEADS_TO_PRUNE:

        pruning_mask = build_pruning_mask(
            model=model,
            rankings=rankings,
            heads_per_layer=heads_per_layer,
        )

        pruning_ratio = (
            100
            * heads_per_layer
            / model.config.n_head
        )

        print(
            f"\nPruning:"
            f" {heads_per_layer} heads/layer"
            f" ({pruning_ratio:.2f}%)"
        )

        ppl = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            examples=evaluation_examples,
            head_mask=pruning_mask,
        )

        results[
            f"{pruning_ratio:.2f}%"
        ] = ppl

        print(
            f"GSM8K perplexity: "
            f"{ppl:.4f}"
        )

    # ========================================================
    # Save results
    # ========================================================

    results_path = (
        OUTPUT_DIR /
        "shapley_gsm8k_perplexity.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL SHAPLEY HEAD PRUNING RESULTS")
    print("=" * 70)

    for ratio, ppl in results.items():

        print(
            f"{ratio:>8} pruning "
            f"-> PPL = {ppl:.4f}"
        )

    print("=" * 70)

    print(
        "\nResults saved to:"
    )

    print(results_path)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()