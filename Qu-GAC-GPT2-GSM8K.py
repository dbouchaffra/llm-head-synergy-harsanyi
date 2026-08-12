from __future__ import annotations

import json
import math
import random
from contextlib import contextmanager
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
# Calibration data
# ------------------------------------------------------------

# Keep identical to the Shapley experiment.
NUM_CALIBRATION_EXAMPLES = 32
CALIBRATION_MAX_LENGTH = 256

# ------------------------------------------------------------
# GSM8K perplexity evaluation
# ------------------------------------------------------------

EVAL_MAX_LENGTH = 512

# None = complete GSM8K test set (1319 examples)
NUM_EVAL_EXAMPLES = None

# ------------------------------------------------------------
# Qu et al. GAC hyperparameters
#
# Original paper:
# alpha = 0.1
# beta  = 0.1
# ------------------------------------------------------------

GAC_ALPHA = 0.1
GAC_BETA = 0.1

# ------------------------------------------------------------
# Matched intervention budgets
#
# GPT-2:
# 12 heads/layer
#
# 1 head  = 8.33%
# 2 heads = 16.67%
# 3 heads = 25%
# ------------------------------------------------------------

HEADS_TO_CALIBRATE = [0, 1, 2, 3]

OUTPUT_DIR = Path("qu_gac_gsm8k_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. Load GPT-2
# ============================================================

def load_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        clean_up_tokenization_spaces=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    # We need explicit eager attention because the experiment
    # modifies attention probabilities through dropout hooks.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
    ).to(DEVICE)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    return model, tokenizer


# ============================================================
# 4. GSM8K language-modeling representation
# ============================================================

def build_lm_text(example):
    """
    IMPORTANT:
    This is intentionally identical to the representation
    used in the Shapley perplexity experiment.
    """

    return (
        f"Question: {example['question']}\n"
        f"Answer: {example['answer']}"
    )


# ============================================================
# 5. Utility: uniform causal attention
# ============================================================

def make_uniform_causal_attention(attention_probs):
    """
    Construct approximately undifferentiated attention.

    Qu et al. mask a head by multiplying QK scores by
    epsilon = 1e-5 before softmax, causing the attention
    distribution to approach uniform attention over valid
    positions.

    For GPT-2 causal attention, we directly construct the
    corresponding uniform causal distribution.

    attention_probs:
        [batch, heads, query_length, key_length]
    """

    batch_size, _, q_len, k_len = attention_probs.shape

    device = attention_probs.device
    dtype = attention_probs.dtype

    # GPT-2 causal structure.
    causal = torch.tril(
        torch.ones(
            q_len,
            k_len,
            device=device,
            dtype=dtype,
        ),
        diagonal=k_len - q_len,
    )

    # Normalize every query row.
    denom = causal.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)

    uniform = causal / denom

    return uniform.unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
    )


# ============================================================
# 6. Coalition masking hook
# ============================================================

def make_coalition_mask_hook(masked_heads):
    """
    Replace selected heads with approximately undifferentiated
    attention, following the masking idea in Eq. (4) of Qu et al.

    Heads NOT in the coalition are masked.
    """

    masked_heads = set(masked_heads)

    def hook(module, inputs, output):

        if not masked_heads:
            return output

        probs = output.clone()

        uniform = make_uniform_causal_attention(
            probs
        )

        for head in masked_heads:

            probs[:, head, :, :] = uniform

        return probs

    return hook


# ============================================================
# 7. Context manager for one coalition
# ============================================================

@contextmanager
def coalition_attention_context(
    model,
    layer_index,
    active_heads,
):
    """
    All heads outside active_heads in the selected layer are
    converted to approximately uniform attention.

    Other transformer layers remain untouched.
    """

    num_heads = model.config.n_head

    active_heads = set(active_heads)

    masked_heads = [
        h
        for h in range(num_heads)
        if h not in active_heads
    ]

    attention_module = (
        model.transformer.h[layer_index].attn
    )

    handle = attention_module.attn_dropout.register_forward_hook(
        make_coalition_mask_hook(masked_heads)
    )

    try:
        yield

    finally:
        handle.remove()


# ============================================================
# 8. Mean loss for one coalition
# ============================================================

@torch.no_grad()
def coalition_loss(
    model,
    tokenizer,
    calibration_texts,
    layer_index,
    active_heads,
):
    """
    Coalition value:

        v(S) = - L(S)

    where heads outside S in the target layer use
    approximately undifferentiated attention.
    """

    total_loss = 0.0

    with coalition_attention_context(
        model=model,
        layer_index=layer_index,
        active_heads=active_heads,
    ):

        for text in calibration_texts:

            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=CALIBRATION_MAX_LENGTH,
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded,
                labels=encoded["input_ids"],
                use_cache=False,
            )

            total_loss += outputs.loss.item()

    return total_loss / len(calibration_texts)


# ============================================================
# 9. Estimate pairwise Harsanyi interactions
# ============================================================

def estimate_layer_pairwise_harsanyi(
    model,
    tokenizer,
    calibration_texts,
    layer_index,
):
    """
    Second-order adaptation of Qu et al.'s interaction analysis.

    For every pair (i,j):

        Delta_ij
        =
        v({i,j})
        - v({i})
        - v({j})
        + v(empty)

    with:

        v(S) = -Loss(S)

    Positive:
        cooperative pair

    Negative:
        competitive interaction
    """

    num_heads = model.config.n_head

    print(
        f"\n{'=' * 70}\n"
        f"Layer {layer_index}: estimating pairwise "
        f"Harsanyi interactions\n"
        f"{'=' * 70}"
    )

    # --------------------------------------------------------
    # Empty coalition
    # --------------------------------------------------------

    empty_loss = coalition_loss(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        layer_index=layer_index,
        active_heads=[],
    )

    v_empty = -empty_loss

    print(
        f"Empty-coalition loss: {empty_loss:.6f}"
    )

    # --------------------------------------------------------
    # Singleton coalitions
    # --------------------------------------------------------

    singleton_values = {}

    for head in tqdm(
        range(num_heads),
        desc=f"Layer {layer_index} singletons",
    ):

        loss = coalition_loss(
            model=model,
            tokenizer=tokenizer,
            calibration_texts=calibration_texts,
            layer_index=layer_index,
            active_heads=[head],
        )

        singleton_values[head] = -loss

    # --------------------------------------------------------
    # Pair coalitions
    # --------------------------------------------------------

    interactions = torch.zeros(
        num_heads,
        num_heads,
        dtype=torch.float64,
    )

    pairs = [
        (i, j)
        for i in range(num_heads)
        for j in range(i + 1, num_heads)
    ]

    for i, j in tqdm(
        pairs,
        desc=f"Layer {layer_index} pairs",
    ):

        pair_loss = coalition_loss(
            model=model,
            tokenizer=tokenizer,
            calibration_texts=calibration_texts,
            layer_index=layer_index,
            active_heads=[i, j],
        )

        v_pair = -pair_loss

        delta = (
            v_pair
            - singleton_values[i]
            - singleton_values[j]
            + v_empty
        )

        interactions[i, j] = delta
        interactions[j, i] = delta

    return interactions


# ============================================================
# 10. Identify salient group + competition ranking
# ============================================================

def analyze_layer_interactions(
    interaction_matrix,
):
    """
    Qu et al. preserve a salient cooperative group before
    calibrating other heads.

    Their original maximum-Harsanyi criterion searches
    arbitrary coalitions.

    This GPT-2 adaptation uses the strongest positive
    SECOND-ORDER coalition (pair) as the salient group.

    The remaining heads are ranked according to:

        C_i = sum_j Delta_ij

    Lowest C_i:
        strongest overall competitive / weakest cooperative
        participation

    These heads are calibrated first.
    """

    num_heads = interaction_matrix.shape[0]

    # --------------------------------------------------------
    # Find maximum positive pairwise Harsanyi dividend
    # --------------------------------------------------------

    max_value = -float("inf")
    salient_pair = None

    for i in range(num_heads):
        for j in range(i + 1, num_heads):

            value = interaction_matrix[i, j].item()

            if value > max_value:
                max_value = value
                salient_pair = (i, j)

    # If no positive pair exists, do not protect a pair.
    if max_value <= 0:
        salient_heads = set()
    else:
        salient_heads = set(salient_pair)

    # --------------------------------------------------------
    # Aggregate cooperation / competition score
    # --------------------------------------------------------

    scores = interaction_matrix.sum(dim=1)

    candidate_heads = [
        h
        for h in range(num_heads)
        if h not in salient_heads
    ]

    # Lowest interaction score first.
    ranking = sorted(
        candidate_heads,
        key=lambda h: scores[h].item(),
    )

    return {
        "salient_heads": sorted(salient_heads),
        "salient_value": max_value,
        "scores": scores,
        "calibration_ranking": ranking,
    }


# ============================================================
# 11. GAC attention calibration
# ============================================================

def calibrate_single_head_attention(
    probs,
    alpha,
    beta,
):
    """
    Apply the redistribution mechanism from Qu et al.

    probs:
        [batch, query_length, key_length]

    Paper equations:

        focused tokens:
            a_i[t] > alpha

        diminished attention:
            A_hat[k,s] = A[k,s] * beta

        removed mass is redistributed proportionally among
        non-focused tokens.

    The first token is never calibrated.
    """

    calibrated = probs.clone()

    batch_size, q_len, k_len = calibrated.shape

    for b in range(batch_size):

        head_probs = calibrated[b]

        # ----------------------------------------------------
        # Eq. (9):
        #
        # a_i[m] =
        # sum_{n=1}^{m} A_i[m,n] / m
        #
        # m is one-based in the paper.
        # ----------------------------------------------------

        average_scores = torch.zeros(
            q_len,
            device=head_probs.device,
            dtype=head_probs.dtype,
        )

        for query_index in range(q_len):

            valid_length = min(
                query_index + 1,
                k_len,
            )

            average_scores[query_index] = (
                head_probs[
                    query_index,
                    :valid_length
                ].sum()
                / valid_length
            )

        # ----------------------------------------------------
        # Eq. (10)
        #
        # Do NOT calibrate first token.
        # ----------------------------------------------------

        focused = torch.nonzero(
            average_scores > alpha,
            as_tuple=False,
        ).flatten()

        focused = focused[
            focused != 0
        ]

        # Only key positions that actually exist.
        focused = focused[
            focused < k_len
        ]

        if focused.numel() == 0:
            continue

        focused_mask = torch.zeros(
            k_len,
            dtype=torch.bool,
            device=head_probs.device,
        )

        focused_mask[focused] = True

        nonfocused_mask = ~focused_mask

        # ----------------------------------------------------
        # Operate on every attention row.
        # ----------------------------------------------------

        for query_index in range(q_len):

            row = head_probs[
                query_index
            ].clone()

            # Future causal positions already contain zero.
            original_focused_mass = (
                row[focused_mask].sum()
            )

            # Eq. (11)
            new_focused = (
                row[focused_mask] * beta
            )

            new_focused_mass = (
                new_focused.sum()
            )

            removed_mass = (
                original_focused_mass
                - new_focused_mass
            )

            row[focused_mask] = new_focused

            # ------------------------------------------------
            # Eq. (12):
            # redistribute removed mass proportionally
            # according to existing non-focused weights.
            # ------------------------------------------------

            remaining_mass = (
                row[nonfocused_mask].sum()
            )

            if (
                removed_mass > 0
                and remaining_mass > 0
            ):

                proportions = (
                    row[nonfocused_mask]
                    / remaining_mass
                )

                row[nonfocused_mask] += (
                    removed_mass
                    * proportions
                )

            # Numerical normalization.
            row_sum = row.sum()

            if row_sum > 0:
                row = row / row_sum

            head_probs[query_index] = row

        calibrated[b] = head_probs

    return calibrated


# ============================================================
# 12. Calibration hook for one layer
# ============================================================

def make_gac_calibration_hook(
    selected_heads,
    alpha,
    beta,
):

    selected_heads = list(selected_heads)

    def hook(module, inputs, output):

        if not selected_heads:
            return output

        probs = output.clone()

        for head in selected_heads:

            probs[:, head, :, :] = (
                calibrate_single_head_attention(
                    probs[:, head, :, :],
                    alpha=alpha,
                    beta=beta,
                )
            )

        return probs

    return hook


# ============================================================
# 13. Install GAC hooks across GPT-2
# ============================================================

@contextmanager
def gac_context(
    model,
    heads_by_layer,
    alpha,
    beta,
):
    """
    heads_by_layer:

        {
            0: [heads to calibrate],
            1: [...],
            ...
        }
    """

    handles = []

    try:

        for layer_index, heads in (
            heads_by_layer.items()
        ):

            if len(heads) == 0:
                continue

            module = (
                model.transformer
                .h[layer_index]
                .attn
                .attn_dropout
            )

            handle = module.register_forward_hook(
                make_gac_calibration_hook(
                    selected_heads=heads,
                    alpha=alpha,
                    beta=beta,
                )
            )

            handles.append(handle)

        yield

    finally:

        for handle in handles:
            handle.remove()


# ============================================================
# 14. Build matched-budget intervention
# ============================================================

def build_gac_budget(
    layer_analyses,
    heads_per_layer,
):
    """
    Calibrate exactly k heads per layer.

    Heads belonging to the strongest positive cooperative
    pair are protected.

    Among remaining heads, the most competitive heads are
    calibrated first.
    """

    result = {}

    for layer, analysis in (
        layer_analyses.items()
    ):

        if heads_per_layer == 0:
            result[layer] = []
            continue

        ranking = analysis[
            "calibration_ranking"
        ]

        result[layer] = ranking[
            :heads_per_layer
        ]

    return result


# ============================================================
# 15. Token-weighted GSM8K perplexity
# ============================================================

@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    examples,
    heads_by_layer,
):
    """
    Same token-weighted perplexity definition used by the
    Shapley experiment.
    """

    total_nll = 0.0
    total_tokens = 0

    with gac_context(
        model=model,
        heads_by_layer=heads_by_layer,
        alpha=GAC_ALPHA,
        beta=GAC_BETA,
    ):

        for example in tqdm(
            examples,
            desc="Evaluating PPL",
        ):

            text = build_lm_text(
                example
            )

            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=EVAL_MAX_LENGTH,
            )

            input_ids = (
                encoded["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                encoded["attention_mask"]
                .to(DEVICE)
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_cache=False,
            )

            # GPT-2 predicts tokens 1...N
            num_tokens = int(
                attention_mask[
                    :, 1:
                ].sum().item()
            )

            total_nll += (
                outputs.loss.item()
                * num_tokens
            )

            total_tokens += num_tokens

    mean_nll = (
        total_nll
        / total_tokens
    )

    return math.exp(mean_nll)


# ============================================================
# 16. Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "QU et al. GAC ADAPTATION — GPT-2 / GSM8K"
    )
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(
        f"Calibration examples: "
        f"{NUM_CALIBRATION_EXAMPLES}"
    )
    print(
        f"GAC alpha: {GAC_ALPHA}"
    )
    print(
        f"GAC beta:  {GAC_BETA}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, tokenizer = (
        load_model_and_tokenizer()
    )

    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    print("\nModel loaded.")
    print(f"Layers: {num_layers}")
    print(f"Heads/layer: {num_heads}")

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
    # SAME fixed calibration sampling logic as Shapley
    # --------------------------------------------------------

    generator = random.Random(SEED)

    calibration_indices = (
        generator.sample(
            range(len(train_set)),
            NUM_CALIBRATION_EXAMPLES,
        )
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
    # Evaluation examples
    # --------------------------------------------------------

    if NUM_EVAL_EXAMPLES is None:

        evaluation_examples = list(
            test_set
        )

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
    # Estimate Harsanyi interactions
    # ========================================================

    interaction_matrices = {}
    layer_analyses = {}

    for layer in range(num_layers):

        matrix = (
            estimate_layer_pairwise_harsanyi(
                model=model,
                tokenizer=tokenizer,
                calibration_texts=calibration_texts,
                layer_index=layer,
            )
        )

        interaction_matrices[
            layer
        ] = matrix

        analysis = (
            analyze_layer_interactions(
                matrix
            )
        )

        layer_analyses[
            layer
        ] = analysis

        print(
            f"\nLayer {layer}"
        )

        print(
            "Salient cooperative heads:",
            analysis["salient_heads"],
        )

        print(
            "Maximum pairwise Harsanyi:",
            f"{analysis['salient_value']:+.8f}",
        )

        print(
            "Calibration ranking:",
            analysis[
                "calibration_ranking"
            ],
        )

        # Save after EVERY layer so a crash does not
        # destroy hours of completed computation.
        torch.save(
            interaction_matrices,
            OUTPUT_DIR
            / "pairwise_harsanyi_partial.pt",
        )

    # ========================================================
    # Save complete interaction results
    # ========================================================

    torch.save(
        interaction_matrices,
        OUTPUT_DIR
        / "gpt2_gsm8k_qu_pairwise_harsanyi.pt",
    )

    # JSON-compatible summary.
    json_analysis = {}

    for layer, analysis in (
        layer_analyses.items()
    ):

        json_analysis[str(layer)] = {
            "salient_heads":
                analysis[
                    "salient_heads"
                ],

            "salient_value":
                analysis[
                    "salient_value"
                ],

            "interaction_scores":
                analysis[
                    "scores"
                ].tolist(),

            "calibration_ranking":
                analysis[
                    "calibration_ranking"
                ],
        }

    with open(
        OUTPUT_DIR
        / "qu_gac_layer_analysis.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_analysis,
            f,
            indent=2,
        )

    # ========================================================
    # Evaluate matched budgets
    # ========================================================

    results = {}

    print("\n" + "=" * 70)
    print("GSM8K PERPLEXITY — GAC ADAPTATION")
    print("=" * 70)

    for heads_per_layer in (
        HEADS_TO_CALIBRATE
    ):

        ratio = (
            100.0
            * heads_per_layer
            / num_heads
        )

        intervention = (
            build_gac_budget(
                layer_analyses,
                heads_per_layer,
            )
        )

        print(
            f"\nCalibration budget: "
            f"{heads_per_layer} heads/layer "
            f"({ratio:.2f}%)"
        )

        if heads_per_layer > 0:

            for layer in range(
                num_layers
            ):

                print(
                    f"Layer {layer:2d}: "
                    f"{intervention[layer]}"
                )

        ppl = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            examples=evaluation_examples,
            heads_by_layer=intervention,
        )

        results[
            f"{ratio:.2f}%"
        ] = ppl

        print(
            f"GSM8K perplexity: "
            f"{ppl:.4f}"
        )

    # ========================================================
    # Save final results
    # ========================================================

    results_path = (
        OUTPUT_DIR
        / "qu_gac_gsm8k_perplexity.json"
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
    # Final output
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "FINAL QU-GAC ADAPTATION RESULTS"
    )
    print("=" * 70)

    for ratio, ppl in results.items():

        print(
            f"{ratio:>8} intervention "
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