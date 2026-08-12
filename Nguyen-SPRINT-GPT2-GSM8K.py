from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "openai-community/gpt2"

# Sentence encoder used as phi(x).
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

# ------------------------------------------------------------
# SAME calibration size as Shapley / Qu-GAC
# ------------------------------------------------------------

NUM_CALIBRATION_EXAMPLES = 32
CALIBRATION_MAX_LENGTH = 256

# ------------------------------------------------------------
# SAME GSM8K perplexity evaluator
# ------------------------------------------------------------

EVAL_MAX_LENGTH = 512
NUM_EVAL_EXAMPLES = None  # None = complete 1319-example test set

# ------------------------------------------------------------
# SPRINT-style embedding training
# ------------------------------------------------------------

EMBED_DIM = 128

TRAIN_EPOCHS = 400
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# For each calibration question, heads whose removal produces
# lower loss than the unmodified model are considered positive.
#
# If no head improves loss, use the best-performing heads as
# fallback positives so every example provides supervision.
FALLBACK_POSITIVE_HEADS = 8

# ------------------------------------------------------------
# Matched intervention budgets
#
# GPT-2:
#   12 layers
#   12 heads/layer
#   144 total layer-head configurations
#
# 8.33%  = 12 dynamically selected heads
# 16.67% = 24 dynamically selected heads
# 25%    = 36 dynamically selected heads
# ------------------------------------------------------------

INTERVENTION_RATIOS = [
    0.0,
    8.33,
    16.67,
    25.0,
]

OUTPUT_DIR = Path("sprint_gpt2_gsm8k_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. LOAD GPT-2
# ============================================================

def load_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        clean_up_tokenization_spaces=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
    ).to(DEVICE)

    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return model, tokenizer


# ============================================================
# 4. GSM8K REPRESENTATION
# ============================================================

def build_lm_text(example):
    """
    SAME language-modeling representation used in the
    Shapley and Qu-GAC controlled evaluations.
    """

    return (
        f"Question: {example['question']}\n"
        f"Answer: {example['answer']}"
    )


def build_question_text(example):
    """
    SPRINT's selector receives the question itself rather
    than the complete question+answer sequence.
    """

    return example["question"]


# ============================================================
# 5. HEAD INDEX UTILITIES
# ============================================================

def flat_to_layer_head(flat_index, num_heads):
    """
    0 ... 143 -> (layer, head)
    """

    layer = flat_index // num_heads
    head = flat_index % num_heads

    return layer, head


def layer_head_to_flat(layer, head, num_heads):

    return layer * num_heads + head


# ============================================================
# 6. HEAD-MASK CONSTRUCTION
# ============================================================

def make_head_mask(
    num_layers,
    num_heads,
    pruned_flat_indices=None,
):
    """
    Hugging Face GPT-2 supports head_mask.

    head_mask[l,h] = 0:
        disable that attention head.

    head_mask[l,h] = 1:
        leave that attention head unchanged.

    This zeros the corresponding head contribution before
    the attention output is combined, matching the structured
    head-removal concept used in SPRINT.
    """

    mask = torch.ones(
        num_layers,
        num_heads,
        device=DEVICE,
        dtype=torch.float32,
    )

    if pruned_flat_indices is not None:

        for flat_index in pruned_flat_indices:

            layer, head = flat_to_layer_head(
                int(flat_index),
                num_heads,
            )

            mask[layer, head] = 0.0

    return mask


# ============================================================
# 7. PER-EXAMPLE LM LOSS
# ============================================================

@torch.no_grad()
def example_loss(
    model,
    tokenizer,
    example,
    head_mask=None,
):
    """
    Language-model loss for one GSM8K example.
    """

    text = build_lm_text(example)

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
        head_mask=head_mask,
        use_cache=False,
    )

    return outputs.loss.item()


# ============================================================
# 8. BUILD SPRINT-STYLE CALIBRATION MATRIX
# ============================================================

def build_calibration_matrix(
    model,
    tokenizer,
    calibration_examples,
):
    """
    Original SPRINT evaluates individually pruned
    layer-head configurations and obtains an outcome vector
    for each question.

    Original paper:
        z_ij = 1 if pruning configuration j results in a
        correct answer for question i.

    PPL adaptation:
        score_ij =
            baseline_loss_i - pruned_loss_ij

    Positive score:
        pruning that head improves language-model loss.

    This preserves SPRINT's question-specific supervision
    while aligning it to the metric used in our controlled
    experiment.
    """

    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    total_heads = num_layers * num_heads

    num_examples = len(calibration_examples)

    improvement_matrix = np.zeros(
        (num_examples, total_heads),
        dtype=np.float32,
    )

    baseline_losses = np.zeros(
        num_examples,
        dtype=np.float32,
    )

    print("\n" + "=" * 70)
    print("BUILDING SPRINT-STYLE CALIBRATION MATRIX")
    print("=" * 70)

    for example_index, example in enumerate(
        calibration_examples
    ):

        print(
            f"\nCalibration example "
            f"{example_index + 1}/{num_examples}"
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline_loss = example_loss(
            model=model,
            tokenizer=tokenizer,
            example=example,
            head_mask=None,
        )

        baseline_losses[example_index] = baseline_loss

        print(
            f"Baseline loss: {baseline_loss:.6f}"
        )

        # ----------------------------------------------------
        # Evaluate every individually pruned layer-head pair
        # ----------------------------------------------------

        for flat_index in tqdm(
            range(total_heads),
            desc="Single-head pruning",
        ):

            mask = make_head_mask(
                num_layers=num_layers,
                num_heads=num_heads,
                pruned_flat_indices=[flat_index],
            )

            pruned_loss = example_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                head_mask=mask,
            )

            improvement = (
                baseline_loss
                - pruned_loss
            )

            improvement_matrix[
                example_index,
                flat_index
            ] = improvement

        # Save continuously.
        np.save(
            OUTPUT_DIR
            / "sprint_improvement_matrix_partial.npy",
            improvement_matrix,
        )

        np.save(
            OUTPUT_DIR
            / "sprint_baseline_losses_partial.npy",
            baseline_losses,
        )

    return (
        improvement_matrix,
        baseline_losses,
    )


# ============================================================
# 9. CREATE POSITIVE SETS
# ============================================================

def build_positive_mask(
    improvement_matrix,
):
    """
    Original SPRINT:

        M_i+ = configurations that solve question i.

    PPL adaptation:

        M_i+ = configurations whose pruning decreases
        question-level language-model loss.

    If there are no improving heads for an example,
    use the best FALLBACK_POSITIVE_HEADS heads.
    """

    num_examples, total_heads = (
        improvement_matrix.shape
    )

    positive_mask = np.zeros(
        (num_examples, total_heads),
        dtype=np.float32,
    )

    print("\nPositive configurations per calibration example:")

    for i in range(num_examples):

        scores = improvement_matrix[i]

        positive = np.where(
            scores > 0.0
        )[0]

        if len(positive) == 0:

            positive = np.argsort(
                scores
            )[-FALLBACK_POSITIVE_HEADS:]

        positive_mask[
            i,
            positive
        ] = 1.0

        print(
            f"Example {i:2d}: "
            f"{len(positive)} positive heads"
        )

    return positive_mask


# ============================================================
# 10. SPRINT SELECTOR
# ============================================================

class SprintSelector(nn.Module):
    """
    Adaptation of SPRINT's learned embedding space.

    q_i = theta(phi(x_i))

    Each layer-head pair j has a learned embedding v_j.

    Selection at inference:
        nearest v_j to q_i.
    """

    def __init__(
        self,
        sentence_dim,
        embed_dim,
        total_heads,
    ):

        super().__init__()

        self.projection = nn.Linear(
            sentence_dim,
            embed_dim,
        )

        self.head_embeddings = nn.Parameter(
            torch.randn(
                total_heads,
                embed_dim,
            )
            * 0.02
        )

    def question_embedding(
        self,
        sentence_embeddings,
    ):

        q = self.projection(
            sentence_embeddings
        )

        return F.normalize(
            q,
            dim=-1,
        )

    def normalized_head_embeddings(self):

        return F.normalize(
            self.head_embeddings,
            dim=-1,
        )

    def distances(
        self,
        sentence_embeddings,
    ):

        q = self.question_embedding(
            sentence_embeddings
        )

        v = self.normalized_head_embeddings()

        # squared Euclidean distance
        distances = (
            q.unsqueeze(1)
            - v.unsqueeze(0)
        ).pow(2).sum(dim=-1)

        return distances


# ============================================================
# 11. CONTRASTIVE SPRINT LOSS
# ============================================================

def sprint_contrastive_loss(
    distances,
    positive_mask,
):
    """
    First term of Nguyen et al.'s SPRINT objective:

          -log(
             sum_{j in M_i+} exp(-d_ij)
             --------------------------------
             sum_j exp(-d_ij)
          )

    We implement it numerically through logsumexp.
    """

    logits = -distances

    all_logsumexp = torch.logsumexp(
        logits,
        dim=1,
    )

    masked_logits = logits.masked_fill(
        positive_mask == 0,
        float("-inf"),
    )

    positive_logsumexp = torch.logsumexp(
        masked_logits,
        dim=1,
    )

    loss = (
        all_logsumexp
        - positive_logsumexp
    ).mean()

    return loss


# ============================================================
# 12. TRAIN SELECTOR
# ============================================================

def train_sprint_selector(
    sentence_encoder,
    question_texts,
    positive_mask_np,
    total_heads,
):
    """
    Train the SPRINT-style projection and layer-head embeddings.

    SentenceTransformer.encode() may create inference tensors.
    To avoid PyTorch autograd errors, encode to NumPy first and
    then create a fresh ordinary torch.Tensor for training.
    """

    print("\n" + "=" * 70)
    print("TRAINING SPRINT-STYLE SELECTOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Encode questions with the frozen sentence encoder.
    # --------------------------------------------------------

    question_features_np = sentence_encoder.encode(
        question_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )

    # Fresh tensor: safe for autograd through self.projection.
    question_features = torch.tensor(
        question_features_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    sentence_dim = question_features.shape[-1]

    positive_mask = torch.tensor(
        positive_mask_np,
        device=DEVICE,
        dtype=torch.bool,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    selector = SprintSelector(
        sentence_dim=sentence_dim,
        embed_dim=EMBED_DIM,
        total_heads=total_heads,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Full-batch training (32 examples).
    # --------------------------------------------------------

    selector.train()

    for epoch in range(1, TRAIN_EPOCHS + 1):
        optimizer.zero_grad()

        distances = selector.distances(question_features)

        loss = sprint_contrastive_loss(
            distances,
            positive_mask,
        )

        loss.backward()
        optimizer.step()

        if (
            epoch == 1
            or epoch % 25 == 0
            or epoch == TRAIN_EPOCHS
        ):
            print(
                f"Epoch {epoch:4d}/{TRAIN_EPOCHS} "
                f"loss = {loss.item():.6f}"
            )

    selector.eval()
    return selector


# ============================================================
# 13. DYNAMIC HEAD SELECTION
# ============================================================

@torch.no_grad()
def select_heads_for_question(
    selector,
    sentence_encoder,
    question,
    budget,
):
    """
    SPRINT-style dynamic selection.

    Every input can receive a DIFFERENT set of heads.

    The selected configurations are the nearest learned
    layer-head embeddings to the current question embedding.
    """

    if budget == 0:
        return []

    feature_np = sentence_encoder.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )

    feature = torch.tensor(
        feature_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    distances = selector.distances(
        feature
    )[0]

    selected = torch.topk(
        distances,
        k=budget,
        largest=False,
    ).indices

    return selected.cpu().tolist()


# ============================================================
# 14. TOKEN-WEIGHTED GSM8K PERPLEXITY
# ============================================================

@torch.no_grad()
def evaluate_dynamic_perplexity(
    model,
    tokenizer,
    sentence_encoder,
    selector,
    examples,
    budget,
):
    """
    SAME token-weighted perplexity definition as the
    Shapley and Qu-GAC experiments.

    Difference:
        head mask is selected dynamically for EVERY
        GSM8K example.
    """

    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    total_nll = 0.0
    total_tokens = 0

    selection_counts = np.zeros(
        num_layers * num_heads,
        dtype=np.int64,
    )

    for example in tqdm(
        examples,
        desc=f"Evaluating dynamic budget={budget}",
    ):

        question = build_question_text(
            example
        )

        selected_heads = (
            select_heads_for_question(
                selector=selector,
                sentence_encoder=sentence_encoder,
                question=question,
                budget=budget,
            )
        )

        for flat_index in selected_heads:
            selection_counts[flat_index] += 1

        if budget == 0:

            head_mask = None

        else:

            head_mask = make_head_mask(
                num_layers=num_layers,
                num_heads=num_heads,
                pruned_flat_indices=selected_heads,
            )

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
            head_mask=head_mask,
            use_cache=False,
        )

        # GPT-2 predicts tokens 1...N.
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

    ppl = math.exp(
        mean_nll
    )

    return (
        ppl,
        selection_counts,
    )


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "NGUYEN et al. SPRINT-STYLE ADAPTATION"
    )
    print(
        "GPT-2 / GSM8K / TOKEN-WEIGHTED PERPLEXITY"
    )
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(
        f"Calibration examples: "
        f"{NUM_CALIBRATION_EXAMPLES}"
    )

    # ========================================================
    # Load GPT-2
    # ========================================================

    model, tokenizer = (
        load_model_and_tokenizer()
    )

    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    total_heads = (
        num_layers * num_heads
    )

    print("\nGPT-2 loaded.")
    print(f"Layers: {num_layers}")
    print(
        f"Heads/layer: {num_heads}"
    )
    print(
        f"Total layer-head configurations: "
        f"{total_heads}"
    )

    # ========================================================
    # Load sentence encoder
    # ========================================================

    print(
        "\nLoading sentence encoder..."
    )

    sentence_encoder = SentenceTransformer(
        SENTENCE_MODEL_NAME,
        device=str(DEVICE),
    )

    # Freeze sentence encoder.
    for parameter in (
        sentence_encoder.parameters()
    ):
        parameter.requires_grad = False

    # ========================================================
    # GSM8K
    # ========================================================

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
    )

    train_set = dataset["train"]
    test_set = dataset["test"]

    # SAME fixed sampling seed as other experiments.
    generator = random.Random(
        SEED
    )

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

    calibration_questions = [
        build_question_text(example)
        for example in calibration_examples
    ]

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
        f"\nCalibration examples: "
        f"{len(calibration_examples)}"
    )

    print(
        f"Evaluation examples: "
        f"{len(evaluation_examples)}"
    )

    # ========================================================
    # Build per-question / per-head calibration outcomes
    # ========================================================

    matrix_path = (
        OUTPUT_DIR
        / "sprint_improvement_matrix.npy"
    )

    baseline_path = (
        OUTPUT_DIR
        / "sprint_baseline_losses.npy"
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    partial_matrix_path = (
        OUTPUT_DIR / "sprint_improvement_matrix_partial.npy"
    )
    partial_baseline_path = (
        OUTPUT_DIR / "sprint_baseline_losses_partial.npy"
    )

    if matrix_path.exists() and baseline_path.exists():
        print("\nExisting calibration matrix found.")
        print("Loading instead of recomputing...")

        improvement_matrix = np.load(matrix_path)
        baseline_losses = np.load(baseline_path)

    elif partial_matrix_path.exists() and partial_baseline_path.exists():
        candidate_matrix = np.load(partial_matrix_path)
        candidate_baselines = np.load(partial_baseline_path)

        expected_shape = (NUM_CALIBRATION_EXAMPLES, total_heads)
        partial_is_complete = (
            candidate_matrix.shape == expected_shape
            and candidate_baselines.shape == (NUM_CALIBRATION_EXAMPLES,)
            and np.all(np.isfinite(candidate_matrix))
            and np.all(np.isfinite(candidate_baselines))
            and np.all(candidate_baselines > 0)
        )

        if partial_is_complete:
            print("\nComplete partial calibration files found.")
            print("Loading them instead of recomputing...")

            improvement_matrix = candidate_matrix
            baseline_losses = candidate_baselines

            # Promote completed partial files to canonical files.
            np.save(matrix_path, improvement_matrix)
            np.save(baseline_path, baseline_losses)
        else:
            print("\nPartial calibration files are incomplete; recomputing.")
            improvement_matrix, baseline_losses = build_calibration_matrix(
                model=model,
                tokenizer=tokenizer,
                calibration_examples=calibration_examples,
            )
            np.save(matrix_path, improvement_matrix)
            np.save(baseline_path, baseline_losses)

    else:
        improvement_matrix, baseline_losses = build_calibration_matrix(
            model=model,
            tokenizer=tokenizer,
            calibration_examples=calibration_examples,
        )

        np.save(matrix_path, improvement_matrix)
        np.save(baseline_path, baseline_losses)

    # ========================================================
    # Positive head configurations
    # ========================================================

    positive_mask = (
        build_positive_mask(
            improvement_matrix
        )
    )

    np.save(
        OUTPUT_DIR
        / "sprint_positive_mask.npy",
        positive_mask,
    )

    # ========================================================
    # Train SPRINT selector
    # ========================================================

    selector = train_sprint_selector(
        sentence_encoder=sentence_encoder,
        question_texts=calibration_questions,
        positive_mask_np=positive_mask,
        total_heads=total_heads,
    )

    torch.save(
        selector.state_dict(),
        OUTPUT_DIR
        / "sprint_selector.pt",
    )

    # ========================================================
    # Matched budgets
    # ========================================================

    budgets = {}

    for ratio in INTERVENTION_RATIOS:

        if ratio == 0:

            budget = 0

        else:

            budget = int(
                round(
                    total_heads
                    * ratio
                    / 100.0
                )
            )

        budgets[ratio] = budget

    print("\nMatched dynamic budgets:")

    for ratio, budget in budgets.items():

        print(
            f"{ratio:5.2f}%"
            f" -> "
            f"{budget} layer-head pairs/example"
        )

    # ========================================================
    # Evaluate
    # ========================================================

    results = {}

    selection_statistics = {}

    print("\n" + "=" * 70)
    print(
        "GSM8K PERPLEXITY — SPRINT-STYLE ADAPTATION"
    )
    print("=" * 70)

    for ratio in INTERVENTION_RATIOS:

        budget = budgets[ratio]

        print(
            f"\nIntervention budget: "
            f"{ratio:.2f}% "
            f"({budget}/{total_heads} heads per input)"
        )

        ppl, counts = (
            evaluate_dynamic_perplexity(
                model=model,
                tokenizer=tokenizer,
                sentence_encoder=sentence_encoder,
                selector=selector,
                examples=evaluation_examples,
                budget=budget,
            )
        )

        results[
            f"{ratio:.2f}%"
        ] = ppl

        # ----------------------------------------------------
        # Save head-selection frequencies
        # ----------------------------------------------------

        frequency_dict = {}

        for flat_index, count in enumerate(
            counts
        ):

            if count == 0:
                continue

            layer, head = (
                flat_to_layer_head(
                    flat_index,
                    num_heads,
                )
            )

            key = (
                f"L{layer}_H{head}"
            )

            frequency_dict[key] = int(
                count
            )

        selection_statistics[
            f"{ratio:.2f}%"
        ] = frequency_dict

        print(
            f"GSM8K perplexity: "
            f"{ppl:.4f}"
        )

        # Save after every budget.
        with open(
            OUTPUT_DIR
            / "sprint_gsm8k_perplexity_partial.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                results,
                f,
                indent=2,
            )

    # ========================================================
    # Save final results
    # ========================================================

    results_path = (
        OUTPUT_DIR
        / "sprint_gsm8k_perplexity.json"
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

    with open(
        OUTPUT_DIR
        / "sprint_selection_statistics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selection_statistics,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "FINAL SPRINT-STYLE ADAPTATION RESULTS"
    )
    print("=" * 70)

    for ratio, ppl in results.items():

        print(
            f"{ratio:>7} intervention "
            f"-> PPL = {ppl:.4f}"
        )

    print("=" * 70)

    print(
        "\nResults saved to:"
    )

    print(
        results_path
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()