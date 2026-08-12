from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "openai-community/gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 has no padding token by default.
    tokenizer.pad_token = tokenizer.eos_token

    # Eager attention is the safest option when using differentiable head masks.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
    ).to(DEVICE)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    return model, tokenizer


def compute_gradient_head_importance(
    model,
    tokenizer,
    texts: Sequence[str],
    max_length: int = 512,
) -> torch.Tensor:
    """
    Estimate gradient-based attention-head importance:

        I_h = E_x[|dL(x) / d xi_h|]

    Returns:
        Tensor of shape [num_layers, num_heads].
    """
    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    importance = torch.zeros(
        num_layers,
        num_heads,
        device=DEVICE,
        dtype=torch.float32,
    )

    # Keep model parameters differentiable so the loss has a valid graph.
    model.requires_grad_(True)
    model.eval()

    print("Global grad mode:", torch.is_grad_enabled())
    print("Inference mode:", torch.is_inference_mode_enabled())

    for index, text in enumerate(texts):
        model.zero_grad(set_to_none=True)

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

        head_mask = torch.ones(
            num_layers,
            num_heads,
            device=DEVICE,
            dtype=torch.float32,
            requires_grad=True,
        )

        with torch.enable_grad():
            outputs = model(
                **encoded,
                labels=encoded["input_ids"],
                head_mask=head_mask,
                use_cache=False,
            )

            loss = outputs.loss

            print(
                f"Example {index + 1}: "
                f"loss.requires_grad={loss.requires_grad}, "
                f"head_mask.requires_grad={head_mask.requires_grad}"
            )

            if not loss.requires_grad:
                raise RuntimeError(
                    "The loss still has no gradient graph. "
                    "Check that the function is not called inside "
                    "torch.no_grad() or torch.inference_mode()."
                )

            loss.backward()

        if head_mask.grad is None:
            raise RuntimeError(
                "The loss has gradients, but GPT-2 did not propagate them "
                "to head_mask. This Transformers version may not support "
                "differentiable GPT-2 head masks."
            )

        importance += head_mask.grad.detach().abs()

        print(
            f"Processed {index + 1}/{len(texts)} "
            f"| loss={loss.item():.4f} "
            f"| mask grad mean={head_mask.grad.abs().mean().item():.6f}"
        )

    importance /= max(len(texts), 1)

    layer_norms = torch.linalg.vector_norm(
        importance,
        ord=2,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)

    importance = importance / layer_norms

    return importance.cpu()
if __name__ == "__main__":
    print(f"Using device: {DEVICE}", flush=True)

    model, tokenizer = load_model_and_tokenizer()
    print("Model loaded.", flush=True)

    # Small test dataset.
    calibration_texts = [
        (
            "Question: John has 3 apples and buys 2 more apples. "
            "How many apples does he have? Answer: 5."
        ),
        (
            "Question: A box contains 12 pencils. Four pencils are removed. "
            "How many pencils remain? Answer: 8."
        ),
        (
            "Question: Sarah has 10 euros and spends 6 euros. "
            "How many euros remain? Answer: 4."
        ),
    ]

    importance = compute_gradient_head_importance(
        model=model,
        tokenizer=tokenizer,
        texts=calibration_texts,
        max_length=128,
    )

    print("\nGradient-based head importance:", flush=True)
    print(importance, flush=True)
    print("\nShape:", importance.shape, flush=True)

    output_path = "gpt2_gradient_head_importance.pt"
    torch.save(importance, output_path)

    print(f"\nResults saved to: {output_path}", flush=True)