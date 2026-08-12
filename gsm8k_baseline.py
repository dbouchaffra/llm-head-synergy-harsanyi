import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "gpt2"

# Shorter generation to reduce prompt repetition / drift
MAX_NEW_TOKENS = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# 2. Load GPT-2
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    clean_up_tokenization_spaces=False
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


# ============================================================
# 3. Load GSM8K
# ============================================================

dataset = load_dataset("openai/gsm8k", "main")
test_set = dataset["test"]

print("Number of GSM8K test examples:", len(test_set))


# ============================================================
# 4. Fixed few-shot prompt
# ============================================================

FEW_SHOT_PROMPT = """
Question: There are 15 trees in a garden and 6 are removed. How many trees remain?
Answer: 15 - 6 = 9. The answer is 9.

Question: A box contains 4 rows of 7 apples. How many apples are there?
Answer: 4 * 7 = 28. The answer is 28.

Question: Sarah has 24 candies and gives 8 candies to her friend. How many candies does Sarah have left?
Answer: 24 - 8 = 16. The answer is 16.

Question: A train travels 60 miles per hour for 3 hours. How many miles does it travel?
Answer: 60 * 3 = 180. The answer is 180.

"""


def build_prompt(question):
    return (
        FEW_SHOT_PROMPT
        + f"Question: {question}\n"
        + "Answer:"
    )


# ============================================================
# 5. Number normalization
# ============================================================

def normalize_number(value):
    if value is None:
        return None

    value = value.strip()
    value = value.replace(",", "")
    value = value.replace("$", "")

    try:
        number = float(value)

        # Convert 42.0 -> 42
        if number.is_integer():
            return str(int(number))

        # Convert 42.50 -> 42.5
        return str(number)

    except ValueError:
        return value


# ============================================================
# 6. Extract GSM8K ground-truth answer
# ============================================================

def extract_ground_truth(answer):
    """
    Official GSM8K answers contain the final answer after ####.
    Example:
        ... reasoning ...
        #### 18
    """

    match = re.search(
        r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)",
        answer
    )

    if match is None:
        return None

    return normalize_number(match.group(1))


# ============================================================
# 7. Keep only the model's first answer
# ============================================================

def isolate_first_answer(text):
    """
    GPT-2 may begin generating another Question: after its answer.

    Example:
        $18.

        Question: ...
        Answer: ...

    Everything beginning with the second Question: is discarded.
    """

    text = re.split(
        r"\n\s*(?:Question:|Q:)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    return text.strip()


# ============================================================
# 8. Extract predicted final answer
# ============================================================

def extract_prediction(text):
    """
    1. Keep only the first generated answer.
    2. Prefer explicit phrases such as:
           'The answer is 18'
           'Answer: 18'
    3. Otherwise use the last numerical value appearing in
       that first answer.
    """

    text = isolate_first_answer(text)

    # --------------------------------------------------------
    # First preference:
    # "The answer is 18"
    # "the answer is $18"
    # --------------------------------------------------------

    explicit_patterns = [
        r"the\s+answer\s+is\s*:?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"final\s+answer\s*(?:is|:)\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"answer\s*:\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    ]

    for pattern in explicit_patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if matches:
            return normalize_number(matches[-1])

    # --------------------------------------------------------
    # Fallback:
    # take the final number in the first answer only
    # --------------------------------------------------------

    numbers = re.findall(
        r"[-+]?\$?\d[\d,]*(?:\.\d+)?",
        text
    )

    if not numbers:
        return None

    return normalize_number(numbers[-1])


# ============================================================
# 9. Evaluation
# ============================================================

correct = 0
total = 0
no_prediction = 0

examples = []

for example in tqdm(test_set):

    question = example["question"]
    reference_text = example["answer"]

    gold = extract_ground_truth(reference_text)

    prompt = build_prompt(question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,

            # Limit answer length
            max_new_tokens=MAX_NEW_TOKENS,

            # Deterministic decoding
            do_sample=False,

            # GPT-2 padding / EOS
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,

            # Avoid warning
            use_cache=True,
        )

    # ========================================================
    # Decode ONLY newly generated tokens
    # ========================================================

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = output[0, input_length:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    first_answer = isolate_first_answer(generated_text)

    prediction = extract_prediction(generated_text)

    if prediction is None:
        no_prediction += 1

    is_correct = (
        prediction is not None
        and gold is not None
        and prediction == gold
    )

    correct += int(is_correct)
    total += 1

    # ========================================================
    # Save first 20 examples for manual inspection
    # ========================================================

    if len(examples) < 20:
        examples.append({
            "question": question,
            "gold": gold,
            "prediction": prediction,
            "first_answer": first_answer,
            "full_generation": generated_text,
            "correct": is_correct,
        })


# ============================================================
# 10. Final result
# ============================================================

accuracy = 100.0 * correct / total

print("\n")
print("========================================")
print("GPT-2 GSM8K BASELINE")
print("========================================")
print(f"Correct:        {correct}")
print(f"Total:          {total}")
print(f"No prediction:  {no_prediction}")
print(f"Accuracy:       {accuracy:.4f}%")
print("========================================")


# ============================================================
# 11. Print examples for verification
# ============================================================

print("\nSample predictions:\n")

for x in examples[:10]:

    print("Question:")
    print(x["question"])

    print("\nGold:")
    print(x["gold"])

    print("\nPrediction:")
    print(x["prediction"])

    print("\nCorrect:")
    print(x["correct"])

    print("\nFirst answer used for evaluation:")
    print(repr(x["first_answer"]))

    print("\nFull generation:")
    print(repr(x["full_generation"][:500]))

    print("\n" + "-" * 80 + "\n")