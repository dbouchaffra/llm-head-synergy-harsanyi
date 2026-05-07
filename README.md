# LLM Head Synergy via Harsanyi Dividends

This repository contains the official code for the paper:

> **Higher‑Order Synergy in Large Language Models: A Harsanyi Dividend Analysis of Attention Heads**  
> Djamel Bouchaffra  
> *Submitted to Neural Networks* (2026)

The code implements a principled framework based on the **Game‑Theoretic Free Energy Principle (GT‑FEP)** to quantify synergy and redundancy among attention heads in transformer models. It computes Harsanyi dividends for coalitions of heads, identifies synergistic pairs/triples, and prunes redundant heads without significant accuracy loss.

## 🔬 What this repository does

- Computes **pairwise Harsanyi dividends** for attention heads in GPT‑2, BERT, and TinyLlama (Llama‑like) on the GSM8K reasoning dataset.
- Derives **Shapley‑value importance scores** for each head (`φ(i) = d({i}) + ½ Σ_{j≠i} d({i,j})`).
- Performs **head pruning** experiments (removing 5%, 10%, 20% of heads with lowest scores) and compares against random and magnitude‑based pruning.
- Generates **task‑dependent synergy heatmaps** (Jaccard similarity of positive coalitions across reasoning, memorization, and coding tasks).
- Produces publication‑ready figures: pairwise dividend heatmaps, head importance histograms, pruning masks, perplexity curves, and task similarity matrices.

### Scripts and their outputs

- `gpt_bert_analysis.py` → generates `gpt_pair_dividends_heatmap.png`, `bert_pair_dividends_heatmap.jpg`, `histograms-layers.png`, `pruning_masks.png`, `perplexity_curve.png`.
- `llama_analysis.py` → generates `llama_head_scores_heatmap.png`.
- `task_similarity.py` → generates `task_heatmap.png`.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Hugging Face `transformers` and `datasets`
- NumPy, Matplotlib, Seaborn, tqdm
- (Optional) CUDA‑capable GPU (NVIDIA GeForce RTX or higher)

Install all dependencies with:

```bash
pip install torch transformers datasets matplotlib seaborn numpy scipy tqdm
## Usage
(Add brief usage instructions here, e.g., how to run each script.)

📁 Repository structure
(Optional – you can list main files and folders.)

📖 Citation
If you use this code or data in your research, please cite our paper:

bibtex
@article{Bouchaffra2026Synergy,
  title   = {Higher‑Order Synergy in Large Language Models: A Harsanyi Dividend Analysis of Attention Heads},
  author  = {Bouchaffra, Djamel},
  journal = {Neural Networks},
  year    = {2026},
  note    = {Submitted}
}
📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

🤝 Contact
For questions or feedback, please open an issue or contact Djamel Bouchaffra at djamel.bouchaffra@uvsq.fr.
