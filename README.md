LLM Head Synergy via Harsanyi Dividends

This repository contains the official code for the paper:

A Coalition-Based Game-Theoretic Framework for Higher-Order Attention Interactions and Structured Head Pruning
Djamel Bouchaffra
Submitted to Pattern Recognition (2026)

The code implements a coalition-based framework derived from the Game-Theoretic Free Energy Principle (GT-FEP) for analyzing higher-order interactions among attention heads and performing structured head pruning in transformer models. Attention heads are modeled as players in a cooperative game, while arbitrary subsets of heads form coalitions whose values are characterized through Harsanyi interaction terms.

The repository supports both information-theoretic interaction analysis and performance-based structured pruning.

🔬 What this repository does
Models transformer attention heads as coalitions in a cooperative game and computes Harsanyi interaction terms describing dependencies among heads.
Performs information-theoretic interaction analysis using attention-derived random variables, where each head variable records the key position receiving maximum attention for a given query token.
Estimates singleton, pairwise, and selected higher-order Harsanyi interactions in BERT, GPT-2, and TinyLlama on GSM8K.
Computes second-order Harsanyi-Shapley interaction scores:

\Delta({i})
+
\frac{1}{2}
\sum_{j\neq i}\Delta({i,j})
$$

as well as non-cancelling interaction measures based on absolute Harsanyi magnitudes.

Implements a performance-based structured pruning score:

\operatorname{Norm}(g_i)
+
\lambda
\operatorname{Norm}
\left(
\sum_{j\neq i}|\Gamma_{ij}|
\right)
$$

Performs permanent structured head pruning of GPT-2 and evaluates the resulting models using token-weighted perplexity on Penn Treebank and GSM8K.
Compares the proposed pruning criterion against random, gradient-based, and Monte Carlo Shapley-based static pruning under controlled pruning ratios.
Includes matched-intervention experiments with GPT-2 adaptations of GAC and SPRINT, distinguishing predictive preservation from permanent structural head removal.
Generates figures for analyzing interaction structure across heads and layers, including Harsanyi interaction heatmaps, head-score distributions, pruning results, and perplexity curves.
Scripts and their outputs
gpt_bert_analysis.py → performs information-theoretic interaction analysis for GPT-2 and BERT and generates pairwise interaction heatmaps and head-level statistics.
llama_analysis.py → analyzes attention-head interaction structure across TinyLlama layers and generates head-interaction visualizations.
Pruning experiments → compute performance-based singleton and pairwise head-removal effects, rank heads using the proposed coalition score, and evaluate structured pruning through token-weighted perplexity.
Baseline experiments → evaluate random, gradient-based, and Shapley-based static pruning, together with matched GAC and SPRINT-style intervention protocols where applicable.
Requirements
Python 3.10+
PyTorch 2.0+
Hugging Face transformers and datasets
NumPy
SciPy
Matplotlib
Seaborn
tqdm
Optional: CUDA-capable NVIDIA GPU for faster interaction and pruning experiments

Install the main dependencies with:

pip install torch transformers datasets matplotlib seaborn numpy scipy tqdm
Usage

The repository contains scripts for information-theoretic attention-head interaction analysis and performance-based structured pruning.

Run the main analysis scripts with:

python gpt_bert_analysis.py
python llama_analysis.py

Additional pruning and baseline scripts can be run independently to reproduce the structured-pruning experiments and comparison methods described in the manuscript.

Depending on the experiment, pretrained model checkpoints and datasets are downloaded automatically through Hugging Face.

GPU execution is recommended for the pruning experiments, although the scripts can also be adapted for CPU execution.

📁 Repository structure

The main repository components include:

.
├── gpt_bert_analysis.py
├── llama_analysis.py
├── task_similarity.py
├── pruning/
├── baselines/
├── figures/
├── requirements.txt
├── LICENSE
└── README.md

The exact structure may vary as additional experiment and baseline scripts are added.

📖 Citation

If you use this code or build upon the framework in your research, please cite the accompanying manuscript:

@article{Bouchaffra2026Coalition,
  title   = {A Coalition-Based Game-Theoretic Framework for Higher-Order Attention Interactions and Structured Head Pruning},
  author  = {Bouchaffra, Djamel},
  journal = {Pattern Recognition},
  year    = {2026},
  note    = {Submitted}
}
📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

🤝 Contact

For questions, feedback, or issues related to the implementation, please open a GitHub issue or contact:

Djamel Bouchaffra
djamel.bouchaffra@uvsq.fr
