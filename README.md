# LLM Implementation and Fine-tuning

This project covers three core aspects of Large Language Models: **Tokenization**, **LLM Implementation** (GPT-2), and **LoRA Fine-tuning**. It provides hands-on implementations from building a BPE tokenizer from scratch to pretraining a GPT-2 model and performing parameter-efficient fine-tuning with LoRA.

## Project Structure

```
├── hw1.pdf                        # Full report (Chinese)
├── README.md
└── hw1-code/
    ├── bpe/
    │   ├── tokenizer.ipynb        # BPE tokenizer implementation
    │   └── manual.txt             # Chinese text corpus for training
    ├── pretrain/
    │   ├── pretrain.py            # GPT-2 model & training pipeline
    │   ├── fineweb.py             # FineWeb-Edu dataset downloader
    │   ├── hellaswag.py           # HellaSwag benchmark evaluator
    │   ├── input.txt              # Training data
    │   └── play.ipynb             # Interactive demo
    └── Lora (1).ipynb             # LoRA fine-tuning experiments
```

## Part 1: Tokenization (BPE)

A Byte-Pair Encoding (BPE) tokenizer is implemented from scratch. The algorithm starts with a vocabulary of 256 single-byte entries and iteratively merges the most frequent adjacent byte pairs until reaching the target vocabulary size (1024).

### Implementation

The `Tokenizer` class (`hw1-code/bpe/tokenizer.ipynb`) provides:

- **`train()`** — Iteratively finds the most frequent byte pair via `get_stats()`, merges it into a new token via `merge()`, and records the merge rule. After training, the `vocab` dictionary maps token IDs to byte sequences.
- **`encode(text)`** — Converts text to UTF-8 bytes, then applies learned merge rules in priority order (earliest merges first).
- **`decode(ids)`** — Maps token IDs back to byte sequences via `vocab` and decodes to UTF-8 text.

### Key Findings

- **Round-trip fidelity**: Encoding then decoding `manual.txt` reproduces the original content exactly.
- **Chinese vs. English efficiency**: The custom tokenizer (trained on Chinese text) encodes Chinese more compactly than GPT-2's tokenizer (158 vs. 248 tokens), but is less efficient on English (645 vs. 239 tokens) due to fewer English merge rules in the training corpus.
- **Why tokenization matters for LLMs**: Tokenizer design directly affects model capabilities — it explains why LLMs struggle with character-level tasks (e.g., string reversal), simple arithmetic, non-English languages, and Python indentation handling.

## Part 2: LLM Implementation (GPT-2)

A full GPT-2 model is implemented from scratch in PyTorch (`hw1-code/pretrain/pretrain.py`), following the standard Transformer decoder architecture.

### Architecture

| Component | Details |
|---|---|
| Layers | 12 Transformer blocks |
| Attention heads | 12 per block |
| Embedding dim | 768 |
| Context length | 1024 tokens |
| Vocab size | 50304 (padded from 50257 for efficiency) |

Each Transformer block follows: **LayerNorm → Multi-Head Causal Self-Attention → Residual → LayerNorm → MLP → Residual**

- **CausalSelfAttention** — Computes Q, K, V projections, applies scaled dot-product attention with a causal mask, and uses Flash Attention for efficiency.
- **MLP** — Two linear layers with GELU activation (`fc → GELU → proj`).
- **Final LayerNorm** (`ln_f`) is applied before the output linear head.

### Training Pipeline

The training pipeline includes several optimization techniques:

- **Mixed precision (FP16)** — Reduces memory usage and leverages Tensor Cores for faster matrix operations. A `GradScaler` ensures numerical stability.
- **`torch.compile()`** — Eliminates Python dispatch overhead and fuses kernels for faster execution.
- **Flash Attention** — Kernel-level optimization for attention computation, reducing memory from O(n^2) to O(n) and improving speed.
- **Gradient accumulation** — Simulates large batch training with limited GPU memory by accumulating gradients across multiple forward-backward passes before updating.
- **Learning rate schedule** — Linear warmup followed by cosine decay.
- **Checkpoint & resume** — Periodically saves model weights, optimizer state, and scheduler state for fault-tolerant training.
- **Data parallelism** — Multi-GPU training via `DistributedDataParallel` with gradient synchronization across devices.

### Evaluation

The model is evaluated on the **HellaSwag** benchmark for commonsense reasoning.

## Part 3: LoRA Fine-tuning

Parameter-efficient fine-tuning is performed using **Low-Rank Adaptation (LoRA)** (`hw1-code/Lora (1).ipynb`). Experiments are conducted on two datasets:

1. **Movie review dataset** — Sentiment-oriented text generation.
2. **Alpaca dataset** — Instruction-following tasks (prompt = instruction + input, target = output; loss is computed only on the output portion).

### Hyperparameter Study

Starting from baseline settings, systematic experiments were conducted to find optimal hyperparameters:

| Parameter | Values Tested | Finding |
|---|---|---|
| **Learning rate** | 1e-3, 1e-4 | 1e-4 is optimal; 1e-3 causes loss to diverge |
| **LoRA rank (r)** | 32, 64 | Moderate impact; higher r helps on more complex tasks (e.g., instruction-following) |
| **lora_alpha** | 32, 64, 128 | Minimal impact when scaled proportionally with r (maintaining alpha/r ratio) |
| **target_modules** | `["query_key_value"]`, `["query_key_value", "dense"]` | Including `dense` layers improves performance noticeably |
| **lora_dropout** | 0.05, 0.1 | Slight improvement at 0.1 |
| **bias** | `"lora_only"`, `"none"` | Negligible difference (model has few bias-containing layers) |

### Results Summary

| Configuration | Trainable Params | % of Total | Notes |
|---|---|---|---|
| Baseline (r=32, attention only) | 466,944 | 0.08% | Loss diverges with high lr; incoherent output |
| Optimized (r=32, lr=1e-4) | 466,944 | 0.08% | Stable training, coherent movie reviews |
| r=64 + dense layers | 9,535,488 | 1.68% | Best performance on instruction-following |

### Key Takeaways

- **Learning rate is the most critical parameter** — even more impactful than LoRA-specific hyperparameters.
- **Including dense layers** in `target_modules` consistently improves results across both tasks.
- **Higher rank (r)** benefits more complex tasks (instruction-following) more than simpler tasks (sentiment).
- **LoRA enables effective fine-tuning with 0.08%–1.68% trainable parameters**, making it practical for resource-constrained settings.

## Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers & Datasets
- PEFT (for LoRA)
- tiktoken (for GPT-2 tokenizer comparison)

## License

See [hw1-code/pretrain/LICENSE](hw1-code/pretrain/LICENSE).
