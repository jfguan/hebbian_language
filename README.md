# Hebbian Memory for Mamba

## Core Idea

Add a persistent memory matrix W ∈ ℝ^{d×d} alongside Mamba's fast recurrent state at each layer. W accumulates associative key-value pairs via Hebbian outer-product updates. At inference, the model improves on a domain simply by reading more of it — no gradient updates required.

**Key finding:** W learns useful associations with standard training — no resets or special training tricks needed. Just add W to Mamba and train normally.

## Why Mamba, Not Transformers

Transformers have lossless KV-cache memory within the context window. A lossy memory matrix cannot compete with exact attention and gets ignored during training. Mamba's state is already lossy, so a complementary slow memory fills a real gap.

## Update Rules (per layer)

Let r_t be the residual-stream vector after Mamba's recurrent update.

**Read:** $\text{read}_t = W \cdot r_{t-1}$

**Write:** $W \leftarrow \sigma(\lambda) \cdot W + \text{proj}_{\text{write}}(r_t) \cdot r_{t-1}^\top$

**Inject:** $r_t \leftarrow r_t + \alpha \cdot \text{proj}_{\text{read}}(\text{read}_t)$

Learned parameters per layer: one scalar λ (decay), two projections, one fixed scalar α = 0.03. The write rule is deliberately simple — one matrix, one decay, one outer product.

## Design Decisions

Three changes from the initial design, each discovered through experiments:

**1. Split read and write keys.** Originally both read and write used the shifted (previous) output as key. This caused 40% recall on a synthetic associative task — the read key came from a different context than the write key after state resets. Fix: write key = `r_{t-1}` (shifted, prevents information leakage), read key = `r_t` (current, enables proper retrieval). Result: 100% recall.

**2. Fixed α = 0.03.** Originally α was a learned gate initialized near zero. Problem: early in training, W is noise, and unscaled memory reads destabilize learning of basic language patterns. Fix: hardcode α = 0.03 to bound noise while preserving gradient flow through proj_read. The model learns to use W through the projections, not through gating.

**3. PG-19 instead of tiny Shakespeare.** Shakespeare is ~350K tokens of repetitive verse — a small Mamba memorizes it in weights, leaving no role for W. Novels have genuine long-range dependencies (characters, settings, plot) that benefit from persistent associative memory.

## Architecture

- 8-layer Mamba (d_model=512, d_state=16, expand=2)
- Hebbian memory W ∈ ℝ^{512×512} at each layer
- BPE tokenizer (vocab_size=512) trained on PG-19 novels
- ~18M parameters total

## Dataset

PG-19 (Project Gutenberg novels) streamed via HuggingFace. ~10M chars train, ~1M chars val. Tokenized and cached as `.npy` files.

## Results

### Resets Are Not Needed

The original hypothesis was that resetting Mamba's state during training would force reliance on W. Experiments show W actually works *better* without resets:

| Training condition | W delta (upd vs frz) | ±stderr |
|---|---|---|
| **Memory, no resets** | **+0.103** | **0.012** |
| Memory, with resets | +0.092 | 0.010 |

W learns useful associations through standard training. No tricks needed — the 3% residual connection provides enough gradient signal for W to become useful on its own.

### W Updating vs W Frozen (within-model test, no resets)

Same model, two eval passes over 4096 val tokens (4 random windows, averaged):

| | Avg Loss | PPL |
|---|---|---|
| **W updating** | **3.074** | **~21.6** |
| W frozen | 3.177 | ~24.0 |

**Overall delta: +0.103 ± 0.012** (8.6 standard errors from zero).

Per-segment breakdown (512-token segments):

| Segment | Delta | ±stderr |
|---------|-------|---------|
| 0-512 | +0.089 | 0.029 |
| 512-1024 | +0.109 | 0.031 |
| 1024-1536 | +0.116 | 0.031 |
| 1536-2048 | +0.118 | 0.047 |
| 2048-2560 | +0.084 | 0.029 |
| 2560-3072 | +0.087 | 0.024 |
| 3072-3584 | +0.110 | 0.037 |
| 3584-4096 | +0.108 | 0.028 |

W helps at every position, including beyond the 2048-token training length.

### Full 2×2 Factorial

All models trained for 1000 steps, batch_size=1, seq_len=2048. No-memory baselines use d_model=576 (~17.4M params) to match the memory model's 18M params. Evaluated on 4 random windows of 4096 val tokens each, no resets.

| | No resets | Resets |
|---|---|---|
| **No memory (d=576)** | 3.122 | 3.123 |
| **Memory (d=512)** | **3.074** | 3.083 |

Key findings:

1. **W beats the param-matched baseline in both conditions.** The no-resets comparison is the cleanest: identical training (full 2048-token sequences), only difference is W vs wider Mamba. Memory wins by 0.048 loss.
2. **No resets is the best config.** The original hypothesis — that resets force reliance on W — was wrong. W learns useful associations through standard training, and resets slightly hurt by limiting Mamba to 256-token training contexts.
3. **Resets don't matter for the baseline.** The no-memory model gets 3.122 with or without resets (3.123), confirming that the reset effect is specific to W.

### Memory vs Param-Matched Baselines (no resets, clean comparison)

All models trained identically on full 2048-token sequences. No tricks, no chunking. Two baselines: wider Mamba (d=576, 8 layers) and deeper Mamba (d=512, 10 layers), both param-matched to the memory model.

| Model | Params | Avg Loss | PPL |
|---|---|---|---|
| **Memory (W updating, d=512, 8 layers)** | **18.0M** | **3.074** | **~21.6** |
| Baseline wide (no memory, d=576, 8 layers) | 17.4M | 3.122 | ~22.7 |
| Baseline deep (no memory, d=512, 10 layers) | 17.2M | 3.162 | ~23.6 |
| Memory (W frozen, d=512, 8 layers) | 18.0M | 3.177 | ~24.0 |

W beats both wider and deeper baselines. The params are better spent on associative memory than on either more width or more depth.

With W frozen, the memory model is *worse* than both baselines (narrower Mamba than wide baseline, fewer layers than deep baseline, and no memory to compensate) — confirming W is actively contributing, not just a parameter artifact.

### Context Length Scaling (trained at 2K, evaluated at 4K–32K)

W's benefit grows with sequence length — the longer the context, the more associations W accumulates, and the bigger the gap over baseline. All models trained on 2048-token sequences.

| Eval length | Memory (W upd) | Baseline (d=576) | Gap | vs training length |
|---|---|---|---|---|
| 4K | 3.074 | 3.122 | 0.048 | 2× |
| 16K | 3.085 | 3.170 | 0.085 | 8× |
| **32K** | **3.091** | **3.186** | **0.095** | **16×** |

The gap nearly doubles from 4K to 32K. W generalizes far beyond training length with no degradation — at 32K tokens (16× training length), every 2048-token segment shows positive delta.

32K within-model test (W updating vs W frozen, 2 windows):

| | Avg Loss | Delta |
|---|---|---|
| **W updating** | **3.091** | |
| W frozen | 3.192 | **+0.101 ± 0.007** |

### Learned Decay Rates

All layers learned decay σ(λ) ≈ 0.986–0.990, meaning W retains ~99% of its content per step. Information persists for hundreds of tokens. No stratification across layers at this scale — likely due to limited capacity (d=512) forcing all layers to maximize retention.

## Scaling Outlook

**Capacity scales as d².** W stores associations in a d×d matrix. With random d-dimensional keys, SNR ≈ √(d/k) where k is the number of live associations. At current scale (d=512), interference is noticeable. At d=4096, interference becomes negligible — the matrix can comfortably hold thousands of associations.

| d_model | W capacity (d²) | ~Live associations (γ=0.99) | SNR |
|---------|-----------------|---------------------------|-----|
| 512 | 262K | ~100 | 2.3 |
| 1024 | 1M | ~100 | 3.2 |
| 4096 | 16M | ~100 | 6.4 |
| 8192 | 67M | ~100 | 9.0 |

With lower decay at larger d (more capacity to spare), live associations could scale to thousands, effectively creating a growing working memory that scales with model width.

**Parameter overhead is ~33% per layer** (two d×d projections). This is equivalent to ~n/3 extra Mamba layers across n layers. The argument: those params buy a qualitatively different capability (associative recall at d capacity) that more Mamba layers cannot provide (each bottlenecked at d_state=16).

**Hierarchical memory** is a natural extension: multiple W matrices with different decay rates, sharing projections. Slow W (γ≈0.999) accumulates long-term patterns; fast W (γ≈0.99) tracks recent context. Both are parallel-scannable linear recurrences.

## Next Steps (priority order)

**1. Scale model size (d=1024, ~100M params).** The core theory: W's capacity scales as d², so its benefit should grow with model width. This is the make-or-break experiment. If the delta grows from 5% to 10%+, we have a scaling law. If it stays flat, W is a curiosity. Runnable overnight on a Mac Mini (24GB).

**2. Sequence length (train at 4K-8K, eval at 16K+).** W's value proposition is persistent memory across long ranges. Need to train on longer sequences so W gets gradient signal from longer-range dependencies. Coupled with model size — small models can't learn 16K-token patterns regardless of memory.

**3. Training data complexity.** PG-19 is novels — relatively uniform text. Code (variable bindings across files), multi-document QA (facts from doc A used in doc B), or dialogue (speaker identity over long conversations) would test whether W helps with different types of associative recall.

**4. Training data amount / steps.** Current model may be undertrained (1000 steps). More training won't change the fundamental scaling question but could widen the gap if W's projections haven't fully converged.

## Risks

1. **Interference.** Rank-1 updates with a single scalar decay may cause catastrophic forgetting as W accumulates. Mitigated by scaling d.
2. **Subsidization.** W may absorb work Mamba could handle alone, weakening the base model. The factorial design measures this directly.
3. **O(d²) per token per layer.** Manageable with low-rank projections or applying memory at every nth layer.

## Discovery Timeline

1. **Shared read/write keys failed.** Both read and write used the shifted (previous) output as key. Synthetic associative recall plateaued at 40%. Fix: read with current output, write with previous. Recall jumped to 100%.

2. **Unscaled memory reads destabilized training.** Early in training W is noise. Full-strength reads injected garbage into the residual stream. Fix: hardcode α = 0.03 to bound noise. Training stabilized.

3. **Tiny Shakespeare was too easy.** W showed no benefit — the text was small and repetitive enough for Mamba to memorize in weights alone. Switched to PG-19 novels.

4. **W benefit is statistically significant on PG-19.** Within-model test (W updating vs W frozen) showed +0.092 ± 0.010 across 4 random windows. Every 512-token segment showed positive delta.

5. **W beats param-matched wider Mamba.** d=576 baseline (17.4M params, no memory) scored 3.122. Memory model (18M params, d=512 + W) scored 3.074. The extra params are better spent on W than on width.

6. **Resets aren't needed.** Expected resets to be essential for training W. Instead, the no-resets model showed a *larger* W delta (+0.103 vs +0.092). Standard training works better — the simplest version of the idea is the one that works.

7. **W beats param-matched deeper Mamba.** 10-layer baseline (17.2M params) scored 3.162 — worse than both the wide baseline and the memory model. W is a better use of parameters than either more width or more depth.

8. **W benefit scales with context length.** Evaluated at 4K, 16K, and 32K tokens (trained at 2K). Gap over baseline grew from 0.048 to 0.095 — nearly doubling. W generalizes 16× beyond training length with no degradation.

## Usage

```bash
# Train with memory
uv run python train.py --steps 1000 --batch-size 1

# Train param-matched baseline
uv run python train.py --steps 1000 --batch-size 1 --no-memory --d-model 576

# Evaluate (4 random windows, 4096 tokens each)
uv run python eval_memory.py --model model_mem1.pt

# Evaluate at longer context (32K tokens, 2 windows)
uv run python eval_memory.py --model model_mem1.pt --tokens 32768 --windows 2 --segment 2048

# Curriculum: pretrain without memory, then introduce W
uv run python train.py --steps 500 --batch-size 1 --no-memory
uv run python train.py --steps 500 --batch-size 1 --resume model_mem0.pt

# Synthetic associative recall test
uv run python eval_recall.py
```
