# Hebbian Linear Models

Hebbian linear models are a minimal yet extremely effective linear architecture for general modeling.

It's similar to other linear architectures like  Mamba and Gated Delta Net, but is much simpler - each layer is just a convolution, MLP, and memory block. The memory block which is an associative matrix - a fixed cost soft KV cache.

There are a few very small but critical changes:
1. The convolution + MLP is the main processor - the memory is additive with a residual skip.
2. A token shift during memory write allows removing the QK projections.
3. The memory matrix is not split into heads to effectively increase storage space.

Hebbian linear versus baseline mamba shows a modest improvement on pg19 prose (-0.1 loss), but a massive improvement (-0.9 loss) on the_stack code due to code's natural associative structure in variable assignments.

# Attention Basic Review (feel free to skip)
The attention mechanism cost in the transformer is quadratic - longer sequences are much more costly, due to the QKV structure.

In a simplified single layer text example with word tokens, we need to predict the blank:
The dog barked. The man saw the dog ____?

To predict what comes after 'dog', the model creates a Query vector (q). Every previous word has a Key vector (k) associated. The dot product of each Q and K tells us how related the two words are.

For example, 'dog' is likely more related to the previous 'dog', 'barked' words, and less so to 'the' and 'saw'.

The Value vector (v) contains the contexualized meaning of the word - bark is an dog's action, not the bark of a tree for example.

q*k tells us the relation strength, which we use to scale the influence of a previous word's v for our current prediction. The attention mechanism gives us the appropriate context for prediction.

The main problem is with really long text, you need to compute Q*K for every word, and hence need to cache the KV values for every previous word to avoid recomputing them.

This is really expensive, so people have tried many things like combining different attention types like sliding window (keep only last X tokens), strided(keep every Y tokens), with global to keep the cost down.

A different research direction is linear models, which instead have a fixed cost regardless of sequence length. 

# Linear Models Basic Review (feel free to skip)
Many different linear architectures exist (Mamba 1/2/3, Gated Delta Net, Kimi Linear, etc.), the simplified problem is that the infinite series of Key and Value vectors of attention must be compressed into a fixed size matrix, let's just call M for memory.

The original linear attention publishes the core insight:
For each k,v vectors, take the outer product v⊗k or (v · k^T).
Multiplying v⊗k by k again, we get v · (k^T @ k) = v · ‖k‖². You get v back.
Essentially, storing kv together allows retrieving v with k. 

After adding v⊗k to M, multiplying M by k retrieves v. M is your KV cache.
However, M is fixed size, so continually adding v⊗k start to overlap. Instead of a clean v retrieval, you get a weighted combination of all v's in M, which could be useful.

Every new token, we multiple M by decay (γ) so old keys fade and "make room" for new keys.

# Linear Models Issues
Fundementally, compared to full attention which retains all kv vectors crisply, linear attention compression can only approximate. The cost of faster speed is worse recall, which many hybrids like Jamba, Olmo Hybrid, Kimi Linear mitigate by using linear and full attention layers in a 3:1 ratio for example, to try and reduce the cost.


# Convolutions are mostly what you need
Mambaout showed for vision models, the SSM block doesn't contribute much and simple convolutions are most of the contribution.
It makes sense for language also. If you stack 6 convolution + MLP layers and each conv has a width of 4 tokens, you actually have a 4 + (4-1) * 5 = 19 nonlinear token window

If layer 1 has information on token t and past three tokens, layer 2 mixes t and past 6 tokens
Layer 1: [t to t-3]
Layer 2: [t to t-3, t-1 to t-4, t-2 to t-5, t-3 to t-6]
...

Simple convolutions work well because a lot language modeling is immediate local context, influenced by a few keywords in past context. Suspecting attention is overkill because perfect recall for filler words likely is clogging memory - tricks like attention sinks are needed to actively forget irrelelvant words.

However, beyond immediate local context, you still need a mechanism for long term recall, which this model adds with the hebbian matrix.

# Hebbian Linear Token Shift
Originally, the hebbian memory matrix augmented Mamba to mitigate recall issues, stacking layers of MambaBlock -> HebbianBlock.

Critically, this matrix uses a new token shift Opus 4.5 seredipitously discovered, which reduces parameter cost by 12.5% to 25% per layer.

Linear architectures all create the q/k/v vectors via projections for the memory matrix. However, during prediction, we're storing v⊗k, which is "symmetric". We're storing the key for the new predicted token with the "thinking state" together, which isn't very useful.

Going back to the example:
The dog barked. The man saw the dog ____?

A symmetric memory matrix contains the key value pairs of:
"dog": "thinking about dog"
"barked": "thinking about barking"

We need to predict "bark", so query vector learns to transform into something like "barked", retrieving the "bark thinking state" to predict "bark". A big responsibility of Q/K projections are spent breaking the symmetry, although they're more expressive as well.

The Hebbian token shift breaks the symmetry by uses the previous word directly as the key. The key value pairs look like:

"the": "thinking about dog"
"dog": "thinking about barking"

To know what I should be thinking about now, just use the previous token as a primer.
Predicting the final word, we use the previous token "dog" as the key, which last time was followed by the "thinking about barking" state.

This skips the Q and K projections which is 12.5% of the layer parameters at MLP expand factor of 4, or 25% at expand factor of 2.

I believe only Mamba 3 uses trapazoidal keys, and RWKV-7's lerp uses weighted averages but both still keep a form of the QK


## Results
A full-rank $d \times d$ associative memory matrix augmenting each layer of a language model. The memory accumulates outer-product associations over the context at $O(Td^2)$ cost — linear in sequence length — and injects retrieved content into the residual stream via a skip connection.

The architecture pairs a gated causal convolution (local token mixing) with the Hebbian memory (long-range associative binding) at every layer. No attention, no SSM — just conv + memory.


Hebbian vs Mamba baseline (parameter-matched, same data, same training):

| Setting | Hebbian | Mamba | Gap |
|---|---|---|---|
| 18M, The Stack (10M tokens) | **1.73** | 2.66 | **+0.93** |
| 18M, PG-19 (10M tokens) | **3.38** | 3.50 | **+0.12** |

The gap widens over training — Hebbian improves faster than Mamba at every checkpoint. Code (The Stack) benefits more than prose (PG-19), likely because code has more repeated key-value patterns (variable names, function signatures, imports).

## Replication

### Setup

```bash
uv sync
```

Data is cached automatically on first run from HuggingFace.

### Synthetic recall sanity check

Verifies that $W$ achieves near-perfect associative recall before training:

```bash
uv run eval_recall/eval_recall.py
```

### 18M prose (PG-19)

```bash
# Memory model
uv run train.py --dataset pg19

# Wide baseline (~17.4M, d=576)
uv run train.py --dataset pg19 --no-memory --d-model 576

# Deep baseline (~17.2M, 10 layers)
uv run train.py --dataset pg19 --no-memory --n-layers 10
```

### 18M code (codeparrot)

```bash
uv run train.py --dataset code
uv run train.py --dataset code --no-memory --d-model 576
uv run train.py --dataset code --no-memory --n-layers 10
```

### 100M code (codeparrot)

```bash
uv run train_100M.py --dataset code
uv run train_100M.py --dataset code --no-memory
```

### 100M multilingual (The Stack)

```bash
uv run train_stack100M.py
uv run train_stack100M.py --no-memory
```

### Evaluation

**W ablation** (updating vs frozen $W$, confirms gains come from $W$ not extra params):

```bash
uv run eval_memory/eval_memory.py --model checkpoints/model_code_memory.pt --dataset code --tokens 16384 --windows 8
```

**Long-context eval** (per-segment loss up to 16K tokens):

```bash
# Run from repo root
uv run eval_long/eval_long.py \
  --models checkpoints/model_code_memory.pt checkpoints/model_code_deep.pt \
  --dataset code --tokens 16384 --windows 8 \
  --out eval_long/my_eval.txt
```

**Compile paper:**

```bash
/usr/local/texlive/2025/bin/universal-darwin/pdflatex \
  -output-directory=paper paper/paper.tex
```

## Architecture

Each Mamba layer is augmented with:

```
W_t = γ · W_{t-1} + proj_write(r_{t-1}) ⊗ proj_write(r_{t-1})ᵀ   # write
read_t = W_t · proj_read(r_t)                                        # read
r_t ← r_t + α · proj_read(read_t)                                   # inject
```

where `r_t` is the post-Mamba residual, `γ = σ(decay)` is a learned per-layer scalar, and `α = 0.03` is fixed. Two key design choices: (1) separate write/read projections — necessary for correct retrieval after Mamba state resets; (2) fixed `α` — prevents memory noise from destabilizing early training.

## Experiments (March 2026)

Explored the delta rule, alpha tuning, component ablations, and layer patterns. All at 18M params, 1221 steps on The Stack (code).

### Results

| Model | Val Loss | Step time | Architecture |
|---|---|---|---|
| Hebbian + delta deep, alpha=0.2 | **1.711** | 2992ms | 6 conv+hebb + 2 conv+delta (layers 6,7) |
| Hebbian, alpha=0.2 | 1.726 | ~2800ms | 8 conv+hebb |
| Hebbian, alpha=0.1 | 1.744 | ~2800ms | 8 conv+hebb |
| Hebbian, alpha=0.03 (original) | 1.779 | ~2800ms | 8 conv+hebb |
| Conv + 2 hebb + 1 delta | 1.779 | 2015ms | 7 conv + 2 hebb + 1 delta |
| Conv only (10 layers) | 1.874 | 1887ms | 10 conv+MLP |
| Conv + 2 delta (no hebb) | 1.980 | 1756ms | 8 conv + 2 delta |
| All-delta (simplified) | ~2.05 | — | 8 conv+delta |
| Mamba | 2.45 | 38017ms | 10 Mamba layers |
| Memory only (no conv) | 2.573 | 2489ms | 8 MLP+hebb |
| All-delta (full GDN-style) | 2.77 | — | 8 conv+delta (input-dep decay, beta) |

### What works

- **Alpha=0.2 is the biggest win.** Going from 0.03 to 0.2 gives +0.053 nats — more than any architectural change. At 0.03, memory contributes too little signal for gradients to increase it. Learned alpha stays near its init (confirmed on 100M FineWeb run at 19K steps: alpha settled at 0.01–0.07 per layer).
- **Sparse delta at deep layers.** 2 delta layers at positions 6,7 on top of plain Hebbian gives +0.015 nats. Delta works best on abstract representations where key normalization loses less information.
- **Memory at every layer.** Removing memory from first 4 layers costs ~0.15 nats. Early layers build associations that later layers refine — conv+MLP can't substitute.
- **Conv+MLP is the backbone.** 10 conv+MLP layers get 1.874 — ~70% of the way to full Hebbian. The memory adds long-range associations on top.

### What doesn't work

- **Delta at every layer.** WY forward-substitution + forced key normalization hurts optimization. The normalized key space compresses token information that raw keys preserve.
- **Delta without Hebbian priming.** Delta layers at the end need earlier Hebbian layers to populate the memory. Conv-only → delta gives worse results than conv-only → more conv.
- **Alpha ≥ 0.3.** Destabilizes training — memory overwhelms the residual stream before useful representations form.
- **Negative eigenvalues (beta [0,2]).** NaN with normalized keys and WY correction.
- **Memory in residual stream (no skip).** Activations explode layer-over-layer without the skip connection to anchor magnitudes. GDN solves this with output gate + RMSNorm; plain Hebbian uses the skip connection instead.

### Architecture insights

**Plain Hebbian's simplicity is its strength.** No WY correction, no key normalization, raw hidden states as keys (magnitude carries information), just batched matmuls in a short chunk loop. No custom kernel needed at any scale — standard cuBLAS handles everything. The chunkwise parallel form (32 iterations for T=2048, C=64) is 30–40x faster than sequential recurrence on MPS.

**The skip connection replaces GDN's output gate.** GDN stabilizes memory reads with `norm(o) * silu(gate)` — a learned valve. Plain Hebbian stabilizes with a skip connection — simpler, fewer params, same effect.

**Delta rule's niche: precise recall at deep layers.** The error correction (`v - W·k`) helps when the model needs exact associations (variable names, function signatures). But it requires normalized keys for numerical stability, which limits its use to layers where representations are abstract enough that normalization doesn't lose important information.

**Conv handles local, memory handles global.** Conv (d_conv=4) captures syntax and local patterns. Memory (D×D matrix) captures cross-file associations. Neither can substitute for the other — removing conv costs more than removing memory.

## Checkpoints

| File | Params | Dataset |
|---|---|---|
| `checkpoints/model_mem1.pt` | 18M | PG-19 prose |
| `checkpoints/model_code_memory.pt` | 18M | codeparrot |
| `checkpoints/model_code_deep.pt` | 17.2M | codeparrot (baseline) |
| `checkpoints/model_code100M_memory.pt` | 105.7M | codeparrot |
| `checkpoints/model_code100M_deep.pt` | 107.2M | codeparrot (baseline) |
| `checkpoints/model_stack100M_memory.pt` | 97.5M | The Stack |
| `checkpoints/model_stack100M_deep.pt` | 101.1M | The Stack (baseline) |
