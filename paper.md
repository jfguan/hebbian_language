# Paper Plan: Hebbian Memory for Mamba

## Story
Simple 32-line mechanism → works on synthetic recall → beats param-matched baselines → scales with context length → scales with training compute → generalizes to code → scales with model size.

## Experiments

### 1. Synthetic KV Recall [DONE]
- eval_recall.py: 100% vs chance (1/16)
- Proves W carries associations across Mamba state resets
- Baseline: no memory, same architecture

### 1b. Memory Capacity Curve [DONE]
- Train on 32 KV pairs (d_model=128, 4 layers), eval at [4, 8, 16, 32, 64, 128, 256]
- 256 unique keys, 16 unique values, Mamba state reset between store and query phases
- Results (recall accuracy):

| Pairs | Accuracy | vs chance (6.2%) |
|-------|----------|------------------|
| 4     | 100.0%   | 16× chance        |
| 8     | 100.0%   | 16× chance        |
| 16    | 100.0%   | 16× chance        |
| 32    | 99.8%    | 16× chance        |
| 64    | 96.0%    | 15× chance        |
| 128   | 55.7%    | 9× chance         |
| 256   | 18.8%    | 3× chance         |

- Perfect recall up to 32 pairs (training distribution), 96% at 64 (2× training count)
- Naive theory predicts SNR ≈ √(d/k) ≈ 1.4 at k=64 — should give ~50-70% with random keys. Getting 96% means learned projections structure keys to reduce interference beyond the random-key bound.
- 4 layers with independent W matrices appear to distribute associations: 18.8% at 256 pairs = ~48 correct, or ~12 per layer — well within single-matrix capacity. A single-layer model should show the cliff much earlier.
- Chart: eval_recall/capacity.png

### 2. Param-Matched Baselines at 18M [DONE]
- Memory (d=512, 8 layers, 18M) vs wide baseline (d=576, 8 layers, 17.4M) vs deep baseline (d=512, 10 layers, 17.2M)
- Memory wins both: 3.074 vs 3.122 (wide) vs 3.162 (deep)
- W frozen control: 3.177 (worse than both baselines, confirming W actively contributes)

### 3. Context Length Scaling [DONE]
- Trained at 2K, evaluated at 4K / 16K / 32K
- Gap grows: 0.048 → 0.085 → 0.095
- W generalizes 16x beyond training length

### 4. Training Compute Scaling [RUNNING]
- 1000 steps (~2M tokens) → 5000 steps (~10M tokens)
- Show gap holds or grows with more training
- Both models trained on identical data

### 5. Code Dataset [TODO]
- Add code corpus (e.g. The Stack subset or similar)
- Train memory + baseline from scratch on code
- Eval: variable bindings are pure associative recall — strongest case for W
- Shows mechanism generalizes beyond novels

### 6. Model Size Scaling: 18M → 100M [TODO]
- d=1024, 12 layers (memory) vs d=1024, ~16 layers (baseline, param-matched)
- Train on Mac Mini (24GB), seq_len=1024 or 2048
- The critical experiment: W capacity scales as d², so benefit should grow
- If gap doubles → scaling law. If flat → W is a curiosity.

## Figures
1. Synthetic recall accuracy (memory vs no-memory)
1b. Memory capacity curve (recall vs number of KV pairs, d=128)
2. Loss comparison table: memory vs wide vs deep baselines
3. Context length scaling plot (gap vs eval length)
4. Training compute curve (gap vs steps)
5. Per-segment loss at 32K (W updating vs W frozen)
6. Code dataset results
7. 100M results

## Key Claims
1. A 32-line addition to Mamba gives it persistent associative memory
2. No special training needed — standard language modeling objective suffices
3. Params better spent on W than on width or depth
4. Benefit grows with context length (constant memory cost, unlike KV-cache)
5. Generalizes across domains (novels, code)
6. Benefit grows with model scale (d² capacity)
