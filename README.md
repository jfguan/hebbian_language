# Linear Hebbian Models

Hebbian linear models are a minimal yet extremely effective linear architecture for general modeling.

It's similar to other linear architectures like  Mamba and Gated Delta Net, but is much simpler - each layer is:

4 token convolution -> MLP -> memory block -> next layer

The memory block which is an associative matrix - a fixed cost soft KV cache similar to Gated Delta Net but uses the full matrix instead of block diagonal with divided heads.

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
Fundementally, compared to full attention which retains all kv vectors crisply, linear attention compression can only approximate. The cost of faster speed is worse recall, which many hybrids like Jamba, Olmo, Kimi Linear mitigate by full attention layers only every few layers.

# Linear Hebbian Token Shift
The hebbian matrix is very similar to the GDN matrix, except it uses a new token shift Opus 4.6 discovered, reducing parameter cost 12.5% - 25% per layer.

Gated Delta Net creates the q/k/v vectors via projections for the memory matrix. However, during prediction, we're storing v⊗k, which is "symmetric". We're storing the key for the new predicted token with the "thinking state" together, which isn't very useful.

In our example:
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

Similar models in literature include Mamba 3 with trapazoidal SSM, and RWKV-7's lerp token lerp.

# Linear Hebbian Residual Skip
In this architecture, memory is not the primary mechanism, but instead an augmentation to the convolution+MLP.

While Gated Delta Net's output which directly feeds into an MLP, the Linear Hebbian matrix adds a residual skip, so the convolution + MLP output can somewhat "ignore" the memory if desired. Furthermore, an alpha scalar controls memory influence "strength", which after training settles from .05 to .10. Despite only having a 10% influence per layer, across layers the nudges from memory influence the final prediction significantly.

# Results
## 18M Scale Testing
Disclaimer - all models are heavily undertrained, and limited windows, so noise variation could be significant. However, the nat gaps seem quite large and are at least indicative.

We train a baselines Mamba and convolution models to compare against Linear Hebbian on a coding dataset, The Stack. Coding datasets help exaggerate recall differences since code is high density variable and function associations.

We see that the convolution + MLP model performs .3 nats worse than Mamba, but Linear Hebbian performs almost ~.9 nats better than Mamba. 

As we see on prose pg19, Linear Hebbian performs only ~.11 nats better, which makes sense, prose needs less recall.

## Convolutions are mostly all you need
The MambaOut paper showed for vision models, the SSM doesn't contribute much and just convolutions can approximate it. Similar results in language, where a big part is predicting from immediate local context, influenced by keywords farther in the past.

Stacking 6 convolution + MLP layers with conv width of 4 tokens, you actually have a 4 + (4-1) * 5 = 19 nonlinear local token window to work with.
If layer 1 has information on token t and past 3 tokens, layer 2 mixes t and past 6 tokens
Layer 1: [t to t-3]
Layer 2: [t to t-3, t-1 to t-4, t-2 to t-5, t-3 to t-6]
...

I suspect attention is overkill - perfect recall for filler words likely is clogging memory that need to be "forgotten" with tricks like attention sinks.

However, a mechanism for long term recall is still needed, which the hebbian matrix fills.

## Sanity Checks
Let's also sanity check the hebbian matrix works. A synthetic task trains a small hebbian model to memorize key/value pairs and retrieve them, which it does. 

After training, we can see the model has a natural memory capacity curve.

We also freeze the memory matrix to see if it's critical - without it, performance drops dramatically.

## 100M Scale Testing
Scaling up to 100M, again we show that hebbian maintains a sigifnicant nat improvement around ~.5 over baseline Mamba, theorized just due to larger state to store memory in.

In addition, we create a new layer type delta hebbian, which is just hebbian with the delta rule. It's very close to Gated Delta Net, and splits the matrix block diagonally with heads for hardware efficiency. Placing a few delta hebbian layers at the end of the model further improves performance, theorizing they play a role similar to full attention layers in hybrid models like Olmo. The delta rule allows erasing values before rewriting, allowing cleaner key writes as well as preserving keys indefinitely for critical long term keywords.

## TODO - train multihead attention and GDN as well

## TODO

## Generalization Testing
To test that the architecture generalizes, we use Karpathy's nano-gpt setup, training a 124M model on fineweb-edu data for 10B tokens. With no tuning we see validation loss is ~.10 worse than the GPT-2.


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
