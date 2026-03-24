# Sleep Consolidation: Memory → MLP Transfer

The memory matrix has finite capacity. Common patterns should be absorbed into MLP weights, freeing memory for novel associations.

## Procedure

1. **Generate**: produce tokens with memory active (higher quality than MLP alone)
2. **Consolidate**: disable memory, train full model on generated tokens (cross-entropy)
3. **Resume**: re-enable memory, continue normal training on real data

The MLP fails on tokens where it relied on memory. Gradients push it to internalize those patterns. Memory disabled during consolidation is essential — otherwise the MLP "cheats" by reading from memory instead of learning.

Self-generated tokens bias toward common patterns, naturally prioritizing frequent associations. No external data needed.

Analogous to RL: memory-augmented model is the reward model, MLP-only is the policy. The rollout is self-generation. The update is training without memory.
