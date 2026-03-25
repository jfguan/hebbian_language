# Sleep Consolidation: Memory Decay Distillation

Transfer knowledge from the delta Hebbian memory matrix M into the MLP, without using training data.

## Idea

1. **Freeze writes** to M
2. **Generate tokens** from the model itself
3. **Decay M** gradually toward zero
4. **Train the MLP** to produce what M was reading out — the memory read `r = gate * (M @ rk)` is the supervisory signal
5. When M hits zero, distillation is complete — the MLP has absorbed M's associations

No training data needed. The model's own generations provide the input distribution. The memory tells the MLP what to output, then fades away.

## Why it works

M is a linear associative memory (key → value via matrix multiply). The MLP is a nonlinear associative memory with higher capacity. Absorbing a linear map is trivial for the MLP.

As the MLP shifts its output to absorb `r`, the hidden states change, which changes the memory reads. But since M is decaying, the target shrinks to zero regardless — the loop is self-correcting.

## Key knob

**Decay rate.** Too fast and the MLP can't absorb without disrupting what it already does. Too slow and you waste compute. Could be made adaptive: slow down when loss spikes, speed up when stable.

## Continual learning cycle

1. **Wake**: model sees new data, M absorbs associations quickly
2. **Sleep**: freeze writes, decay M, MLP absorbs via self-generation
3. **Clear M**, ready for new learning
