# Architecture

## Goal

`ts_trans` is a small educational time-series transformer project. The purpose is to learn the architecture by building the important pieces explicitly and inspecting their mathematics, tensor shapes, inputs, and outputs.

The model should remain intentionally small. The user will build most transformer/model pieces directly, usually from short snippets or guided steps, rather than asking an agent to generate the whole model.

AI assistance is especially appropriate for:

- constructing controlled test inputs;
- running focused tests;
- checking tensor shapes and dimensions;
- inspecting intermediate outputs;
- checking gradients and parameter counts;
- writing small test helpers or wrappers;
- diagnosing errors;
- verifying CPU/GPU parity where useful;
- committing and pushing completed, verified units of work.

AI should not replace the learning exercise by implementing the entire transformer stack unless explicitly requested.

## Initial forecasting problem

Begin with a univariate next-step forecasting problem.

Given a sequence

`[x_1, ..., x_N]`

predict

`x_(N+1)`.

A batch of scalar sequences can be represented as shape:

`(B, N, 1)`

where:

- `B` is batch size;
- `N` is sequence length;
- the final dimension is one because the raw series is univariate.

## Initial model shape

Use a deliberately small model so every operation remains inspectable.

Suggested starting values:

- model dimension `d_model = 16` or `32`;
- one attention head initially;
- one transformer block initially;
- feed-forward dimension `d_ff = 2 * d_model` or similarly small;
- scalar one-step output.

These are starting choices for learning, not fixed requirements.

The conceptual flow is:

`scalar sequence -> input representation -> positional/time information -> attention -> residual -> feed-forward network -> residual -> forecast head -> scalar forecast`

Each component should be introduced separately and tested before adding the next component.

## Implementation principle

Prefer explicit implementations over high-level transformer convenience classes while learning the mechanics.

For example, early attention code should make the following objects visible where practical:

- input representation `X`;
- query `Q`;
- key `K`;
- value `V`;
- attention scores `S`;
- normalized attention weights `A`;
- attention output.

Likewise, feed-forward code should make its affine maps, activation, dimensions, and residual connection clear.

Once the mechanics are understood, higher-level PyTorch components may be compared against the explicit implementation.

## Testing philosophy

Tests are part of the learning process rather than merely regression guards.

For each new component, prefer small deterministic examples that can verify properties such as:

- expected input and output shapes;
- finite outputs;
- row sums of softmax attention weights;
- causal masking when introduced;
- parameter dimensions and counts;
- residual shape compatibility;
- gradients reaching learned parameters;
- reproducibility under a fixed seed;
- CPU/GPU consistency within numerical tolerance.

Use AI help freely to create these inputs, run the checks, and inspect the resulting tensors. The model implementation itself should remain user-directed.

## Hardware target

Development machine:

- 32 GB system RAM;
- NVIDIA GeForce RTX 5060 Ti with 16 GB VRAM;
- CUDA-capable GPU available for PyTorch.

This hardware is far beyond what the initial educational model requires. The first implementations should optimize for transparency rather than GPU utilization. GPU acceleration can be introduced once the CPU implementation is understood and tested.

The architecture should avoid assumptions that require large-memory hardware; small examples and tests should remain practical on CPU.

## Scope

Do not build a generalized forecasting framework at the outset. Add capabilities only when they support a specific learning goal.

Possible later additions include multiple heads, multiple blocks, richer embeddings, calendar features, multivariate inputs, larger datasets, training utilities, diagnostic plots, or Dash interfaces, but none are part of the initial architecture unless explicitly requested.
