# Mini LLM Inference Engine Project Plan

## Project goal

Build a lightweight large language model inference engine from scratch to understand the core mechanisms of inference and serving. Start with single-request, single-GPU inference, then incrementally explore scheduling, hybrid architectures, mixture-of-experts, speculative decoding, and multi-GPU execution.

The project prioritizes correctness, a minimal first implementation, and incremental development. Model computations are implemented independently; Hugging Face Transformers is used for checkpoint and tokenizer interoperability and reference validation, not for the engine's forward pass. See [AGENTS.md](AGENTS.md) for development rules.

## Current status

As of October 31, 2026, **M0 (environment and reference baseline) is complete**; **Phase 0 is the active milestone** and M1 is the next step:

- [README.md](README.md) describes the project direction, Phase 0 scope, setup, the M0 validation record, and roadmap; [AGENTS.md](AGENTS.md) defines development and correctness requirements.
- M0 environment: Python 3.14.6 venv (`.venv/`, gitignored) with pinned `torch==2.15.0+cu132`, `transformers==5.22.1`, `tokenizers==0.23.3`, `safetensors==0.8.0`, `numpy==2.5.4`, `pytest==9.2.0` ([requirements.txt](requirements.txt)).
- M0 baseline: `Qwen/Qwen3-0.6B` revision `main` @ `c1899de289a04d12100db370d81485cdf75e47ca`, BF16, `sdpa`, greedy, prompt `"The capital of France is"`, `max_new_tokens=32`, run on an NVIDIA GeForce RTX 4060 Laptop GPU (CUDA device 1 on this machine). Reference logits and greedy token IDs (cached `generate` path plus a manual uncached path) are saved to the gitignored `reference/` directory by [scripts/make_reference_baseline.py](scripts/make_reference_baseline.py); two runs were byte-identical. Recorded finding: the cached and uncached bf16 paths first diverge at decode step 5 (exact top-2 logit tie on the cached path); max/mean logit abs diff before divergence is 0.2188 / 0.0291.
- Tests: 7/7 pass (`.venv/bin/python -m pytest tests/`), validating the saved baseline artifacts on CPU.
- There is still no inference engine, model loading, KV cache, or generation code in this repository.

The next step is M1 (basic layers: RMSNorm, RoPE, SwiGLU MLP) with deterministic reference comparisons.

## Phase 0: Minimal inference engine

**Goal:** Build the smallest correct, functional single-request, single-GPU BF16 inference engine.

Core work:

- Load a Hugging Face checkpoint and tokenizer.
- Independently implement the Transformer forward pass, including RMSNorm, RoPE, grouped-query attention (GQA), MLP, and LM head.
- Implement causal prefill, autoregressive decoding, a contiguous KV cache, and greedy sampling.
- Compare logits and greedy outputs against the matching Hugging Face model and configuration.
- Use deterministic tests for causal masking, positions, cache bounds, and GQA, including multi-token prefill followed by decode.

Acceptance criteria:

1. Successfully load and run one explicitly identified small Qwen dense model.
2. Meet a documented numerical tolerance against reference logits, reporting maximum and mean errors.
3. Match reference greedy-decoding tokens.
4. Verify cached versus uncached logits and tokens.
5. Pass deterministic unit and integration tests and record the model revision, dtype, input, and test environment.

Continuous batching, paged attention, prefix caching, chunked prefill, custom CUDA/Triton kernels, quantization, speculative decoding, MoE, Gated DeltaNet, tensor parallelism, and HTTP serving are outside this milestone.

### Phase 0 milestones

Each milestone has a focused acceptance test and can be committed independently. Use `Qwen/Qwen3-0.6B` as the target; pin the model revision and compatible dependency versions when establishing the baseline. Record tolerances and report errors rather than assuming bitwise equality across hardware or backends.

- [x] **M0 — Environment and reference baseline.** Set up the environment and run the Hugging Face reference on one BF16-capable GPU. Fix the model and tokenizer revision, prompt or token IDs, generation settings, dtype, attention backend, and software and hardware versions. Save reference logits and greedy token IDs locally, with reproducible commands. **Acceptance:** Reference text generation succeeds on the GPU and the baseline can be reproduced. Do not commit downloaded weights or generated artifacts.
- [ ] **M1 — Basic layers.** Implement RMSNorm, RoPE, and the SwiGLU MLP as independent components. **Acceptance:** Deterministic inputs produce outputs close to matching reference components under documented tolerances; test positions and relevant configuration details.
- [ ] **M2 — GQA attention.** Implement Q/K/V projections, Q/K normalization as required by Qwen3, RoPE, grouped-query attention, causal masking, and output projection using standard PyTorch operations. **Acceptance:** Compare against the corresponding reference attention layer with matching weights and configuration; include deterministic GQA, causal masking, and position tests.
- [ ] **M3 — Transformer model.** Assemble decoder layers, embeddings, final RMSNorm, and LM head using the reference configuration and an independently implemented forward pass. For this milestone, test-only copying of reference weights into the custom model is acceptable; the production checkpoint loader belongs to M4. **Acceptance:** Full-model logits for fixed input are close to the reference, with maximum and mean errors and tolerances recorded.
- [ ] **M4 — Weight loader.** Load the real checkpoint from safetensors into the custom model, checking names, shapes, dtype handling, weight sharing where applicable, and complete expected parameter coverage. **Acceptance:** The actual checkpoint loads without silently missing or ignoring unexpected weights, and the independently loaded model passes the M3 reference-logit comparison. Explicitly document any allowed exceptions.
- [ ] **M5 — Uncached greedy generation.** Implement single-request prefill and token-by-token generation by recomputing the full sequence, with explicit stopping and context limits. **Acceptance:** Greedy token IDs match the reference for fixed inputs and generation settings.
- [ ] **M6 — Contiguous KV cache.** Store each attention layer's K/V for prefill and decode only new tokens. A `torch.cat` implementation may be an intermediate step; finish this milestone with a bounded, preallocated contiguous cache. **Acceptance:** Cached and uncached logits are close and greedy tokens match for multi-token prefill and subsequent decode; test positions, causal masking, and cache bounds, including overflow rejection.
- [ ] **M7 — Correctness suite and baseline benchmark.** Expand integration coverage across prompts, lengths, and boundaries, and measure prefill latency, decode tokens per second, and peak GPU memory. **Acceptance:** All required correctness checks pass and a reproducible benchmark records the model revision, environment, workload, timing method, and results. Keep generated benchmark artifacts out of Git; document how to reproduce them.

Tests and documentation are added with each milestone, not deferred until M7. GPU or reference validation is reported as pending if it cannot actually be run.

## Roadmap

The roadmap covers Phase 0 through Phase 6 and a possible multimodal extension. **Only Phase 0 is currently in scope for implementation**; the later phases are planned, not implemented. Verify specific architecture and model names, checkpoint formats, and reference implementations when each phase begins.

### Phase 1: KV cache and basic serving

Add KV cache allocation and lifecycle management, paged KV cache and block tables, request state, basic continuous batching, waiting/running/finished queues, and a minimal HTTP inference API.

### Phase 2: Scheduling and cache optimization

Add chunked prefill, token-budget scheduling, cache eviction and request preemption, prefix caching and shared KV blocks, CUDA Graphs, and benchmark-driven kernel optimization. Compare scheduling policies across latency, throughput, and GPU memory use.

### Phase 3: Qwen hybrid architecture support

Explore the target Qwen3.8-27B hybrid architecture, including Gated DeltaNet recurrent state, full and linear attention execution, and unified KV cache and recurrent state lifecycle management. Validate BF16 first, optionally explore INT8/INT4, and compare against the official reference implementation. Verify the model name and architecture before implementation.

### Phase 4: Mixture-of-experts

Implement expert routing and gating, top-K selection, expert dispatch and aggregation, MoE checkpoint loading and execution, and numerical validation against a reference MoE model.

### Phase 5: Speculative decoding

Implement draft and target model execution, token verification and rejection, KV cache rollback and state consistency. Explore multi-token prediction (MTP) and DFlash/DFlash2-style block drafting, and measure acceptance rate and latency. Stochastic speculative decoding must preserve the required sampling distribution.

### Phase 6: Tensor parallelism

Implement distributed weight partitioning, column- and row-parallel linear layers, NCCL communication, and attention and MLP tensor parallelism. Extend multi-GPU execution to hybrid and MoE models and measure communication overhead and scaling.

### Future extension: Multimodal inference

Potential work includes vision encoders, image/audio/video processing, multimodal input representations, attention and position embeddings, and Omni model inference and serving. This is explicitly outside the initial scope.

## Testing and benchmarking

Correctness checks grow with the engine: reference logit errors, greedy token consistency, cached versus uncached execution, and later batching, speculative decoding, and single- versus multi-GPU consistency.

Performance measurements may include time to first token (TTFT), time per output token (TPOT), decode and aggregate throughput, GPU memory use, KV cache utilization, prefix cache hit rate, and prefill and decode latency. Comparisons with Hugging Face Transformers, nano-vLLM, vLLM, and SGLang should use controlled, reproducible workloads and environments. No performance claims should be made without measurements.
