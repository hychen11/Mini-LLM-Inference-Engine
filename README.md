# Mini LLM Inference Engine

A correctness-first LLM inference engine built from scratch, developed in small, independently testable stages. The intended code style is simple and direct, taking inspiration from nano-vLLM. See [AGENTS.md](AGENTS.md) for development-agent instructions.

See the [project plan](PROJECT_PLAN.md) for the full roadmap and the current repository status.

**Status: M0 (environment and reference baseline) is complete.** The inference engine itself does not exist yet; M1+ are not started.

## Current milestone: Phase 0

Build the smallest correct single-request, single-GPU BF16 inference engine for a small Qwen dense model:

- Load a Hugging Face checkpoint and tokenizer.
- Implement the Transformer forward pass, including RMSNorm, RoPE, grouped-query attention, MLP, and LM head.
- Support causal prefill, autoregressive decoding, a contiguous KV cache, and greedy sampling.
- Compare logits and greedy outputs with the matching Hugging Face Transformers reference.

Phase 0 is complete when a documented model runs successfully, reference logits meet a documented numerical tolerance, greedy output tokens match, cached and uncached decoding agree, and deterministic unit and integration tests pass. Tolerances and the exact model revision will be chosen and recorded with the implementation and validation environment.

Continuous batching, paged attention, serving, quantization, custom kernels, speculative decoding, MoE, hybrid attention, and multi-GPU execution are outside this milestone.

## Setup

Python 3.14.6 (Linux x86_64). The `torch` PyPI wheel bundles the CUDA runtime (cu132).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Pinned versions are in [requirements.txt](requirements.txt).

## Phase 0 baseline (M0)

Generate the local Hugging Face reference baseline (downloads `Qwen/Qwen3-0.6B` to the HF cache; artifacts are written to the gitignored `reference/` directory):

```bash
# On this machine the RTX 4060 is CUDA device 1 (CUDA ordering differs from nvidia-smi).
.venv/bin/python scripts/make_reference_baseline.py --device cuda:1
```

Run the deterministic artifact tests (CPU-only, skip if `reference/` is missing):

```bash
.venv/bin/python -m pytest tests/
```

### M0 validation record

- **Model:** `Qwen/Qwen3-0.6B`, revision `main` at commit `c1899de289a04d12100db370d81485cdf75e47ca`, BF16, `attn_implementation="sdpa"`.
- **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, BF16-capable), CUDA device 1; driver 595.99.02, CUDA runtime 13.2.
- **Software:** Python 3.14.6, torch 2.15.0+cu132, transformers 5.22.1, tokenizers 0.23.3, safetensors 0.8.0, numpy 2.5.4, pytest 9.2.0.
- **Input:** prompt `"The capital of France is"` (token IDs `[785, 6722, 315, 9625, 374]`), greedy (`do_sample=False`, `num_beams=1`), `max_new_tokens=32`, stop on EOS (`151645`).
- **Reference:** `model.generate` with KV cache (`output_scores=True`) is the canonical reference; a manual uncached greedy loop is recorded as a secondary artifact. Reference text: `" Paris. The capital of France is also the capital of the Republic of France. ..."` (32 tokens, no EOS).
- **Reproducibility:** two consecutive runs produced byte-identical artifacts (verified with `cmp`); `metadata.json` differs only in `created_utc`.
- **Cached vs uncached:** the two bf16 paths first diverge at decode step 5, where the cached path has an exact top-2 logit tie; before divergence, max/mean logit abs diff is 0.2188 / 0.0291. Each path is deterministic on its own. This is the baseline M5 (uncached) and M6 (KV cache) will validate against.
- **Tests:** 7/7 pass (`.venv/bin/python -m pytest tests/`), checking metadata, shapes/dtypes, argmax consistency of both token sequences, EOS/limit stopping, and artifact hashes.
- Downloaded weights stay in the HF cache and generated artifacts stay in `reference/`; neither is committed.

## Roadmap

| Phase | Scope |
| --- | --- |
| 0 | Minimal dense-model inference and reference validation |
| 1 | KV cache lifecycle, paged allocation, request queues, continuous batching, minimal HTTP API |
| 2 | Chunked prefill, scheduling, preemption, prefix caching, CUDA graphs, measured optimization |
| 3 | Target Qwen hybrid architectures, including full attention and Gated DeltaNet state |
| 4 | Mixture-of-Experts routing, checkpoint loading, execution, and validation |
| 5 | Speculative decoding, verification, rollback, MTP and block drafting |
| 6 | Tensor parallelism, NCCL communication, and multi-GPU validation |
| Future | Multimodal inference, outside the initial scope |

Architecture and model names, checkpoint formats, and reference implementations will be verified when their respective phases begin. Each phase should first establish correctness before performance work.

## Validation and benchmarks

Correctness checks will grow with the engine: reference logit error and greedy tokens, cached versus uncached execution, then batching, speculative sampling, and multi-GPU consistency as those features arrive. Performance measurements may include TTFT, time per output token, throughput, memory use, and cache utilization. Comparisons with Transformers, nano-vLLM, vLLM, and SGLang should use documented, reproducible workloads and environments; no performance claims are made yet.
