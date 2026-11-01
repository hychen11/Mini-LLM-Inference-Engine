# Agent development guide

## Project goal

Build a small, understandable LLM inference engine incrementally. Correctness comes before performance. The active milestone is **Phase 0** in `README.md`; do not implement later phases unless explicitly requested.

## Working principles

- Prefer short, readable Python and explicit data flow, inspired by nano-vLLM's simplicity. Do not copy code without respecting its license.
- Build the smallest working change that satisfies the current milestone. Avoid speculative abstractions, framework layers, configuration systems, and dependencies.
- Keep model math independently implemented. Hugging Face Transformers may be used for checkpoint/tokenizer interoperability and as a reference, not as the engine's forward pass.
- Separate model execution, checkpoint loading, KV cache, and generation when the implementation warrants it. Introduce interfaces only when there is a concrete need.
- State supported model families, hardware, dtypes, context limits, and assumptions explicitly. Reject unsupported configurations rather than silently producing incorrect results.
- Preserve existing user changes. Do not expand scope or change public behavior without a reason.

## Phase 0 correctness

- Target one small, explicitly identified Qwen dense checkpoint, one request, and one GPU with BF16 support. CPU execution may be used for small deterministic tests where practical.
- Implement RMSNorm, RoPE, grouped-query attention, MLP, LM head, causal prefill, autoregressive decode, contiguous KV cache, and greedy sampling.
- Validate against the matching Hugging Face model and configuration. Document the model revision, dtype, input, numerical tolerance, and test environment; report maximum and mean logit errors.
- Check reference greedy tokens and cached versus uncached logits/tokens, including multi-token prefill and subsequent decode. Cover causal masking, positions, cache bounds, and GQA with deterministic tests.
- Do not claim GPU or reference validation passed unless it was actually run. Explain missing hardware, model access, or other blockers.

## Development workflow

- Before changing code, inspect the relevant files and keep changes focused on the requested milestone.
- Add or update focused tests for behavior changes; run the smallest relevant checks first. Record commands and results when reporting work.
- Keep setup and usage instructions current in `README.md`. Pin or constrain dependencies when they are introduced and document any compatibility requirements.
- Do not commit model weights, downloaded checkpoints, secrets, virtual environments, or generated benchmark artifacts.
- Optimize only after correctness is established and a reproducible benchmark identifies a need.
