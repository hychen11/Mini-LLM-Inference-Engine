"""Generate the local Hugging Face reference baseline for Phase 0 (M0).

Loads the pinned Qwen3-0.6B revision in BF16 on a CUDA GPU and runs one fixed
prompt through:

  1. a plain causal-prefill forward (full logits for every prompt position),
  2. model.generate (greedy, KV cache) with output_scores=True -- the
     canonical reference tokens and per-step decode logits,
  3. a manual uncached greedy loop (full recompute each step) -- recorded as
     a secondary artifact because bf16 cached and uncached execution can
     diverge at near-tie logits (this is what M5/M6 validate later).

Saved under ./reference (gitignored):

  prefill_logits.npy        float32 [seq_len, vocab]   exact bf16 values
  decode_logits.npy         float32 [n_steps, vocab]   cached path (generate)
  greedy_tokens.npy         int64   [n_steps]          canonical reference
  greedy_text.txt           decoded generated text
  uncached_decode_logits.npy float32 [n_steps, vocab]  manual recompute path
  uncached_tokens.npy       int64   [n_steps]
  metadata.json             revision, settings, versions, divergence stats

Logits are stored as float32 upcasts of the bf16 values (lossless), so later
milestones compare against exact reference numbers.

Usage (RTX 4060 is CUDA device 1 on this machine):
  .venv/bin/python scripts/make_reference_baseline.py --device cuda:1
"""

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, Qwen3ForCausalLM

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
DTYPE = torch.bfloat16
ATTN_IMPLEMENTATION = "sdpa"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def uncached_greedy(model, input_ids, eos_ids, max_new_tokens):
    """Greedy decode by full-sequence recompute (no KV cache)."""
    device = input_ids.device
    tokens: list[int] = []
    logits: list[np.ndarray] = []
    current = input_ids
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            step = model(current).logits[0, -1]
        logits.append(step.float().cpu().numpy())
        next_id = int(step.argmax())
        tokens.append(next_id)
        if next_id in eos_ids:
            break
        current = torch.cat([current, torch.tensor([[next_id]], device=device)], 1)
    return tokens, np.stack(logits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--output-dir", type=Path, default=Path("reference"))
    args = ap.parse_args()

    device = torch.device(args.device)
    if not torch.cuda.is_available() or not device.type.startswith("cuda"):
        raise SystemExit("M0 baseline requires a CUDA GPU; pass --device cuda:N.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to(device).eval()

    commit_sha = HfApi().repo_info(args.model, revision=args.revision).sha
    eos_ids = model.config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    # 1. Causal prefill: full logits for every prompt position.
    with torch.inference_mode():
        prefill_logits = model(input_ids).logits[0].float().cpu().numpy()

    # 2. Canonical reference: greedy generate with KV cache and step scores.
    with torch.inference_mode():
        gen = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    ref_tokens = gen.sequences[0, input_ids.shape[1]:].tolist()
    decode_logits = np.stack([s[0].float().cpu().numpy() for s in gen.scores])

    # Internal consistency: saved scores must reproduce the saved tokens.
    for i, tok in enumerate(ref_tokens):
        if int(decode_logits[i].argmax()) != tok:
            raise SystemExit(f"generate scores/tokens inconsistent at step {i}")

    # 3. Secondary artifact: manual uncached greedy loop.
    unc_tokens, unc_logits = uncached_greedy(
        model, input_ids, eos_ids, args.max_new_tokens
    )

    # Cached vs uncached divergence statistics (up to first divergence).
    n = min(len(ref_tokens), len(unc_tokens))
    first_divergence = next(
        (i for i in range(n) if ref_tokens[i] != unc_tokens[i]), None
    )
    cmp_steps = n if first_divergence is None else first_divergence
    max_diff = float(np.max(np.abs(decode_logits[:cmp_steps] - unc_logits[:cmp_steps]))) \
        if cmp_steps else 0.0
    mean_diff = float(np.mean(np.abs(decode_logits[:cmp_steps] - unc_logits[:cmp_steps]))) \
        if cmp_steps else 0.0

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "prefill_logits.npy", prefill_logits)
    np.save(outdir / "decode_logits.npy", decode_logits)
    np.save(outdir / "greedy_tokens.npy", np.asarray(ref_tokens, dtype=np.int64))
    (outdir / "greedy_text.txt").write_text(tokenizer.decode(ref_tokens))
    np.save(outdir / "uncached_decode_logits.npy", unc_logits)
    np.save(outdir / "uncached_tokens.npy", np.asarray(unc_tokens, dtype=np.int64))

    files = {
        name: sha256(outdir / name)
        for name in (
            "prefill_logits.npy", "decode_logits.npy", "greedy_tokens.npy",
            "greedy_text.txt", "uncached_decode_logits.npy", "uncached_tokens.npy",
        )
    }
    metadata = {
        "model": args.model,
        "revision": args.revision,
        "commit_sha": commit_sha,
        "prompt": args.prompt,
        "prompt_token_ids": input_ids[0].tolist(),
        "max_new_tokens": args.max_new_tokens,
        "generation": {
            "sampling": "greedy (argmax)",
            "do_sample": False,
            "num_beams": 1,
            "eos_token_ids": eos_ids,
            "stop_on_eos": True,
            "reference_path": "model.generate with KV cache (output_scores=True)",
        },
        "dtype": "bfloat16",
        "attn_implementation": ATTN_IMPLEMENTATION,
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "config": model.config.to_dict(),
        "cached_vs_uncached": {
            "first_divergence_step": first_divergence,
            "compared_steps": cmp_steps,
            "max_logit_abs_diff": max_diff,
            "mean_logit_abs_diff": mean_diff,
            "note": "bf16 cached (generate) and uncached (recompute) paths can "
                    "differ at near-tie logits; each path is deterministic.",
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "driver": driver_version(),
            "transformers": __import__("transformers").__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "safetensors": __import__("safetensors").__version__,
            "numpy": np.__version__,
        },
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": files,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"model: {args.model} @ {commit_sha}")
    print(f"device: {torch.cuda.get_device_name(device)} ({args.device})")
    print(f"prompt: {args.prompt!r} -> {input_ids.shape[1]} tokens")
    print(f"reference (cached) tokens: {len(ref_tokens)} "
          f"(stopped on EOS: {ref_tokens and ref_tokens[-1] in eos_ids})")
    print(f"uncached tokens: {len(unc_tokens)}")
    print(f"first divergence (cached vs uncached): step {first_divergence}")
    print(f"max/mean logit diff before divergence: {max_diff:.4f} / {mean_diff:.6f}")
    print(f"reference text: {tokenizer.decode(ref_tokens)!r}")
    print(f"artifacts: {outdir.resolve()}")


if __name__ == "__main__":
    main()
