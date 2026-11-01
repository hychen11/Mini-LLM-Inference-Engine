"""Deterministic checks on the saved M0 reference baseline artifacts.

These tests validate artifact self-consistency (metadata fields, shapes,
dtypes, greedy argmax agreement, file hashes) on CPU. They do not require a
GPU or network access. They skip if the artifacts have not been generated
yet; run scripts/make_reference_baseline.py first.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

REF = Path(__file__).resolve().parent.parent / "reference"
REQUIRED_METADATA = (
    "model", "revision", "commit_sha", "prompt", "prompt_token_ids",
    "max_new_tokens", "generation", "dtype", "attn_implementation",
    "device", "config", "cached_vs_uncached", "versions", "files",
)
ARTIFACTS = (
    "prefill_logits.npy", "decode_logits.npy", "greedy_tokens.npy",
    "greedy_text.txt", "uncached_decode_logits.npy", "uncached_tokens.npy",
)


@pytest.fixture(scope="module")
def baseline():
    if not (REF / "metadata.json").exists():
        pytest.skip(
            "reference/ artifacts missing; run "
            "scripts/make_reference_baseline.py first"
        )
    meta = json.loads((REF / "metadata.json").read_text())
    return {
        "meta": meta,
        "prefill": np.load(REF / "prefill_logits.npy"),
        "decode": np.load(REF / "decode_logits.npy"),
        "tokens": np.load(REF / "greedy_tokens.npy"),
        "uncached_decode": np.load(REF / "uncached_decode_logits.npy"),
        "uncached_tokens": np.load(REF / "uncached_tokens.npy"),
    }


def test_metadata_fields(baseline):
    meta = baseline["meta"]
    for key in REQUIRED_METADATA:
        assert key in meta, f"metadata missing {key!r}"
    assert meta["model"] == "Qwen/Qwen3-0.6B"
    assert re.fullmatch(r"[0-9a-f]{40}", meta["commit_sha"])
    assert meta["dtype"] == "bfloat16"
    assert meta["attn_implementation"] == "sdpa"
    assert meta["generation"]["do_sample"] is False
    assert meta["generation"]["num_beams"] == 1
    assert len(meta["prompt_token_ids"]) > 0


def test_logit_shapes_and_dtypes(baseline):
    meta = baseline["meta"]
    vocab = meta["config"]["vocab_size"]
    for arr in (baseline["prefill"], baseline["decode"], baseline["uncached_decode"]):
        assert arr.dtype == np.float32
    assert baseline["prefill"].shape == (len(meta["prompt_token_ids"]), vocab)
    assert baseline["decode"].shape == (len(baseline["tokens"]), vocab)
    assert baseline["uncached_decode"].shape == (len(baseline["uncached_tokens"]), vocab)
    assert baseline["tokens"].dtype == np.int64
    assert baseline["uncached_tokens"].dtype == np.int64
    assert 1 <= len(baseline["tokens"]) <= meta["max_new_tokens"]
    assert 1 <= len(baseline["uncached_tokens"]) <= meta["max_new_tokens"]


def test_greedy_argmax_consistency(baseline):
    """Saved tokens must equal argmax of the saved reference logits."""
    for logits, tokens in (
        (baseline["decode"], baseline["tokens"]),
        (baseline["uncached_decode"], baseline["uncached_tokens"]),
    ):
        for i, tok in enumerate(tokens):
            assert int(logits[i].argmax()) == int(tok), f"step {i}"


def test_prefill_last_position_matches_first_decode_step(baseline):
    """The prefill last-position logits select the same first token."""
    first = int(baseline["tokens"][0])
    assert int(baseline["prefill"][-1].argmax()) == first


def test_stops_only_on_eos_or_limit(baseline):
    meta = baseline["meta"]
    for tokens in (baseline["tokens"], baseline["uncached_tokens"]):
        if len(tokens) < meta["max_new_tokens"]:
            assert int(tokens[-1]) in meta["generation"]["eos_token_ids"]


def test_cached_vs_uncached_recorded(baseline):
    c = baseline["meta"]["cached_vs_uncached"]
    first = c["first_divergence_step"]
    tokens, unc = baseline["tokens"], baseline["uncached_tokens"]
    n = min(len(tokens), len(unc))
    expected = next((i for i in range(n) if tokens[i] != unc[i]), None)
    assert first == expected
    if first is None:
        assert c["max_logit_abs_diff"] >= 0.0


def test_file_hashes(baseline):
    for name in ARTIFACTS:
        expected = baseline["meta"]["files"][name]
        actual = hashlib.sha256((REF / name).read_bytes()).hexdigest()
        assert actual == expected, f"hash mismatch for {name}"
