"""Dump per-step torch reference activations for Qwen3.5-0.8B text decode.

Runs the HF model one token at a time with a live cache, so it exercises the
same recurrent path the FPGA will. Captures logits and per-layer hidden states.

Output: artifacts/ref.npz  -- consumed by verify.py (numpy check) and later by
the HLS csim testbenches as golden vectors.

Run with the project venv:  ./.venv/bin/python golden/dump_torch_ref.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import find_snapshot  # noqa: E402

PROMPT = "The capital of France is"
N_NEW = 6


def main():
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DynamicCache,
        Qwen3_5ForConditionalGeneration,
    )

    snap = find_snapshot()
    tok = AutoTokenizer.from_pretrained(snap)
    print("loading torch model in fp32 (exact widening of the bf16 checkpoint)...")
    # NB: AutoModelForCausalLM resolves to Qwen3_5ForCausalLM, which wants a *text*
    # config; the checkpoint declares the multimodal wrapper. Instantiate it directly.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(snap, dtype=torch.float32)
    model.eval()

    text_model = model.model.language_model
    cfg = model.config.text_config
    layers = text_model.layers

    captured: dict[str, np.ndarray] = {}
    step = {"i": 0}

    hooks = []
    for li, layer in enumerate(layers):
        def mk(li):
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[f"s{step['i']}_layer{li}"] = h.detach().float().numpy().reshape(-1)
            return hook
        hooks.append(layer.register_forward_hook(mk(li)))

    ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()
    print(f"prompt {PROMPT!r} -> {len(ids)} tokens: {ids}")

    cache = Qwen3_5DynamicCache(cfg)
    all_logits, fed = [], []
    seq = list(ids)

    with torch.no_grad():
        for n in range(len(ids) + N_NEW):
            t = seq[n]
            fed.append(t)
            out = model(
                input_ids=torch.tensor([[t]]),
                past_key_values=cache,
                cache_position=torch.tensor([n]),
                use_cache=True,
            )
            lg = out.logits[0, -1].float().numpy()
            all_logits.append(lg)
            captured[f"s{step['i']}_logits"] = lg
            step["i"] += 1
            if n + 1 >= len(ids):
                seq.append(int(np.argmax(lg)))

    for h in hooks:
        h.remove()

    gen = seq[len(ids):]
    print("continuation:", repr(tok.decode(gen)))

    # final recurrent + conv state, for DeltaNet state checks
    lt = cfg.layer_types
    lin = [i for i, k in enumerate(lt) if k == "linear_attention"]
    for j, li in enumerate(lin):
        rs, cs = cache.recurrent_states[li], cache.conv_states[li]
        if rs is not None:
            captured[f"rec{j}"] = rs.detach().float().numpy().reshape(-1)
        if cs is not None:
            captured[f"conv{j}"] = cs.detach().float().numpy().reshape(-1)

    captured["fed_tokens"] = np.array(fed, np.int64)
    captured["prompt_len"] = np.array([len(ids)], np.int64)
    captured["generated"] = np.array(gen, np.int64)

    os.makedirs("artifacts", exist_ok=True)
    np.savez("artifacts/ref.npz", **captured)
    print(f"wrote artifacts/ref.npz  ({len(captured)} arrays, "
          f"{sum(a.nbytes for a in captured.values())/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
