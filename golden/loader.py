"""Load Qwen3.5-0.8B text-LM weights from safetensors into fp32 numpy.

numpy has no bfloat16, and safetensors' numpy backend refuses BF16 outright.
We parse the header ourselves and widen BF16 -> FP32 by the only conversion
that is exactly representable: shift the 16 bits into the high half of an
FP32. No rounding, no loss.

Vision tower (`model.visual.*`) and the MTP head (`mtp.*`) are dropped -- the
FPGA target is text-only decode.
"""

import glob
import json
import os
import struct

import numpy as np

DEFAULT_REPO = "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"


def find_snapshot(repo: str = DEFAULT_REPO) -> str:
    hits = glob.glob(os.path.expanduser(f"{repo}/snapshots/*/config.json"))
    if not hits:
        raise FileNotFoundError(f"no snapshot with config.json under {repo}")
    return os.path.dirname(sorted(hits)[0])


def _bf16_to_f32(raw: bytes, shape) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    # BF16 is literally the top 16 bits of an FP32. Widen, don't convert.
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32).reshape(shape)


_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}


def load_text_weights(snapshot: str, verbose: bool = True) -> dict[str, np.ndarray]:
    """Return {tensor_name: fp32 array} for the text LM only."""
    st = glob.glob(f"{snapshot}/*.safetensors")
    if not st:
        raise FileNotFoundError(f"no .safetensors in {snapshot}")

    out: dict[str, np.ndarray] = {}
    skipped = 0
    for path in sorted(st):
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
            hdr.pop("__metadata__", None)
            base = 8 + hlen
            for name, meta in hdr.items():
                if name.startswith("model.visual") or name.startswith("mtp."):
                    skipped += 1
                    continue
                lo, hi = meta["data_offsets"]
                f.seek(base + lo)
                raw = f.read(hi - lo)
                dt = meta["dtype"]
                if dt == "BF16":
                    arr = _bf16_to_f32(raw, meta["shape"])
                elif dt in _DTYPES:
                    arr = np.frombuffer(raw, dtype=_DTYPES[dt]).reshape(meta["shape"]).astype(np.float32)
                else:
                    raise ValueError(f"unhandled dtype {dt} for {name}")
                # strip the `model.language_model.` prefix; it buys us nothing
                out[name.replace("model.language_model.", "")] = np.ascontiguousarray(arr)

    if verbose:
        n = sum(a.size for a in out.values())
        mb = sum(a.nbytes for a in out.values()) / 2**20
        print(f"loaded {len(out)} text tensors, {n:,} params, {mb:.0f} MB fp32 "
              f"(dropped {skipped} vision/mtp tensors)")
    return out


def load_config(snapshot: str) -> dict:
    with open(f"{snapshot}/config.json") as f:
        return json.load(f)["text_config"]
