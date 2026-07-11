"""Export a softmax case plus the exp lookup table the RTL uses. Softmax is
input-generic, so realistic random logits exercise it fully.

  python3 scripts/export_softmax_tv.py  ->  artifacts/tv_softmax.bin
"""
import numpy as np

T = 64
LUTN = 1024
RANGE = 16.0   # exp table covers [-16, 0]


def main():
    rng = np.random.RandomState(0)
    scores = (rng.randn(T) * 3.0).astype(np.float32)
    s = scores - scores.max()
    e = np.exp(s)
    w = (e / e.sum()).astype(np.float32)

    # exp LUT over [-RANGE, 0] at bin centers, values in (0,1]
    idx = np.arange(LUTN)
    xs = -RANGE + (idx + 0.5) * (RANGE / LUTN)
    lut = np.exp(xs).astype(np.float32)

    print(f"T={T}  scores range [{scores.min():.2f},{scores.max():.2f}]  "
          f"w max {w.max():.4f}  w sum {w.sum():.4f}")
    with open("artifacts/tv_softmax.bin", "wb") as f:
        np.array([T, LUTN], "<i4").tofile(f)
        lut.astype("<f4").tofile(f)
        scores.astype("<f4").tofile(f)
        w.astype("<f4").tofile(f)
    print("wrote artifacts/tv_softmax.bin")


if __name__ == "__main__":
    main()
