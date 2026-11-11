# Legacy kernel parity reproducer

This is a minimal source-equivalent C++ reproduction of three independent
correctness issues in `bot/src/evaluation/nnue.rs` at commit
`31713b0bd20c0bddf9dfa11daacdbd2d67888ae4`. It uses the same relevant AVX2
intrinsics and deterministic integer-valued floats. It is not a build of the
Rust repository and does not load a trained checkpoint.

From the repository root:

```bash
g++ -O3 -std=c++17 experiments/spatial-nnue/legacy-parity/kernel_parity.cpp \
    -o experiments/spatial-nnue/data/kernel_parity
experiments/spatial-nnue/data/kernel_parity
```

The executable checks CPU support for AVX2 and FMA before entering vectorized
code. It returns zero only when the old behavior is reproduced and each local
correction agrees with its reference calculation.

| Reproduction | Original result | Locally corrected result |
|---|---:|---:|
| Dense SIMD path omits the scalar path's ReLU | Maximum error 4 | Maximum error 0 |
| `set_ps(w0,...,w7)` reverses weights relative to loaded input lanes | Dot product 11632 | Dot product 11968, matching scalar |
| Prefetched 40-row loop has 38 iterations and one tail addition | 39 rows | 40 rows |

`kernel_parity_results.json` is the recorded output from the available GCC
14.2.0 runtime. These results establish behavior of the isolated kernels,
not the playing strength or complete output difference of a trained bot.

Additional full-network mismatches need separate tests before the original
SIMD entrypoints can be considered equivalent: the newer encoding macros omit
their final window, newer SIMD encoders omit ReLU, and `nnue_new.rs` uses
different scalar/SIMD output normalization. No production Rust source is
modified by this reproducer.
