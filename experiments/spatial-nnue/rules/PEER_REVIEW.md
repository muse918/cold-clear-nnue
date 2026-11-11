# Independent experiment source review

Reviewed `train.py`, `kernels.cpp`, and `generate.cpp` read-only after changing the split to `fold=(game_id//2)%10`. Source/model/library hashes and executed check results are recorded in `peer_review_checks.json`. No training, kernel, or generator source was modified by this review.

## Outcome

No material target-alignment, ranking, pooling, row-seam reuse, or model-export defect was found in the reviewed implementation. The experiment supports conclusions about surrogate-teacher approximation and the measured local evaluator implementation. It does not establish competitive TETR.IO playing strength or a general ranking of neural architectures.

## Confirmed semantics

- **Target and next piece:** Lock records evaluate `continuation(child, queue[step+1])` and carry that same next-piece ID. Garbage records evaluate `continuation(after, queue[step])` before the current piece is placed. All recorded sequential next-piece labels and garbage-to-lock labels agree over the 98,091-record dataset.
- **Action ranking:** The learned value is continuation only. Both true and predicted action ranking add the current move's recorded reward exactly once. Continuation already includes the next move's reward; this is consistent with the explicitly defined two-placement surrogate teacher. It is not an actual attack or win-probability target.
- **Splits:** All 4,408 parent groups belong to one split each. Train/validation/test contain 66/10/20 games and 67,514/10,168/20,409 records. Every split contains both clean and garbage conditions. Adjacent even/odd game IDs are a stratification pair: the generator seeds them differently, so they are not matched runs with identical queues.
- **Checkpoint selection:** Validation RMSE selects epochs; test metrics do not select a checkpoint. Seed resets give each model a deterministic training order. The common epoch budget and optimizer form a controlled initial comparison, not an equally tuned or parameter-matched architecture benchmark.
- **Bit order:** Python patch bit `3*dy+dx` matches the native four-row pattern construction. Position features use `10*y+x` in both implementations. The transpose-and-contiguous export of linear weights matches the native input-major readers.
- **Local pooling:** The local CNN pools over the 37 vertical anchors while retaining eight horizontal anchors. Fixed-query attention pools over the same vertical axis and preserves two eight-value heads per horizontal anchor. It does not implement arbitrary content-dependent token-to-token self-attention.
- **Empty baseline:** Native LUT residuals subtract the empty-pattern value and initialize every horizontal accumulator with 37 empty-pattern values. Attention numerator and denominator mass from empty patches are retained. This preserves the trained function up to floating-point summation differences.
- **Row seams:** A four-row strip is reused only when every source row is consecutive and every complete row mask is unchanged. Row mapping is injective, preventing duplicate reuse. Reused local features can move vertically because that axis is pooled; absolute-position features are separately updated from changed occupied cells.

## Executed export check

The native shared library independently evaluated every one of the 98,091 child boards, then outputs were compared with the saved PyTorch predictions. The tolerance was `0.001 + 0.00001 * abs(torch_score)`.

| Model | Weight source | Maximum absolute error | Mean absolute error | Failures |
|---|---|---:|---:|---:|
| Cell | `results/final/seed918` | 0.00006104 | 0.00000386 | 0 |
| Local CNN | `results/final/seed918` | 0.00004578 | 0.00000378 | 0 |
| Fixed-query attention | `results/seed918` pilot | 0.00004578 | 0.00000465 | 0 |

The attention pilot is used only to validate the common arithmetic/export layout. Its predictive quality is excluded from the corrected-split final results. No claim of bit-exact float equality is made.

## Review follow-up and interpretation limits

The review identified an off-by-one validation check accepting next-piece ID 7. **Fixed and confirmed by source inspection:** the dataset reader now rejects `r.next >= 7`, and the C ABI prediction entry point rejects `next[i] >= 7`. Every experiment record already had a valid ID, so this correction does not change the measured predictions or model comparisons.

The initial export check covered shape and finiteness. The implementation has since added direct-convolution-versus-LUT checks over all 4,096 patterns and an independent native encoder check from exported scalar weights. Those checks have their own machine-readable results; the all-record Python/native comparison above remains a separate end-to-end layout check.

One-edge transition parity starts from a freshly computed parent and by itself does not bound drift along long accumulator chains. The implementation now also provides selected-path chained parity, whose transition counts, segment lengths and numeric errors are reported separately. The export comparison in this review concerns float32 arithmetic; integer quantization and complete one-ply expansion timing are separate experiments. None of these numerical checks establish match strength.

The candidate dataset has 2.25% line-clear edges, while the selected generator trajectories have 30.73% clearing moves (`metadata.json`). Both are valid but answer different workload questions. Overall candidate-edge speed should not be presented as a real-match event frequency. Likewise, aggregate sample errors weight candidates, while teacher regret averages parent decision groups; the distinction is appropriate and should remain explicit.
