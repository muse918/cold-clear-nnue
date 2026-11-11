# Quantized incremental evaluator experiment

`quantized.cpp` performs **post-training quantization of the first layer and compiled local-pattern table only**. It reads the same float `.snn` models as `kernels.cpp`; it quantizes weights on load and does not export a new integer model-file format. Activation, attention division, the 32-unit head, and output normalization stay float32. There is no quantization-aware training or Rust integration in this experiment.

## Representation and safety

Each output channel has its own positive scale. Weights use symmetric nearest rounding, with halfway values rounded away from zero:

| Mode | Weight range | Accumulator | Unsafe range handling |
|---|---:|---:|---|
| `--bits 8` | −127…127 | int16 | Reject; retry the q16/int32 baseline |
| `--bits 16` | −32767…32767 | int32 | Reject |

The position branch quantizes each of its 400 cell embeddings and its bias. For channel `j`, the implementation computes the conservative bounds

```text
lower[j] = q_bias[j] + sum_i min(0, q_weight[i,j])
upper[j] = q_bias[j] + sum_i max(0, q_weight[i,j]).
```

This covers every occupancy bitboard, including all 40 rows, independently of the evaluation dataset. Merely choosing int8 weights does not prove that an int16 accumulator is safe.

The local CNN/attention table contains all 4,096 binary 3×4 patterns. `Model::load` separates the empty-pattern embedding `E(0)` and stores residuals `R(p) = E(p) − E(0)`. Each of the eight horizontal positions has 37 vertical windows. Its quantized pool is

```text
q_pool[x,j] = round(37 * E(0,j) / scale[j]) + sum_y q_R(pattern(x,y),j).
```

The empty residual is exactly zero. The bound is the quantized empty baseline plus 37 times the minimum/maximum residual, including zero. Windows overlap, but that cannot invalidate this conservative bound. It also covers intermediate sums after removing any subset of residuals. Every replacement removes the old contribution before adding the new one; arithmetic uses int32 before a checked-range int16 store. No accumulator saturates or clips. Weight rounding uses the stated symmetric range.

The attention model's two denominator channels additionally require a **strictly positive lower bound across all patterns**. Unsafe quantization is rejected; no epsilon or denominator clamp is inserted. All three final attention models passed this check at both precisions. Their q8 denominator lower bounds were 26.1200, 31.9758, and 10.2947 for seeds 918, 919, and 920.

## Incremental paths

The evaluator reuses the exact board-delta logic in `kernels.cpp`:

- Full refresh reconstructs the accumulator from the child board.
- Refresh-events updates ordinary four-cell locks incrementally and rebuilds after line clears or garbage rises.
- Row-seams preserves unchanged consecutive four-row strips, subtracts old boundary windows, and adds new boundary windows. The absolute-position branch uses parent/child occupancy differences.

The test compares both incremental accumulators and scores **bit for bit** against quantized full refresh. It also carries an accumulator along selected game trajectories, including forced garbage transitions, instead of rebuilding its parent before every edge.

## Measured accuracy and parity

The final models use the corrected balanced validation split in `results/final/seed{918,919,920}`. All measurements below use the 29,346 held-out transitions and 1,265 groups with at least two lock candidates. “Same choice” compares `reward + quantized_value` with `reward + float_value`; it is not a win rate or teacher agreement metric.

| Model / seed | q8 score MAE vs float | q8 same choice | q16 score MAE vs float | q16 same choice |
|---|---:|---:|---:|---:|
| Cell / 918 | 0.268673 | 97.4704% | 0.00137616 | 100% |
| Cell / 919 | 0.270396 | 97.3123% | 0.00105363 | 100% |
| Cell / 920 | 0.250936 | 97.9447% | 0.00100669 | 100% |
| CNN / 918 | 0.314774 | 98.1818% | 0.00170623 | 100% |
| CNN / 919 | 0.181598 | 98.8933% | 0.00105838 | 100% |
| CNN / 920 | 0.446636 | 98.8142% | 0.000927755 | 100% |
| Attention / 918 | 0.355627 | 98.0237% | 0.000781621 | 100% |
| Attention / 919 | 0.862514 | 96.8379% | 0.00341774 | 100% |
| Attention / 920 | 0.245575 | 98.2609% | 0.000774343 | 99.9209% |

The single q16 attention choice difference for seed 920 had float-score regret 0.00048828125. CNN q8 maximum choice regret was 0.529282, 0.436495, and 0.218388 across the three seeds. Exact agreement at q16 on these CNN groups does not establish agreement on unseen states.

All 18 model/precision combinations passed the all-board integer bounds and had **zero mismatched accumulator components or scores** for either incremental method. The carried trajectory check covered 1,314 selected/garbage transitions in 16 games, with no discontinuities and a longest chain of 88 transitions. The q8 all-board bounds over these nine models fit inside [−13,125, 10,296].

All six seed-918 configurations also passed 2,400 separate adversarial transitions in `data/stress/records.bin`, including line clears, garbage rises, buffer occupancy, and rows 38/39. These are independent one-edge fixtures, not long games. They have no multi-candidate ranking groups, so their ranking fields are `null`. CNN stress score MAE versus float was 0.357890 at q8 and 0.00205499 at q16.

The dataset and rollouts use the explicitly documented hard-drop surrogate. These results establish numerical behavior for this experiment; they do not measure official TETR.IO match strength or quantized closed-loop play.

## Memory accounting

For the final CNN, dimensions and payload sizes are the same across seeds:

| Payload | q8/int16 | q16/int32 |
|---|---:|---:|
| Quantized position and LUT weights | 78,336 B | 156,672 B |
| Quantized biases and scales | 384 B | 384 B |
| Unchanged float head | 21,636 B | 21,636 B |
| Total parameter payload | 100,356 B | 178,692 B |
| Active accumulator entries | 320 B | 640 B |
| Current generic aligned accumulator allocation | 576 B | 1,088 B |

The source float CNN file is 335,152 B, including its header. Parameter payload is not process memory: this validation harness retains the float reference model as well. The generic accumulator reserves maximum dimensions; its allocation is larger than the active entries copied and evaluated for the CNN.

## Measured latency and decision

Final seed-918 measurements used all 29,346 held-out transitions, nine repetitions, serial benchmark processes, and no CPU pinning. Other experiment jobs were paused. Values are median microseconds per candidate; both implementations exclude parent construction and include child update plus head. Float values come from `*_native_heldout_benchmark.json`; integer values come from `*_quantized_q{8,16}_timing.json` in the same directory.

| Model / precision | Full refresh | Incremental, refresh events | Incremental, row seams |
|---|---:|---:|---:|
| Cell / float32 | 1.066 | 0.619 | 0.608 |
| Cell / q8-int16 | 0.889 | 0.696 | 0.642 |
| Cell / q16-int32 | 3.900 | 0.902 | 0.868 |
| CNN / float32 | 1.424 | 0.829 | 0.824 |
| CNN / q8-int16 | 1.899 | 1.048 | 1.041 |
| CNN / q16-int32 | 2.489 | 1.132 | 1.078 |
| Attention / float32 | 1.673 | 0.947 | 0.936 |
| Attention / q8-int16 | 2.082 | 1.127 | 1.122 |
| Attention / q16-int32 | 2.576 | 1.122 | 1.069 |

**Keep the float evaluator as this prototype's performance baseline.** The tested CNN q8 and q16 row-seam paths were respectively 26.4% and 30.9% slower than the float implementation. Quantization reduced the parameter payload and removed accumulator rounding drift, but did not improve incremental evaluation latency here. These are mixed integer/float paths with dequantization and an unchanged float head; the precise cost breakdown was not profiled. Explicit integer SIMD kernels or a quantized head are possible follow-up experiments, not measured improvements in this result.

The shared host produced some repetition outliers; the JSON retains minimum and maximum times. Do not interpret small differences between the event strategies as a stable ranking. The timing dataset also has a different event distribution from selected game trajectories: most evaluated candidates do not clear lines.

## Reproduction

From `experiments/spatial-nnue`:

```sh
g++ -O3 -std=c++17 -march=native -Wall -Wextra quantized.cpp -o data/quantized
data/quantized --model results/final/seed918/cnn.snn --data data/heldout/records.bin --bits 8 --output results/final/seed918/cnn_quantized_q8.json --predictions results/final/seed918/cnn_quantized_q8.f32 --repeats 0
data/quantized --model results/final/seed918/cnn.snn --data data/heldout/records.bin --bits 16 --output results/final/seed918/cnn_quantized_q16.json --predictions results/final/seed918/cnn_quantized_q16.f32 --repeats 0
```

Substitute the seed/model name for the other configurations. For stress validation, use `data/stress/records.bin` and an output name ending in `_stress.json`. Prediction files contain one native little-endian float32 value per record, in input order. The JSON reports bounds, payload sizes, parity failures, score error, and ranking sensitivity. `results/final/quantized_multiseed_stress_summary.json` collects the correctness results.

`--repeats 0` collects no timings. With a positive repeat count, the benchmark reports full-refresh, refresh-events, and row-seams latency over the entire input stream. Parent accumulators are prepared outside the timed region; each child includes its active accumulator copy where applicable, dirty-window extraction, integer updates, dequantization, and the unchanged float head. The report therefore measures a cached-parent evaluator, not a complete search or a network client. Collect timings on an otherwise idle CPU and compare with the matching cached-parent float measurements.

To reproduce the timing run, use `--repeats 9` and a separate output filename such as `results/final/seed918/cnn_quantized_q8_timing.json`.

The measured sensitivity supports q16/int32 as a conservative numerical baseline for future integer-kernel work. q8/int16 offers the smaller payload but changes roughly 1.1–1.8% of the CNN choices in this sample; deployment should evaluate that tradeoff with quantization-aware training or closed-loop search measurements. The present measurements support retaining float inference.
