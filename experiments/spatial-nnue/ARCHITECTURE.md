# Spatial NNUE: architecture and incremental inference specification

This document specifies the models and update algorithms implemented in
[`train.py`](train.py), [`kernels.cpp`](kernels.cpp), and
[`quantized.cpp`](quantized.cpp). The exported model and these implementations are
the reference for the experiment. Measurements below refer to the final seed918
exports, not the earlier pilot models.

The experiment's `cell` identifier denotes a new controlled positional-cell
baseline. It does not denote the repository's existing NNUE, which already uses
position-specific 3-by-4 and 10-by-1 categorical features; see
[`bot/src/evaluation/nnue.rs`](../../bot/src/evaluation/nnue.rs). The new models
change sharing, feature dimensions, and capacity. Their storage sizes therefore
are not an isolated compression result for an otherwise identical old model.

## 1. Board, context, and value contract

The binary board is `B[y, x]`, with width 10, height 40, `y = 0` at the bottom,
and `x = 0` at the left. A row occupies the low ten bits of a `uint16_t`. All 40
rows, including the upper buffer, are represented. Local windows are valid
windows: anchor coordinates are `x = 0..7` and `y = 0..36`, for 296 tokens. There
is no padding outside the board.

The network also receives one next-piece ID in `0..6`, encoded as seven one-hot
features. In the dataset the order is `I, O, T, L, J, S, Z`. The network input does
not include hold, combo, back-to-back, a pending garbage queue, or a longer piece
queue. The experiment's move generator and teacher are restricted surrogates;
their scope is recorded in [`data/train/metadata.json`](data/train/metadata.json).
These evaluators are research prototypes, not complete TETR.IO state evaluators.

The returned value is an estimate for the supplied board and next piece. A final
affine transform restores the training target's scale:

$$
\widehat V(B,n)=\mu+\sigma\left[w_o^\mathsf T
\operatorname{ReLU}(W_h f(B,n)+b_h)+b_o\right].
$$

The first head layer has 32 units. Immediate transition reward is external to
this function; [`search_bench.cpp`](search_bench.cpp) evaluates a candidate as
`reward + predicted_value`.

## 2. Exact local convolution-to-table compilation

The 12-bit index of a 3-wide, 4-high window is

$$
p_{x,y}=\sum_{d_y=0}^{3}\sum_{d_x=0}^{2}
B[y+d_y,x+d_x]\,2^{3d_y+d_x}.
$$

Let `bits(p)` be the resulting length-12 binary vector. The shared local encoder
is

$$
h(p)=\operatorname{ReLU}\left(
A_2\operatorname{ReLU}(A_1\operatorname{bits}(p)+b_1)+b_2\right),
$$

with `A1: 16 x 12`, `A2: 16 x 16`, and 16-element biases. On a board this is exactly
a valid `Conv2d(1,16,kernel_size=(4,3))`, ReLU, `Conv2d(16,16,kernel_size=1)`, ReLU,
using the same weights and bit ordering. The first kernel's dimensions are
height then width in PyTorch. There are only `2^12 = 4096` possible inputs, so the
entire encoder can be compiled into

$$
T[p]=h(p),\qquad p=0,\ldots,4095.
$$

At inference, four row loads, shifts, and masks form the index, followed by a
16-value table access. The local affine layers and their ReLUs are absent from
the inference hot path. Compilation preserves this local network's mathematical
function; floating-point evaluation order can cause small rounding differences.
It is not an approximation of a different, larger CNN. Quantization is a later,
explicit approximation step.

The trainable encoder has 480 parameters, whereas its compiled float table has
65,536 entries. Training the shared encoder constrains the table; the experiment
does not train 4,096 unrelated embedding vectors. A wider effective receptive
field increases the exact table exponentially: a 5-by-5 binary receptive field
would require `2^25` entries per output channel. That is why this implementation
keeps nonlinear spatial computation local and combines its outputs in a small
head.

[`train.py::check_compiled_encoder`](train.py) independently evaluates the
equivalent convolutions for all 4,096 patterns. The native
`--encoder-source PATH` option independently executes the source affine layers
and activations, then compares every compiled table component.

## 3. Eight horizontal groups and the absolute-position branch

### CNN pooling

The local CNN model sums vertically at each horizontal anchor:

$$
S_x=\sum_{y=0}^{36}T[p_{x,y}],\qquad g_x=S_x/37.
$$

Concatenating `g0 ... g7` gives 128 features. Horizontal location is retained by
the eight groups. The local branch does not encode a token's absolute height;
moving an intact window vertically within the same x group leaves its
contribution unchanged. It also cannot distinguish every possible arrangement
with the same per-group multiset of local patterns. This invariance enables row
reuse but limits expressiveness.

The separate positional branch is a learned sparse affine sum over occupied
cells:

$$
a_{\rm pos}(B)=b_{\rm pos}+
\sum_{y=0}^{39}\sum_{x=0}^{9}B[y,x]W_{10y+x},
\qquad z_{\rm pos}=\operatorname{ReLU}(a_{\rm pos}).
$$

`W_i` is a vector of length 32 for CNN/attention and 128 for the cell baseline.
Only occupied cells contribute during refresh. The accumulator stores the
pre-activation sum; ReLU is applied when constructing head features, not while
updating it. Absolute height and cell position therefore remain available to
the head even though the local pooled branch is vertically invariant. This
small branch does not guarantee that the combined representation is injective.

The feature layouts are:

| Model | Feature order | Head input width |
|---|---|---:|
| Cell | `relu(position[128]), next_onehot[7]` | 135 |
| CNN | `x0..x7: mean_local[16], relu(position[32]), next_onehot[7]` | 167 |
| Attention | `x0..x7: head0_value[8], head1_value[8], relu(position[32]), next_onehot[7]` | 167 |

The next-piece context is added after the board accumulators. Changing that
context alone requires another small head evaluation, not rebuilding board
features.

## 4. Fixed-query attention as additive sufficient statistics

The attention variant uses the same local encoder `h(p)`. For head `r = 0,1`, it
computes a local score and an eight-component local value:

$$
\ell_r(p)=\operatorname{clip}(a_r^\mathsf T h(p)+b_r,-5,5),
\qquad v_r(p)=\tanh(C_rh(p)+d_r).
$$

The clamp and tanh are part of the trained model. Define

$$
d_r(p)=\exp(\ell_r(p)),\qquad n_r(p)=d_r(p)v_r(p).
$$

Each table entry contains two blocks of `[numerator[8], denominator[1]]`, hence
18 float values. For each x group, the cached state is

$$
N_{x,r}=\sum_{y=0}^{36}n_r(p_{x,y}),\qquad
D_{x,r}=\sum_{y=0}^{36}d_r(p_{x,y}),\qquad
g_{x,r}=N_{x,r}/D_{x,r}.
$$

This is exactly softmax-weighted pooling over the 37 tokens of each group, using
the specified clamped logits. It needs 16 scalar reciprocals and 128 scalar
multiplications after the sums are updated. **All 37 tokens participate,
including empty windows.** The real-valued denominator satisfies
`37*exp(-5) <= D <= 37*exp(5)` and must remain finite and positive in the native
evaluator.

The two learned score vectors are fixed queries: neither query depends on the
current board, next piece, or another token. Keys and values depend only on the
local binary window. Consequently, a changed token alters only its own
numerator/denominator contribution. There is no all-token attention matrix at
inference, and no exponentiation or tanh in the inference hot path.

This model provides content-dependent pooling, not a full Transformer. A
content-dependent self-attention layer produces a different query for every
token; changing a key/value generally affects every query's result. Stacking
such layers also changes subsequent keys and queries. The fixed-query model
deliberately omits those interactions so its pooled statistics stay additive.
Its global head can combine the pooled groups, position features, and context,
but it cannot recover all the token-to-token relationships discarded by pooling.

## 5. Folding the empty pattern without removing tokens

Both local models use the same algebraic optimization. Let

$$
c=T[0],\qquad \Delta T[p]=T[p]-c.
$$

Since every x group always contains 37 windows,

$$
S_x=37c+\sum_{y=0}^{36}\Delta T[p_{x,y}],
\qquad \Delta T[0]=0.
$$

The loader converts the table to residuals in place and keeps `c` separately.
Refresh starts with `37*c`, then skips empty windows' zero residuals. Replacing
one pattern uses `DeltaT[new] - DeltaT[old]`. Neither an activation nor a change
in token count is involved.

For attention this applies to all numerator and denominator channels. Omitting
empty windows' original denominator contributions would change the normalized
function. Baseline folding preserves them exactly in real arithmetic. Clearing
rows does not reduce the number of windows: the 40-row board, including its
newly empty upper rows, still produces 37 windows per x group.

## 6. Parent-to-child update algorithms

### Ordinary lock: sparse local changes

An ordinary lock has `event == 0`, `clears == 0`, and `garbage == 0`. The update
copies the active parent accumulator, then updates the positional branch using
the actual parent/child cell difference.

For a changed cell `(cx, cy)`, affected local anchors satisfy

$$
x\in[\max(0,c_x-2),\min(7,c_x)],\qquad
y\in[\max(0,c_y-3),\min(36,c_y)].
$$

Eight 64-bit masks deduplicate the y anchors for each x. A four-cell placement
has a conservative upper bound of 48 anchor incidences before deduplication;
overlap and board boundaries reduce it. For every dirty anchor, the kernel
extracts the old and new pattern indices and applies one table-vector
difference. It does not store all 296 pattern indices or token embeddings per
search node.

The four supplied pre-clear placed-cell coordinates are used only on this
ordinary-lock path. If they are unavailable, dirty anchors are derived from
row XORs instead. A caller must supply coordinates that describe the transition
when using the supplied-coordinate fast path.

### Absolute-position changes: compare coordinates, not row identities

For each row index, the implementation computes

```text
removed = parent_rows[y] & ~child_rows[y] & 1023
added   = child_rows[y]  & ~parent_rows[y] & 1023
```

It subtracts `W[10*y+x]` for every removed bit and adds the corresponding vector
for every added bit. This is the signed form of an occupancy XOR update. It
handles clears and garbage shifts correctly because weights are tied to the
destination's absolute coordinate. The positional branch cannot reuse a whole
translated row's sum as if its height were unchanged.

### Event refresh path

`incremental_refresh_events` performs the sparse ordinary-lock update and uses
a complete child refresh for every other transition. It provides a compact
correct baseline when clears and garbage are uncommon among evaluated edges.

### Row/seam path: retain intact four-row strips

`row_source[child_y]` names the originating parent row, or `-1` for a newly
created row. The mapping preserves row order and is injective for nonnegative
entries. A row may retain its source identity even if placement changed bits
within it, so source identity alone is insufficient.

A child strip anchored at `y` can reuse parent strip `q = row_source[y]` only if
`0 <= q <= 36` and, for every `dy = 0..3`, both conditions hold:

```text
row_source[y + dy] == q + dy
child_rows[y + dy] == parent_rows[q + dy]
```

These checks establish consecutive source rows and unchanged complete row
masks. They imply `p_child[x,y] == p_parent[x,q]` for all eight x groups. Because
the local encoder and pool have no y-dependent weights, their contributions
can stay in the accumulator even when `y != q`.

The algorithm marks reusable old and new strips in two bitmasks. Starting from
the parent's local accumulator, it subtracts residual contributions for every
unretained old strip, then adds residual contributions for every unretained
new strip. The number of retained old and new strips is equal because the row
map is injective. New upper emptiness, removed top rows, garbage rows, placement
changes, and windows crossing a deletion seam are handled by those complements.

Matching complete row masks is conservative: a changed bit outside a particular
x window can force recomputation of all eight windows in its strip. This may
do extra work but cannot incorrectly retain a changed window. Reuse depends
on verified row mapping and contents, not just the clear count or a global
height offset. Noncontiguous clear masks therefore require no special-case
assumption.

## 7. Cost and storage

Let `M` be the number of occupied cells, `P` the positional width, `D` the local
table width, `K` the number of dirty ordinary-lock anchors, and `R_old/R_new`
the numbers of unretained old/new strips. Let `Delta` be the number of changed
absolute occupancy bits. Vector arithmetic costs are:

| Operation | Main work before the head |
|---|---|
| Full positional refresh | `M*P` additions, plus a scan of 40 row masks |
| Full local refresh | Extract 296 indices; add `D` values for each nonempty pattern; initialize `8*D` sums |
| Ordinary incremental update | Copy `P+8*D` values; scan 40 row pairs; update `Delta*P` positional values and at most `K*D` local-vector components |
| Event row/seam update | Same active copy and positional XOR; at most `37*4` strip comparisons; up to `8*(R_old+R_new)*D` residual-vector components |
| Head | `135*32` or `167*32` dense first-head terms, 32 ReLUs, and 32 output terms |

The seam path is not guaranteed to be cheaper than refresh for every board.
In the worst case it removes and adds every strip, and absolute-position XOR
can change many cells after a clear. No unconditional `O(number_of_cleared_rows)`
claim is made. An adaptive refresh threshold is a possible later optimization;
it is not implemented or measured here.

The float model payload sizes include the 44-byte model header and all exported
arrays, but not allocator overhead or other process memory:

| Model | Local table shape | Local table bytes | Serialized model bytes | Active float accumulator bytes |
|---|---|---:|---:|---:|
| Cell | None | 0 | 222,896 | 512 |
| CNN | `4096 x 16` | 262,144 | 335,152 | 640 |
| Attention | `4096 x 18` | 294,912 | 367,920 | 704 |

**The generic float `Accumulator` actually occupies 1,088 bytes for every model.**
It has `pos[128]` and `local[144]`, aligned to 64 bytes. Copy/update loops operate
only on the active prefix, but containers still allocate the full generic
struct. Active bytes are not the current search-node allocation. Specializing
the struct by model would reduce this allocation; that change is not included
in these measurements.

The precomputed-parent benchmark reports `number_of_parents * 1088` separately:
4,795,904 bytes for train and 1,429,632 bytes for heldout. Those numbers cover
accumulator structs only, not dataset records, group metadata, child indices,
or total resident memory. The sibling benchmark instead refreshes a parent
once for its candidate group during the timed operation.

### Integer first-layer variant

`quantized.cpp` applies per-channel symmetric quantization to the positional
weights and residual local table, with separately quantized bias/baseline.
The q8 path uses int8 weights and a range-checked int16 accumulator; q16 uses
int16 weights and a range-checked int32 accumulator. It rejects a configuration
whose conservative bounds do not establish safety, including positivity of
the attention denominator. The activation, normalized pooling, head, and output
scaling remain float32 after dequantization. There is no saturating accumulator
arithmetic.

For q8/int16, active accumulator bytes are 256/320/352 for cell/CNN/attention,
but the generic aligned struct occupies 576 bytes. For q16/int32, active bytes
are the same as the float table above and the allocated struct is 1,088 bytes.
The quantization validation harness also retains the float master model;
reported quantized parameter payload is not its resident-process footprint.
Bit-exact fresh-versus-incremental parity applies to the same quantized model.
It does not imply equality to the original float model.

## 8. Binary interfaces and validation

The model header is 16-byte magic `SPATIALNNUEv1` plus three zero bytes, followed
by five little-endian uint32 fields (`kind`, `localdim`, `posdim`, `featdim`,
`headdim`) and float32 `output_mean`, `output_std`. Arrays follow as float32:

```text
local_lut[4096, localdim]       # absent for cell
posW[400, posdim], posB[posdim]
headW[featdim, 32], headB[32]
outW[32], outB[1]
```

Weights are serialized input-major even though the formulas above use the
usual output-by-input matrix notation. `kind` is 0/1/2 for cell/CNN/attention;
`localdim` is 0/16/18, `posdim` is 128/32/32, and `featdim` is 135/167/167.
Current v1 native loaders require a little-endian machine.

Dataset records have a fixed 240-byte packed layout; the exact dtype is recorded
in the dataset metadata and the `Record` declaration. The record includes both
boards, row sources, four placed-cell IDs, clear/garbage flags, piece context,
and selected-path identifiers. The evaluator validates row bounds, enum ranges,
and order-preserving injective row mapping.

The native kernel exposes model load/free, batch full-refresh prediction, and
thread-local error reporting through a minimal C ABI. A loaded float model is
immutable; prediction calls own their scratch accumulator. It is used by the
Python rollout harness, not a networked bot client.

Validation has three distinct levels:

1. Exhaustively compare the local encoder with all 4,096 compiled table entries.
2. Compare each independent parent-to-child update with a fresh child evaluation.
3. Carry the actual updated accumulator through selected trajectories and compare
   each step with fresh evaluation, so repeated-addition drift is observable.

Final float train and heldout reports contain no parity failures. Heldout chained
validation covers 1,314 transitions in 16 uninterrupted segments, with maximum
length 88. A separate 2,400-record official-oracle-validated stress set covers
clear counts 1..4, noncontiguous clears, clear plus garbage, upper-buffer locks,
and garbage movement; float parity also passes there. Stress records are
independent transitions, so that suite does not establish long-path behavior.

Float comparisons use explicit absolute-plus-relative tolerances, not bitwise
equality: accumulator `1e-4 + 3e-5*abs(reference)`, score
`1e-3 + 1e-5*abs(reference)`. Encoder comparison uses
`2e-5 + 3e-5*abs(reference)`. The native seed918 exhaustive maximum table errors
are `3.5763e-7` for CNN and `3.8147e-6` for attention. Validation results and
tolerances are included in each native JSON report; `--verify-only` explicitly
marks timings as unmeasured.

## 9. Rare-event decision and measurement limits

The relevant frequency is the frequency among evaluated search edges. In the
training trace, 2,204 of 97,907 lock candidates clear lines, or 2.251%, while
1,298 of 4,224 selected locks clear lines, or 30.729%. These are different
distributions. This experiment supports treating clears as a cold path during
candidate evaluation, not assuming that executed gameplay rarely clears.

The heldout trace contains 28,633 no-clear candidates, 575 singles, 83 doubles,
5 triples, 1 quad, and 49 garbage records. Triple/quad measurements have too few
samples to characterize general event latency. Correctness stress tests
deliberately increase event coverage but do not estimate natural event frequency.

The nine-repeat float benchmark includes accumulator copying and head execution.
The precomputed-parent measurement excludes parent construction; the sibling
pipeline includes one parent refresh per candidate group, averaging 22.33
children on heldout. Move generation and full search bookkeeping are outside
both kernel scopes. Timings are medians of repetition-average time per child,
not per-child latency quantiles. Runs were serial, but CPU affinity and clock
frequency were not fixed.

Heldout sibling-pipeline measurements, microseconds per evaluated child:

| Model | Full child refresh | Ordinary incremental, event refresh | Ordinary incremental, row seams |
|---|---:|---:|---:|
| Cell | 1.1323 | 0.6469 | 0.6390 |
| CNN | 1.4102 | 0.8903 | 0.8772 |
| Attention | 1.6428 | 0.9997 | 0.9948 |

Sources:
[`cell_native_heldout_benchmark.json`](results/final/seed918/cell_native_heldout_benchmark.json),
[`cnn_native_heldout_benchmark.json`](results/final/seed918/cnn_native_heldout_benchmark.json),
[`attention_native_heldout_benchmark.json`](results/final/seed918/attention_native_heldout_benchmark.json).

Row seams reduce event-specific work: for example, CNN single-clear cost is
1.434 microseconds with event refresh and 0.946 with seams; garbage is 1.387
versus 0.853. However, the overall sibling improvement over event refresh is
about 1.5% for CNN and 0.5% for attention. Repetition ranges overlap: CNN seams
is 0.843..0.917 microseconds and event refresh is 0.872..0.910.

The supported engineering decision is to retain sparse ordinary-lock updates
as the essential fast path. Event refresh remains a simple reference and a
reasonable default. Keep the validated row/seam path available for event-heavy
search distributions and further profiling, but this experiment does not
establish a material overall throughput benefit from seams on the sampled
distribution. Architecture quality and the CNN-versus-attention choice require
the separate multi-seed prediction and rollout results; kernel timing alone
does not settle that choice.
