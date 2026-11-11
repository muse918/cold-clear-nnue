# Placement search: Cold Clear reference and TETR.IO implementation

## Reference audit

The inspected upstream is [`MinusKelvin/cold-clear`](https://github.com/MinusKelvin/cold-clear), commit **`279edd7c3177ff8077f6a930193397814b281f27`** (2022-01-18, “don't count long moves that are part of tspin doubles”). These links pin the examined source, rather than relying on a moving default branch:

- [`bot/src/dag.rs`](https://github.com/MinusKelvin/cold-clear/blob/279edd7c3177ff8077f6a930193397814b281f27/bot/src/dag.rs): generations, state deduplication, leaf selection, speculative queue branches, backpropagation and root advancement.
- [`bot/src/modes/normal.rs`](https://github.com/MinusKelvin/cold-clear/blob/279edd7c3177ff8077f6a930193397814b281f27/bot/src/modes/normal.rs): thinker dispatch, minimum/maximum work, candidate selection and reset handling.
- [`bot/src/evaluation/standard.rs`](https://github.com/MinusKelvin/cold-clear/blob/279edd7c3177ff8077f6a930193397814b281f27/bot/src/evaluation/standard.rs): selection weights and value/reward semantics.

The upstream calls its algorithm Monte Carlo Tree Search. Its storage is a generation-indexed DAG, where a generation contains states with the same number of locked pieces. It is **not a conventional fixed-width beam search**. Leaf traversal uses randomized weighted selection from evaluated children, marking leaves to prevent multiple thinkers from expanding the same leaf. The standard evaluator weights a child by `(value - minimum_value + 10)^2 / (rank^2 + 1)`.

Backpropagation sorts child moves and applies `Value::improve`. In the standard evaluator, that maximizes board value and largest spike separately; those maxima may come from different continuations. Unknown future pieces are represented by speculative child groups. Their propagated value averages the possible remaining bag pieces, with dead possibilities penalized relative to the worst surviving possibility. A newly observed preview resolves the corresponding speculative generation while retaining node links.

The upstream deduplication key combines a compressed occupied board with combo, B2B, bag information, a reserve piece and `reserve_is_hold`. The last field matters because the same apparent reserve can represent either a held piece or an observed next piece while hold is empty. Playing a selected move advances the root and discards the old generation. A board reset rebuilds the DAG; if the change looks like a garbage rise, the normal mode can shift its previous plan and request analysis of that plan again. These details are in the pinned DAG and normal-mode sources above.

## What this implementation takes from that design

`search.cpp` is a new C++17 implementation using the ported TETR.IO environment. It takes the following structural ideas from Cold Clear: memoized game states, exact hold/queue accounting, separate immediate rewards and future-state evaluations, per-node evaluation caches, and reuse of a retained descendant when the observed root is exactly that state.

The traversal is deliberately a **deterministic generation beam over a DAG**. That makes work caps, exhaustive small-depth comparisons, and incremental NNUE integration straightforward. The algorithm does not implement upstream randomized leaf sampling, speculative bag expectation, multithreaded thinkers, opening books, or perfect-clear-loop search. No old PPT attack, movement or garbage simulator is embedded in this searcher. The geometry and battle modules implement the supplied TETR.IO rules.

| Concern | Pinned Cold Clear | This implementation |
|---|---|---|
| Traversal | Weighted random leaf selection with backpropagation | Complete a generation, merge dominated identical states, retain the best beam |
| State graph | Generations and interned simplified states | Canonical full-state keys and cached child edges |
| Unknown pieces | Speculation over remaining bag possibilities | Stop at the observed queue horizon |
| Hold | Reserve plus hold/next distinction | Explicit current, held piece, observed queue and hold-used flag |
| Scores | Evaluator-specific value/reward with separate spike propagation | Scalar sum of edge rewards plus leaf value |
| Reuse | Advance generations; resolve speculative previews | Re-root only on an exact cached state; an unseen observation resets the graph |
| Work limits | Thinker dispatch and minimum/maximum node work | Hard processed-edge cap, cache-node cap, cooperative deadline |
| Physics | Upstream movement/environment implementation | TETR.IO geometric reachability and lock/battle transitions; no frame input simulation |

## Public search contract

`Searcher` owns immutable rules and refers to an evaluator that must outlive it. `SearchEvaluator::evaluate` receives the resulting state and, for a new child, its parent state, parent evaluation and authoritative transition. `Evaluation` includes 272 opaque float slots for the existing native accumulator. The search never compares accumulator bytes to decide state equivalence; state identity is the complete canonical state key.

The default options are depth 4, beam width 64, 12,000 processed edges, 32,768 cached nodes and 50 ms. `frames_per_piece` defaults to zero. A caller may supply an external elapsed-frame estimate; the search does not derive it from the placement witness or simulate control, gravity, DAS or lock delay. Depth and beam width are independent from the work cap: requesting depth 4 does not imply that four complete layers fit within the budget.

For a path ending at state `s`, the score is:

`sum(reward(transition_i)) + evaluate(s)`.

At each layer, routes to the same canonical state are merged by keeping the route with the larger cumulative score. This is valid because the state key and immutable rules contain every modeled fact that can affect future transitions and evaluations. Different root actions can therefore share a descendant without evaluating its NNUE accumulator again. Stable insertion order breaks score ties, and unordered maps are used for lookup rather than traversal order.

Scores are not maximized across incompatible horizons. After a complete generation, a root candidate's value is replaced by its best value at that generation. Candidates removed by beam pruning retain an explicitly reported shallower `depth`. The selected action comes from the best surviving path in the **last fully completed layer**. If the first layer cannot be completed, the search returns the best legal root action processed so far. An interrupted deeper layer cannot displace the completed-layer result merely because some of its branches happened to be visited first.

Unknown and dead leaves are not expanded. An unknown leaf retains its current evaluation; a dead leaf receives a large terminal penalty. `no_surviving_continuation` reports that the selected searched continuation is dead. With beam pruning, it is not a proof that every possible root action loses. `dead` is reserved for an authoritative dead root, and `no_legal_actions` describes a root whose generated action set is empty.

### Visible queue and hold

`State::queue` excludes `current` and contains only observed previews. An ordinary lock draws the next current from its front. Swapping with an occupied hold slot also draws once after the lock. Using an empty hold first consumes the front preview as the piece to place, and then consumes another preview to draw the next current. A missing observed draw becomes `Piece::None`; the search never substitutes a hidden bag draw.

A client should append newly revealed previews only after processing the corresponding real lock or observation. A newly extended queue is a different state key. This implementation intentionally invalidates its graph when that complete new root is absent from the cache. It does not claim the speculative-preview reuse implemented by upstream Cold Clear. Queue-independent accumulator reuse across such observations would be a useful separate optimization, but is not implemented here.

### State identity and cache lifetime

`state_key` includes occupied rows, garbage/solid provenance, current/hold/observed queue, hold-used and last-clear flags, death, combo/B2B, piece count, cumulative sent count, frame and last-tank time, dynamic garbage multiplier/cap, garbage RNG state, hole-change state, target context and the ordered incoming packet list. Packet identity includes amount, hole, width, readiness, hardening, shielding and packet ID. This is intentionally more conservative than merging equal-looking bitboards.

The map stores and compares complete key bytes, so a collision of its internal hash does not merge unequal states. Rules are immutable for a searcher's lifetime. Changing hold permission, deduplication mode or assumed elapsed frames clears the graph because those settings change edge generation or transitions. The evaluator and its parameters must also remain unchanged for the searcher's lifetime.

An exact repeated root can reuse generated edges and all retained accumulators. Re-rooting to an exact cached descendant keeps reachable descendants, compacts their IDs, and drops unrelated states. Unexpected garbage, timing or newly observed previews are safe cache misses. `cache_hits` counts reused child states/edges inside a search as well as retained work. `reused_nodes` counts nodes retained at call entry; it is the metric that distinguishes cross-call reuse. A cache-node limit is not an RSS guarantee: state vectors, action witnesses and allocator capacity have their own memory costs.

### Budget and deadline semantics

`max_nodes` is a hard cap on processed candidate edges in the current call, including cached edges. The initial root evaluation is additional and is counted in `evaluations`. Fresh child evaluations, geometry enumerations, deduplicated paths, beam pruning and cutoffs have separate counters. Repeating a cached query still consumes traversal budget, so cached and fresh searches obey the same cap.

The deadline is cooperative, checked between candidates. Geometry enumeration, one child transition/evaluation, and root setup are not preemptible. The first legal root fallback is allowed even if root setup already exceeded the deadline. Consequently, 50 ms is an allocation of work, not a strict real-time guarantee. `time_budget_ms=0` disables this deadline for deterministic node-budget comparisons. The cached-node limit stops growth independently and still preserves an already discovered root fallback.

## NNUE integration

`BotEvaluator` uses the existing native network and keeps its accumulators in the search node. New ordinary placements receive a sparse incremental update. With the default `RefreshEvents` policy, a line-clear or garbage event takes the existing full-refresh path. `RowSeams` is exposed as an alternative for experiments that measure when event-specific seam updates are preferable. A full refresh is used for new observed roots.

The trained network's piece input is the next piece to place, which is `State::current` at a post-lock state. It is not `queue.front()`. At a depleted visible horizon, the adapter falls back to the board heuristic instead of inventing a future piece or passing an untrained empty context. The included attention model was trained on earlier surrogate labels, not on this battle environment or measured match wins. The default hybrid uses 25% NNUE board value, 75% board heuristic and explicit hand-set battle-context terms. The adapter also gives configurable rewards for sent/cancelled attack, cleared lines and perfect clears.

## Verification and reproducibility

`search_tests.cpp` contains independent exhaustive recursion that does not call the searcher, memoize states or reuse its accumulator code. It compares exact values and selected root actions against wide, uncapped two-ply searches with and without hold, and a three-ply search. Additional tests cover empty/occupied hold consumption, depleted previews, a one-edge fallback, interrupted-layer selection, deadline fallback, cache limits, exact repetition/re-rooting, preview invalidation, time-option invalidation, deduplication equivalence, official cancellation, no-legal/dead roots, and 31 mutations of battle-state key fields. The test evaluator also checks that a child's accumulator is based on its actual parent's board. The recorded result is `tests/search_results.json`.

`gameplay_bench.cpp` is a complete offline placement-game smoke test. Each of three policies plays eight common seeded 7-bag streams for up to 120 locks: greedy heuristic, depth-3/beam-16 heuristic, and the same search with the seed918 attention hybrid. All policies have a 2,000-edge cap and no wall-clock deadline. Four games are clean. Four receive two ready rows with a known hole before locks 12, 24, ..., 120, using an independent precomputed garbage schedule. The schedule and hidden queue suffix belong to the referee and are never passed to search.

Only five previews are visible. After each actual lock, the referee verifies hold consumption and refills the observation window. A fixed 30-frame elapsed-time value per lock supplies an explicit 2 PPS clock; this does not simulate movement or gravity. Every selected action is matched against the generated legal set and applied through the battle environment. The runner checks piece accounting, frame advancement, incoming-row conservation and the hard search cap. Results include per-game trajectory checksums, actual completed-depth distributions, attacks, cancellations, received garbage, holds, spins, cache counts and elapsed times in `tests/gameplay_report.json`.

These are capped offline games with no opponent. Their survival and attack totals are evidence of functioning multi-ply gameplay and integration, not competitive strength. The eight seeds are too small for policy-strength claims, and the shared-host timings are not isolated kernel benchmarks. Full-state preview invalidation also means little or no cross-lock graph reuse is expected in this particular harness; within-call DAG reuse is reported separately.

### Recorded smoke-run result

All 24 games reached the 120-lock cap alive. Across 2,880 selected actions, the legal-action, queue/hold, frame, incoming-row accounting and hard-budget checks passed. The recorded aggregate results are:

| Policy | Locks | Lines | Sent | Cancelled | Received | Mean search ms |
|---|---:|---:|---:|---:|---:|---:|
| Greedy heuristic | 960 | 431 | 90 | 4 | 73 | 0.333 |
| Depth-3 beam, heuristic | 960 | 430 | 202 | 11 | 64 | 11.364 |
| Depth-3 beam, 25% attention NNUE hybrid | 960 | 425 | 183 | 16 | 62 | 12.921 |

Both beam policies completed three layers on 949 of 960 decisions. Eleven decisions per policy returned the last completed two-layer result after reaching the 2,000-edge cap. The heuristic beam recorded 108,368 cached-child hits; the hybrid recorded 109,920. Both had zero nodes retained across real locks because the referee revealed a new preview each time. This is consistent with the conservative cache contract above.

The hybrid sent fewer rows than the heuristic beam on this fixture, while cancelling more incoming rows. This does not justify promoting the surrogate-trained NNUE weights as a stronger battle policy. The current result supports keeping the heuristic search as a measured baseline and evaluating newly trained battle-aware models against it. Model inference time is only part of the reported end-to-end search time; these figures cannot be used as NNUE kernel speed comparisons.

From the repository root, the standalone commands are:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tetrio-bot/engine.cpp tetrio-bot/geometry.cpp tetrio-bot/battle.cpp tetrio-bot/search.cpp tetrio-bot/search_tests.cpp -o tetrio-bot/build/search_tests
tetrio-bot/build/search_tests > tetrio-bot/tests/search_results.json
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic tetrio-bot/engine.cpp tetrio-bot/geometry.cpp tetrio-bot/battle.cpp tetrio-bot/search.cpp tetrio-bot/evaluator.cpp tetrio-bot/gameplay_bench.cpp -o tetrio-bot/build/gameplay_bench
tetrio-bot/build/gameplay_bench experiments/spatial-nnue/results/final/seed918/attention.snn > tetrio-bot/tests/gameplay_report.json
```
