# JSONL protocol v1

The executable reads one JSON object per nonempty line and writes one JSON
reply per request. Successful replies have `ok:true`; errors have `ok:false`
and `error`. A supplied `id` is echoed. Invalid JSON or an invalid request does
not terminate the process. Unknown fields and unsupported options are rejected.
Startup/model errors go to stderr and return a nonzero exit status.

## Commands

| `cmd` | Additional request fields | Behavior |
|---|---|---|
| `info` | none | Current rules/options, evaluator, version and coordinate contract |
| `configure` | optional `rules`, `search`, `evaluation`, `model` | Partial options update; validates/loads before replacing evaluator/search cache |
| `state` | optional complete `state` | Replace snapshot when supplied; return current snapshot |
| `actions` | optional complete `state`, `allow_hold` | Return all generated legal target placements |
| `search` | optional complete `state`, per-call `search` options | Return action/PV/candidates/stats; does not lock a piece |
| `apply` | `action` | Match to current legal candidates, simulate the lock, return state/events/delta |
| `enqueue` | `packet` | Receive one new network-reconciled arrival, apply receive multiplier and cap |
| `advance` | integer `frames` | Advance mutable frame-based values; no movement, ACK or readiness simulation |
| `clear_cache` | none | Discard the retained search graph |

`state` replacement is complete, not a field patch. Omitted fields take the
documented defaults from `engine.hpp`; send the full returned snapshot when
updating only packet readiness or adding a newly observed preview. A new-piece
decision boundary is required. A partially moved falling pose is not modeled.

## State

`current` is required: a standard piece letter or `null` for an unknown horizon.
All other fields are optional for constructing simple offline fixtures.
Production snapshots should supply the authoritative modeled state.

| Fields | Representation |
|---|---|
| `board.rows`, `.garbage`, `.solid` | Arrays of up to 40 masks, 0..1023, bottom-first; padded with zeros |
| `current`, `hold` | `I/O/T/L/J/S/Z` (either case); `null` allowed |
| `queue` | Known piece letters, excludes current, at most `rules.preview_count` |
| `hold_used`, `last_clear`, `dead` | Booleans |
| `combo`, `b2b`, `pieces_placed` | Nonnegative official counter state; reset combo/B2B is 0 |
| `total_sent` | Cumulative outgoing amount used by opener cancellation, not raw attack/APM |
| `frame`, `last_tank` | Nonnegative integer frames; `last_tank <= frame` |
| `current_multiplier`, `current_garbage_cap` | Mutable current values, distinct from rule bases |
| `rng_seed` | Park–Miller state in 1..2147483646; offline default 1 |
| `last_hole`, `hole_changed` | Last column (-1 or null for none); changed requires a known column |
| `enemies`, `garbage_bonus` | Target-bonus context |
| `incoming` | Ordered remaining local packets, with receive effects already applied |

`garbage` and `solid` must be subsets of occupancy. They distinguish clearable
garbage from permanent material. A packet's `hardened` flag is not solid material.
Rows are whole-row identities for clear compaction; columns are bits with x=0
at the left edge. All forty rows participate in state identity and simulation.

At frame zero, omitted mutable rates initialize to the configured rule bases.
If an increasing rate has already passed its margin, a snapshot without its
current value is rejected. The engine cannot reconstruct unknown earlier room
changes from a frame number. Supplying current values on every live snapshot is
the unambiguous contract. `advance` and action `elapsed_frames` perform repeated
per-frame additions, preserving the official order.

Example packet:

```json
{"amount":4,"hole":null,"width":1,"active":true,"spawn_ready":false,"hardened":false,"shielded":false,"id":12}
```

`amount` is required and positive. `hole:null` uses the supplied local RNG;
a fixed 0..9 column is accepted when known. Width is 1..10. The default booleans
are active/ready true, hardened/shielded false, suitable for an explicit offline
ready-packet fixture. A live bridge must supply real status. The numeric ID is a
stable local mapping for the transport identity, not a guessed remote sequence.

**Snapshot versus arrival:** `state.incoming` is copied without applying receive
multiplication again. `enqueue.packet` is a new arrival after network ACK
reconciliation but before local receive multiplier/cap. Confirmation changes
replace the snapshot; do not enqueue the same packet a second time. Neither
command implements raw network IGE/ACK/passthrough processing.

## Action and output

Actions contain `piece`, `x`, `y`, `rotation`, `spin`, `use_hold`,
`elapsed_frames`, `last_rotation`, `rotated180`, `kick_index`, `path`, and `cells`.
Coordinates are matrix top-left anchors: occupied cells are `(x+shape_x,y-shape_y)`.
Rotation is 0..3. Spin is `none`, `mini`, or `full`.

To apply an action, send the action returned by `search` or `actions`:

```json
{"cmd":"apply","action":{"piece":"O","x":0,"y":1,"rotation":0,"spin":"none","use_hold":false}}
```

This particular target is valid only in a matching current/hold/board state.
`apply` requires piece/x/y/rotation; missing spin, hold and elapsed frames default
to none/false/0. Supplied path, cell list and rotation metadata are descriptive
only: the engine replaces them with the matching generated action. A spin class
must itself match a legal candidate. There is no trusted-spin bypass.

`path` uses `left/right/down/cw/ccw/180`. It is a geometric witness, not executable
frame timing. `elapsed_frames` is externally supplied time to advance before
the lock; it is not derived from path length. The maximum per request is 216,000.

The `apply` reply includes `events` (lines, garbage cleared, spin, PC, attack,
sent, cancelled, received, surge and ordered outgoing packet sizes), complete
`state`, and `delta`. `delta.placed_cells` stores `y*10+x` before row events.
`delta.row_source[y]` is the original parent row for child row y, or -1 for a new
row. `clear_mask`, `clears`, and actual inserted `garbage` are also returned.

## Search and evaluation settings

`info` returns every supported field and current value. Common search settings:

| Field | Default | Meaning |
|---|---:|---|
| `max_depth` | 4 | Maximum locked-piece horizon, also limited by observations |
| `beam_width` | 64 | Retained frontier paths after each completed layer |
| `max_nodes` | 12000 | Processed candidate-edge cap, including cached edges |
| `max_cache_nodes` | 32768 | Retained graph-node cap, not an RSS guarantee |
| `time_budget_ms` | 50 | Cooperative deadline; 0 disables it |
| `allow_hold` | true | Include hold actions |
| `deduplicate` | true | Merge equivalent states/paths |
| `frames_per_piece` | 0 | External elapsed-time assumption per searched lock |

The result has `action` (or null), scalar `value`, `pv`, `candidates`, `stats`,
`termination`, and the effective evaluator name. Candidate `depth` values can
differ after beam pruning; do not rank those numbers as if every candidate was
evaluated to the same horizon. The selected move follows the completed-layer
contract in [SEARCH.md](SEARCH.md).

`stats.reused_nodes` measures cross-call retained nodes. `cache_hits` also counts
sharing within the current search. The default visible-preview replenishment
normally changes the root key and starts a new graph. A deadline is cooperative
and permits a legal root fallback even after root setup exceeded the budget.

Evaluation modes are `heuristic`, `nnue`, `hybrid`; updates are `full`,
`refresh_events`, `row_seams`. Hybrid's `nnue_weight` defaults to 0.25. Rewards
default to `sent_reward:4`, `cancelled_reward:3`, `line_reward:0.5`, `pc_reward:2`.
These weights and battle-context terms are hand set. A missing model means
heuristic fallback for hybrid; explicit `nnue` requires a model. Unknown current
uses the board heuristic in every mode. `configure.model` loads a local `.snn`
path once; null removes the model.

Rules use native snake_case names. The named `preset:"tetra_league"` resets to
the pinned default before applying overrides. Other `configure.rules` objects
patch current rules. All supported rule fields are reported by `info`; see
[rules audit](tests/rules/RULES_AUDIT.md) for their official source semantics.
Changing configuration clears the search cache and preserves any valid stored
snapshot, including its explicit dynamic values.
