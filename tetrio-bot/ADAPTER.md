# Triangle integration boundary

This process accepts a reconciled battle snapshot and returns a **target
placement** with its spin class and geometric witness. It does not execute
keys, simulate gravity, or connect to TETR.IO. The transport is newline-delimited
JSON, one request and reply per line. `info` reports the supported commands and
coordinate contract; see `protocol.cpp` for the validated schema.

The reference wrapper is pinned to
[halp1/triangle aa0698ab8b4299b85c5f87e9b54793e1b7d9d380](https://github.com/halp1/triangle/tree/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380).
The native rule authority is the supplied TETR.IO Beta 1.7.8 / engine 19 client.
Triangle provides useful transport and snapshot fields, but its engine is not
assumed to be an exact rule oracle. In particular, the stock wrapper's compact
state message loses information needed for exact garbage and hold state.

## Board and piece coordinates

Triangle's [`Board.state`](https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/engine/board/index.ts)
is already **bottom-first**. Its line-clear code removes an array row and pushes
an empty row at the end; garbage is inserted at the beginning. Do **not** reverse
these rows. Read all `board.fullHeight` rows, not just the twenty visible rows.
For this port, require `board.width === 10`, `board.height === 20`, and
`board.fullHeight === 40` for the Tetra League preset.

By contrast, the supplied official client's `gsm.board` is top-first. A direct
official snapshot needs `native_y = 39 - official_row_index`. This conversion
must not also be applied to a Triangle snapshot.

| Source tile | Native occupancy/provenance |
|---|---|
| `null` | Clear the corresponding bit |
| Triangle `i/o/t/l/j/s/z` | Set `board.rows[y]` bit `x` |
| Triangle `gb` | Set both `board.rows[y]` and `board.garbage[y]` |
| Official `gbd` | Set `board.rows[y]` and `board.solid[y]`; preserve its permanent-material meaning |
| Triangle `bomb` or another unsupported tile | Reject this room/snapshot for this port; do not silently convert it to a normal mino or permanent garbage |

Triangle stores a tile as `{mino, connections}`; the stock adapter projects
`tile?.mino ?? null`. Connection bits are rendering metadata and are not needed.
The pinned [`Mino` enum](https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/engine/queue/types.ts)
has `gb` and `bomb`, but no `gbd` value. An official `gbd` mapping therefore must
come from an official-format or explicitly extended snapshot, not an invented
Triangle enum member.

The native protocol accepts either case for standard piece names and emits
uppercase `I/O/T/L/J/S/Z`. Send `null` for an empty hold. Current-piece `null`
means an unknown search horizon, not an extra bag draw. Queue entries must be
known standard tetrominoes; an empty queue expresses unknown previews.

Native action `(x,y,rotation)` uses the same matrix anchor as Triangle's
[`Tetromino`](https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/engine/utils/tetromino/index.ts):

```text
native x        = triangle.falling.location[0]
native y        = floor(triangle.falling.location[1])
native rotation = triangle.falling.rotation
occupied cell   = (x + shape_x, y - shape_y)
```

Rotation indices are `0,1,2,3`; native witnesses name CW, CCW and 180 actions
separately. O uses its two-column matrix, while I uses its four-column matrix.
Do not subtract an additional pivot offset when handing a native target to a
planner that already expects Triangle's matrix-anchor coordinates.

## Snapshot timing, hold and previews

Submit a snapshot at the new-piece decision boundary, with any previous lock,
clear, garbage change, and queue update already reconciled. The native search
starts a fresh piece at spawn, including clutch when the previous lock cleared.
It does not accept an arbitrary partly moved falling pose as a search root.

The following fields are directly present in the pinned
[`Engine`](https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/engine/index.ts).
They must all describe the same decision boundary.

| Native field | Triangle source or reconciliation rule |
|---|---|
| `current` | `engine.falling.symbol` |
| `hold` | `engine.held` |
| `hold_used` | `engine.holdLocked` |
| `last_clear` | `engine.lastWasClear` |
| `queue` | The actually observed prefix of `engine.queue.raw()`, excluding current |
| `pieces_placed` | `engine.stats.pieces` after the previous lock completes |
| `frame` | `engine.frame` at the chosen integer-frame boundary |
| `current_multiplier` | `engine.dynamic.garbageMultiplier.get()` |
| `current_garbage_cap` | `engine.dynamic.garbageCap.get()` |
| `dead` | Authoritative terminal state or the resolved lock's top-out result; no `engine.dead` property is assumed |

[`Queue.raw()`](https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/engine/queue/index.ts)
returns the entire remaining internal array. It is not a guarantee that every
entry is visible to the player. Limit the native queue to the observed preview
count, normally five for this preset. The native parser rejects a longer queue.
Do not fill an unknown horizon from the queue seed or bag snapshot.

The stock wrapper listens for `queue.add`, but those notifications concern
internal replenishment. A custom bridge should maintain the visible prefix at
each decision boundary rather than append every internally generated piece to
the search horizon. Empty-hold play consumes the first observed queue piece;
an existing hold swaps the current piece without this extra queue consumption.

## Counter conventions and rule authority

Triangle initializes both `stats.combo` and `stats.b2b` to `-1`. This port follows
the supplied official client's nonnegative counter state: reset is zero and
the first corresponding clear increments to one. At an ordinary aligned
snapshot boundary, convert with:

```text
native.combo = triangle.stats.combo + 1
native.b2b   = triangle.stats.b2b + 1
```

This is a representation conversion, not a promise that Triangle's subsequent
rule evolution always matches the official client. Spin modes, perfect-clear
options, and surge behavior must use the official rule configuration and be
resynchronized when the transport's predicted counters disagree. The actual
counter state affects the next move's combo/B2B reward; never clamp both
Triangle `-1` and `0` to native zero.

The native `total_sent` is the official-style cumulative outgoing amount used
by the opener-phase rule. Triangle exposes both `stats.garbage.sent` and
`garbageQueue.sent`. They serve different callers in the wrapper. Feed the
reconciled quantity expected by the official battle model, and do not replace
it with APM, raw attack before cancellation, or total incoming garbage.

Set the actual room rules explicitly when departing from Tetra League. The
stock adapter configuration is not a complete copy of the official room
options. In particular, the pinned
[`configFromEngine`](https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/utils/adapters/core/index.ts)
contains a concrete field error: it fills `garbageCap` from the **multiplier**
getter. Its `playFromEngine` uses the correct cap getter. Read the live
`engine.dynamic.garbageCap.get()` or the correctly sourced play field; do not
forward the configuration's `garbageCap` blindly. Preserve fractional cap state
when using the live getter; the play message floors it.

## Packet confirmation and readiness stay outside the search engine

The stock
[`stateFromEngine`](https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/utils/adapters/core/index.ts)
reduces incoming garbage to `number[]` amounts. That loses packet identities,
confirmation, timing, width and readiness. It also omits hold lock, last clear,
dynamic state and RNG history. It is insufficient for exact native battle
reconstruction without additional bridge data.

The pinned
[`GarbageQueue`](https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/engine/garbage/index.ts)
keeps incoming fields `frame`, `amount`, `size`, `cid`, `gameid`, and
`confirmed`. The engine's IGE handler first reconciles interaction amounts,
including network passthrough and `ackiid`, and later applies an
`interaction_confirm` to the matching `(cid,gameid)` packet. Keep that network
reconciliation in Triangle/the transport layer. Do not send raw IGE messages
straight to native `enqueue`.

The native packet boundary is explicit:

| Native packet field | Bridge responsibility |
|---|---|
| `amount` | Remaining amount after authoritative reconciliation and cancellation |
| `width` | Packet hole width; Triangle calls this `size` |
| `id` | Stable local numeric identity; retain a mapping to `(gameid,cid)` when compound identities are needed |
| `active` | Resolved activation status from the official-compatible transport model |
| `spawn_ready` | Resolved eligibility to spawn at this decision boundary |
| `hardened`, `shielded` | Official packet protection flags when the room uses them |
| `hole` | A known fixed hole, or `null` for the model's local hole RNG; incoming Triangle packets do not already contain a `column` |

`confirmed` alone is not an exact substitute for both `active` and
`spawn_ready`. Triangle's own `tank` implementation contains a TODO about its
difference from the real active system. Do not hide that gap behind a guessed
boolean conversion. Resolve readiness in the transport boundary, or supply an
explicit planning scenario and rehydrate after the authoritative event.

The native `advance` command advances frame-based dynamic values; it does not
receive ACKs or turn packets active merely because time passed. When an
existing packet is confirmed or otherwise changes readiness, send an updated
full `state` snapshot. Do not `enqueue` the same packet again as if it were a
new arrival. The bridge must deduplicate network events before enqueueing.

Distinguish the two ingestion paths. `state.incoming` is a complete snapshot of
the **remaining local queue**, after receive multiplication, caps and any
previous cancellations. The `state` parser preserves those amounts. `enqueue`
accepts a **new** network-reconciled arrival before the local receive multiplier
and absolute cap, and applies those two settings itself. Sending an already
multiplied local queue entry through `enqueue` would multiply it twice. Neither
path performs the network ACK/passthrough reconciliation.

For deterministic hole prediction, preserve the actual RNG state and history
represented by `rng_seed`, `last_hole`, `hole_changed`, and `last_tank`.
Triangle's garbage snapshot has related `seed`, `lastColumn`,
`hasChangedColumn`, and `lastTankTime` fields, but copying names does not prove
identical draw/event order. A fresh default seed is suitable for a controlled
offline scenario, not evidence that a live opponent's future holes are known.
When authoritative holes arrive, update the board/provenance and decision
snapshot. Permanent board material (`solid`) is distinct from a packet's
`hardened` cancellation protection.

## Placement output versus the stock key wrapper

[`BotWrapper`](https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/utils/bot-wrapper/index.ts)
expects `adapter.play()` to return `keys`, then converts those keys into frame
events. It explicitly requires ARR 0 and SDF 41. The new native protocol is a
target-placement protocol, so it is **not** a drop-in implementation of that
key-returning interface.

Use the returned `piece`, `use_hold`, `x`, `y`, `rotation`, `spin`, and occupied
`cells` as the target for an existing placement-to-input executor. Preserve spin
provenance while selecting its route. `path` is a geometric proof, not that
executor's timed key list: mapping every native `down` to a Triangle
`softDrop` under SDF 41 would change the route. Frame controls are deliberately
outside this port, as requested.

The public `apply` command resolves the requested action against the current
generated legal action set and uses the generated spin/rotation metadata. It
does not trust user-supplied `last_rotation`, kick index, path, or cell lists as
proof of legality. `apply` advances the native simulation; it does not submit a
move to TETR.IO. After the actual executor locks the piece, reconcile against
the real board/counters before reusing a predicted descendant as authoritative.

## Minimal protocol exchange

This is a valid empty-board offline scenario. A live bridge replaces every
field with its reconciled state and supplies pending packets when present.

```json
{"cmd":"configure","rules":{"preset":"tetra_league"},"search":{"max_depth":4,"beam_width":64,"max_nodes":5000,"time_budget_ms":20},"evaluation":{"mode":"heuristic"}}
{"cmd":"state","state":{"board":{"rows":[],"garbage":[],"solid":[]},"current":"T","hold":null,"queue":["I","O","S","Z","L"],"hold_used":false,"last_clear":false,"combo":0,"b2b":0,"pieces_placed":0,"total_sent":0,"frame":0,"current_multiplier":1,"current_garbage_cap":8,"incoming":[]}}
{"cmd":"search"}
```

The reply contains the selected `action`, a principal variation, candidate
values, search statistics and termination reason. Unknown previews remain an
unknown horizon. `state` can replace the complete snapshot after an observed
event, while `clear_cache` explicitly discards reusable search state when the
bridge needs a clean restart.
