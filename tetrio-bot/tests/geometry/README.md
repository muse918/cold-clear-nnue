# TETR.IO placement geometry audit

The geometry port is checked against the user-supplied **TETR.IO Beta 1.7.8,
engine 19** client, SHA-256
`6b682a1226dd2ae66ce19a08e944df92e7f175bccaa48cca7269bcd78f39a27a`.
The client itself is not included. `official_geometry_oracle.mjs` reads it at
runtime, checks the hash, and isolates its actual board, `ComputeKick`,
`IsTSpin`, and `ConsiderBlockout` methods. It does not initialize the client,
make network requests, access an account, or launch a renderer.

The bot wrapper reference is pinned to
[Triangle aa0698ab8b4299b85c5f87e9b54793e1b7d9d380](https://github.com/halp1/triangle/tree/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380).
The official client is the authority when its behavior differs from this wrapper.

## Geometry contract

`Board::rows[0]` is the bottom row. There are ten columns and forty rows,
including the hidden buffer. `Pose{x,y,rotation}` is the **top-left matrix
anchor**, with a cell located at `(x+shape_x, y-shape_y)`. For a normal twenty-row
visible board, the initial anchor is `(3,22,0)`, except O uses `(4,22,0)`.
All standard tetrominoes have `dy=1`; O has `dx=0`, and the others `dx=1`.

To compare a native pose with the official pivot coordinates:

```text
official_x = anchor_x + matrix.dx
official_y = 39 + matrix.dy - anchor_y - 0.04
```

`collides` checks integer board occupancy. `rotate_pose`, spin classification,
spawn and generated placements conservatively use the ordinary noninteger
falling-y phase represented by the formula above. The official collision
method rejects a negative fractional y before taking its ceiling. Consequently,
the topmost hidden row can be integer-valid but invalid at this phase. The
generator excludes such frame-dependent ceiling placements; it does not claim
to model every possible frame fraction. The board still retains all forty rows
for occupancy, top-out, garbage and row-event handling.

SRS and SRS+ are separate selectable tables. A zero-displacement rotation is
tested first; an accepted displaced kick reports its zero-based index in the
official kick list. `kick_index=-1` means the zero-displacement trial. SRS+ I
kick ordering is different from SRS, and SRS+ has special I 180 kicks. O can
rotate its orientation but cannot use displaced kicks in these two systems.

Clutch is considered only when the previous lock cleared a line and clutch is
enabled. A blocked spawn is raised until it is legal, subject to the same board
ceiling. A blocked spawn otherwise produces no placements.

## Reachability and spin provenance

The generator traverses legal translations, down steps, and CW/CCW/180 rotations
from spawn. It emits grounded placements and retains separate `None`, `Mini`,
and `Full` spin states at an identical pose. The emitted set is deduplicated by
the four occupied cells **and spin class**, so two equal footprints with
different attack consequences remain separate actions. BFS expansion and path
selection have deterministic ordering.

Each rotation computes spin immediately on the unchanged parent board. A
successful translation or down step clears its rotation/spin provenance. This
matches the relevant official positional-action behavior. In particular, a
rotation above a slot followed by a drop does not automatically qualify as a
spin just because the final footprint has three occupied corners. An official
hard drop preserves rotation flags, but a spin detected before that drop was
already grounded, so a positive-distance drop cannot carry a previously
detected spin into a new landing position.

`Placement::path` is a geometric witness using `Left`, `Right`, `Down`, `CW`,
`CCW`, and `Half`. It is not a timing-feasible key script. Gravity, DAS/ARR,
soft-drop speed, lock delay, lock-reset budgets, and rotation-reset timing are
outside this port's scope. The caller may use the target placement with its bot
adapter or verify the witness with `rotate_pose`.

`classify_spin` assumes a successful rotation has just happened at the supplied
pose; callers must not infer this from an arbitrary locked board. Public user
actions should be matched against generated placements before applying them.

## Spin modes

The following rules are implemented from the official `IsTSpin` switch.
“Corner rule” means grounded after rotation, at least three occupied corners,
with T minis determined by the two front corners. A zero-based kick index of
3 upgrades a corner mini to full, including 180-degree rotations.

| Mode | T | Other standard pieces |
|---|---|---|
| `none` | No spin | No spin |
| `T-spins` | Corner rule | No spin |
| `T-spins+` | Corner rule, then immobility fallback as mini | No spin |
| `all` | Corner rule only | Four-direction immobility gives full |
| `all+` | Corner rule, then immobility fallback as mini | Four-direction immobility gives full |
| `all-mini` | Corner rule only | Four-direction immobility gives mini |
| `all-mini+` | Corner rule, then immobility fallback as mini | Four-direction immobility gives mini |
| `mini-only` | Corner success is forced to mini; immobility fallback also mini | Four-direction immobility gives mini |
| `handheld` | T corner rule | S/Z/L/J use their own corner tables; I/O receive none |
| `stupid` | Grounded after rotation gives full | Grounded after rotation gives full |

Three concrete differences from the pinned Triangle wrapper were found in its
[`src/engine/index.ts`](https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/engine/index.ts):

1. Triangle's `all` and `all-mini` permit a T immobility fallback after the
   corner rule fails. Official engine 19 returns the failed corner result for
   T in these two non-plus modes.
2. Triangle promotes T minis using particular quarter-turn kick vectors.
   Official engine 19 checks the kick-list index, including 180 rotations.
3. Triangle's initial T-corner mode list omits `mini-only`. Official engine 19
   applies the T corner rule and forces a successful result to mini.

The native port follows the official client in these cases. The oracle covers
the rule distinctions and all standard pieces; it does not import Triangle's
spin detector as a substitute reference.

## Reproducing the checks

From `tetrio-bot`, an isolated build needs only C++17 and the checked-in JSON
header:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -DTETRIO_GEOMETRY_STANDALONE geometry.cpp geometry_tests.cpp \
  -o build/geometry_tests
build/geometry_tests
node tests/geometry/verify_official_geometry.mjs \
  /path/to/the/supplied/tetrio.js build/geometry_tests \
  tests/geometry/official_geometry_report.json
```

In the integrated build, link the normal board implementation and omit
`TETRIO_GEOMETRY_STANDALONE`. The macro supplies only an isolated board occupancy
method for this test target; production code does not use it.

The recorded run in `official_geometry_report.json` passed with **zero
mismatches**:

| Check | Count |
|---|---:|
| Shape/rotation definitions | 28 |
| Integer collision probes, including walls/floor/ceiling | 3,592 |
| Ordered rotation/kick probes | 5,822 |
| Spin classification probes across all ten modes | 18,200 |
| Actual official clutch/blockout probes | 560 |
| Complete reachable placement-set comparisons | 84 |
| Total primitive and placement-set cases | 28,286 |
| Emitted paths replayed through official primitives | 2,245 |
| Additional native regression paths replayed | 4,864 |

Random tests use a fixed seed and include dense local cavities as well as
column-shaped stacks. The report retains per-piece kick-index coverage and
per-mode spin-class counts. Passing these checks establishes agreement with
the pinned primitives and the stated geometric graph; it is not a claim of
complete live-game or frame-timing equivalence.
