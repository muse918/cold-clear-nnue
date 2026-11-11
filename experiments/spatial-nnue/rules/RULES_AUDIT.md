# TETR.IO row-transition and Triangle adapter audit

## Pinned sources

- User-supplied `upload/tetrio.js`, SHA-256 `6b682a1226dd2ae66ce19a08e944df92e7f175bccaa48cca7269bcd78f39a27a`.
- Uploaded client identifies Beta 1.7.8, client commit `e84ebccd2`, build `20261109T191625`, engine version 19.
- Triangle commit `aa0698ab8b4299b85c5f87e9b54793e1b7d9d380`, package `@haelp/teto` 4.2.8. README's last-tested Beta 1.7.8 matches the upload.
- Cold Clear NNUE inspected at `31713b0bd20c0bddf9dfa11daacdbd2d67888ae4`.

## Confirmed uploaded-client rules

The uploaded client is minified into four lines, so search tokens and the file hash are more useful locators than line numbers.

1. `const y=19,x=11,A=new Map,v=function`: engine version 19 and default dimensions `W:10,H:20,B:20,T:40`. `OptionsList` defaults also give width 10, visible height 20, buffer 20; dimensions are configurable.
2. `F=function(e)` is the board factory. The official board is **top-first**. `GetFullLines:function` returns rows whose cells are non-null and are not `gbd` (permanent/bomb garbage).
3. `RemoveLinesFromStack:function` shallow-copies the row arrays, sorts full row indices descending, splices whole rows, and prepends empty rows. Surviving row objects are reused. It also updates rendering-edge metadata and actor rows; those metadata are irrelevant for occupancy-only NNUE but not necessarily for full rendering fidelity.
4. `PushLine` (`g` inside the board factory) discards the top row and inserts a garbage row above the configured un-clearable/permanent floor. With ordinary garbage and no permanent blocks, it inserts at the bottom. It refuses an insertion if the very top row is entirely occupied. The normal-game specialized fast path must not silently replace custom modes with permanent/bomb garbage.
5. `ClearLines:function` first finds full rows and records garbage provenance/downstack, then removes rows, computes all-clear, attacks and cancellation, and conditionally tanks garbage. A lock can therefore contain both a clear event and a later garbage-rise event, depending on rule settings.
6. The `tetra league` preset in this uploaded file explicitly sets `7-bag`, `all-mini+`, `SRS+`, 180-degree rotation, five previews, ARE=0, lineclear ARE=0, combo multiplier, B2B charging, all-clear garbage=5, special garbage bonus enabled, openerphase=14, and no lockout. Generic option defaults differ from this preset; do not use generic defaults as a League ruleset.
7. Garbage bonus code includes `garbagespecialbonus && garbageRowsCleared && (lines===4 || spin)` before the attack cap. Combo/B2B/surge, garbage provenance, and action spin history are distinct from the binary board input.

## Verified isolated differential checks

Run with Node 24:

```bash
node experiments/spatial-nnue/rules/verify_row_events.mjs /absolute/path/to/tetrio.js
```

The harness **does not execute the browser client**. It extracts the exact board-factory function and invokes selected row methods in an isolated VM with rendering/actor/timer calls replaced by no-op stubs. Triangle's downloaded Board source is transpiled using Node's built-in TypeScript transformer; no npm installation or online account is used.

- 1,000 deterministic arbitrary-board line-clear cases matched official vs Triangle after converting top-first/bottom-first coordinates. Checks include garbage-cleared counts and non-clearable cells (`gbd` in the client, `bomb` in Triangle).
- Independently of Triangle, all 1,000 official clear results matched stable filtering of deleted rows plus empty top padding; all surviving row objects retained identity.
- 1,000 deterministic arbitrary-board ordinary-garbage-rise cases found 751 mismatches in the pinned Triangle Board implementation. This is a deliberately broad primitive correctness test with occupied buffer rows; it is **not** an estimate of real-match bug frequency.
- The pinned Triangle source truncates after insertion using `splice(fullHeight - amount - 1, amount)`. A local experimental replacement with `splice(fullHeight, amount)` matched the official ordinary-garbage transition in all 1,000 cases.
- The discrepancy is a buffer/boundary issue, not a claim that ordinary low boards are broken. All 168 tests with stack heights 0..20 and garbage amounts 1..8 matched the unmodified wrapper. Minimal mismatch at cap 8: a marker in bottom-first row 23 should move to row 31, but the wrapper deletes it. At cap 1, row 37 should move to 38 but is deleted; a marker at top row 39 should be discarded, but the wrapper retains it.
- No complete-game, replay, input-reachability, attack, cancellation, timing, or spin equivalence claim is made. The local correction does not claim support for permanent/bomb or special Zenith rules.
- Complete machine-readable results are in `row_event_results.json`.

`official_row_oracle.mjs` is a reusable dependency-free JSONL oracle that takes the uploaded client as an external input. It exports `loadOfficialRowOracle(path)` and supports 40 bottom-first bitmask rows, optional garbage provenance bitmasks, prevalidated added cells, and explicit ordinary-garbage holes. It does not claim to validate movement reachability, spin, cancellation, or timing. A vertical-I quad on four 9-cell rows was smoke-tested to produce an empty board.

```bash
node experiments/spatial-nnue/rules/official_row_oracle.mjs /path/to/tetrio.js < events.jsonl
```

The extracted official board factory occupies UTF-16 character offsets `[830092, 842485)` and the official shape-table literal occupies `[751416, 754883)` in the uploaded file. The harness checks recognizable boundaries and requires the audited file hash; a different client needs re-audit before changing the pin. These are character offsets, not byte offsets; the original minified file contains Unicode. The uploaded client remains an external local input and is not copied into this repository. Every reported check uses width 10, visible height 20, buffer height 20, and total height 40.

## Generated dataset parity

Run from the repository root with the external uploaded client:

```bash
node experiments/spatial-nnue/rules/validate_dataset.mjs \
  experiments/spatial-nnue/data/train/records.bin \
  /absolute/path/to/tetrio.js \
  experiments/spatial-nnue/rules/dataset_parity.json --all
```

The deterministic default selects up to 10,000 evenly spaced records and adds every garbage event and every three-or-four-line clear. `--all` validates every record. The validator reads the packed 240-byte schema from `generate.cpp` (40 parent rows, 40 child rows, row-source mapping, four placed cell IDs, and event metadata). It derives ordinary garbage holes from the recorded child's inserted rows, because the binary schema does not store a separate hole field.

All **98,091** training records passed against the isolated official row primitives: **97,907 locks and 184 ordinary garbage events**, with zero failures. Child occupancy, line-clear count/mask, and all 98,091 row-source mappings matched. All 97,907 locks used a footprint found in the uploaded official shape table; every footprint had a collision-free vertical descent from the upper buffer in its fixed orientation and was grounded at the recorded placement. The saved full parity pass took 8.60 seconds on the experiment machine. Dataset SHA-256: `7ac56f13fb812998e62aa03c8a365d6978af3bfd6b255e36ba7480cc311f75ad`. Complete results are recorded in `dataset_parity.json`.

The clear histogram for these candidate lock records is `[95,703, 1,958, 213, 29, 4]` for zero through four clears: 2,204/97,907 (2.25%) candidate locks clear a line. This is a distribution over the generated search candidates, not a measured real-game clear frequency or the frequency along selected trajectories.

This check covers exact row primitives and fixed-orientation vertical-drop geometry. It does **not** establish an SRS/SRS+ input path from the actual spawn orientation, hold legality, rotation/spin classification, attack values, cancellation, timing, topout policy, or playing strength. The generator's separate synthetic garbage records also do not validate the official timing and cancellation process that decides when garbage rises. Full League compatibility therefore remains outside this experiment's validation scope.

### Heldout and adversarial boundary checks

The same official oracle also passed all **29,346 heldout records** (29,297 locks and 49 garbage rises), recorded in `heldout_dataset_parity.json`. Its dataset hash is `b803c509f30e117af9ba1ec9b68c90066ab40c0eed8cfc8be3b4edc95a6bc5d9`.

An additional **2,400 adversarial records** passed, recorded in `stress_parity.json`: 600 clear events, 600 clear-then-garbage events, 600 O locks in rows 38/39, and 600 garbage rises with occupied buffer rows. The 1,200 clearing records evenly cover one through four clears, including noncontiguous deleted rows. Combined events apply the official clear first and then ordinary garbage insertion; successive hole inputs are derived by reversing the final bottom-row sequence, so distinct holes per inserted row retain their order. These cases exercise terminal/high-buffer geometry and are **not** samples of gameplay frequency or model quality. The stress dataset hash is `fab2500f549754e8f475179597b76329b570deb24fcc82e91fa611b2d6d49ae8`.

Across training, heldout and stress files, **129,837 records** were checked without a mismatch. These extra cases extend boundary and combined-event coverage while retaining the row-primitive-only rule scope above.

## Vendored source notice

`triangle_board.ts` is the unmodified `src/engine/board/index.ts` at the pinned Triangle commit above. Its MIT license is retained in `TRIANGLE_LICENSE.md`. The harness applies its experimental truncation correction only to an in-memory copy, so the recorded upstream counterexample remains reproducible.

## Triangle adapter integration

Primary URLs (replace `main` with pinned commit for reproducibility):

- https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/utils/bot-wrapper/index.ts
- https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/utils/adapters/core/index.ts
- https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/utils/adapters/adapter-io.ts
- https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/utils/adapters/types/messages.ts
- https://github.com/halp1/triangle/blob/aa0698ab8b4299b85c5f87e9b54793e1b7d9d380/src/engine/board/index.ts

`AdapterIO` provides newline-delimited JSON over a child process's stdin/stdout. Outgoing messages are config, state, pieces, play; incoming messages are info and move with key sequences. `BotWrapper` requires ARR 0 and SDF 41, schedules actions by target PPS, converts DAS and drop keys to frame/subframe keydown+keyup events, and resynchronizes adapter state on garbage events.

Important integration observations:

- Triangle Board state is **bottom-first**, unlike the uploaded official client. Its tile types also differ (`bomb` versus `gbd`). Preserve coordinate/type conversion at the boundary.
- `configFromEngine` sets `garbageCap` from `engine.dynamic.garbageMultiplier.get()` in the pinned source. `playFromEngine` uses the garbage-cap tracker. The configuration value is a static source-level mismatch requiring correction or overriding.
- Default state messages contain incoming garbage **amounts only**, omitting packet frame/confirmed state and other timing metadata. Extend the custom `data` field or custom adapter for time-aware search.
- Default messages forward `engine.queue.raw()`. Enforce the declared observation contract (five visible previews for the audited League preset) rather than exposing longer seeded queues to the evaluator/teacher.
- Hold lock availability, opener counters, own sent amount, and detailed garbage timing are not all present in the default State message. Exact search must supply required metadata separately.
- The Triangle replay-test converter appears to swap width and height field names. Production `Game.createEngine` uses width/height in their expected order. Do not copy the replay helper without validating its configuration mapping.

## Existing Cold Clear scope

`libtetris/src/board.rs` stores B2B as a boolean, uses a fixed combo table and perfect-clear garbage=10. It contains no per-cell garbage provenance. Its movement/result types use T-spin status, not the uploaded all-mini+ ruleset with all-piece spin behavior. Its mechanics and evaluator test stream are useful for geometry/search experiments, but its attack and battle result cannot be labeled current TETR.IO League-equivalent without adaptation.

## Implication for NNUE experiments

The normal fast path should update only local cells/row encodings. A clear should carry a compact row map (`old row -> new row`, deleted rows, inserted empties), so row-local cached encodings move by identity. Any absolute-position component must be recomputed or adjusted; moving a cached row cannot make absolute-position weights invariant.

Treat clear frequency as a measured workload property. With no incoming garbage and stable occupancy, conservation gives roughly `10 * cleared_lines = 4 * placed_pieces`; with 1..4-line clears, clear-event frequency can be roughly 10..40%, even before garbage effects. The distribution over searched candidate edges can differ substantially from the distribution over chosen game moves. Profile both, and report synthetic frequency sweeps separately from empirical workloads.
