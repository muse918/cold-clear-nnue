# Cold Clear

Modern Tetris Versus bot.

## TETR.IO environment and NNUE search

The new [tetrio-bot](tetrio-bot/README.md) module is a standalone C++17
placement-level TETR.IO environment and deterministic beam/DAG searcher. It
supports the pinned Tetra League rules, configurable all-spin modes, battle
garbage state, a JSONL interface, and the existing native spatial NNUE kernels.
See its [protocol](tetrio-bot/PROTOCOL.md), [Triangle integration contract](tetrio-bot/ADAPTER.md),
and [source/experiment audit](tetrio-bot/SEARCH.md). Build with
`make -C tetrio-bot`; no frame controls or gravity simulation is included.

The earlier [spatial NNUE experiments](experiments/spatial-nnue/README.md)
contain the model format, row-event kernels, training experiments and recorded
measurements. The supplied surrogate-trained weights remain experimental;
the new engine defaults to a heuristic evaluation when no model is supplied.

## Original Cold Clear

You can play against it [in your browser](https://minuskelvin.net/cold-clear)
or [on your desktop](https://github.com/MinusKelvin/cold-clear/releases).

## Usage

### As a Rust library

```toml
# Cargo.toml

[dependencies]
cold-clear = { git = "https://github.com/MinusKelvin/cold-clear" }
```

### As a C library

Clone the repository and run `cargo build --release -p c-api` to build. You
can find both static and shared libraries in `target/release`, and the API is
described in [`c-api/coldclear.h`](c-api/coldclear.h).

### Running the desktop client from source

Clone the repository and run `cargo run --release` fromt he project directory.

## License

Cold Clear is licensed under [MPLv2](LICENSE).
