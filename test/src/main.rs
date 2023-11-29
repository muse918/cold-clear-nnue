#![feature(test)]
use core::time;
use std::{fs, thread};

use cold_clear::evaluation::nnue::*;
use libtetris::Board;
use serde::*;

extern crate test;

#[cfg(test)]
mod tests {
    use test::Bencher;

    use super::*;

    #[bench]
    fn bench_eval(b: &mut Bencher) {
        let net = test::black_box(Nnue::with_random());
        let board = test::black_box(Board::new());
        b.iter(|| net.forward(&board));
    }

    #[bench]
    fn bench_eval_simd(b: &mut Bencher) {
        let net = test::black_box(Nnue::with_random());
        let board = test::black_box(Board::new());
        b.iter(|| net.forward_simd(&board));
    }
}

fn main() {}
