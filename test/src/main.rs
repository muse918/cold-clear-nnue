#![feature(test)]
use core::arch::x86_64::{_MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2};
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

    #[bench]
    fn bench_eval_simd_2(b: &mut Bencher) {
        let net = test::black_box(Nnue::with_random());
        let board = test::black_box(Board::new());
        b.iter(|| net.forward_simd_2(&board));
    }
}

fn main() {}
