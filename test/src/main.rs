use core::time;
use std::thread;

use cold_clear::evaluation::nnue::*;
use serde::*;
use std::fs;

fn read_nnue(dir: String) -> Nnue {
    let nnue_str = fs::read_to_string(dir).expect("failed to read nnue");
    serde_json::from_str(&nnue_str).unwrap()
    // serde_json::from_str(&nnue_str).unwrap()
}

fn main() {
    let model = Nnue::default();
    fs::write("D:\\muse918\\cold-clear-nnue-train\\src\\net\\net_tmp.json", serde_json::to_string(&model).unwrap()).unwrap();
    read_nnue("D:\\muse918\\cold-clear-nnue-train\\src\\net\\net_tmp.json".to_string());
    // read_nnue("D:\\muse918\\cold-clear-nnue-train\\src\\net\\net_20231031.json".to_string());
}
