use core::time;
use std::thread;

use cold_clear::evaluation::nnue::*;
use serde::*;

fn main() {
    println!("Hello!");
    let x = Nnue::with_random(); // 613mb
    println!("begin serialization");
    let y = serde_json::to_string(&x).unwrap();
    println!("{}", y.len()); // 641770680 for default, 864320177 for with_random
                             // println!("{}", y.chars().take(50).collect::<String>());
    println!("{}", y);
}
