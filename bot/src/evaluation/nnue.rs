use std::arch::x86_64::*;
use std::{default, fs, os};

use lazy_static::lazy_static;
use libtetris::*;
use serde::de::{Error, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};

const STANDARD: Standard = Standard {
    back_to_back: 52,
    bumpiness: -24,
    bumpiness_sq: -7,
    row_transitions: -5,
    height: -39,
    top_half: -150,
    top_quarter: -511,
    jeopardy: -11,
    cavity_cells: -173,
    cavity_cells_sq: -3,
    overhang_cells: -34,
    overhang_cells_sq: -1,
    covered_cells: -17,
    covered_cells_sq: -1,
    tslot: [8, 148, 192, 407],
    well_depth: 57,
    max_well_depth: 17,
    well_column: [20, 23, 20, 50, 59, 21, 59, 10, -10, 24],

    move_time: -3,
    wasted_t: -152,
    b2b_clear: 104,
    clear1: -143,
    clear2: -100,
    clear3: -58,
    clear4: 390,
    tspin1: 121,
    tspin2: 410,
    tspin3: 602,
    mini_tspin1: -158,
    mini_tspin2: -93,
    perfect_clear: 999,
    combo_garbage: 150,

    use_bag: true,
    timed_jeopardy: true,
    stack_pc_damage: false,
    sub_name: None,
};

struct LayerVisitor<const I: usize, const O: usize>;
impl<'de, const I: usize, const O: usize> Visitor<'de> for LayerVisitor<I, O> {
    type Value = (Vec<[f32; O]>, [f32; O]);
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an valid single precision floating point number")
    }
    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut weights = vec![[0.; O]; I];
        let mut biases = [0.; O];
        let mut encountered = 0;
        for weight in weights.iter_mut().flatten() {
            match seq.next_element()? {
                Some(v) => *weight = v,
                None => return Err(Error::invalid_length(encountered, &self)),
            }
            encountered += 1;
        }
        for bias in biases.iter_mut() {
            match seq.next_element()? {
                Some(v) => *bias = v,
                None => return Err(Error::invalid_length(encountered, &self)),
            }
            encountered += 1;
        }

        Ok((weights, biases))
    }
}

const ENCODE_LEN: usize = ((8 * 37) << 12) + ((1 * 40) << 10) + 7 + 1 + 20 + 7 * 5 + 7; // TODO: change later
                                                                                        // const ENCODE_LEN: usize = 0;
type VecT = f32;

#[derive(Clone, Debug)]
pub struct EncodeLayer<const O: usize, const MIN: i32, const MAX: i32> {
    weights: Vec<[VecT; O]>,
    biases: [VecT; O],
}
impl<const O: usize, const MIN: i32, const MAX: i32> Default for EncodeLayer<O, MIN, MAX> {
    fn default() -> Self {
        Self {
            weights: vec![[0.; O]; ENCODE_LEN],
            biases: [0.; O],
        }
    }
}

impl<const O: usize, const MIN: i32, const MAX: i32> EncodeLayer<O, MIN, MAX> {
    // [0, 10 - M] x [0, 40 - N]
    fn convolute<const M: usize, const N: usize>(field: &[[bool; 10]; 40]) -> [[i32; 10]; 40] {
        let mut ret = field.map(|x| x.map(|y| y as i32));
        for row in ret.iter_mut() {
            for i in 0..=10 - M {
                for j in 1..M {
                    row[i] += row[i + j] << j;
                }
            }
        }
        for j in 0..10 {
            for i in 0..=40 - N {
                for k in 1..N {
                    ret[i][j] += ret[i + k][j] << (M * k);
                }
            }
        }
        ret
    }
    fn forward(&self, board: &Board) -> [VecT; O] {
        let mut ret = [0.; O];
        let field = board.get_field();
        let conv_3x4 = Self::convolute::<3, 4>(&field);
        for (i, v) in conv_3x4
            .iter()
            .take(41 - 4)
            .flat_map(|x| x.iter().take(11 - 3))
            .cloned()
            .enumerate()
        {
            for j in 0..O {
                ret[j] += self.weights[(i << 12) + v as usize][j];
            }
        }
        let conv_10x1 = Self::convolute::<10, 1>(&field);
        for (i, v) in conv_10x1
            .iter()
            .take(41 - 1)
            .flat_map(|x| x.iter().take(11 - 10))
            .cloned()
            .enumerate()
        {
            for j in 0..O {
                ret[j] += self.weights[((8 * 37) << 12) + (i << 10) + v as usize][j];
            }
        }
        //TO-DO: 큐, 기타 인코딩
        for v in board.bag.iter().map(|x| match x {
            Piece::I => 0,
            Piece::O => 1,
            Piece::T => 2,
            Piece::L => 3,
            Piece::J => 4,
            Piece::S => 5,
            Piece::Z => 6,
        }) {
            for j in 0..O {
                ret[j] += self.weights[((8 * 37) << 12) + (1 * 40 << 10) + v as usize][j];
            }
        }
        if board.b2b_bonus {
            for j in 0..O {
                ret[j] += self.weights[((8 * 37) << 12) + (1 * 40 << 10) + 7][j];
            }
        }
        for j in 0..O {
            ret[j] += self.weights
                [((8 * 37) << 12) + (1 * 40 << 10) + 7 + 1 + (board.combo).min(19) as usize][j];
        }
        for x in board.next_pieces.iter().take(5).enumerate() {
            let idx = match x.1 {
                Piece::I => 0,
                Piece::O => 1,
                Piece::T => 2,
                Piece::L => 3,
                Piece::J => 4,
                Piece::S => 5,
                Piece::Z => 6,
            } + 7 * x.0;
            for j in 0..O {
                ret[j] += self.weights[((8 * 37) << 12) + (1 * 40 << 10) + 7 + 1 + 20 + idx][j];
            }
        }
        if let Some(hold) = board.hold_piece {
            let idx = match hold {
                Piece::I => 0,
                Piece::O => 1,
                Piece::T => 2,
                Piece::L => 3,
                Piece::J => 4,
                Piece::S => 5,
                Piece::Z => 6,
            };
            for j in 0..O {
                ret[j] +=
                    self.weights[((8 * 37) << 12) + (1 * 40 << 10) + 7 + 1 + 20 + 7 * 5 + idx][j];
            }
        }

        for (v, bias) in ret.iter_mut().zip(self.biases) {
            *v = (*v + bias).max(0.);
        }

        ret
    }
    fn with_random() -> Self {
        let mut x = Self::default();
        let mut state: i64 = 0x42;
        for v in x.weights.iter_mut().flatten().chain(x.biases.iter_mut()) {
            state ^= state.wrapping_shr(3);
            state ^= state.wrapping_shl(3);
            *v = 1e15 / (state as f32);
        }
        x
    }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_simd(&self, board: &Board) -> [__m256; O / 8] {
        let mut ret = [_mm256_setzero_ps(); O / 8];
        macro_rules! add_ret {
            ($(($source:expr, $index:expr)),*) => {
                $(
                for j in 0..O / 8 {
                    ret[j] = _mm256_add_ps(ret[j], _mm256_loadu_ps(&$source[$index][8 * j]))
                }
                )*
            };
            ($($index:expr),*) => {
                add_ret!($((&self.weights, $index)),*)
            };
            ($source:expr; clamp) => {
                for j in 0..O / 8 {
                    ret[j] = _mm256_max_ps(_mm256_add_ps(
                        ret[j],
                        _mm256_loadu_ps(&$source[8 * j]),
                    ), _mm256_setzero_ps())
                }
            };
        }
        let field = board.get_field();
        let conv_3x4 = Self::convolute::<3, 4>(&field);
        for (i, v) in conv_3x4
            .iter()
            .take(41 - 4)
            .flat_map(|x| x.iter().take(11 - 3))
            .cloned()
            .enumerate()
        {
            add_ret!((i << 12) + v as usize);
        }
        let conv_10x1 = Self::convolute::<10, 1>(&field);
        for (i, v) in conv_10x1
            .iter()
            .take(41 - 1)
            .flat_map(|x| x.iter().take(11 - 10))
            .cloned()
            .enumerate()
        {
            add_ret!(((8 * 37) << 12) + (i << 10) + v as usize);
        }
        for v in board.bag.iter().map(|x| match x {
            Piece::I => 0,
            Piece::O => 1,
            Piece::T => 2,
            Piece::L => 3,
            Piece::J => 4,
            Piece::S => 5,
            Piece::Z => 6,
        }) {
            add_ret!(((8 * 37) << 12) + (1 * 40 << 10) + v as usize);
        }
        if board.b2b_bonus {
            add_ret!(((8 * 37) << 12) + (1 * 40 << 10) + 7);
        }
        add_ret!(((8 * 37) << 12) + (1 * 40 << 10) + 7 + 1 + (board.combo).min(19) as usize);
        for x in board.next_pieces.iter().take(5).enumerate() {
            let idx = match x.1 {
                Piece::I => 0,
                Piece::O => 1,
                Piece::T => 2,
                Piece::L => 3,
                Piece::J => 4,
                Piece::S => 5,
                Piece::Z => 6,
            } + 7 * x.0;
            add_ret!(((8 * 37) << 12) + (1 * 40 << 10) + 7 + 1 + 20 + idx);
        }
        if let Some(hold) = board.hold_piece {
            let idx = match hold {
                Piece::I => 0,
                Piece::O => 1,
                Piece::T => 2,
                Piece::L => 3,
                Piece::J => 4,
                Piece::S => 5,
                Piece::Z => 6,
            };
            add_ret!(((8 * 37) << 12) + (1 * 40 << 10) + 7 + 1 + 20 + 7 * 5 + idx);
        }
        // add bias
        add_ret!(self.biases; clamp);
        ret
    }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_simd_2<const STRATEGY: i32>(&self, board: &Board) -> [__m256; O / 8] {
        let mut ret = [_mm256_setzero_ps(); O / 8];
        macro_rules! simd_helper {
            ($(($source:expr, $index:expr)),*) => {
                $(
                for j in 0..O / 8 {
                    ret[j] = _mm256_add_ps(ret[j], _mm256_loadu_ps(&$source[$index][8 * j]))
                }
                )*
            };
            ($($index:expr),*) => {
                simd_helper!($((&self.weights, $index)),*)
            };
            ($source:expr; clamp) => {
                for j in 0..O / 8 {
                    ret[j] = _mm256_max_ps(_mm256_add_ps(
                        ret[j],
                        _mm256_loadu_ps(&$source[8 * j]),
                    ), _mm256_setzero_ps())
                }
            };
            ($source:expr; prefetch) => {
                for j in 0..O / 16 {
                    _mm_prefetch::<STRATEGY>(
                        &$source[16 * j] as *const f32 as *const i8,
                    );
                }
            }
        }
        let field = board.get_field();
        let conv_3x4 = Self::convolute::<3, 4>(&field);
        let mut it = conv_3x4
            .iter()
            .take(41 - 4)
            .flat_map(|x| x.iter().take(11 - 3))
            .cloned();
        let mut i = 0;
        let mut cur = it.next().unwrap();
        for _ in 0..(41 - 4) * (11 - 3) - 1 {
            let next = it.next().unwrap();
            simd_helper!(self.weights[4096 + (i << 12) + next as usize]; prefetch);
            simd_helper!((i << 12) + cur as usize);
            cur = next;
            i += 1;
        }
        simd_helper!((i << 12) + cur as usize);

        let conv_10x1 = Self::convolute::<10, 1>(&field);
        let mut it = conv_10x1
            .iter()
            .take(41 - 1)
            .flat_map(|x| x.iter().take(11 - 10))
            .cloned();
        let mut i = 0;
        let mut cur = it.next().unwrap();
        for _ in 0..(41 - 1) * (11 - 10) - 2 {
            let next = it.next().unwrap();
            simd_helper!(self.weights[((8 * 37) << 12) + 1024 + (i << 10) + next as usize]; prefetch);
            simd_helper!(((8 * 37) << 12) + (i << 10) + cur as usize);
            cur = next;
            i += 1;
        }
        simd_helper!(((8 * 37) << 12) + (i << 10) + cur as usize);

        for v in board.bag.iter().map(|x| match x {
            Piece::I => 0,
            Piece::O => 1,
            Piece::T => 2,
            Piece::L => 3,
            Piece::J => 4,
            Piece::S => 5,
            Piece::Z => 6,
        }) {
            simd_helper!(((8 * 37) << 12) + (1 * 40 << 10) + v as usize);
        }
        if board.b2b_bonus {
            simd_helper!(((8 * 37) << 12) + (1 * 40 << 10) + 7);
        }
        simd_helper!(((8 * 37) << 12) + (1 * 40 << 10) + 7 + 1 + (board.combo).min(19) as usize);
        for x in board.next_pieces.iter().take(5).enumerate() {
            let idx = match x.1 {
                Piece::I => 0,
                Piece::O => 1,
                Piece::T => 2,
                Piece::L => 3,
                Piece::J => 4,
                Piece::S => 5,
                Piece::Z => 6,
            } + 7 * x.0;
            simd_helper!(((8 * 37) << 12) + (1 * 40 << 10) + 7 + 1 + 20 + idx);
        }
        if let Some(hold) = board.hold_piece {
            let idx = match hold {
                Piece::I => 0,
                Piece::O => 1,
                Piece::T => 2,
                Piece::L => 3,
                Piece::J => 4,
                Piece::S => 5,
                Piece::Z => 6,
            };
            simd_helper!(((8 * 37) << 12) + (1 * 40 << 10) + 7 + 1 + 20 + 7 * 5 + idx);
        }
        // add bias
        simd_helper!(self.biases; clamp);
        ret
    }
}

impl<const O: usize, const MIN: i32, const MAX: i32> Serialize for EncodeLayer<O, MIN, MAX> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(O * ENCODE_LEN + O))?;
        for weight in self.weights.iter().flatten() {
            seq.serialize_element(weight)?;
        }
        for bias in self.biases.iter() {
            seq.serialize_element(bias)?;
        }
        seq.end()
    }
}

impl<'de, const O: usize, const MIN: i32, const MAX: i32> Deserialize<'de>
    for EncodeLayer<O, MIN, MAX>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = deserializer.deserialize_seq(LayerVisitor::<ENCODE_LEN, O>)?;
        Ok(EncodeLayer {
            weights: v.0,
            biases: v.1,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct LinearClampLayer<const I: usize, const O: usize, const MIN: i32, const MAX: i32> {
    weights: Vec<[VecT; O]>,
    biases: [VecT; O],
}

impl<const I: usize, const O: usize, const MIN: i32, const MAX: i32> Serialize
    for LinearClampLayer<I, O, MIN, MAX>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_seq(Some(I * O + O))?;
        for weight in self.weights.iter().flatten() {
            map.serialize_element(weight)?;
        }
        for bias in self.biases {
            map.serialize_element(&bias)?;
        }
        map.end()
    }
}

impl<'de, const I: usize, const O: usize, const MIN: i32, const MAX: i32> Deserialize<'de>
    for LinearClampLayer<I, O, MIN, MAX>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = deserializer.deserialize_seq(LayerVisitor::<I, O>)?;
        Ok(LinearClampLayer {
            weights: v.0,
            biases: v.1,
        })
    }
}

impl<const I: usize, const O: usize, const MIN: i32, const MAX: i32> Default
    for LinearClampLayer<I, O, MIN, MAX>
{
    fn default() -> Self {
        LinearClampLayer {
            weights: vec![[0.; O]; I],
            biases: [0.; O],
        }
    }
}

impl<const I: usize, const O: usize, const MIN: i32, const MAX: i32>
    LinearClampLayer<I, O, MIN, MAX>
{
    fn forward(&self, inp: [VecT; I]) -> [VecT; O] {
        let mut ret = [0.; O];
        for (i, x) in inp.iter().enumerate() {
            for (j, y) in self.weights[i].iter().enumerate() {
                ret[j] += x * y;
            }
        }
        for (x, bias) in ret.iter_mut().zip(self.biases) {
            // *x = (*x + bias).clamp(MIN as VecT, MAX as VecT);
            *x = (*x + bias).max(0.);
        }
        return ret;
    }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_simd(&self, inp: [__m256; I / 8]) -> [__m256; O / 8] {
        let mut ret = [_mm256_setzero_ps(); O / 8];
        for i in 0..O / 8 {
            ret[i] = _mm256_loadu_ps(&self.biases[8 * i]);
        }
        for i in 0..I / 8 {
            let mut tmp = [0_f32; 8];
            _mm256_storeu_ps(&mut tmp as *mut f32, inp[i]);
            for (j, &x) in tmp.iter().enumerate() {
                for k in 0..O / 8 {
                    ret[k] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(&self.weights[8 * i + j][8 * k]),
                        _mm256_set1_ps(x),
                        ret[k],
                    );
                }
            }
        }
        ret
    }
    fn forward_non_clamp(&self, inp: [VecT; I]) -> [VecT; O] {
        let mut ret = [0.; O];
        for (i, x) in inp.iter().enumerate() {
            for (j, y) in self.weights[i].iter().enumerate() {
                ret[j] += x * y;
            }
        }
        for (x, bias) in ret.iter_mut().zip(self.biases) {
            *x += bias
        }
        ret
    }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_reduce(&self, inp: [__m256; I / 8]) -> f32 {
        macro_rules! load {
            ($source:expr, $base:expr, $($off:tt),*) => {
                _mm256_set_ps($($source[$base + $off][0]),*)
            };
        }
        let mut ret_m256i = _mm256_setzero_ps();
        for i in 0..I / 8 {
            let weight = load!(self.weights, 8 * i, 0, 1, 2, 3, 4, 5, 6, 7);
            ret_m256i = _mm256_fmadd_ps(weight, inp[i], ret_m256i);
        }
        let hi = _mm256_extractf128_ps::<1>(ret_m256i);
        let lo = _mm256_castps256_ps128(ret_m256i);
        let s1 = _mm_add_ps(hi, lo);
        let s2 = _mm_movehl_ps(s1, s1);
        let s3 = _mm_add_ps(s1, s2);
        let hi = _mm_shuffle_ps::<1>(s3, s3);
        let lo = s3;
        let sum = _mm_add_ss(lo, hi);
        _mm_cvtss_f32(sum) + self.biases[0]
    }
    fn with_random() -> Self {
        let mut x = Self::default();
        let mut state: i64 = 0x42;
        for v in x.weights.iter_mut().flatten().chain(x.biases.iter_mut()) {
            state ^= state.wrapping_shr(3);
            state ^= state.wrapping_shl(3);
            *v = 1e15 / state as f32;
        }
        x
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Nnue {
    encode: EncodeLayer<128, 0, 1>, // encode & do first matmul
    linear1: LinearClampLayer<128, 64, 0, 1>,
    linear2: LinearClampLayer<64, 32, 0, 1>,
    linear3: LinearClampLayer<32, 1, -1, 1>,
}

impl Nnue {
    // (board, placement) values
    pub fn forward(&self, board: &Board) -> f32 {
        let l1 = self.encode.forward(board);
        let l2 = self.linear1.forward(l1);
        let l3 = self.linear2.forward(l2);
        let l4 = self.linear3.forward_non_clamp(l3);
        l4[0]
    }
    pub fn forward_simd(&self, board: &Board) -> f32 {
        unsafe {
            let l1 = self.encode.forward_simd(board);
            let l2 = self.linear1.forward_simd(l1);
            let l3 = self.linear2.forward_simd(l2);
            self.linear3.forward_reduce(l3)
        }
    }
    pub fn forward_simd_2(&self, board: &Board) -> f32 {
        unsafe {
            let l1 = self.encode.forward_simd_2::<_MM_HINT_T0>(board);
            let l2 = self.linear1.forward_simd(l1);
            let l3 = self.linear2.forward_simd(l2);
            self.linear3.forward_reduce(l3)
        }
    }
    pub fn with_random() -> Self {
        Self {
            encode: EncodeLayer::with_random(),
            linear1: LinearClampLayer::with_random(),
            linear2: LinearClampLayer::with_random(),
            linear3: LinearClampLayer::with_random(),
        }
    }
}

fn read_nnue(dir: String) -> Nnue {
    let nnue_str = fs::read_to_string(dir).expect("failed to read nnue");
    serde_json::from_str(&nnue_str).unwrap()
}

lazy_static! {
    static ref MODEL: Nnue = read_nnue(
        "D:\\muse918\\cold-clear-nnue-train\\src\\net\\net_20231102_trained.json".to_string()
    );
}

use super::standard::{Reward, Value};
use super::Standard;
use crate::evaluation::Evaluator;

#[derive(Debug, Default, Serialize, Deserialize, Copy, Clone)]
pub struct NnueEvaluator;

impl Evaluator for NnueEvaluator {
    type Value = Value;
    type Reward = Reward;
    fn name(&self) -> String {
        "NNUE".to_string()
    }

    fn pick_move(
        &self,
        candidates: Vec<crate::dag::MoveCandidate<Self::Value>>,
        incoming: u32,
    ) -> crate::dag::MoveCandidate<Self::Value> {
        let mut backup = None;
        for mv in candidates.into_iter() {
            if incoming == 0
                || mv.board.column_heights()[3..6]
                    .iter()
                    .all(|h| incoming as i32 - mv.lock.garbage_sent as i32 + h <= 20)
            {
                return mv;
            }

            match backup {
                None => backup = Some(mv),
                Some(c) if c.evaluation.spike < mv.evaluation.spike => backup = Some(mv),
                _ => {}
            }
        }

        return backup.unwrap();
    }
    fn evaluate(
        &self,
        lock: &LockResult,
        board: &Board,
        move_time: u32,
        placed: Piece,
    ) -> (Self::Value, Self::Reward) {
        let mut acc_eval = 0;

        if lock.perfect_clear {
            acc_eval += STANDARD.perfect_clear;
        }
        if STANDARD.stack_pc_damage || !lock.perfect_clear {
            if lock.b2b {
                acc_eval += STANDARD.b2b_clear;
            }
            if let Some(combo) = lock.combo {
                let combo = combo.min(11) as usize;
                acc_eval += STANDARD.combo_garbage * libtetris::COMBO_GARBAGE[combo] as i32;
            }
            match lock.placement_kind {
                PlacementKind::Clear1 => {
                    acc_eval += STANDARD.clear1;
                }
                PlacementKind::Clear2 => {
                    acc_eval += STANDARD.clear2;
                }
                PlacementKind::Clear3 => {
                    acc_eval += STANDARD.clear3;
                }
                PlacementKind::Clear4 => {
                    acc_eval += STANDARD.clear4;
                }
                PlacementKind::Tspin1 => {
                    acc_eval += STANDARD.tspin1;
                }
                PlacementKind::Tspin2 => {
                    acc_eval += STANDARD.tspin2;
                }
                PlacementKind::Tspin3 => {
                    acc_eval += STANDARD.tspin3;
                }
                PlacementKind::MiniTspin1 => {
                    acc_eval += STANDARD.mini_tspin1;
                }
                PlacementKind::MiniTspin2 => {
                    acc_eval += STANDARD.mini_tspin2;
                }
                _ => {}
            }
        }

        if placed == Piece::T {
            match lock.placement_kind {
                PlacementKind::Tspin1 | PlacementKind::Tspin2 | PlacementKind::Tspin3 => {}
                _ => acc_eval += STANDARD.wasted_t,
            }
        }

        // magic approximation of line clear delay
        let move_time = if lock.placement_kind.is_clear() {
            (move_time / 10) as i32 + 40
        } else {
            (move_time / 10) as i32
        };
        acc_eval += STANDARD.move_time * move_time;

        let highest_point = *board.column_heights().iter().max().unwrap() as i32;

        acc_eval += STANDARD.jeopardy
            * (highest_point - 10).max(0)
            * if STANDARD.timed_jeopardy {
                move_time
            } else {
                10
            }
            / 10;

        let value = MODEL.forward_simd(board);
        (
            Value {
                value: (value) as i32,
                spike: 0,
            },
            Reward {
                value: acc_eval as i32,
                attack: if lock.placement_kind.is_clear() {
                    lock.garbage_sent as i32
                } else {
                    -1
                },
            },
        )
    }
}
