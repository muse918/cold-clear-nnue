use std::default;

use lazy_static::lazy_static;
use libtetris::*;
use serde::{
    de::{Error, Visitor},
    ser::SerializeSeq,
    Deserialize, Serialize,
};

struct LayerVisitor<const I: usize, const O: usize>;
impl<'de, const I: usize, const O: usize> Visitor<'de> for LayerVisitor<I, O> {
    type Value = ([[f32; O]; I], [f32; O]);
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an valid single precision floating point number")
    }
    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut weights = [[0.; O]; I];
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

const ENCODE_LEN: usize = 45050; // TODO: change later
type VecT = f32;

#[derive(Clone, Debug)]
struct EncodeLayer<const O: usize, const MIN: i32, const MAX: i32> {
    weights: [[VecT; O]; ENCODE_LEN],
    biases: [VecT; O],
}
impl<const O: usize, const MIN: i32, const MAX: i32> Default for EncodeLayer<O, MIN, MAX> {
    fn default() -> Self {
        Self {
            weights: [[0.; O]; ENCODE_LEN],
            biases: [0.; O],
        }
    }
}

impl<const O: usize, const MIN: i32, const MAX: i32> EncodeLayer<O, MIN, MAX> {
    // [0, 10 - M] x [0, 40 - N]
    fn convolute<
        const M: usize,
        const N: usize,
        const ELEVENMINUSM: usize,
        const FOURTYONEMINUSN: usize,
    >(
        field: &[[bool; 10]; 40],
    ) -> [[i32; ELEVENMINUSM]; FOURTYONEMINUSN] {
        // let mut ret = field.map(|x| x.map(|y| y as i32));
        let mut ret = [[0; ELEVENMINUSM]; FOURTYONEMINUSN];
        for row in ret.iter_mut() {
            for i in 0..=10 - M {
                for j in 1..=M {
                    row[i] += row[i + j] << j;
                }
            }
        }
        for j in 0..10 {
            for i in 0..=40 - N {
                for k in 1..=N {
                    ret[i][j] += ret[i + k][j] << (M * k);
                }
            }
        }
        ret
    }
    fn forward(&self, board: &Board) -> [VecT; O] {
        let mut ret = [0.; O];
        let field = board.get_field();
        let conv_3x4 = Self::convolute::<3, 4, 8, 37>(&field);
        for (i, v) in conv_3x4.iter().flatten().cloned().enumerate() {
            for j in 0..O {
                ret[j] += self.weights[(i << 12) + v as usize][j];
            }
        }
        let conv_1x10 = Self::convolute::<1, 10, 10, 31>(&field);
        for (i, v) in conv_1x10.iter().flatten().cloned().enumerate() {
            for j in 0..O {
                ret[j] += self.weights[((8 * 37) << 12) + (i << 10) + v as usize][j];
            }
        }

        for (v, bias) in ret.iter_mut().zip(self.biases) {
            *v = (*v + bias).clamp(MIN as VecT, MAX as VecT);
        }

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
    weights: [[VecT; O]; I],
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
            weights: [[0.; O]; I],
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
            *x = (*x + bias).clamp(MIN as VecT, MAX as VecT);
        }
        ret
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Nnue {
    layer1: EncodeLayer<128, 0, 1>, // encode & do first matmul
    linear1: LinearClampLayer<128, 64, 0, 1>,
    linear2: LinearClampLayer<64, 32, 0, 1>,
    linear3: LinearClampLayer<32, 2, 0, 1>,
}

impl Nnue {
    // (board, placement) values
    fn forward(&self, board: &Board) -> (f32, f32) {
        let l1 = self.layer1.forward(board);
        let l2 = self.linear1.forward(l1);
        let l3 = self.linear2.forward(l2);
        let l4 = self.linear3.forward(l3);
        (l4[0], l4[1])
    }
}

lazy_static! {
    static ref MODEL: Nnue = Default::default();
}

use crate::evaluation::Evaluator;

use super::standard::{Reward, Value};
impl Evaluator for Nnue {
    type Value = Value;
    type Reward = Reward;
    fn name(&self) -> String {
        "Nnue".to_string()
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
        let value = MODEL.forward(board);
        (
            Value {
                value: value.0 as i32,
                spike: 0,
            },
            Reward {
                value: value.1 as i32,
                attack: if lock.placement_kind.is_clear() {
                    lock.garbage_sent as i32
                } else {
                    -1
                },
            },
        )
    }
}
